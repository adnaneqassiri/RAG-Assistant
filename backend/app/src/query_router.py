import json
import re
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    max_tokens=600
)

STRUCTURAL_PATTERNS = [
    r"\bquestion\s*\d+\b",
    r"\bexercice\s*\d+\b",
    r"\bexercise\s*\d+\b",
    r"\btd[-\s]?\d+\b",
    r"\bchapitre\s*\d+\b",
    r"\bchapter\s*\d+\b",
    r"\bsection\s*\d+\b",
    r"\bpart\s*[a-z0-9]+\b",
    r"\bfigure\s*\d+\b",
    r"\btable\s*\d+\b",
]

DOC_HINTS = [
    "document", "pdf", "file", "fichier",
    "cours", "course", "td", "chapter",
    "according to", "based on", "uploaded",
    "ce document", "ce fichier", "this document"
]


def strong_rules_need_retrieval(query: str) -> bool:
    q = query.lower()

    for pattern in STRUCTURAL_PATTERNS:
        if re.search(pattern, q):
            return True

    if any(hint in q for hint in DOC_HINTS):
        return True

    return False


def extract_json_from_response(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("LLM response content is not a string.")

    text = text.strip().lstrip("\ufeff").strip()

    # Remove markdown code fences first
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first complete JSON object using brace counting
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {repr(text[:300])}")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_text = text[start:i+1]
                return json.loads(json_text)

    raise ValueError(f"Unbalanced braces in response: {repr(text[:300])}")
  
def llm_planner(query: str):
    prompt = f"""
You are a retrieval planner for a RAG system.

Return ONLY valid JSON.
Do not use markdown.
Do not wrap the JSON in triple backticks.
Do not add explanations.

Your job:
1. Decide if retrieval is needed
2. Rewrite the query for search (short, keyword optimized)
3. Rewrite the query for answer generation
4. Extract metadata candidates if possible
5. Assign confidence (0 to 1)
6. Choose search profile

SEARCH PROFILES:
- keyword_heavy
- balanced
- semantic_heavy

RULES:
- If the query refers to specific content inside documents, retrieval_needed = true
- Preserve important tokens like numbers, TD, chapter, section, course names
- Do not invent metadata
- If unsure, return empty metadata fields
- If the query contains structured references or numbers, prefer keyword_heavy

RETRIEVAL DECISION RULES:
- retrieval_needed = FALSE if:
  * The query is a general knowledge question (definitions, facts, geography, concepts)
  * Examples: "What is RAG?", "What is the capital of France?", "What is JDBC?"
  * The answer is common knowledge and does not depend on any uploaded document

- retrieval_needed = TRUE only if:
  * The query explicitly references a document, file, course, TD, chapter, or exercise
  * The query asks about specific content that only exists in uploaded documents
  * Examples: "From the SISR document...", "In chapter 3...", "Question 2 of TD1..."

FEW-SHOT EXAMPLES:

Query: "What is the capital of Canada?"
{{"retrieval_needed": false, "search_query": "", "generation_query": "What is the capital of Canada?", "metadata_candidates": {{"course": "", "doc_type": "", "sheet_number": "", "question_number": ""}}, "metadata_confidence": {{"course": 0.0, "doc_type": 0.0, "sheet_number": 0.0, "question_number": 0.0}}, "search_profile": "balanced"}}

Query: "What is RAG?"
{{"retrieval_needed": false, "search_query": "", "generation_query": "What is RAG?", "metadata_candidates": {{"course": "", "doc_type": "", "sheet_number": "", "question_number": ""}}, "metadata_confidence": {{"course": 0.0, "doc_type": 0.0, "sheet_number": 0.0, "question_number": 0.0}}, "search_profile": "balanced"}}

Query: "How is ESRGAN trained? from the SISR document"
{{"retrieval_needed": true, "search_query": "ESRGAN training SISR", "generation_query": "How is ESRGAN trained in the context of SISR?", "metadata_candidates": {{"course": "", "doc_type": "SISR document", "sheet_number": "", "question_number": ""}}, "metadata_confidence": {{"course": 0.0, "doc_type": 0.8, "sheet_number": 0.0, "question_number": 0.0}}, "search_profile": "keyword_heavy"}}

Query: "Explain question 3 of TD2"
{{"retrieval_needed": true, "search_query": "question 3 TD2", "generation_query": "Explain question 3 of TD2", "metadata_candidates": {{"course": "", "doc_type": "TD", "sheet_number": "2", "question_number": "3"}}, "metadata_confidence": {{"course": 0.0, "doc_type": 0.9, "sheet_number": 0.9, "question_number": 0.9}}, "search_profile": "keyword_heavy"}}


Return exactly this JSON schema:
{{
  "retrieval_needed": true,
  "search_query": "...",
  "generation_query": "...",
  "metadata_candidates": {{
    "course": "",
    "doc_type": "",
    "sheet_number": "",
    "question_number": ""
  }},
  "metadata_confidence": {{
    "course": 0.0,
    "doc_type": 0.0,
    "sheet_number": 0.0,
    "question_number": 0.0
  }},
  "search_profile": "balanced"
}}

User query:
{json.dumps(query, ensure_ascii=False)}
"""

    response = llm.invoke(prompt)

    try:
        return extract_json_from_response(response.content)
    except Exception as e:
        print("LLM parsing error:", e)
        print("RAW:", response.content)

        return {
            "retrieval_needed": True,
            "search_query": query,
            "generation_query": query,
            "metadata_candidates": {},
            "metadata_confidence": {},
            "search_profile": "balanced"
        }


def retrieval_router(query: str):
    rule_hit = strong_rules_need_retrieval(query)
    llm_result = llm_planner(query)

    final_retrieval = rule_hit or llm_result.get("retrieval_needed", False)

    llm_result["retrieval_needed"] = final_retrieval
    llm_result["rule_triggered"] = rule_hit
    llm_result["decision_source"] = "rule_override" if rule_hit else "llm"

    return llm_result