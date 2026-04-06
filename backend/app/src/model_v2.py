import os
import math
import re
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


class AdvancedRAGPipline:
    VALID_DOC_TYPES = {"td", "tp", "course", "exam"}

    def __init__(self, model_name , vector_store, embeddings_manager):
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.history = []
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required.")
        
        self.llm = ChatGroq(
            model = self.model_name or "llama-3.3-70b-versatile",
            api_key = self.api_key,
            temperature= 0.3,
            max_tokens= 2048
        )
        
        print(f"Initialized GROQ LLM with model: {self.model_name}")

    @staticmethod
    def _tokenize(text):
        return set(re.findall(r"\w+", (text or "").lower()))

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_metadata_preferences(self, metadata, metadata_confidence):
        metadata = metadata or {}
        metadata_confidence = metadata_confidence or {}
        preferences = {}

        for field, value in metadata.items():
            if value in (None, ""):
                continue

            confidence = self._safe_float(metadata_confidence.get(field), default=0.0)
            normalized_value = str(value).strip()
            normalized_field = field

            if field == "doc_type":
                lowered = normalized_value.lower()
                if lowered in self.VALID_DOC_TYPES:
                    normalized_value = lowered
                elif "." in lowered:
                    normalized_field = "source_file"
                else:
                    normalized_field = "source_stem"

            preferences[normalized_field] = {
                "value": normalized_value,
                "confidence": confidence,
                "strict": confidence >= 0.75,
            }

        return preferences

    def _build_chroma_filter(self, metadata_preferences):
        strict_filters = {}
        for field, payload in metadata_preferences.items():
            value = payload["value"]
            if not payload["strict"]:
                continue
            if field in {"sheet_number", "question_number", "doc_type"}:
                strict_filters[field] = value.lower()
            elif field == "source_file":
                strict_filters["source_file_lower"] = value.lower()
            elif field == "source_stem":
                strict_filters["source_stem_lower"] = value.lower()
        return strict_filters or None

    def _metadata_match_score(self, doc_metadata, metadata_preferences):
        if not metadata_preferences:
            return 0.0

        normalized_source = " ".join([
            str(doc_metadata.get("source_file_lower", "")),
            str(doc_metadata.get("source_stem_lower", "")),
        ]).strip()

        score = 0.0
        matched = 0
        for field, payload in metadata_preferences.items():
            expected = payload["value"].lower()
            actual = str(doc_metadata.get(field, "")).lower()
            confidence = payload["confidence"]

            is_match = actual == expected
            if field in {"source_file", "source_stem"}:
                is_match = expected in normalized_source
            if not is_match and field in {"doc_type", "course"} and expected in normalized_source:
                is_match = True

            if is_match:
                matched += 1
                score += 0.5 + (confidence * 0.5)

        if matched == 0:
            return 0.0

        return min(score / max(len(metadata_preferences), 1), 1.0)

    def _keyword_score(self, search_query, document, metadata):
        query_tokens = self._tokenize(search_query)
        if not query_tokens:
            return 0.0

        document_tokens = self._tokenize(document)
        metadata_tokens = self._tokenize(
            f"{metadata.get('source_file', '')} {metadata.get('source_stem', '')} "
            f"{metadata.get('doc_type', '')} {metadata.get('sheet_number', '')} {metadata.get('question_number', '')}"
        )

        document_overlap = len(query_tokens & document_tokens) / len(query_tokens)
        metadata_overlap = len(query_tokens & metadata_tokens) / len(query_tokens)
        return min((document_overlap * 0.8) + (metadata_overlap * 0.2), 1.0)

    def _semantic_score(self, distance):
        if distance is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (distance / 2.0)))

    def _profile_weights(self, search_profile):
        if search_profile == "keyword_heavy":
            return {"semantic": 0.35, "keyword": 0.45, "metadata": 0.20}
        if search_profile == "semantic_heavy":
            return {"semantic": 0.65, "keyword": 0.20, "metadata": 0.15}
        return {"semantic": 0.50, "keyword": 0.25, "metadata": 0.25}
    
    
    def retrieve(self, search_query, top_k, score_threshold, metadata=None, metadata_confidence=None, search_profile="balanced"):
        print(f"Retrieving documents for search_query: '{search_query}'")
        print(f"Top K: {top_k}, score threshold: {score_threshold}, profile: {search_profile}")

        metadata_preferences = self._build_metadata_preferences(metadata, metadata_confidence)
        chroma_filter = self._build_chroma_filter(metadata_preferences)
        print(f"Metadata preferences: {metadata_preferences}")
        print(f"Chroma filter: {chroma_filter}")
        
        query_embeddings = self.embeddings_manager.generate_embeddings([search_query])[0]
        
        try:
            candidate_size = max(top_k * 4, 12)
            results = self.vector_store.collection.query(
                query_embeddings=[query_embeddings.tolist()],
                n_results=candidate_size,
                where=chroma_filter
            )
            initial_candidates = len(results.get("ids", [[]])[0]) if results.get("ids") else 0
            print(f"Initial vector candidates: {initial_candidates}")

            if chroma_filter and initial_candidates == 0:
                print("No candidates matched the strict metadata filter. Retrying without filter.")
                results = self.vector_store.collection.query(
                    query_embeddings=[query_embeddings.tolist()],
                    n_results=candidate_size
                )
                initial_candidates = len(results.get("ids", [[]])[0]) if results.get("ids") else 0
                print(f"Fallback vector candidates: {initial_candidates}")

            ranked_candidates = {}
            if results["documents"] and results["documents"][0]:
                for doc_id, document, doc_metadata, distance in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    ranked_candidates[doc_id] = {
                        "id": doc_id,
                        "content": document,
                        "metadata": doc_metadata or {},
                        "distance": distance,
                    }

            if metadata_preferences:
                metadata_where = chroma_filter if ranked_candidates else None
                for candidate in self.vector_store.get_documents_by_metadata(where=metadata_where, limit=max(top_k * 6, 20)):
                    ranked_candidates.setdefault(candidate["id"], {
                        "id": candidate["id"],
                        "content": candidate["content"],
                        "metadata": candidate["metadata"],
                        "distance": None,
                    })

            print(f"Candidates before reranking: {len(ranked_candidates)}")

            weights = self._profile_weights(search_profile)
            retrieved_docs = []

            for candidate in ranked_candidates.values():
                doc_metadata = candidate["metadata"]
                distance = candidate.get("distance")
                semantic_score = self._semantic_score(distance)
                keyword_score = self._keyword_score(search_query, candidate["content"], doc_metadata)
                metadata_score = self._metadata_match_score(doc_metadata, metadata_preferences)

                hybrid_score = (
                    semantic_score * weights["semantic"] +
                    keyword_score * weights["keyword"] +
                    metadata_score * weights["metadata"]
                )

                passes_distance_gate = distance is None or distance <= score_threshold
                passes_hybrid_gate = hybrid_score >= 0.15
                if not (passes_distance_gate or passes_hybrid_gate or metadata_score >= 0.5):
                    continue

                candidate["semantic_score"] = semantic_score
                candidate["keyword_score"] = keyword_score
                candidate["metadata_score"] = metadata_score
                candidate["similarity_score"] = hybrid_score
                retrieved_docs.append(candidate)

            if retrieved_docs:
                print("Top candidate preview:")
                for candidate in retrieved_docs[:3]:
                    print(
                        f"  - {candidate['metadata'].get('source_file', 'unknown')} "
                        f"(hybrid={candidate['similarity_score']:.3f}, "
                        f"semantic={candidate['semantic_score']:.3f}, "
                        f"keyword={candidate['keyword_score']:.3f}, "
                        f"metadata={candidate['metadata_score']:.3f})"
                    )

            retrieved_docs.sort(
                key=lambda item: (
                    item["similarity_score"],
                    item["metadata_score"],
                    item["keyword_score"],
                    -item["distance"] if item["distance"] is not None else math.inf,
                ),
                reverse=True,
            )

            for rank, candidate in enumerate(retrieved_docs[:top_k], start=1):
                candidate["rank"] = rank

            print(f"Retrieved {len(retrieved_docs[:top_k])} documents after hybrid reranking")
            return retrieved_docs[:top_k]
        except Exception as e:
            print(f"Error During retrieval: {e}")
            return []

    def generate_response(self, generation_query, context):
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful AI assistant. Use the following context, to answer the question accurately and concisely.

            Context: {context}
            
            Question: {question}
            
            Answer: Provide a clear and informative answer based on the context above. if the context doesnt contain enough information to answer the question, please say so.
            """
        )
        
        formatted_prompt = prompt_template.format(context=context, question=generation_query)
        
        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating the response: {str(e)}"
    
    def query(self, query, generation_query, search_query, search_profile, metadata, metadata_confidence):
        results = self.retrieve(
            search_query=search_query,
            top_k=5,
            score_threshold=2.0,
            metadata=metadata,
            metadata_confidence=metadata_confidence,
            search_profile=search_profile,
        )
        if not results:
            answer = "No relevant context found"
            sources = []
        else:
            context = '\n\n'.join([doc['content']  for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unkown')),
                'page': doc['metadata'].get('page', 'unkown'),
                'score': doc['similarity_score'],
                'semantic_score': doc.get('semantic_score', 0.0),
                'keyword_score': doc.get('keyword_score', 0.0),
                'metadata_score': doc.get('metadata_score', 0.0),
                'preview': doc['content']
            } for doc in results]
            
            answer = self.generate_response(generation_query, context)
    
        self.history.append({
            'question': query,
            'answer': answer,
            'sources': sources,
        })
        
        return {
            'question': query,
            'answer': answer,
            'sources': sources,
            'history': self.history
        }
