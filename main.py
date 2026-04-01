from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.trace import Tracer
from typing import List 
import asyncio
import hashlib

# Obtain a tracer instance from open telemetry
# All spans created with this tracer will be the part of the same 
# distributed tracing system and will be exported to the configured backend
tracer: Tracer = trace.get_tracer(__name__)

# Initialise the fastapi application
app = FastAPI()

# Helper function used by the observable endpoint
async def retrieve_document(query: str) -> List[str]:
    """ 
        Simulate document retrieval (e.g. vector search or knowledge base lookup)
        This lookup function represents the retrieval stage in a RAG pipeline
        In real systems, this might query a vector database 
    """
    await asyncio.sleep(0.5)
    return [
        "FastAPI enables high-performance async APIs.",
        "OpenTelemetry provides vendor-neutral observability.",
        "LLM observability requires tracing prompts and tokens.",
    ]
    
def build_prompt(query: str, documents: List[str]) -> str:
    """
        Construct the final prompt from the retrieved documents and the user query
        Prompt construction is kept seperate so it can be observed or modified 
        independently if needed (for example, to measure prompt assembly latency)
    """
    context = "\n".join(documents)
    return f"Context: {context}, Question: {query}"

class LLMResponse:
    """ 
        Minimal abstractions for an LLM response.
        This keeps the example self-contained while still allowing us to attach 
        token usage, and other metadata for observability
    """
    
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
    
    
async def call_llm(prompt: str) -> LLMResponse:
    """ 
        Simulate an LLM API call.
        In a real implementation this would call OpenAI, Anthropic, or another
        provider. The artifical delay represents model latency
    """
    await asyncio.sleep(2)
    response_text = "FastAPI and OpenTelemetry enable end-to-end LLM observability."
    # Token count is approximated here for demonstration purposes.

    prompt_tokens = len(prompt.split())
    completion_tokens = len(response_text.split())
    
    return LLMResponse(prompt, prompt_tokens, completion_tokens)

def summarise_response(response: LLMResponse) -> str:
    """
    Example post-processing step.
    Post-processing is separated into its own phase so any additional latency
    or errors are not incorrectly attributed to the LLM itself.
    """
    return response.text

def summarize_response(response: LLMResponse) -> str:
    """
    Example post-processing step.
    Post-processing is separated into its own phase so any additional latency
    or errors are not incorrectly attributed to the LLM itself.
    """
    return response.text

# Observable endpoint
@app.post("/query")
async def rag_query(request: Request, query: str):
    """
        Handles a single RAG-style request with explicit opentelemetry spans.
        This endpoint demonstrates how to create one trace per request with child
        spans for retrieval, LLM invocation, and post processing
    """
    
    # Create a top-level span for the HTTP request
    # Even if fastapi auto-instrumentation is enabled, defining this explicitly 
    # allows us to attach domain-specific metadata
    with tracer.start_as_current_span("http.request") as http_span:
        http_span.set_attribute("http.method", "POST") 
        http_span.set_attribute("http.route", "/query")
        
        # Retrieval phase
        # This span isolates the retrieval step so that the relevance issues can be 
        # debugged independently of LLM behaviour
        with tracer.start_as_current_span("rag.retrieval") as retrieval_span:
            retrieval_span.set_attribute("rag.top_k", 5)
            retrieval_span.set_attribute("rag.similarity_threshold", 0.8)
            documents = await retrieve_document(query)
            
            # Record how many documents were returned
            # This is a key signal when diagnosing hallucinations
            # or missing context in the final response
            
            retrieval_span.set_attribute(
                "rag.documents_returned",
                len(documents)
            )
            
        # LLM invocation phase 
        # This spans wraps the actual LLM call and is the primary anchor for 
        # latency, cost and prompt related analysis
        with tracer.start_as_current_span("llm.call") as llm_span:
            llm_span.set_attribute("llm.provider", "example")
            llm_span.set_attribute("llm.model", "example=llm")
            llm_span.set_attribute("llm.temperature", 0.7)
            llm_span.set_attribute("llm.prompt_template_id", "rag_v1")

            # Build the final prompt using retrieved context 
            # The raw prompt is intentionally not stored as a span attribute
            prompt = build_prompt(query, documents)
            
            # Prompt metadata
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            llm_span.set_attribute("llm.prompt_hash", prompt_hash)
            llm_span.set_attribute("llm.prompt_length", len(prompt))

            response = await call_llm(prompt)

            # Hash the response, instead of storing raw texts
            # This allows coorelation across traces without exposing content
            response_hash = hashlib.sha256(
                response.text.encode()
            ).hexdigest()
            llm_span.set_attribute("llm.response_hash", response_hash)
            
            # Record token usage to enable cost attribution 
            # and capacity planning
            llm_span.set_attribute("llm.usage.prompt_tokens", response.prompt_tokens)
            llm_span.set_attribute("llm.usage.completion_tokens", response.completion_tokens)
            llm_span.set_attribute("llm_usage.total_tokens", response.total_tokens)
            
            # example price per token
            estimated_cost = response.total_tokens * 0.000002
            llm_span.set_attribute("llm.cost_estimated_usd", estimated_cost)
            
            
        # Post-processing phase
        # Any transformation after the LLM response is captured here,
        # ensuring inference latency is not overstated.
        with tracer.start_as_current_span("llm.postprocess") as post_span:
            summary = summarize_response(response)
            post_span.set_attribute(
                "llm.summary_length",
                len(summary),
            )

    # Return the final response to the client.
    # All spans above belong to the same distributed trace.
    return {"summary": summary}