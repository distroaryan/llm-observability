# Harry Potter RAG + LLM Observability

Building a RAG application as a side project is one thing, but taking it to production is a whole different beast. If the app starts crashing, burning through tokens, or spitting out vague responses, you need a way to trace through the errors and figure out what went wrong.

That's what this repo is about: building an end-to-end LLM observability system.

### Why is this hard?
Traditional systems are deterministic. LLMs are probabilistic. They just act differently. To actually observe a RAG model, we need to borrow concepts from distributed tracing and separate our concerns.

Think about the distinct steps in a RAG pipeline:
1. Input Validation
2. Document retrieval
3. Prompt construction
4. Model execution
5. Response handling

Because these are all distinct steps, if we separate them out, the whole thing becomes much easier to test, observe, and evolve. This aligns perfectly with distributed tracing concepts—each block cleanly maps to a trace span and can be debugged independently. 

At the end of the day, the goal is the same. In traditional distributed systems, we trace a user's request across multiple microservices. Here, we trace the request across each stage of the LLM pipeline. Same principles, different implementation.

### What's in this repo?
I built a simple Harry Potter RAG pipeline for this. It pulls data from a Harry Potter PDF and gives back an answer. 

We trace the user's request across 1 major span and 3 child spans like this:

```text
http.request
 ├── rag.retrieval
 ├── llm.call
 └── llm.postprocess
```
Tech Stack Used: Python, FastAPI, Opentelemetry, langchain

### Read more
This project was inspired by / based on this great article:
[Build End-to-End LLM Observability in FastAPI with OpenTelemetry](https://www.freecodecamp.org/news/build-end-to-end-llm-observability-in-fastapi-with-opentelemetry/)
