# Code Delivery for Module #3 | Local semantic retrieval (RAG)

Async Python pipeline that answers a question from a local Chroma index. Files in `data/` are cleaned, split into token-sized chunks, and embedded once. A LangChain LCEL chain retrieves the closest chunks, asks the model to answer only from that context, and attaches the source filenames itself.

`get_rag_response()` returns a `RAGResponse`:

- `answer`: the model answer. If the context does not contain it, the model says it does not have access to that information.
- `sources`: filenames of the retrieved chunks. These come from document metadata, not from the model.
- `retrieved_fragments`: how many chunks were passed into the prompt (`top_k` 4, kept between 3 and 5).

The chain adds keys as it goes:

1. A lambda on `RunnablePassthrough.assign` stores the retrieved `documents`.
2. A second assign formats those documents into `context`.
3. `build_chain` calls `.with_structured_output(LLMAnswer)` (there is no `{formato}` variable), then checks the payload and retries truncated output.
4. A final `RunnableLambda` builds the `RAGResponse` and its source list.

Indexing and search use the same embedding model (`sentence-transformers/all-MiniLM-L6-v2`). If the collection was built with a different model, the store refuses to open it. A non-empty collection is left as-is on the next run; delete `vectorstore/` to reindex.

The default chat provider is OpenAI (`gpt-4o-mini`). Pass `-p anthropic` or `-p gemini` to switch. Model names follow the Module 2 delivery (`claude-haiku-4-5-20251001`, `models/gemini-3.8-flash`). Gemini runs with at least `max_tokens=1024` so thinking tokens are less likely to truncate the object.

## Requirements

- **Python 3.12.13**
- An API key for the chat provider you call:
  - `OPENAI_API_KEY` when `-p openai` (the default)
  - `ANTHROPIC_API_KEY` when `-p anthropic`
  - `GEMINI_API_KEY` when `-p gemini`

Embeddings run locally. They do not use an API key.

## Setup

From this directory, with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the env template and set the key for the provider you want to call. `.env` is gitignored.

```bash
cp .env.dist .env
```

The first query downloads `sentence-transformers/all-MiniLM-L6-v2` if it is not already cached.

## Project layout

| File | Role |
|------|------|
| `main.py` | Asks the user for a question, or runs one question / the two demo questions |
| `samples.py` | Grounded question and trap question |
| `chain.py` | LCEL chain, `build_chain`, and `get_rag_response()` |
| `prompts.py` | Grounded system prompt. Context and question are template variables |
| `ingestion.py` | Reads `data/*.md` and `data/*.txt`, chunks them, and skips a populated index |
| `processing.py` | `DocumentProcessor`: clean text, then split at 500 tokens with 50 of overlap |
| `vectorstore.py` | Persistent Chroma client, upsert, similarity search, retriever |
| `llm.py` | `OpenAIClient`, `AnthropicClient`, and `GeminiClient` |
| `factory.py` | Builds a client from `LLMConfig` |
| `schemas.py` | `LLMAnswer`, `RAGResponse`, `Provider`, and `LLMConfig` |
| `constants.py` | Chunk size, `top_k`, paths, retry settings |
| `data/` | TechCorp policy notes used as the corpus |
| `tests/` | Chunking, index persistence, chain propagation, and retry cases |
| `requirements.txt` | Dependencies |
| `.env.dist` | Env template with empty secrets |

## Usage

Run from this directory. The index is opened before the prompt. With no arguments the program then reads questions from the terminal. An empty line exits.

```bash
python main.py

python main.py -p anthropic
python main.py --provider gemini

# One question, then exit
python main.py -q "¿Cuántos días de vacaciones corresponden a un empleado con menos de 5 años?"

# Spec checks: one answer that is in the documents, and one that is not
python main.py --demo
```

Unit tests do not call a chat provider. The index tests load the local embedding model.

```bash
pytest
```

### Docker

Create a `.env` first. The compose file mounts `./vectorstore` so the index survives the container.

```bash
docker compose build
docker compose run --rm app
docker compose run --rm app python main.py --demo
docker compose run --rm app python main.py -p gemini
```

## Example

A question about vacation days should come back with the figure from `politica_vacaciones.md` (14 consecutive days under 5 years of tenure) and that file in `sources`.

A question about the CEO's salary is not in the corpus. The answer has to say that the information is not available. It must not invent a number. `sources` still lists the chunks that were retrieved, because those are the fragments the model saw.
