# RAG Assistant

Real-time personal knowledge base powered by local embeddings + LLM.

Ask questions during a job interview or German lesson — the assistant instantly finds relevant context from your CV, tech stack, or grammar notes and gives a concise answer.

## How it works

```
Your question → embed (local, ~50ms) → ChromaDB search → relevant chunks → LLM → answer
```

No personal data leaves your machine during retrieval. Only the final prompt (question + matched chunks) goes to the LLM API.

## Modes

| Mode | Knowledge base | Use case |
|------|---------------|----------|
| `interview` | CV, tech stack, Q&A answers | Job interviews |
| `german` | Grammar rules, vocabulary | German lessons / calls |

## Quick start

**1. Install**
```bat
install.bat
```

**2. Add your OpenAI key**
```
copy .env.example .env
# edit .env and paste your key
```

**3. Fill in your knowledge base**

Edit the files in `knowledge_base/personal/` — these are Markdown files, never committed to git:

```
knowledge_base/personal/
├── cv.md              # your CV, background, typical Q&A answers
├── tech_stack.md      # technologies, projects
└── german/
    ├── grammar.md     # grammar rules you've studied
    └── vocabulary.md  # words you know
```

**4. Load into database**
```bat
python ingest.py
```

**5. Run**
```bat
start_interview.bat
# or
start_german.bat
```

## Knowledge base format

Files use `##` headers as chunk boundaries — each section becomes a searchable unit:

```markdown
## Verben mit festen Präpositionen
sich interessieren für + Akkusativ — интересоваться чем-то
warten auf + Akkusativ — ждать кого-то
```

Keep sections focused on one topic. The smaller and cleaner the chunk, the better the retrieval.

## CLI reference

```bash
python main.py                   # interview mode
python main.py --mode german     # german mode
python main.py --stats           # show chunk counts per collection
python ingest.py                 # ingest all files
python ingest.py --collection german        # ingest one collection
python ingest.py --reset german             # clear + re-ingest
```

## Project structure

```
rag-assistant/
├── rag_assistant/
│   ├── embedder.py      # sentence-transformers (all-MiniLM-L6-v2, runs locally)
│   ├── knowledge.py     # ChromaDB wrapper — add / search chunks
│   └── assistant.py     # RAG pipeline — retrieve context, call LLM
├── knowledge_base/
│   ├── personal/        # your data — gitignored, never committed
│   └── examples/        # template files for reference
├── ingest.py            # load markdown files into ChromaDB
├── main.py              # interactive CLI
├── install.bat
├── start_interview.bat
└── start_german.bat
```

## Requirements

- Python 3.11+
- OpenAI API key (uses `gpt-4o-mini` by default)
- ~500 MB disk for the embedding model (downloaded once)
- No GPU required

## Privacy

- `knowledge_base/personal/` and `db/` are in `.gitignore` — never committed
- Embeddings are computed locally, no data sent to external services during retrieval
- Only the LLM prompt (your question + matched text snippets) goes to OpenAI
