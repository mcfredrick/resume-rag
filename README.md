# resume-rag

Client-side RAG agent for Matthew Fredrick's resume. Runs entirely in the browser — no server, no API keys, no data leaves your machine.

**Live site:** https://mcfredrick.github.io/resume-rag/

## How it works

This is a minimal, dependency-free RAG (Retrieval-Augmented Generation) pipeline built on [Transformers.js](https://huggingface.co/docs/transformers.js), which runs ML models directly in the browser via WebAssembly and WebGPU.

### Architecture

```
docs/context.md          ← curated Q&A pairs (source of truth)
      │
      ▼ npm run generate-embeddings
docs/embeddings.json     ← pre-computed vectors shipped with the site
      │
      ▼ (at query time, in a Web Worker)
[1] Embed query          ← all-MiniLM-L6-v2 (384-dim, ~25MB)
[2] Cosine similarity    ← rank all chunks, filter score < 0.3, keep top 4
[3] Build prompt         ← system prompt + retrieved chunks as context
[4] Generate answer      ← SmolLM2-360M-Instruct (~200MB, q4/q4f16)
```

**Models:**
- **Embedding:** `Xenova/all-MiniLM-L6-v2` — used both offline (embedding generation) and at query time (query embedding)
- **LLM:** `HuggingFaceTB/SmolLM2-360M-Instruct` — runs in a Web Worker; uses WebGPU (`q4f16`) if available, falls back to WASM (`q4`)

**Knowledge base:** `docs/context.md` contains hand-curated Q&A pairs covering career history, skills, and achievements. Each blank-line-separated block becomes one chunk. This replaces raw resume chunking — curated Q&A pairs embed more precisely and give the tiny LLM cleaner context.

**First load:** ~25MB (embedding model) + ~200MB (LLM weights) — both cached in the browser after that.

### Key files

| File | Purpose |
|------|---------|
| `docs/context.md` | Source Q&A pairs — edit this to update content |
| `docs/embeddings.json` | Pre-computed embeddings — regenerate after editing context.md |
| `docs/worker.js` | Web Worker: loads models, embeds queries, runs generation |
| `docs/main.js` | Main thread: cosine similarity retrieval, UI state |
| `docs/index.html` | Single-page UI |
| `scripts/generate-embeddings.mjs` | Offline script to re-embed context.md |

## Local development

```bash
npm run dev
```

Opens `http://localhost:8080` with live reload. No build step — edits to `docs/worker.js`, `docs/main.js`, or `docs/index.html` take effect on the next browser refresh.

> Note: the browser downloads ~225MB of model weights from HuggingFace on first load. They're cached after that.

## Updating content

Edit `docs/context.md`, then regenerate embeddings:

```bash
npm install
npm run generate-embeddings
```

Commit both `docs/context.md` and the updated `docs/embeddings.json`.

## Deployment

The site is served directly from the `docs/` directory via GitHub Pages. Push to `main` and it's live.

The `coi-serviceworker.min.js` in `docs/` sets the `Cross-Origin-Isolation` headers required for SharedArrayBuffer (needed by the WASM/WebGPU runtimes) without requiring server configuration.
