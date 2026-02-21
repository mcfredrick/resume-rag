# resume-rag

Client-side RAG agent for Matthew Fredrick's resume. Runs entirely in the browser — no server required.

**Live site:** https://mcfredrick.github.io/resume-rag/

## How it works

1. The resume is split into chunks and embedded offline with `all-MiniLM-L6-v2`
2. Those embeddings are shipped as `docs/embeddings.json`
3. At query time, the browser embeds the question with the same model, finds the top-5 most relevant chunks via cosine similarity, and feeds them to SmolLM2-360M-Instruct for a grounded answer
4. Both models run via [Transformers.js](https://huggingface.co/docs/transformers.js) — WebGPU if available, WASM fallback

**First load:** ~25MB (embeddings model) + ~200MB (SmolLM2 q4) — cached in browser after that.

## Regenerating embeddings

If `resume.md` changes, re-run:

```bash
npm install
npm run generate-embeddings
```

Then commit the updated `docs/embeddings.json`.
