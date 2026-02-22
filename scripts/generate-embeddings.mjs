import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const CONTEXT_PATH = join(__dirname, '..', 'docs', 'context.md');
const OUTPUT_PATH = join(__dirname, '..', 'docs', 'embeddings.json');
const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';

// Each blank-line-separated block in context.md becomes its own chunk.
function chunkContext(content) {
  return content
    .split(/\n\n+/)
    .map(block => block.trim())
    .filter(block => block && !block.startsWith('#'));
}

async function main() {
  console.log('Loading embedding model (downloads ~25MB on first run)...');
  const embedder = await pipeline('feature-extraction', EMBED_MODEL, {
    progress_callback: (info) => {
      if (info.status === 'progress') {
        process.stdout.write(`\r  ${info.file}: ${Math.round(info.progress ?? 0)}%   `);
      } else if (info.status === 'done') {
        process.stdout.write(`\r  ${info.file}: done           \n`);
      }
    },
  });

  const contextContent = readFileSync(CONTEXT_PATH, 'utf-8');
  const allChunks = chunkContext(contextContent);
  console.log(`\nLoaded ${allChunks.length} context chunks`);
  console.log(`Embedding ${allChunks.length} chunks total...`);

  const results = [];
  for (let i = 0; i < allChunks.length; i++) {
    const text = allChunks[i];
    const output = await embedder(text, { pooling: 'mean', normalize: true });
    results.push({ text, embedding: Array.from(output.data) });
    process.stdout.write(`\r  Embedding chunk ${i + 1}/${allChunks.length}...`);
  }

  console.log(`\nSaving to ${OUTPUT_PATH}`);
  mkdirSync(dirname(OUTPUT_PATH), { recursive: true });
  writeFileSync(OUTPUT_PATH, JSON.stringify(results));
  console.log(`Done — ${results.length} chunks, ${results[0].embedding.length}-dim embeddings`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
