import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const RESUME_PATH = '/Users/matt/Job Search/Resume/2026-02_general/resume.md';
const OUTPUT_PATH = join(__dirname, '..', 'docs', 'embeddings.json');
const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const CHUNK_SIZE = 512;

function chunkResume(content) {
  const chunks = [];
  const lines = content.split('\n');
  let currentLines = [];
  let currentLength = 0;

  for (const line of lines) {
    if (!line.trim()) continue;
    if (currentLength + line.length > CHUNK_SIZE && currentLines.length > 0) {
      chunks.push(currentLines.join('\n'));
      currentLines = [line];
      currentLength = line.length;
    } else {
      currentLines.push(line);
      currentLength += line.length;
    }
  }

  if (currentLines.length > 0) {
    chunks.push(currentLines.join('\n'));
  }

  return chunks;
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

  const content = readFileSync(RESUME_PATH, 'utf-8');
  const chunks = chunkResume(content);
  console.log(`\nSplit resume into ${chunks.length} chunks`);

  const results = [];
  for (let i = 0; i < chunks.length; i++) {
    const text = chunks[i];
    const output = await embedder(text, { pooling: 'mean', normalize: true });
    results.push({ text, embedding: Array.from(output.data) });
    process.stdout.write(`\r  Embedding chunk ${i + 1}/${chunks.length}...`);
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
