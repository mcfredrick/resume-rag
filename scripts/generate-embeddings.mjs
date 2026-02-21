import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const RESUME_PATH = '/Users/matt/Job Search/Resume/2026-02_general/resume.md';
const EXTRA_PATH = join(__dirname, '..', 'docs', 'extra-context.md');
const OUTPUT_PATH = join(__dirname, '..', 'docs', 'embeddings.json');
const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const CHUNK_SIZE = 512;

// Chunk resume by line length, injecting the nearest section heading at the
// start of each new chunk so bullets never lose their company/section context.
function chunkResume(content) {
  const chunks = [];
  const lines = content.split('\n');
  let currentLines = [];
  let currentLength = 0;
  let currentHeading = null;

  for (const line of lines) {
    if (!line.trim()) continue;

    if (line.startsWith('#')) {
      currentHeading = line.trim();
    }

    if (currentLength + line.length > CHUNK_SIZE && currentLines.length > 0) {
      chunks.push(currentLines.join('\n'));
      // Prepend heading to new chunk so it retains section context
      if (currentHeading && !line.startsWith('#')) {
        currentLines = [currentHeading, line];
        currentLength = currentHeading.length + line.length;
      } else {
        currentLines = [line];
        currentLength = line.length;
      }
    } else {
      currentLines.push(line);
      currentLength += line.length;
    }
  }

  if (currentLines.length > 0) chunks.push(currentLines.join('\n'));
  return chunks;
}

// Each blank-line-separated block in extra-context.md becomes its own chunk.
function chunkExtra(content) {
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

  const resumeContent = readFileSync(RESUME_PATH, 'utf-8');
  const resumeChunks = chunkResume(resumeContent);
  console.log(`\nSplit resume into ${resumeChunks.length} chunks`);

  let extraChunks = [];
  if (existsSync(EXTRA_PATH)) {
    const extraContent = readFileSync(EXTRA_PATH, 'utf-8');
    extraChunks = chunkExtra(extraContent);
    console.log(`Loaded ${extraChunks.length} extra-context chunks`);
  }

  const allChunks = [...resumeChunks, ...extraChunks];
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
