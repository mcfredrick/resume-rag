import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowRemoteModels = true;

const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const LLM_MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct';

let embedder = null;
let generator = null;

function makeProgressCallback(model) {
  return (info) => {
    if (info.status === 'progress' && info.progress != null) {
      self.postMessage({ type: 'progress', payload: { model, progress: info.progress } });
    }
  };
}

async function loadModels() {
  embedder = await pipeline('feature-extraction', EMBED_MODEL, {
    progress_callback: makeProgressCallback('embed'),
  });

  let device = 'wasm';
  try {
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) device = 'webgpu';
    }
  } catch (_) {}

  generator = await pipeline('text-generation', LLM_MODEL, {
    dtype: device === 'webgpu' ? 'q4f16' : 'q4',
    device,
    progress_callback: makeProgressCallback('llm'),
  });

  self.postMessage({ type: 'ready' });
}

async function getEmbedding(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

async function generateAnswer(query, chunks) {
  const context = chunks.map((c, i) => `[${i + 1}] ${c}`).join('\n\n');

  const messages = [
    {
      role: 'system',
      content:
        "You are a helpful assistant answering questions about Matthew Fredrick's resume. " +
        'Answer based only on the provided context. Be concise and accurate.',
    },
    {
      role: 'user',
      content: `Context:\n${context}\n\nQuestion: ${query}`,
    },
  ];

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      self.postMessage({ type: 'token', payload: { text } });
    },
  });

  await generator(messages, {
    max_new_tokens: 300,
    do_sample: false,
    streamer,
  });

  self.postMessage({ type: 'done' });
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;
  try {
    switch (type) {
      case 'load':
        await loadModels();
        break;
      case 'embed': {
        const embedding = await getEmbedding(payload.query);
        self.postMessage({ type: 'embedding', payload: { embedding, query: payload.query } });
        break;
      }
      case 'generate':
        await generateAnswer(payload.query, payload.chunks);
        break;
    }
  } catch (err) {
    self.postMessage({ type: 'error', payload: { message: err.message } });
  }
};
