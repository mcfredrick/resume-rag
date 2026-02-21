const worker = new Worker('./worker.js', { type: 'module' });

const embedFill = document.getElementById('embed-fill');
const embedPct = document.getElementById('embed-pct');
const llmFill = document.getElementById('llm-fill');
const llmPct = document.getElementById('llm-pct');
const statusCard = document.getElementById('status-card');
const questionInput = document.getElementById('question');
const submitBtn = document.getElementById('submit-btn');
const qaForm = document.getElementById('qa-form');
const answerCard = document.getElementById('answer-card');
const answerText = document.getElementById('answer-text');
const chips = document.querySelectorAll('.chip');

let embeddings = null;
let isGenerating = false;

async function loadEmbeddings() {
  const response = await fetch('./embeddings.json');
  embeddings = await response.json();
}

function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

function findTopChunks(queryEmbedding, k = 5) {
  return embeddings
    .map(({ text, embedding }) => ({ text, score: cosineSimilarity(queryEmbedding, embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(({ text }) => text);
}

function setProgress(model, pct, done = false) {
  const fill = model === 'embed' ? embedFill : llmFill;
  const label = model === 'embed' ? embedPct : llmPct;
  fill.style.width = `${pct}%`;
  label.textContent = done ? '✓' : `${Math.round(pct)}%`;
  if (done) fill.classList.add('done');
}

function enableUI() {
  statusCard.querySelector('.status-label').textContent = 'Models ready';
  questionInput.disabled = false;
  submitBtn.disabled = false;
  questionInput.focus();
  chips.forEach((chip) => {
    chip.classList.remove('disabled');
    chip.addEventListener('click', () => {
      if (!isGenerating) {
        questionInput.value = chip.dataset.q;
        questionInput.focus();
      }
    });
  });
}

function startAnswer() {
  isGenerating = true;
  submitBtn.disabled = true;
  submitBtn.textContent = 'Thinking…';
  questionInput.disabled = true;
  answerCard.classList.add('visible');
  answerText.innerHTML = '<span class="cursor"></span>';
}

function finishAnswer() {
  isGenerating = false;
  submitBtn.disabled = false;
  submitBtn.textContent = 'Ask';
  questionInput.disabled = false;
  const cursor = answerText.querySelector('.cursor');
  if (cursor) cursor.remove();
}

worker.onmessage = async ({ data: { type, payload } }) => {
  switch (type) {
    case 'progress':
      setProgress(payload.model, payload.progress);
      break;

    case 'modelInfo':
      document.getElementById('llm-name').textContent = payload.name;
      break;

    case 'ready':
      setProgress('embed', 100, true);
      setProgress('llm', 100, true);
      await loadEmbeddings();
      enableUI();
      break;

    case 'embedding': {
      const topChunks = findTopChunks(payload.embedding);
      startAnswer();
      worker.postMessage({ type: 'generate', payload: { query: payload.query, chunks: topChunks } });
      break;
    }

    case 'token': {
      const cursor = answerText.querySelector('.cursor');
      if (cursor) {
        cursor.insertAdjacentText('beforebegin', payload.text);
      } else {
        answerText.textContent += payload.text;
      }
      break;
    }

    case 'done':
      finishAnswer();
      break;

    case 'error':
      answerText.textContent = `Error: ${payload.message}`;
      finishAnswer();
      break;
  }
};

qaForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const query = questionInput.value.trim();
  if (!query || isGenerating || !embeddings) return;
  worker.postMessage({ type: 'embed', payload: { query } });
});

worker.postMessage({ type: 'load' });
