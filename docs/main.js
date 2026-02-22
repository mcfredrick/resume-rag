const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

if (isMobile) {
  document.getElementById('status-card').style.display = 'none';
  document.getElementById('examples').style.display = 'none';
  document.getElementById('persona-section').style.display = 'none';
  document.getElementById('disclaimer').style.display = 'none';
  document.getElementById('qa-form').style.display = 'none';
  document.getElementById('mobile-notice').style.display = 'block';
} else {
  initApp();
}

function initApp() {
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
  const greetingText = document.getElementById('greeting-text');
  const thinkingDots = document.getElementById('thinking-dots');
  const chips = document.querySelectorAll('#chips .chip');
  const personaInput = document.getElementById('persona-input');

  document.querySelectorAll('#persona-chips .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      personaInput.value = chip.dataset.persona;
    });
  });

  function getPersona() {
    const words = personaInput.value.trim().split(/\s+/).filter(Boolean);
    return words.slice(0, 3).join(' ') || 'helpful assistant';
  }

  let embeddings = null;
  let isGenerating = false;
  let currentPersona = 'helpful assistant';

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

  function findTopChunks(queryEmbedding, k = 4, minScore = 0.3) {
    return embeddings
      .map(({ text, embedding }) => ({ text, score: cosineSimilarity(queryEmbedding, embedding) }))
      .filter(({ score }) => score >= minScore)
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

  const robot = document.getElementById('robot');

  const answerLabel = document.getElementById('answer-label');

  function startAnswer(persona) {
    isGenerating = true;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Thinking…';
    questionInput.disabled = true;
    answerCard.classList.add('visible');
    robot.classList.add('thinking');
    answerLabel.textContent = persona === 'helpful assistant' ? 'Answer' : `Answer · as ${persona}`;
    greetingText.innerHTML = '';
    thinkingDots.classList.remove('visible');
    answerText.innerHTML = persona === 'helpful assistant' ? '<span class="cursor"></span>' : '';
  }

  function finishAnswer() {
    isGenerating = false;
    submitBtn.disabled = false;
    submitBtn.textContent = 'Ask';
    questionInput.disabled = false;
    answerText.querySelector('.cursor')?.remove();
    greetingText.querySelector('.cursor')?.remove();
    thinkingDots.classList.remove('visible');
    robot.classList.remove('thinking');
  }

  worker.onmessage = async ({ data: { type, payload } }) => {
    switch (type) {
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
        currentPersona = getPersona();
        startAnswer(currentPersona);
        worker.postMessage({ type: 'generate', payload: { query: payload.query, chunks: topChunks, persona: currentPersona } });
        break;
      }

      case 'greeting': {
        if (!greetingText.querySelector('.cursor')) {
          greetingText.innerHTML = '<span class="cursor"></span>';
        }
        greetingText.querySelector('.cursor').insertAdjacentText('beforebegin', payload.text);
        break;
      }

      case 'thinking': {
        greetingText.querySelector('.cursor')?.remove();
        thinkingDots.classList.add('visible');
        answerLabel.textContent = `Thinking as ${currentPersona}…`;
        break;
      }

      case 'token': {
        if (!answerText.querySelector('.cursor')) {
          thinkingDots.classList.remove('visible');
          answerLabel.textContent = `Answer · as ${currentPersona}`;
          answerText.innerHTML = '<span class="cursor"></span>';
        }
        answerText.querySelector('.cursor').insertAdjacentText('beforebegin', payload.text);
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
}
