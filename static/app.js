/* MedGuide AI frontend — Clinical Teal UI.
   - Conversations persist in localStorage; the active conversation_id is sent
     as X-Conversation-ID so the backend's SQLite thread memory actually works.
   - Bot answers render markdown-lite, show a route badge, an emergency card
     style for the emergency fast-path, and clickable source chips built from
     the structured `sources` array (the **Sources** footer stays in the raw
     API answer for eval/API consumers, but is stripped from display). */
const $ = id => document.getElementById(id);

/* ---------------- conversation storage ---------------- */
const LS_CONV = 'mg_conversations', LS_ACTIVE = 'mg_active';
const MAX_MSGS = 60, MAX_CONVS = 20;
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : 'id-' + Date.now() + '-' + Math.random().toString(16).slice(2));

function loadConvs() { try { return JSON.parse(localStorage.getItem(LS_CONV)) || {}; } catch { return {}; } }
function saveConvs(c) {
  const ids = Object.keys(c).sort((a, b) => (c[b].updated || 0) - (c[a].updated || 0)).slice(0, MAX_CONVS);
  const pruned = {};
  for (const id of ids) {
    pruned[id] = c[id];
    pruned[id].messages = (c[id].messages || []).slice(-MAX_MSGS);
  }
  localStorage.setItem(LS_CONV, JSON.stringify(pruned));
}
let convs = loadConvs();
let activeId = localStorage.getItem(LS_ACTIVE);
if (!activeId || !convs[activeId]) {
  activeId = uid();
  convs[activeId] = { title: 'New chat', updated: Date.now(), messages: [] };
  saveConvs(convs);
}
localStorage.setItem(LS_ACTIVE, activeId);

let history = [];           // [{role, content}] sent to the API
let imageData = null, recorder = null, audioChunks = [], cameraStream = null, lastAnswer = '';

const headers = () => {
  const h = {};
  const token = $('token').value.trim();
  if (token) h.Authorization = `Bearer ${token}`;
  h['X-Conversation-ID'] = activeId;   // the fix: backend thread memory now actually engages
  return h;
};

/* ---------------- markdown-lite ---------------- */
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function inline(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|\W)\*(.+?)\*/g, '$1<em>$2</em>');
}
function md(src) {
  const lines = src.split('\n');
  let out = '', list = null;
  const closeList = () => { if (list) { out += `</${list}>`; list = null; } };
  for (const ln of lines) {
    const t = ln.trim();
    let m;
    if ((m = t.match(/^###\s+(.*)/))) { closeList(); out += `<h4>${inline(m[1])}</h4>`; }
    else if ((m = t.match(/^##\s+(.*)/))) { closeList(); out += `<h3>${inline(m[1])}</h3>`; }
    else if ((m = t.match(/^[-•]\s+(.*)/))) {
      if (list !== 'ul') { closeList(); out += '<ul>'; list = 'ul'; }
      out += `<li>${inline(m[1])}</li>`;
    }
    else if ((m = t.match(/^\d+[.)]\s+(.*)/))) {
      if (list !== 'ol') { closeList(); out += '<ol>'; list = 'ol'; }
      out += `<li>${inline(m[1])}</li>`;
    }
    else if (t === '') { closeList(); }
    else { closeList(); out += `<p>${inline(t)}</p>`; }
  }
  closeList();
  return out || '<p></p>';
}

/* ---------------- message rendering ---------------- */
const FOOTER_RE = /\n\n\*\*Sources\*\*[\s\S]*$/;
const stripFooter = text => text.replace(FOOTER_RE, '').trimEnd();

function badgeFor(tools, intent, low, score, cached) {
  let b = '';
  if (intent === 'emergency') b = '<span class="badge emg">🚨 Emergency</span>';
  else if ((tools || []).includes('tavily')) b = '<span class="badge web">🌐 Web research</span>';
  else if ((tools || []).includes('pinecone')) {
    const tip = (typeof score === 'number') ? ` title="top KB match: relevance ${score.toFixed(2)}"` : '';
    b = `<span class="badge kb"${tip}>📚 Knowledge base</span>`;
  }
  if (cached) b += (b ? ' ' : '') + '<span class="badge cache">⚡ Cached answer</span>';
  if (low) b += ' <span class="badge low">⚠ Limited evidence</span>';
  return b;
}

function chipsFor(sources) {
  return (sources || []).slice(0, 8).map((x, i) => {
    if (x.type === 'pdf') return `<button type="button" class="chip as-btn" data-src="${i}" title="relevance ${Number(x.score || 0).toFixed(2)} — click to read excerpt">📄 PDF p.${x.page}</button>`;
    const label = x.title || x.url;
    return `<a class="chip" href="${esc(x.url)}" target="_blank" rel="noopener">🔗 ${esc(label.length > 42 ? label.slice(0, 42) + '…' : label)}</a>`;
  }).join('');
}

function bubbleUser(text) {
  const el = document.createElement('article');
  el.className = 'user';
  el.textContent = text;
  $('messages').appendChild(el);
  scrollDown();
  return el;
}

function bubbleBot({ text, sources = [], tools = [], intent = '', low = false, score = null, cached = false }) {
  const el = document.createElement('article');
  el.className = 'bot' + (intent === 'emergency' ? ' emergency' : '');
  const badge = badgeFor(tools, intent, low, score, cached);
  const chips = chipsFor(sources);
  el.innerHTML =
    (intent === 'emergency' ? '<div class="emg-head">🚨 Possible emergency</div>' : '') +
    (badge ? `<div class="meta">${badge}</div>` : '') +
    md(stripFooter(text)) +
    (chips ? `<div class="sources">${chips}</div>` : '');
  el._sources = sources || [];
  $('messages').appendChild(el);
  scrollDown();
  return el;
}

/* ---- source excerpt modal (click a 📄 chip) ---- */
function showExcerpt(src) {
  closeExcerpt();
  const back = document.createElement('div');
  back.className = 'modal-back';
  back.id = 'excerptModal';
  back.innerHTML =
    `<div class="modal" role="dialog" aria-label="Source excerpt">` +
    `<div class="modal-head"><strong>📄 Gale Encyclopedia of Medicine — p.${esc(String(src.page))}</strong>` +
    `<button type="button" class="modal-x" aria-label="Close">×</button></div>` +
    `<div class="meta"><span class="badge kb">relevance ${Number(src.score || 0).toFixed(2)}</span></div>` +
    `<p class="excerpt">${esc(src.text || 'Excerpt not available.')}</p>` +
    `</div>`;
  back.addEventListener('click', e => { if (e.target === back || e.target.closest('.modal-x')) closeExcerpt(); });
  document.addEventListener('keydown', escClose);
  document.body.appendChild(back);
}
function escClose(e) { if (e.key === 'Escape') closeExcerpt(); }
function closeExcerpt() {
  document.removeEventListener('keydown', escClose);
  $('excerptModal')?.remove();
}
document.addEventListener('click', e => {
  const b = e.target.closest('button.chip[data-src]');
  if (!b) return;
  const art = b.closest('article');
  const src = art && art._sources && art._sources[+b.dataset.src];
  if (src && src.type === 'pdf') showExcerpt(src);
});

function bubbleTyping() {
  const el = document.createElement('article');
  el.className = 'bot';
  el.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  $('messages').appendChild(el);
  scrollDown();
  return el;
}

function scrollDown() { const m = $('messages'); m.scrollTop = m.scrollHeight; }

function persist() {
  convs[activeId].updated = Date.now();
  saveConvs(convs);
  renderSidebar();
}

function renderMessages() {
  $('messages').innerHTML = '';
  const msgs = convs[activeId].messages;
  if (!msgs.length) {
    $('messages').innerHTML = '<div class="welcome"><strong>✚ MedGuide AI</strong><br>Ask a medical question, describe symptoms, or upload an image.<br>Answers cite their sources and flag uncertainty.</div>';
    return;
  }
  for (const m of msgs) {
    if (m.role === 'user') bubbleUser(m.text);
    else bubbleBot({ text: m.text, sources: m.sources, tools: m.tools, intent: m.intent, low: m.low, score: m.score, cached: m.cached });
  }
}

function rebuildHistory() {
  history = [];
  for (const m of convs[activeId].messages.slice(-8)) {
    history.push({ role: m.role === 'user' ? 'user' : 'assistant', content: m.text });
  }
}

/* ---------------- sidebar ---------------- */
function renderSidebar() {
  const list = $('convList');
  list.innerHTML = '';
  const ids = Object.keys(convs).sort((a, b) => convs[b].updated - convs[a].updated);
  if (!ids.length) { list.innerHTML = '<div class="conv-empty">No conversations yet.</div>'; return; }
  for (const id of ids) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'conv-item' + (id === activeId ? ' active' : '');
    const label = document.createElement('span');
    label.className = 't';
    label.textContent = convs[id].title || 'New chat';
    label.title = label.textContent;
    const del = document.createElement('span');
    del.className = 'del';
    del.textContent = '×';
    del.title = 'Delete conversation';
    del.onclick = e => {
      e.stopPropagation();
      delete convs[id];
      if (id === activeId) {
        const rest = Object.keys(convs).sort((a, b) => convs[b].updated - convs[a].updated);
        activeId = rest[0] || uid();
        if (!convs[activeId]) convs[activeId] = { title: 'New chat', updated: Date.now(), messages: [] };
        localStorage.setItem(LS_ACTIVE, activeId);
      }
      persist(); rebuildHistory(); renderMessages();
    };
    item.append(label, del);
    item.onclick = () => {
      if (id === activeId) return;
      activeId = id;
      localStorage.setItem(LS_ACTIVE, id);
      closeNav();
      rebuildHistory(); renderMessages(); renderSidebar();
    };
    list.appendChild(item);
  }
}

function newChat() {
  activeId = uid();
  convs[activeId] = { title: 'New chat', updated: Date.now(), messages: [] };
  localStorage.setItem(LS_ACTIVE, activeId);
  history = [];
  persist(); renderMessages(); closeNav();
  $('message').focus();
}

$('newChat').onclick = newChat;

/* mobile nav */
function closeNav() { document.body.classList.remove('nav-open'); $('backdrop').hidden = true; }
$('menuBtn').onclick = () => { document.body.classList.add('nav-open'); $('backdrop').hidden = false; };
$('backdrop').onclick = closeNav;

/* ---------------- status ---------------- */
async function check() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    const s = $('status');
    s.textContent = d.configured ? '● Connected' : '⚠ Configure API keys';
    s.classList.toggle('ok', !!d.configured);
  } catch { $('status').textContent = '● Server offline'; }
}
check();

/* ---------------- image / camera ---------------- */
function setImage(file) {
  if (!file) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 5e6) {
    alert('Use PNG/JPEG/WebP under 5MB');
    return;
  }
  const reader = new FileReader();
  reader.onload = () => { imageData = reader.result; $('thumbnail').src = imageData; $('preview').hidden = false; };
  reader.readAsDataURL(file);
}
$('image').addEventListener('change', e => setImage(e.target.files[0]));
$('remove').onclick = () => { imageData = null; $('preview').hidden = true; $('image').value = ''; };

/* ---------------- chat submit ---------------- */
$('chatForm').addEventListener('submit', async e => {
  e.preventDefault();
  const message = $('message').value.trim();
  if (!message) return;
  const conv = convs[activeId];
  const isFirst = !conv.messages.length;

  bubbleUser(message);
  conv.messages.push({ role: 'user', text: message });
  if (isFirst) conv.title = message.length > 36 ? message.slice(0, 36) + '…' : message;
  $('message').value = '';
  $('send').disabled = true;
  const pending = bubbleTyping();

  try {
    const r = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { ...headers(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: history.slice(-8), image_data: imageData })
    });
    if (!r.ok) {
      let detail = 'Request failed';
      try { detail = (await r.json()).detail || detail; } catch { /* non-JSON error body */ }
      throw new Error(detail);
    }
    // Server-sent events: {type:'token'} chunks, then {type:'done'} with the
    // full payload (sources/tools/intent). Guardrail fast-paths send no tokens.
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '', acc = '', streaming = false, done = null;
    for (;;) {
      const { done: rdDone, value } = await reader.read();
      if (rdDone) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const p of parts) {
        const line = p.trim();
        if (!line.startsWith('data:')) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.type === 'token') {
          acc += ev.text || '';
          if (!streaming) { streaming = true; pending.innerHTML = ''; }
          pending.innerHTML = md(acc);
          scrollDown();
        } else if (ev.type === 'done') { done = ev; }
        else if (ev.type === 'error') { throw new Error(ev.detail || 'Stream failed'); }
      }
      if (done) break;
    }
    if (!done) throw new Error('Stream ended unexpectedly');
    pending.remove();
    const finalText = done.answer || acc;
    const low = !!done.low_confidence, score = (typeof done.kb_top_score === 'number') ? done.kb_top_score : null, cached = !!done.cache_hit;
    bubbleBot({ text: finalText, sources: done.sources, tools: done.tools, intent: done.intent, low, score, cached });
    conv.messages.push({ role: 'bot', text: finalText, sources: done.sources || [], tools: done.tools || [], intent: done.intent || '', low, score, cached });
    lastAnswer = stripFooter(finalText);
    history.push({ role: 'user', content: message }, { role: 'assistant', content: finalText });
    history = history.slice(-10);
    persist();
    $('remove').click();
  } catch (err) {
    pending.remove();
    bubbleBot({ text: 'Error: ' + err.message });
  } finally {
    $('send').disabled = false;
    scrollDown();
  }
});

/* ---------------- voice in / out ---------------- */
$('mic').onclick = async () => {
  if (recorder && recorder.state === 'recording') { recorder.stop(); $('mic').innerHTML = '🎤 <span>Voice</span>'; return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => audioChunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks, { type: recorder.mimeType });
      const data = new FormData();
      data.append('file', blob, 'voice.webm');
      $('mic').disabled = true;
      try {
        const r = await fetch('/api/transcribe', { method: 'POST', headers: headers(), body: data });
        const out = await r.json();
        if (!r.ok) throw Error(out.detail);
        $('message').value = out.text;
      } catch (e) { alert(e.message); }
      finally { $('mic').disabled = false; }
    };
    recorder.start();
    $('mic').innerHTML = '⏹ <span>Stop</span>';
  } catch (e) { alert('Microphone unavailable: ' + e.message); }
};

$('listen').onclick = async () => {
  if (!lastAnswer) return alert('Ask a question first');
  $('listen').disabled = true;
  try {
    const r = await fetch('/api/speak', {
      method: 'POST',
      headers: { ...headers(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: lastAnswer })
    });
    if (!r.ok) throw Error('Speech unavailable');
    const url = URL.createObjectURL(await r.blob());
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play();
  } catch (e) { alert(e.message); }
  finally { $('listen').disabled = false; }
};

/* ---------------- camera ---------------- */
$('camera').onclick = async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    $('video').srcObject = cameraStream;
    $('video').hidden = false;
    $('cameraActions').hidden = false;
  } catch (e) { alert('Camera unavailable: ' + e.message); }
};
$('closeCamera').onclick = () => {
  cameraStream?.getTracks().forEach(t => t.stop());
  cameraStream = null;
  $('video').hidden = true;
  $('cameraActions').hidden = true;
};
$('snap').onclick = () => {
  const v = $('video'), c = $('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  c.toBlob(b => setImage(new File([b], 'camera.jpg', { type: 'image/jpeg' })), 'image/jpeg', .85);
  $('closeCamera').click();
};

/* ---------------- init ---------------- */
rebuildHistory();
renderMessages();
renderSidebar();
