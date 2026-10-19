/* Authorship: agent-written (2026-10-18), adapted from Becemder's web/page/annotate.js.
   review-comments.js: comment threads on the paper in review.html. Theo highlights text and comments; the box at the
   bottom right sends new comments, follow-ups and integrate requests to the page's chat in Becemder. The agent answers
   each comment with review_reply.py, which appends to review-replies.txt, and the page shows those replies in the
   threads.

   Theo's side of each thread lives in localStorage, as annotate.js keeps its comments; the comments that annotate.js
   kept on the first version of this page become threads when the page loads (importOld). A thread is anchored by
   character offsets into the text of the root plus the quoted text and the text around it. When the quote is no longer
   in the PDF, or a short quote no longer has its surroundings, the thread is stale, and its card stays where the text
   was (page, and height within the page). */

const KATEX = 'https://cdn.jsdelivr.net/npm/katex@0.19.0/dist/contrib/auto-render.mjs';
const CTX = window.BECEMDER || {};
const WHO = CTX.user || 'the user';
const PATH = decodeURIComponent(location.pathname);
const STORE_KEY = 'paper-review-threads:v1:' + PATH;
const CONTEXT_KEY = STORE_KEY + ':context';
const RESOLVED_KEY = STORE_KEY + ':show-resolved';
// The first version of this page used Becemder's annotate.js, which kept its comments under this key.
const OLD_KEY = 'report-annotations:v1:' + PATH;
const IMPORTED_KEY = STORE_KEY + ':imported';
const CONTEXT_CHARS = 32;
const MIN_CONTEXT_CHARS = 12;
const MIN_SIDE_CHARS = 4;
const LONG_QUOTE_CHARS = 40;
const MAX_QUOTE_CHARS = 700;
const MAX_EARLIER_CHARS = 600;
const CARD_GAP = 8;
const MARGIN_MIN = 260;  // room right of the pages that the cards need, in px
const MATH = [
  { left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true },
  { left: '$', right: '$', display: false }, { left: '\\(', right: '\\)', display: false }
];

let root = null;
let repliesUrl = '';
let threads = [];
let replies = {};          // thread id -> the agent's replies, from review-replies.txt
let showResolved = false;
const ui = { cards: {} };
let editor = null;         // {tid, mid, ta} while a message is being written; mid is null for a new reply
let active = null;         // id of the open thread
let marginOn = false;
let frame = 0;
let sending = false;
let dockOpen = false;
let statusTimer = 0;
let clearedFor = '';
let onIdle = () => {};     // called when a message is saved or put aside, so that a waiting reload can happen
let mathRender = null;

import(KATEX).then(m => { mathRender = m.default; refresh(); }, () => { /* math stays as TeX source */ });

/* ---------- storage ---------- */

function load() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; } catch (e) { return []; }
}

function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(threads)); } catch (e) { /* storage blocked */ }
  updateDock();
}

function stored(key) {
  try { return localStorage.getItem(key) || ''; } catch (e) { return ''; }
}

// Turns each annotate.js comment into a thread, once; the old key stays as it is.
function importOld() {
  let old;
  try { old = JSON.parse(localStorage.getItem(OLD_KEY)) || []; } catch (e) { return; }
  const done = new Set(stored(IMPORTED_KEY).split(' ').filter(Boolean));
  for (const a of old) {
    if (!a || !a.id || typeof a.quote !== 'string' || done.has(a.id)) continue;
    done.add(a.id);
    const id = a.id.slice(1);
    const m = { id: 'm' + id, kind: 'comment', text: a.comment || '', at: a.created };
    if (a.sent) m.sent = a.sent;
    threads.push({
      id: 'c' + id, start: a.start, end: a.end, quote: a.quote, prefix: a.prefix || '', suffix: a.suffix || '',
      section: a.section || '', page: Number((/page (\d+)/i.exec(a.section || '') || [])[1]) || 1, y: 0,
      created: a.created, imported: true, messages: [m]
    });
  }
  store(IMPORTED_KEY, [...done].join(' '));
  if (!stored(CONTEXT_KEY) && stored(OLD_KEY + ':general')) store(CONTEXT_KEY, stored(OLD_KEY + ':general'));
}

// An imported thread knows only its page; once its text is found, it gets its section and height as a new one would.
function placeImported(t) {
  delete t.imported;
  const m = marksOf(t.id)[0];
  const page = m && m.closest('.page');
  if (!page) return;
  t.section = sectionOf(m);
  t.page = Number(page.dataset.page);
  t.y = (m.getBoundingClientRect().top - page.getBoundingClientRect().top) / page.offsetHeight;
}

function store(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* storage blocked */ }
}

/* ---------- text offsets ---------- */

function skip(node) {
  const parent = node.parentElement;
  return !parent || !!parent.closest('svg, script, style, .rv-ui');
}

function textNodes() {
  const out = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: n => skip(n) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT
  });
  for (let n = walker.nextNode(); n; n = walker.nextNode()) out.push(n);
  return out;
}

function fullText() {
  return textNodes().map(n => n.data).join('');
}

function offsetOf(container, offset) {
  const r = document.createRange();
  r.setStart(root, 0);
  r.setEnd(container, offset);
  let acc = 0;
  for (const n of textNodes()) {
    if (n === container) return acc + offset;
    if (r.comparePoint(n, n.data.length) !== 0) break;
    acc += n.data.length;
  }
  return acc;
}

function sharedEnd(a, b) {
  let n = 0;
  while (n < a.length && n < b.length && a[a.length - 1 - n] === b[b.length - 1 - n]) n++;
  return n;
}

function sharedStart(a, b) {
  let n = 0;
  while (n < a.length && n < b.length && a[n] === b[n]) n++;
  return n;
}

// A short or repeated quote, such as "to", occurs all over the paper, so it only keeps its place where the text on
// both sides of it is unchanged. A long quote found exactly once is the same passage whatever surrounds it.
function locate(t, text) {
  const hits = [];
  for (let i = text.indexOf(t.quote); i !== -1; i = text.indexOf(t.quote, i + 1)) hits.push(i);
  const anywhere = hits.length === 1 && t.quote.length >= LONG_QUOTE_CHARS;
  const needBefore = Math.min(MIN_SIDE_CHARS, t.prefix.length);
  const needAfter = Math.min(MIN_SIDE_CHARS, t.suffix.length);
  const needBoth = Math.min(MIN_CONTEXT_CHARS, t.prefix.length + t.suffix.length);
  let best = -1;
  let bestScore = -Infinity;
  for (const i of hits) {
    const before = sharedEnd(text.slice(Math.max(0, i - t.prefix.length), i), t.prefix);
    const after = sharedStart(text.slice(i + t.quote.length, i + t.quote.length + t.suffix.length), t.suffix);
    if (!anywhere && (before < needBefore || after < needAfter || before + after < needBoth)) continue;
    const score = (before + after) * 1e6 - Math.abs(i - t.start);
    if (score > bestScore) { bestScore = score; best = i; }
  }
  if (best < 0) return false;
  t.start = best;
  t.end = best + t.quote.length;
  return true;
}

function sectionOf(node) {
  let name = '';
  for (const h of root.querySelectorAll('h1, h2, h3, h4')) {
    if (!(h.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING)) break;
    name = h.textContent.trim();
  }
  return name || 'Before the first heading';
}

/* ---------- marks ---------- */

function marksOf(id) {
  return [...root.querySelectorAll(`mark.rv-mark[data-id="${id}"]`)];
}

function wrap(t) {
  const targets = [];
  let acc = 0;
  for (const node of textNodes()) {
    const len = node.data.length;
    if (acc < t.end) {
      const s = Math.max(t.start, acc) - acc;
      const e = Math.min(t.end, acc + len) - acc;
      if (e > s && node.data.slice(s, e).trim()) targets.push([node, s, e]);
    }
    acc += len;
  }
  for (const [node, s, e] of targets) {
    if (e < node.data.length) node.splitText(e);
    const mid = s > 0 ? node.splitText(s) : node;
    const mark = document.createElement('mark');
    mark.className = 'rv-mark';
    mark.dataset.id = t.id;
    mid.parentNode.insertBefore(mark, mid);
    mark.appendChild(mid);
  }
}

function unwrap(id) {
  for (const m of marksOf(id)) {
    const p = m.parentNode;
    while (m.firstChild) p.insertBefore(m.firstChild, m);
    p.removeChild(m);
    p.normalize();
  }
}

/* ---------- threads ---------- */

function byId(id) {
  return threads.find(t => t.id === id) || null;
}

function sorted() {
  return threads.slice().sort((a, b) => a.start - b.start);
}

function shown() {
  return sorted().filter(t => showResolved || !t.resolved);
}

function newId(prefix) {
  return prefix + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function now() {
  return new Date().toISOString();
}

function time(iso) {
  return Date.parse(iso) || 0;
}

function unsent(t) {
  return t.messages.filter(m => !m.sent);
}

function repliesOf(t) {
  return replies[t.id] || [];
}

// Sent messages and replies in the order they were written, then the messages not sent yet.
function timeline(t) {
  const mine = t.messages.filter(m => m.sent).map(m => ({ ...m, by: 'you', when: m.sent }));
  const theirs = repliesOf(t).map(r => ({ ...r, by: 'agent', when: r.at }));
  return [...mine, ...theirs].sort((a, b) => time(a.when) - time(b.when))
    .concat(unsent(t).map(m => ({ ...m, by: 'you', when: null })));
}

function state(t) {
  if (t.resolved) return 'resolved';
  if (unsent(t).length) return 'draft';
  const lastSent = Math.max(...t.messages.map(m => time(m.sent)));
  return repliesOf(t).some(r => time(r.at) >= lastSent) ? 'answered' : 'waiting';
}

function unread(t) {
  return !t.resolved && repliesOf(t).some(r => time(r.at) > time(t.seen || t.created));
}

function numbers() {
  const out = {};
  shown().forEach((t, i) => { out[t.id] = i + 1; });
  return out;
}

function messageText(m) {
  return m.kind === 'integrate' ? 'Please integrate the answer into the paper.' : m.text;
}

function preview(t, n) {
  const last = timeline(t).filter(it => it.kind === 'integrate' || it.text).pop();
  if (!last) return 'Highlight only. Click to add a comment.';
  return (last.by === 'agent' ? 'Agent: ' : 'You: ') + clip(messageText(last).replace(/`([^`\n]+)`/g, '$1'), n);
}

function paintMarks() {
  const nums = numbers();
  for (const t of threads) {
    const marks = marksOf(t.id);
    const cls = 'rv-mark rv-' + state(t) + (unread(t) ? ' rv-unread' : '') + (t.resolved && showResolved ? ' rv-faint' : '');
    marks.forEach((m, j) => {
      m.className = cls;
      if (j === marks.length - 1 && nums[t.id]) m.dataset.n = nums[t.id];
      else delete m.dataset.n;
      m.title = preview(t, 140);
    });
  }
}

function hover(id, on) {
  for (const m of marksOf(id)) m.classList.toggle('rv-hover', on);
  if (ui.cards[id]) ui.cards[id].classList.toggle('rv-card-hover', on);
}

/* ---------- creating, writing, resolving ---------- */

function selectionInRoot() {
  const sel = window.getSelection();
  if (!sel || !sel.rangeCount || sel.isCollapsed) return null;
  const r = sel.getRangeAt(0);
  if (!root.contains(r.commonAncestorContainer)) return null;
  return { sel, range: r };
}

function createThread(range) {
  const text = fullText();
  let s = offsetOf(range.startContainer, range.startOffset);
  let e = offsetOf(range.endContainer, range.endOffset);
  while (s < e && /\s/.test(text[s])) s++;
  while (e > s && /\s/.test(text[e - 1])) e--;
  if (e <= s) return null;
  const box = range.getBoundingClientRect();
  const start = range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentElement;
  const page = start.closest('.page');
  const at = now();
  const t = {
    id: newId('c'), start: s, end: e, quote: text.slice(s, e),
    prefix: text.slice(Math.max(0, s - CONTEXT_CHARS), s), suffix: text.slice(e, e + CONTEXT_CHARS),
    section: sectionOf(range.startContainer),
    page: page ? Number(page.dataset.page) : 1,
    y: page ? (box.top - page.getBoundingClientRect().top) / page.offsetHeight : 0,
    created: at, messages: [{ id: newId('m'), kind: 'comment', text: '', at }]
  };
  threads.push(t);
  wrap(t);
  hideSelectionButton();
  openEditor(t.id, t.messages[0].id);
  return t.id;
}

function commentOnSelection() {
  const found = selectionInRoot();
  if (!found) return null;
  const range = found.range.cloneRange();
  found.sel.removeAllRanges();
  return createThread(range);
}

function activate(id) {
  active = id;
  const t = byId(id);
  if (t && unread(t)) { t.seen = now(); save(); }
  refresh();
}

// Scrolls the open card up when its bottom is behind the dock, without pushing its top out of the window. Runs
// when the open card or the dock changes size, not when the reader scrolls.
function keepClearOfDock() {
  const card = active && ui.cards[active];
  const key = card ? active + ':' + card.offsetHeight + ':' + ui.dock.offsetHeight : '';
  if (key === clearedFor) return;
  clearedFor = key;
  if (card) setTimeout(clearOfDock, 200);
}

function clearOfDock() {
  const card = active && ui.cards[active];
  if (!card || !ui.dock || card.style.display === 'none') return;
  const c = card.getBoundingClientRect();
  const d = ui.dock.getBoundingClientRect();
  if (c.right <= d.left || c.left >= d.right) return;
  const by = Math.min(c.bottom - d.top + 12, c.top - 12);
  if (by > 0) window.scrollBy({ top: by, behavior: 'smooth' });
}

function deactivate() {
  if (!active) return;
  active = null;
  refresh();
}

function openEditor(tid, mid) {
  if (editor && editor.tid === tid && editor.mid === mid) { editor.ta.focus({ preventScroll: true }); return; }
  closeEditor(true);
  const t = byId(tid);
  if (!t) return;
  const m = mid && t.messages.find(x => x.id === mid);
  const ta = el('textarea', 'rv-textarea');
  ta.value = m ? m.text : '';
  ta.placeholder = (m && m === t.messages[0] ? 'Your comment or question (optional).' : 'Your reply.') +
    ' Cmd+Enter saves, Esc cancels.';
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); closeEditor(true); }
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeEditor(false); }
  });
  editor = { tid, mid, ta };
  activate(tid);
  ta.focus({ preventScroll: true });
}

function closeEditor(commit) {
  if (!editor) return;
  const { tid, mid, ta } = editor;
  editor = null;
  const t = byId(tid);
  if (t && commit) {
    const text = ta.value.trim();
    const m = mid && t.messages.find(x => x.id === mid);
    if (m && !m.sent) {
      if (text || m === t.messages[0]) m.text = text;
      else t.messages = t.messages.filter(x => x !== m);
    } else if (!mid && text) {
      t.messages.push({ id: newId('m'), kind: 'comment', text, at: now() });
    }
  }
  save();
  refresh();
  setTimeout(onIdle, 0);
}

function integrate(t) {
  t.messages.push({ id: newId('m'), kind: 'integrate', text: '', at: now() });
  save();
  refresh();
}

function resolve(t, on) {
  if (on) {
    t.resolved = now();
    if (active === t.id) active = null;
    if (editor && editor.tid === t.id) closeEditor(true);
  } else {
    delete t.resolved;
  }
  save();
  refresh();
}

function removeThread(t) {
  const kept = t.messages.some(m => m.sent) || repliesOf(t).length;
  if (kept && !window.confirm('Delete this thread? Resolving it keeps it.')) return;
  if (editor && editor.tid === t.id) editor = null;
  unwrap(t.id);
  threads = threads.filter(x => x !== t);
  if (active === t.id) active = null;
  save();
  refresh();
}

function removeMessage(t, mid) {
  if (t.messages[0].id === mid) { removeThread(t); return; }
  if (editor && editor.mid === mid) editor = null;
  t.messages = t.messages.filter(m => m.id !== mid);
  save();
  refresh();
}

/* ---------- cards ---------- */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function button(label, cls, onclick, title) {
  const b = el('button', cls, label);
  b.type = 'button';
  if (title) b.title = title;
  if (onclick) b.addEventListener('click', onclick);
  return b;
}

function row(...children) {
  const r = el('div', 'rv-row');
  r.append(...children);
  return r;
}

function clip(text, n) {
  return text.length > n ? text.slice(0, n - 1) + '\u2026' : text;
}

function fmtTime(iso) {
  const d = new Date(iso);
  const hm = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  return d.toDateString() === new Date().toDateString() ? hm : d.toLocaleDateString([], { day: 'numeric', month: 'short' }) + ' ' + hm;
}

// Paragraphs, line breaks and `code`; KaTeX renders the math afterwards.
function formatted(text) {
  const div = el('div', 'rv-text');
  for (const para of text.split(/\n\s*\n/)) {
    const p = el('p');
    for (const part of para.split(/(`[^`\n]+`)/)) {
      p.append(/^`[^`]+`$/.test(part) ? el('code', '', part.slice(1, -1)) : document.createTextNode(part));
    }
    div.append(p);
  }
  return div;
}

function renderMath(node) {
  if (!mathRender) return;
  try { mathRender(node, { delimiters: MATH, throwOnError: false }); } catch (e) { /* leave the TeX source */ }
}

function label(t, st) {
  if (st === 'resolved') return 'Resolved ' + fmtTime(t.resolved);
  if (st === 'draft') return 'Not sent yet';
  if (st === 'waiting') return 'Sent ' + fmtTime(Math.max(...t.messages.map(m => time(m.sent))));
  const last = repliesOf(t)[repliesOf(t).length - 1];
  return 'Answered ' + fmtTime(last.at) + (repliesOf(t).some(r => r.changed) ? ', changed the paper' : '');
}

function cardFor(t) {
  let card = ui.cards[t.id];
  if (card) return card;
  card = el('div', 'rv-card');
  card.dataset.id = t.id;
  card.addEventListener('mouseenter', () => hover(t.id, true));
  card.addEventListener('mouseleave', () => hover(t.id, false));
  card.addEventListener('click', e => {
    if (e.target.closest('button, textarea, a')) return;
    if (active !== t.id) activate(t.id);
  });
  ui.cards[t.id] = card;
  ui.margin.appendChild(card);
  return card;
}

function editorBox(t, isFirst) {
  const box = el('div', 'rv-edit');
  const left = isFirst && !t.messages[0].sent && t.messages.length === 1
    ? button('Remove highlight', 'rv-btn rv-btn-danger', () => removeThread(t)) : el('span');
  box.append(editor.ta, row(left, el('span', 'rv-spacer'),
    button('Cancel', 'rv-btn rv-btn-quiet', () => closeEditor(false)),
    button('Save', 'rv-btn rv-btn-primary', () => closeEditor(true))));
  return box;
}

// The messages scroll; the reply box and the buttons under them stay in view.
function threadView(t) {
  const view = el('div', 'rv-open');
  const box = el('div', 'rv-thread');
  if (t.stale) box.append(el('div', 'rv-note', 'The highlighted passage changed since the comment was made. It was: \u201c' + clip(t.quote, 300) + '\u201d'));
  const items = timeline(t);
  const lastReply = items.filter(it => it.by === 'agent').pop();
  const asked = lastReply && t.messages.some(m => m.kind === 'integrate' && time(m.at) >= time(lastReply.at));
  for (const it of items) {
    const mine = it.by === 'you';
    const msg = el('div', 'rv-msg rv-by-' + it.by + (mine && !it.sent ? ' rv-unsent' : ''));
    const who = el('div', 'rv-who');
    who.append(el('span', 'rv-author', mine ? 'You' : 'Agent'),
      el('span', '', mine ? (it.sent ? fmtTime(it.sent) : 'not sent yet') : fmtTime(it.at) + (it.changed ? ', changed the paper' : '')));
    msg.append(who);
    const editing = editor && editor.tid === t.id && editor.mid === it.id;
    if (editing) {
      msg.append(editorBox(t, it.id === t.messages[0].id));
    } else if (it.kind === 'integrate') {
      msg.append(el('div', 'rv-text rv-request', messageText(it)));
    } else if (it.text) {
      msg.append(formatted(it.text));
    } else {
      msg.append(el('div', 'rv-text rv-empty', it.sent ? 'Highlighted, no comment.' : 'No comment yet. Click to write one.'));
    }
    if (mine && !it.sent && !editing) {
      who.append(el('span', 'rv-spacer'), button('\u00d7', 'rv-icon rv-x', e => { e.stopPropagation(); removeMessage(t, it.id); }, 'Remove'));
      if (it.kind !== 'integrate') {
        msg.title = 'Click to edit';
        msg.addEventListener('click', e => { if (!e.target.closest('button')) openEditor(t.id, it.id); });
      }
    }
    if (it === lastReply && !it.changed && !asked && !t.resolved) {
      msg.append(row(button('Integrate into the paper', 'rv-btn', () => integrate(t),
        'Ask the agent to work this answer into the paper (sent with the next Send to chat)')));
    }
    box.append(msg);
  }
  view.append(box);
  if (editor && editor.tid === t.id && editor.mid === null) {
    view.append(editorBox(t, false));
  } else if (!editor || editor.tid !== t.id) {
    view.append(row(button('Reply', 'rv-btn', () => openEditor(t.id, null)), el('span', 'rv-spacer'),
      button(t.resolved ? 'Reopen' : 'Resolve', 'rv-btn', () => resolve(t, !t.resolved))));
  }
  return view;
}

function previewView(t) {
  const box = el('div', 'rv-preview');
  if (t.stale) box.append(el('div', 'rv-note', 'The highlighted text changed.'));
  const text = preview(t, 260);
  box.append(el('div', 'rv-text' + (text.startsWith('Highlight only') ? ' rv-empty' : ''), text));
  return box;
}

function fillCard(card, t, n) {
  const st = state(t);
  const open = active === t.id;
  card.className = 'rv-card rv-card-' + st + (open ? ' rv-card-active' : '') + (t.stale ? ' rv-card-stale' : '');
  const head = el('div', 'rv-card-head');
  head.append(el('span', 'rv-n', n ? String(n) : '\u2013'), el('span', 'rv-state', label(t, st)));
  if (t.stale) {
    const tag = el('span', 'rv-tag', 'Stale');
    tag.title = 'The highlighted text is no longer in the PDF';
    head.append(tag);
  }
  if (unread(t)) head.append(el('span', 'rv-tag rv-tag-new', 'New reply'));
  head.append(
    button(t.resolved ? '\u21ba' : '\u2713', 'rv-icon', e => { e.stopPropagation(); resolve(t, !t.resolved); },
      t.resolved ? 'Reopen this thread' : 'Resolve this thread'),
    button('\u00d7', 'rv-icon rv-x', e => { e.stopPropagation(); removeThread(t); }, 'Delete this thread'));
  if (editor && editor.tid === t.id && card.contains(editor.ta)) {
    card.firstChild.replaceWith(head);
    return;
  }
  const old = card.querySelector('.rv-thread');
  const atBottom = !old || old.scrollTop + old.clientHeight >= old.scrollHeight - 4;
  const body = open ? threadView(t) : previewView(t);
  card.replaceChildren(head, body);
  renderMath(body);
  const box = body.querySelector('.rv-thread');
  if (box) box.scrollTop = atBottom ? box.scrollHeight : old.scrollTop;
}

function renderCards() {
  const keep = new Set();
  const nums = numbers();
  for (const t of shown()) {
    keep.add(t.id);
    fillCard(cardFor(t), t, nums[t.id]);
  }
  for (const id of Object.keys(ui.cards)) {
    if (!keep.has(id)) { ui.cards[id].remove(); delete ui.cards[id]; }
  }
}

/* ---------- layout ---------- */

// Viewport box of a thread's text; a stale thread uses where its text was.
function anchorRect(t, last) {
  const marks = marksOf(t.id);
  if (last) marks.reverse();
  for (const m of marks) {
    const r = m.getClientRects();
    if (r.length) return last ? r[r.length - 1] : r[0];
  }
  const page = root.querySelector(`.page[data-page="${t.page}"]`);
  if (!page) return null;
  const r = page.getBoundingClientRect();
  const y = r.top + (t.y || 0) * r.height;
  return { left: r.left + r.width * 0.1, top: y, bottom: y + 16 };
}

function placeMargin() {
  const r = root.getBoundingClientRect();
  const left = r.right + 28;
  const width = Math.min(300, window.innerWidth - left - 16);
  marginOn = width >= MARGIN_MIN - 40;
  ui.margin.classList.toggle('rv-popup-mode', !marginOn);
  ui.margin.style.left = marginOn ? left + window.scrollX + 'px' : '0px';
  ui.margin.style.width = marginOn ? width + 'px' : '0px';
}

// Cards sit next to their text and push each other down; the open card sits exactly next to its text. In a window
// too narrow for the margin, only the open card shows, under its text.
function positionCards() {
  const items = [];
  for (const t of shown()) {
    const card = ui.cards[t.id];
    if (!card) continue;
    if (!marginOn) {
      card.style.display = t.id === active ? '' : 'none';
      const r = t.id === active && anchorRect(t, true);
      if (r) {
        const width = Math.min(360, window.innerWidth - 16);
        card.style.width = width + 'px';
        card.style.left = Math.max(8, Math.min(r.left, window.innerWidth - width - 8)) + window.scrollX + 'px';
        card.style.top = r.bottom + window.scrollY + 8 + 'px';
      }
      continue;
    }
    card.style.display = '';
    card.style.width = '';
    card.style.left = '';
    const r = anchorRect(t, false);
    if (r) items.push({ id: t.id, card, anchor: r.top + window.scrollY, top: 0, h: card.offsetHeight });
  }
  if (!marginOn) return;
  items.sort((x, y) => x.anchor - y.anchor);
  items.forEach(it => { it.top = it.anchor; });
  let k = items.findIndex(it => it.id === active);
  if (k < 0) k = 0;
  for (let i = k + 1; i < items.length; i++) items[i].top = Math.max(items[i].anchor, items[i - 1].top + items[i - 1].h + CARD_GAP);
  for (let i = k - 1; i >= 0; i--) items[i].top = Math.min(items[i].anchor, items[i + 1].top - items[i].h - CARD_GAP);
  for (let i = 0; i < items.length; i++) {
    const floor = i ? items[i - 1].top + items[i - 1].h + CARD_GAP : 0;
    items[i].top = Math.max(items[i].top, floor);
    items[i].card.style.top = items[i].top + 'px';
  }
}

function layoutRail() {
  const docH = Math.max(document.documentElement.scrollHeight, 1);
  const vh = window.innerHeight;
  const ticks = [];
  for (const t of shown()) {
    const r = anchorRect(t, false);
    if (!r) continue;
    const tick = el('div', 'rv-tick rv-tick-' + state(t) + (unread(t) ? ' rv-tick-unread' : ''));
    tick.style.top = Math.min(vh - 4, (r.top + window.scrollY) / docH * vh) + 'px';
    tick.title = preview(t, 140);
    tick.addEventListener('click', () => reveal(t));
    ticks.push(tick);
  }
  ui.rail.replaceChildren(...ticks);
}

function reveal(t) {
  const r = anchorRect(t, false);
  if (r) window.scrollTo({ top: r.top + window.scrollY - window.innerHeight / 3, behavior: 'smooth' });
  const marks = marksOf(t.id);
  marks.forEach(m => m.classList.add('rv-flash'));
  setTimeout(() => marks.forEach(m => m.classList.remove('rv-flash')), 1400);
  activate(t.id);
}

function layout() {
  cancelAnimationFrame(frame);
  frame = 0;
  placeMargin();
  positionCards();
  layoutRail();
}

function schedule() {
  if (!frame) frame = requestAnimationFrame(layout);
}

function refresh() {
  if (!root) return;
  renderCards();
  paintMarks();
  updateDock();
  layout();
  keepClearOfDock();
}

/* ---------- the selection button ---------- */

function showSelectionButton() {
  const found = selectionInRoot();
  if (!found) { hideSelectionButton(); return; }
  const rects = found.range.getClientRects();
  const rect = rects.length ? rects[rects.length - 1] : found.range.getBoundingClientRect();
  ui.selButton.style.display = 'block';
  const w = ui.selButton.offsetWidth;
  ui.selButton.style.left = Math.max(8, Math.min(rect.right + window.scrollX - w / 2, window.scrollX + window.innerWidth - w - 8)) + 'px';
  ui.selButton.style.top = rect.bottom + window.scrollY + 6 + 'px';
}

function hideSelectionButton() {
  if (ui.selButton) ui.selButton.style.display = 'none';
}

/* ---------- the message to the agent ---------- */

function title() {
  return document.title.replace(/^Paper:\s*/, '');
}

function indent(text, pad) {
  return text.replace(/\n/g, '\n' + pad);
}

function promptItem(t, n) {
  const fresh = unsent(t);
  const isNew = !t.messages[0].sent;
  const kind = isNew ? 'New comment' : fresh.every(m => m.kind === 'integrate') ? 'Integrate request' : 'Follow-up';
  const out = [`${n}. ${kind}, id ${t.id}. Section: ${t.section}`];
  out.push(`   Quote${t.stale ? ' (this passage changed since the comment was made)' : ''}: "` +
    indent(clip(t.quote, MAX_QUOTE_CHARS), '   ') + '"');
  const earlier = timeline(t).filter(it => it.by === 'agent' || it.sent);
  if (earlier.length) {
    out.push('   Earlier in this thread:');
    for (const it of earlier) {
      const who = it.by === 'agent' ? 'You' : WHO;
      const text = messageText(it) || '(highlighted, no comment)';
      out.push(`     ${who} (${fmtTime(it.when)}${it.changed ? ', changed the paper' : ''}): ` + indent(clip(text, MAX_EARLIER_CHARS), '       '));
    }
  }
  for (const m of fresh) {
    if (m.kind === 'integrate') out.push(`   ${WHO} asks you to integrate your last answer above into the paper.`);
    else out.push(`   ${m === t.messages[0] ? 'Comment' : 'Follow-up'}: ` + (m.text ? indent(m.text, '   ') : '(highlighted, no comment)'));
  }
  return out.join('\n');
}

function buildPrompt(context, list) {
  const nums = numbers();
  const out = [`Comments on the paper "${title()}"`,
    `Page: ${CTX.path || PATH}, which shows latex/document.pdf. Follow "Comments on the paper" in AGENTS.md.`, ''];
  if (context.trim()) out.push(`Context from ${WHO}:`, context.trim(), '');
  if (list.length) {
    out.push('Comments, numbered as on the page:', '');
    for (const t of list) out.push(promptItem(t, nums[t.id] || '-'), '');
  }
  out.push('What to do:',
    `- Reply to every comment above with latex/review_reply.py. The reply shows in the comment's thread on the page, which is where ${WHO} reads it:`,
    "    python3 latex/review_reply.py <id> [--changed] <<'EOF'",
    '    <the reply; math as $...$ or $$...$$>',
    '    EOF',
    `- When ${WHO} asks a question or makes a remark and asks for no change, answer it in the reply and change nothing.`,
    `- When ${WHO} asks for a change, or asks you to integrate an answer, change the LaTeX source, rebuild with latex/build.sh, say in the reply what you changed, and pass --changed.`,
    '- Then end your chat reply with one sentence: how many comments you answered, and how many of them changed the paper.');
  return out.join('\n');
}

// What Becemder's chat shows of the message: what the user wrote.
function userLines(context, list) {
  const nums = numbers();
  const out = [];
  if (context.trim()) out.push(context.trim(), '');
  for (const t of list) {
    out.push(`${nums[t.id] || '-'}. \u201c${clip(t.quote, 120)}\u201d`);
    for (const m of unsent(t)) out.push('   ' + (messageText(m) || '(highlighted, no comment)'));
    out.push('');
  }
  return out.join('\n').trim();
}

/* ---------- sending ---------- */

function counts() {
  const c = { comments: 0, replies: 0, integrate: 0, resolved: 0, unread: 0 };
  for (const t of threads) {
    if (t.resolved) c.resolved++;
    if (unread(t)) c.unread++;
    for (const m of unsent(t)) {
      if (m.kind === 'integrate') c.integrate++;
      else if (m === t.messages[0]) c.comments++;
      else c.replies++;
    }
  }
  return c;
}

function plural(n, one, many) {
  return n + ' ' + (n === 1 ? one : many);
}

// With nothing to send, the dock shrinks to a button, so that it covers little of the cards behind it.
function updateDock() {
  if (!ui.count) return;
  const c = counts();
  const pending = c.comments + c.replies + c.integrate;
  const parts = [];
  if (c.comments) parts.push(plural(c.comments, 'new comment', 'new comments'));
  if (c.replies) parts.push(plural(c.replies, 'reply', 'replies'));
  if (c.integrate) parts.push(plural(c.integrate, 'integrate request', 'integrate requests'));
  ui.count.textContent = parts.join(' \u00b7 ');
  ui.resolvedBtn.style.display = c.resolved ? '' : 'none';
  ui.resolvedBtn.textContent = (showResolved ? 'Hide resolved' : 'Show resolved') + ` (${c.resolved})`;
  ui.sendBtn.disabled = sending || (!pending && !ui.context.value.trim());
  ui.unread.style.display = c.unread ? '' : 'none';
  ui.unreadText.textContent = c.unread === 1 ? 'New reply on one comment.' : `New replies on ${c.unread} comments.`;
  const open = dockOpen || sending || pending > 0 || !!ui.context.value.trim() || document.activeElement === ui.context;
  ui.dock.classList.toggle('rv-dock-compact', !open);
  ui.context.style.display = open ? '' : 'none';
  ui.count.style.display = open ? '' : 'none';
  ui.sendBtn.style.display = open ? '' : 'none';
  ui.writeBtn.style.display = open ? 'none' : '';
}

function status(text, isError, chat, hideAfter) {
  clearTimeout(statusTimer);
  if (hideAfter) statusTimer = setTimeout(() => status(''), hideAfter);
  ui.status.className = 'rv-status' + (isError ? ' rv-status-err' : '');
  ui.status.style.display = text ? '' : 'none';
  ui.status.replaceChildren(document.createTextNode(text || ''));
  if (chat && CTX.becemder) {
    const a = el('a', '', 'Open the chat');
    a.href = CTX.becemder + '/#chat=' + chat;
    a.target = '_blank';
    a.rel = 'noopener';
    ui.status.append(' ', a);
  }
}

function copyText(text, done) {
  if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(() => done(true), () => done(false));
  else done(false);
}

function markSent(list) {
  const at = now();
  for (const t of list) for (const m of unsent(t)) m.sent = at;
  ui.context.value = '';
  store(CONTEXT_KEY, '');
  save();
  refresh();
}

function send() {
  closeEditor(true);
  const list = sorted().filter(t => unsent(t).length);
  const context = ui.context.value;
  if (!list.length && !context.trim()) return;
  const text = buildPrompt(context, list);
  if (!CTX.feedback) {
    copyText(text, ok => {
      status(ok ? 'Copied. Paste it into the chat.' : 'Copying failed.', !ok, null, ok && 8000);
      if (ok) markSent(list);
    });
    return;
  }
  sending = true;
  updateDock();
  status('Sending\u2026');
  fetch(CTX.feedback, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, comments: userLines(context, list), ticket: CTX.ticket || null, path: CTX.path || null })
  }).then(async r => {
    const out = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(out.error || 'error ' + r.status);
    return out;
  }).then(out => {
    sending = false;
    markSent(list);
    if (out.draft) {
      status('Becemder has no chat for this page yet, so the comments wait as a draft in its chat pane. Send them ' +
        'from there; later comments go straight to that chat.', false, null, 30000);
    } else {
      status(out.queued ? `Sent. \u201c${out.title}\u201d is still answering, so the comments start when its reply ends.`
        : `Sent to \u201c${out.title}\u201d.`, false, out.thread, 12000);
    }
  }, err => {
    sending = false;
    updateDock();
    copyText(text, ok => status('Sending failed: ' + err.message + '.' + (ok ? ' The prompt is copied, so you can paste it into the chat.' : ''), true));
  });
}

/* ---------- the UI ---------- */

function buildUi() {
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);

  ui.selButton = button('Comment', 'rv-ui rv-selbtn', commentOnSelection);
  ui.selButton.addEventListener('mousedown', e => e.preventDefault());
  document.body.appendChild(ui.selButton);
  ui.margin = el('div', 'rv-ui rv-margin');
  document.body.appendChild(ui.margin);
  ui.rail = el('div', 'rv-ui rv-rail');
  document.body.appendChild(ui.rail);

  const dock = el('div', 'rv-ui rv-dock');
  ui.unreadText = el('span');
  ui.unread = row(ui.unreadText, el('span', 'rv-spacer'), button('Show', 'rv-btn', () => {
    const t = shown().find(unread);
    if (t) reveal(t);
  }));
  ui.unread.classList.add('rv-unread-row');
  ui.context = el('textarea', 'rv-textarea rv-context');
  ui.context.placeholder = 'Context for the agent (optional). Cmd+Enter sends.';
  ui.context.value = stored(CONTEXT_KEY);
  ui.context.addEventListener('input', () => { store(CONTEXT_KEY, ui.context.value); updateDock(); });
  ui.context.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); send(); }
    if (e.key === 'Escape') ui.context.blur();
  });
  ui.context.addEventListener('blur', () => { dockOpen = false; updateDock(); setTimeout(onIdle, 0); });
  ui.writeBtn = button('Write to the agent', 'rv-btn rv-btn-link', () => {
    dockOpen = true;
    updateDock();
    ui.context.focus();
  });
  ui.count = el('span', 'rv-count');
  ui.resolvedBtn = button('', 'rv-btn rv-btn-link', () => {
    showResolved = !showResolved;
    store(RESOLVED_KEY, showResolved ? '1' : '');
    refresh();
  });
  ui.sendBtn = button(CTX.feedback ? 'Send to chat' : 'Copy prompt', 'rv-btn rv-btn-primary', send);
  ui.status = el('div', 'rv-status');
  dock.append(ui.unread, ui.context, row(ui.writeBtn, ui.count, ui.resolvedBtn, el('span', 'rv-spacer'), ui.sendBtn), ui.status);
  ui.dock = dock;
  document.body.appendChild(dock);
  status('');
  updateDock();
}

function bindEvents() {
  document.addEventListener('mouseup', e => {
    if (e.target.closest && e.target.closest('.rv-ui')) return;
    setTimeout(showSelectionButton, 0);
  });
  document.addEventListener('keyup', e => {
    if (e.key === 'Shift' || e.shiftKey) setTimeout(showSelectionButton, 0);
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !editor) deactivate();
  });
  document.addEventListener('mousedown', e => {
    if (e.target.closest && e.target.closest('.rv-ui')) return;
    hideSelectionButton();
    if (e.target.closest && e.target.closest('mark.rv-mark')) return;
    if (editor) closeEditor(true);
    else deactivate();
  });
  root.addEventListener('click', e => {
    const m = e.target.closest && e.target.closest('mark.rv-mark');
    if (!m) return;
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed) return;
    e.preventDefault();
    const t = byId(m.dataset.id);
    if (!t) return;
    if (t.messages.length === 1 && !t.messages[0].sent) openEditor(t.id, t.messages[0].id);
    else activate(t.id);
  });
  root.addEventListener('mouseover', e => {
    const m = e.target.closest && e.target.closest('mark.rv-mark');
    if (m) hover(m.dataset.id, true);
  });
  root.addEventListener('mouseout', e => {
    const m = e.target.closest && e.target.closest('mark.rv-mark');
    if (m) hover(m.dataset.id, false);
  });
  window.addEventListener('resize', schedule);
  new ResizeObserver(schedule).observe(document.body);
}

/* ---------- replies ---------- */

async function loadReplies() {
  let raw = '';
  try {
    const r = await fetch(repliesUrl, { cache: 'no-store' });
    if (r.ok) raw = await r.text();
  } catch (e) { /* no replies yet */ }
  const out = {};
  for (const line of raw.split('\n')) {
    const s = line.trim();
    if (!s || s.startsWith('#')) continue;
    try {
      const r = JSON.parse(s);
      if (r && r.id && typeof r.text === 'string') (out[r.id] ||= []).push(r);
    } catch (e) { /* a broken line */ }
  }
  replies = out;
  const t = active && byId(active);
  if (t && unread(t) && !editor) { t.seen = now(); save(); }
  refresh();
}

/* ---------- start ---------- */

export async function startComments(opts) {
  root = opts.root;
  repliesUrl = opts.replies;
  onIdle = opts.onIdle || (() => {});
  showResolved = stored(RESOLVED_KEY) === '1';
  threads = load().filter(t => t && t.id && Array.isArray(t.messages) && t.messages.length);
  importOld();
  buildUi();
  const text = fullText();
  for (const t of threads) t.stale = !locate(t, text);
  for (const t of sorted()) if (!t.stale) wrap(t);
  for (const t of threads) if (t.imported) placeImported(t);
  save();
  bindEvents();
  // Becemder's live.js and review.html wait with a reload while a message is being written or sent.
  window.ReportAnnotations = { editing: () => !!editor || sending, list: () => threads.slice() };
  await loadReplies();
  return { loadReplies };
}

const CSS = `
mark.rv-mark{color:inherit;border-radius:2px;padding:0 1px;cursor:pointer;background:rgba(255,214,90,.55)}
mark.rv-mark.rv-waiting{background:rgba(0,0,0,.05);box-shadow:inset 0 -1px 0 rgba(0,0,0,.32)}
mark.rv-mark.rv-answered{background:rgba(59,123,191,.13);box-shadow:inset 0 -2px 0 rgba(59,123,191,.7)}
mark.rv-mark.rv-unread{background:rgba(59,123,191,.3)}
mark.rv-mark.rv-resolved{background:none;box-shadow:none;cursor:text}
mark.rv-mark.rv-faint{box-shadow:inset 0 -1px 0 rgba(0,0,0,.18);cursor:pointer}
mark.rv-mark.rv-hover{background:rgba(255,196,0,.7)}
mark.rv-mark.rv-flash{background:rgba(255,196,0,.85);transition:background .3s}
mark.rv-mark[data-n]::after{content:attr(data-n);font:600 11px/1 var(--sans,sans-serif);vertical-align:super;margin-left:2px;color:#8a6d00}
mark.rv-mark.rv-waiting[data-n]::after{color:rgba(0,0,0,.45)}
mark.rv-mark.rv-answered[data-n]::after{color:#2c5f94}
.rv-ui{font-family:var(--sans,sans-serif);font-size:14px;line-height:1.4;color:rgba(0,0,0,.85)}
.rv-btn{font:inherit;font-size:12.5px;border-radius:4px;padding:3px 9px;cursor:pointer;border:1px solid rgba(0,0,0,.18);background:#fff;color:rgba(0,0,0,.8)}
.rv-btn:hover{background:#f3f3f3}
.rv-btn:disabled{opacity:.45;cursor:default}
.rv-btn-primary{background:#5f9b65;border-color:#5f9b65;color:#fff;font-size:13.5px;padding:5px 11px}
.rv-btn-primary:hover{background:#426c46}
.rv-btn-primary:disabled:hover{background:#5f9b65}
.rv-btn-quiet{border-color:transparent;background:transparent}
.rv-btn-danger{border-color:transparent;background:transparent;color:#b5483b;padding-left:0}
.rv-btn-danger:hover{background:transparent;text-decoration:underline}
.rv-btn-link{border:none;background:none;padding:0 4px;color:rgba(0,0,0,.5)}
.rv-btn-link:hover{background:none;color:rgba(0,0,0,.8);text-decoration:underline}
.rv-icon{border:none;background:none;padding:0 3px;font-size:14px;line-height:1;color:rgba(0,0,0,.38);cursor:pointer}
.rv-icon:hover{color:#2c5f94}
.rv-x{font-size:16px}
.rv-x:hover{color:#b5483b}
.rv-selbtn{position:absolute;z-index:1001;display:none;font:inherit;font-family:var(--sans,sans-serif);font-size:13.5px;padding:5px 10px;border-radius:4px;border:1px solid #426c46;background:#5f9b65;color:#fff;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.18)}
.rv-textarea{display:block;box-sizing:border-box;width:100%;font:inherit;font-size:13.5px;border:1px solid rgba(0,0,0,.2);border-radius:4px;padding:7px 9px;background:#fff;resize:vertical;min-height:72px}
.rv-row{display:flex;align-items:center;gap:6px;margin-top:6px}
.rv-spacer{flex:1}
.rv-margin{position:absolute;top:0;z-index:900}
.rv-card{position:absolute;left:0;right:0;box-sizing:border-box;background:#fffdf9;border:1px solid rgba(0,0,0,.13);border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.07);padding:7px 10px 8px;font-size:13.5px;cursor:pointer;transition:top .15s ease,transform .15s ease,box-shadow .15s ease}
.rv-popup-mode .rv-card{right:auto}
.rv-card-hover{border-color:rgba(59,123,191,.5)}
.rv-card-active{transform:translateX(-8px);box-shadow:0 4px 16px rgba(0,0,0,.14);border-color:rgba(59,123,191,.6);z-index:1;cursor:default}
.rv-popup-mode .rv-card-active{transform:none}
.rv-card-answered{border-left:3px solid rgba(59,123,191,.75)}
.rv-card-waiting .rv-preview .rv-text{color:rgba(0,0,0,.6)}
.rv-card-resolved{opacity:.62}
.rv-card-stale{border-style:dashed}
.rv-card-head{display:flex;align-items:center;gap:6px;margin-bottom:3px;font-size:12px;color:rgba(0,0,0,.45)}
.rv-n{font-weight:600;color:#426c46}
.rv-state{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rv-tag{font-size:11px;border-radius:3px;padding:0 5px;background:rgba(0,0,0,.07);color:rgba(0,0,0,.6);white-space:nowrap}
.rv-tag-new{background:rgba(59,123,191,.16);color:#2c5f94}
.rv-preview .rv-text{display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden;white-space:pre-wrap;word-wrap:break-word}
.rv-thread{max-height:55vh;overflow:auto;margin-right:-4px;padding-right:4px}
.rv-msg{margin-top:8px}
.rv-who{display:flex;align-items:center;gap:5px;font-size:12px;color:rgba(0,0,0,.45);margin-bottom:2px}
.rv-author{font-weight:600;color:rgba(0,0,0,.6)}
.rv-by-agent .rv-author{color:#2c5f94}
.rv-text{word-wrap:break-word}
.rv-text p{margin:0 0 6px;white-space:pre-wrap}
.rv-text p:last-child{margin-bottom:0}
.rv-by-agent .rv-text{border-left:2px solid rgba(59,123,191,.5);padding-left:8px}
.rv-unsent{cursor:text}
.rv-unsent .rv-text{color:rgba(0,0,0,.72)}
.rv-request{font-style:italic}
.rv-empty{color:rgba(0,0,0,.45)}
.rv-note{font-size:12.5px;color:#7a5200;background:rgba(224,173,0,.12);border-radius:4px;padding:4px 7px;margin:3px 0}
.rv-text code{font-size:12px;background:rgba(0,0,0,.05);padding:0 3px;border-radius:3px}
.rv-text .katex-display{margin:4px 0;overflow-x:auto;overflow-y:hidden}
.rv-edit{margin-top:4px}
.rv-rail{position:fixed;top:0;right:0;bottom:0;width:10px;z-index:999;pointer-events:none}
.rv-tick{position:absolute;left:1px;right:0;height:4px;border-radius:1px;background:#e0ad00;pointer-events:auto;cursor:pointer}
.rv-tick-waiting{background:rgba(0,0,0,.28)}
.rv-tick-answered{background:rgba(59,123,191,.75)}
.rv-tick-unread{background:#2c5f94;height:6px}
.rv-tick-resolved{background:rgba(0,0,0,.15)}
.rv-tick:hover{transform:scaleY(1.6)}
.rv-dock{position:fixed;right:22px;bottom:22px;z-index:1000;display:flex;flex-direction:column;gap:6px;width:300px;background:#fffdf9;border:1px solid rgba(0,0,0,.15);border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.12);padding:10px 12px}
.rv-dock .rv-row{margin-top:0}
.rv-dock-compact{width:auto;max-width:300px;padding:6px 12px}
.rv-context{height:48px;min-height:30px;resize:vertical}
.rv-count{color:rgba(0,0,0,.6);font-size:13px}
.rv-unread-row{font-size:13px;color:#2c5f94}
.rv-status{font-size:12.5px;color:#426c46}
.rv-status a{color:#426c46}
.rv-status-err{color:#b5483b}
@media print{.rv-ui{display:none!important}mark.rv-mark{background:none;box-shadow:none}mark.rv-mark::after{content:none}}
`;
