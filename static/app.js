/* VarStar Lab - front end */
'use strict';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const fmt = (v, d = 3) => (v === null || v === undefined || !isFinite(v)) ? '—' : Number(v).toFixed(d);
const sci = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? '—' : Number(v).toExponential(d);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const commas = (v, d = 0) => (v === null || v === undefined || !isFinite(v)) ? '—'
  : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

const S = {
  sid: null, meta: null, sources: [], target: null, comps: [], compMags: {},
  shape: [0, 0], preset: 'single_night', mode: 'target',
  img: null, zoom: 1, panX: 0, panY: 0, fitZoom: 1,
  period: null, frames: null, calibrated: false, plotTheme: 'dark',
};

/* --------------------------------------------------------------- plumbing */
let toastTimer = null;
function toast(msg, kind = '', ms = 5200) {
  const old = document.querySelector('.toast');
  if (old) old.remove();
  clearTimeout(toastTimer);
  const t = el('div', 'toast ' + kind, esc(msg));
  document.body.appendChild(t);
  toastTimer = setTimeout(() => t.remove(), ms);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = j.detail || JSON.stringify(j); } catch (e) { }
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('json') ? res.json() : res.text();
}
const post = (path, body) => api(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}),
});

/* Poll a background job until it finishes. */
async function waitJob(progEl, label) {
  const bar = progEl.querySelector('.bar > div');
  const txt = progEl.querySelector('.txt');
  progEl.style.display = 'block';
  for (; ;) {
    await new Promise(r => setTimeout(r, 380));
    let j;
    try { j = await api(`/api/session/${S.sid}/job`); }
    catch (e) { progEl.style.display = 'none'; throw e; }
    const pct = j.total ? Math.round(100 * j.current / j.total) : 0;
    bar.style.width = pct + '%';
    txt.textContent = `${j.stage || label}  ${j.total ? `${j.current}/${j.total}` : ''}  ${j.message || ''}`;
    if (!j.running) {
      progEl.style.display = 'none';
      if (j.error) throw new Error(j.error);
      return j;
    }
  }
}

function unlock(n) {
  const s = $('step' + n);
  s.classList.remove('locked');
  return s;
}
function openStep(n) {
  const s = unlock(n);
  s.classList.add('open');
  setTimeout(() => s.scrollIntoView({ behavior: 'smooth', block: 'start' }), 90);
}
function markDone(n) { $('step' + n).classList.add('done'); }

document.querySelectorAll('section.step > .head').forEach(h => {
  h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
});

function noticeHTML(level, text) {
  const ico = { critical: '!', warning: '▲', ok: '✓', info: 'i' }[level] || 'i';
  return `<div class="notice ${level}"><span class="ico">${ico}</span><span>${esc(text)}</span></div>`;
}

function plotBlock(name, title, why) {
  const bust = Date.now();
  return `<div class="plot">
    <div class="cap"><h4>${esc(title)}</h4><span class="why">${esc(why || '')}</span>
      <a class="small" href="/api/session/${S.sid}/plot/${name}?theme=light&t=${bust}"
         download="${name}.png" style="font-size:11.5px">download for print</a></div>
    <img src="/api/session/${S.sid}/plot/${name}?theme=${S.plotTheme}&t=${bust}"
         alt="${esc(title)}" loading="lazy">
  </div>`;
}

/* ================================================== boot ================ */
async function boot() {
  const r = await post('/api/session');
  S.sid = r.session_id;
  S.meta = r;

  const chip = $('targetChip');
  const t = r.target_default;
  chip.textContent = `${t.name} · ${t.vtype} · V ${t.mag_max}–${t.mag_min} · P ${t.period_cat} d`;
  chip.title = `RA ${t.ra_str}  Dec ${t.dec_str}\n${t.other_names}`;

  const pc = $('presets');
  pc.innerHTML = '';
  Object.entries(r.presets).forEach(([k, v]) => {
    const b = el('button', 'preset' + (k === S.preset ? ' sel' : ''),
      `<span class="pl">${esc(v.label)}</span><span class="pd">${esc(v.description)}</span>`);
    b.onclick = () => {
      S.preset = k;
      pc.querySelectorAll('.preset').forEach(x => x.classList.remove('sel'));
      b.classList.add('sel');
    };
    pc.appendChild(b);
  });

  const rel = $('relation');
  rel.innerHTML = '';
  Object.entries(r.relations).forEach(([k, v]) => {
    const o = el('option');
    o.value = k;
    o.textContent = `${v.label}   (M_V = ${v.slope} log P ${v.intercept >= 0 ? '+' : '−'} ${Math.abs(v.intercept)})`;
    rel.appendChild(o);
  });
  rel.onchange = () => {
    const v = r.relations[rel.value];
    $('relNote').innerHTML = noticeHTML('info', v.note);
  };
  rel.insertAdjacentHTML('afterend', '<div id="relNote"></div>');
  rel.onchange();
}

/* ================================================== step 1 ============== */
$('dropzone').onclick = () => $('fileInput').click();
$('dropzone').addEventListener('dragover', e => {
  e.preventDefault(); $('dropzone').classList.add('over');
});
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('over'));
$('dropzone').addEventListener('drop', e => {
  e.preventDefault(); $('dropzone').classList.remove('over');
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});
$('fileInput').onchange = e => { if (e.target.files.length) uploadFiles(e.target.files); };

async function uploadFiles(files) {
  const fd = new FormData();
  let n = 0;
  for (const f of files) { fd.append('files', f); n++; }
  const prog = $('prog1');
  prog.style.display = 'block';
  prog.querySelector('.txt').textContent = `uploading ${n} file(s)…`;
  prog.querySelector('.bar > div').style.width = '8%';
  try {
    await api(`/api/session/${S.sid}/upload`, { method: 'POST', body: fd });
    await waitJob(prog, 'reading frames');
    await loadFrames();
  } catch (e) { prog.style.display = 'none'; toast('Upload failed: ' + e.message, 'err', 9000); }
}

$('btnDemo').onclick = async () => {
  const body = { preset: S.preset, second_mode: $('demoSecondMode').checked };
  const nf = parseInt($('demoFrames').value); if (nf) body.n_frames = nf;
  const cd = parseFloat($('demoCadence').value); if (cd) body.cadence = cd;
  const sd = parseInt($('demoSeed').value); if (!isNaN(sd)) body.seed = sd;
  $('btnDemo').disabled = true;
  try {
    await post(`/api/session/${S.sid}/demo`, body);
    await waitJob($('prog1'), 'simulating');
    await loadFrames();
    toast('Synthetic run generated. The pipeline does not know the injected values.', 'ok');
  } catch (e) { toast('Simulation failed: ' + e.message, 'err', 9000); }
  $('btnDemo').disabled = false;
};

async function loadFrames() {
  const f = await api(`/api/session/${S.sid}/frames`);
  S.frames = f;
  const out = $('framesOut');
  if (!f.n_frames) { out.innerHTML = ''; return; }

  let html = `<div class="cards">
    <div class="card"><div class="k">frames</div><div class="v">${f.n_frames}</div>
      <div class="n">${f.n_usable} usable</div></div>
    <div class="card"><div class="k">time span</div><div class="v">${fmt(f.span_hours, 2)}<small> h</small></div>
      <div class="n">${f.n_sessions} session${f.n_sessions === 1 ? '' : 's'}</div></div>
    <div class="card"><div class="k">timestamps</div><div class="v">${f.n_with_time}</div>
      <div class="n">${f.frames[0] ? esc(String(f.frames[0].time_source).slice(0, 34)) : ''}</div></div>
    <div class="card"><div class="k">format</div><div class="v" style="font-size:15px">${esc(f.kinds.join(', '))}</div>
      <div class="n">${f.frames[0] ? f.frames[0].shape.join(' × ') : ''} px</div></div>
  </div>`;

  if (f.is_synthetic && f.synthetic_truth) {
    const t = f.synthetic_truth;
    html += noticeHTML('info',
      `Synthetic data. Injected: P = ${t.period} d, peak-to-peak ${fmt(t.amplitude_p2p, 3)} mag, ` +
      `mean V = ${fmt(t.mean_mag, 3)}${t.second_mode ? `, plus an overtone at ${fmt(t.period_overtone, 6)} d` : ''}. ` +
      `Compare these with what the pipeline recovers.`);
  }

  if (f.needs_manual_times) {
    html += noticeHTML('warning',
      `${f.n_frames - f.n_with_time} frame(s) carry no usable timestamp. Photometry can still run, ` +
      `but a period needs times — enter the start of the first exposure and the cadence below.`);
    html += `<div class="row tight" style="margin:10px 0">
      <label class="field"><span class="lbl">First exposure start (UTC)</span>
        <input type="text" id="mtStart" placeholder="2026-07-14T14:10:00" style="width:200px"></label>
      <label class="field"><span class="lbl">Cadence (s)</span>
        <input type="number" id="mtCad" value="45" step="1" style="width:100px"></label>
      <label class="field"><span class="lbl">Exposure (s)</span>
        <input type="number" id="mtExp" value="40" step="1" style="width:100px"></label>
      <button id="btnTimes">Apply timestamps</button></div>`;
  }

  const spanNote = f.span_hours > 0 ? spanAdvice(f.span_hours, f.n_sessions, f.sessions) : '';
  html += spanNote;

  html += `<details><summary>Frame list (${f.n_frames})</summary><div class="dbody">
    <div class="tablewrap"><table><thead><tr>
      <th>#</th><th>file</th><th>JD (mid-exposure)</th><th>exp (s)</th>
      <th>filter</th><th>size</th><th>time from</th></tr></thead><tbody>`;
  f.frames.slice(0, 400).forEach((fr, i) => {
    html += `<tr><td>${i + 1}</td><td>${esc(fr.filename)}</td>
      <td class="num">${fr.jd ? fr.jd.toFixed(6) : '—'}</td>
      <td class="num">${fr.exptime ? fmt(fr.exptime, 1) : '—'}</td>
      <td>${esc(fr.filter_name || '—')}</td>
      <td class="num">${fr.shape.join('×')}</td>
      <td style="text-align:left;color:var(--dim)">${esc(String(fr.time_source).slice(0, 40))}</td></tr>`;
  });
  html += `</tbody></table></div>${f.n_frames > 400 ? '<p class="hint">showing the first 400</p>' : ''}
    </div></details>`;

  out.innerHTML = html;
  if ($('btnTimes')) $('btnTimes').onclick = applyTimes;

  $('hint1').textContent = `${f.n_frames} frames · ${fmt(f.span_hours, 2)} h · ${f.n_with_time} timestamped`;
  markDone(1);
  $('refFrame').max = f.n_frames;
  openStep(2);
}

function spanAdvice(hours, nSessions, sessions) {
  const P = (S.meta && S.meta.target_default.period_cat) || 0.1071934;
  const Ph = P * 24;
  const cycles = hours / Ph;
  let longest = hours;
  if (sessions && sessions.length) longest = Math.max(...sessions.map(s => s.span_days)) * 24;
  const cl = longest / Ph;
  let out = '';
  if (cl < 1.0) {
    out += noticeHTML('warning',
      `Your longest continuous run is ${fmt(longest, 2)} h — only ${fmt(cl, 2)} of the ` +
      `${fmt(Ph, 2)} h catalogued period. A run shorter than one full cycle leaves the period ` +
      `ambiguous at the 1 cycle/day level, and stacking on more short nights does not fix it. ` +
      `If you can still change the observing plan, make one session longer than ${fmt(Ph, 1)} h.`);
  } else {
    out += noticeHTML('ok',
      `Longest continuous run covers ${fmt(cl, 2)} pulsation cycles — enough for the curve shape ` +
      `itself to pin down the frequency.`);
  }
  if (nSessions > 1) {
    out += noticeHTML('info', `${nSessions} sessions spanning ${fmt(cycles, 1)} cycles in total. ` +
      `The long baseline sharpens precision; watch the periodogram for aliases spaced by 1 cycle/day.`);
  }
  return out;
}

async function applyTimes() {
  try {
    await post(`/api/session/${S.sid}/times`, {
      start_utc: $('mtStart').value.trim(),
      cadence_s: parseFloat($('mtCad').value),
      exptime_s: parseFloat($('mtExp').value) || null,
    });
    toast('Timestamps applied.', 'ok');
    await loadFrames();
  } catch (e) { toast(e.message, 'err', 9000); }
}

/* ================================================== step 2 ============== */
$('btnDetect').onclick = async () => {
  const btn = $('btnDetect'); btn.disabled = true;
  try {
    const idx = Math.max(0, (parseInt($('refFrame').value) || 1) - 1);
    const r = await post(`/api/session/${S.sid}/detect`, {
      frame: idx, fwhm: parseFloat($('detFwhm').value),
      thresh_sigma: parseFloat($('detThresh').value),
      channel: $('channel').value,
    });
    S.sources = r.sources; S.shape = r.shape;
    S.target = r.suggested_target;
    S.comps = (r.suggested_comps || []).slice();
    S.compMags = {};
    if (r.measured_fwhm) {
      $('phFwhm').value = r.measured_fwhm.toFixed(2);
      $('detFwhm').value = r.measured_fwhm.toFixed(1);
    }
    if (r.gain) $('phGain').value = r.gain;

    let html = `<div class="cards">
      <div class="card"><div class="k">sources found</div><div class="v">${r.n}</div></div>
      <div class="card"><div class="k">measured FWHM</div><div class="v">${fmt(r.measured_fwhm, 2)}<small> px</small></div>
        <div class="n">from the radial profile</div></div>
      <div class="card"><div class="k">sky level</div><div class="v">${fmt(r.background.median, 1)}<small> ADU</small></div>
        <div class="n">σ = ${fmt(r.background.std, 2)}</div></div>
      <div class="card"><div class="k">apertures</div><div class="v" style="font-size:15px">${fmt(r.aperture.r_ap, 1)} / ${fmt(r.aperture.r_in, 1)}–${fmt(r.aperture.r_out, 1)}</div>
        <div class="n">star / sky annulus, px</div></div></div>`;
    if (!r.n) html += noticeHTML('critical',
      'No stars detected. Lower the threshold, check the FWHM, or try a different channel — ' +
      'and make sure this is a light frame, not a dark or bias.');
    const nsat = r.sources.filter(s => s.saturated).length;
    if (nsat) html += noticeHTML('warning',
      `${nsat} detected star(s) are saturated (peak ≥ 97% of full well). Saturated pixels ` +
      `respond non-linearly and cannot be used — they are drawn dimmed and are excluded from auto-pick.`);
    $('detectOut').innerHTML = html;

    $('canvasWrap').style.display = 'block';
    await loadPreview(idx);
    renderPicks();
  } catch (e) { toast('Detection failed: ' + e.message, 'err', 9000); }
  btn.disabled = false;
};

async function loadPreview(idx) {
  const url = `/api/session/${S.sid}/preview?frame=${idx}&channel=${$('channel').value}` +
    `&stretch=${$('stretch').value}&t=${Date.now()}`;
  await new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => { S.img = im; res(); };
    im.onerror = () => rej(new Error('preview failed'));
    im.src = url;
  });
  zoomFit();
}

const cv = $('fieldCanvas');
const ctx = cv.getContext('2d');

function zoomFit() {
  if (!S.img) return;
  const w = cv.parentElement.clientWidth;
  cv.width = w;
  cv.height = Math.round(w * S.img.height / S.img.width);
  S.fitZoom = 1; S.zoom = 1; S.panX = 0; S.panY = 0;
  draw();
}

/* image pixel -> canvas pixel */
function toCanvas(x, y) {
  const sc = (cv.width / S.shape[1]) * S.zoom;
  return [x * sc + S.panX, y * sc + S.panY];
}
function toImage(cx, cy) {
  const sc = (cv.width / S.shape[1]) * S.zoom;
  return [(cx - S.panX) / sc, (cy - S.panY) / sc];
}

function draw() {
  if (!S.img) return;
  ctx.fillStyle = '#05080d';
  ctx.fillRect(0, 0, cv.width, cv.height);
  const sc = (cv.width / S.shape[1]) * S.zoom;
  ctx.imageSmoothingEnabled = S.zoom < 3;
  ctx.drawImage(S.img, S.panX, S.panY, S.shape[1] * sc, S.shape[0] * sc);

  const rAp = Math.max(4, (parseFloat($('phAp').value) || 1.5) * (parseFloat($('phFwhm').value) || 4) * sc);
  const showAll = $('showAll').checked;

  S.sources.forEach((s, i) => {
    const [x, y] = toCanvas(s.x, s.y);
    if (x < -40 || y < -40 || x > cv.width + 40 || y > cv.height + 40) return;
    const isT = i === S.target;
    const ci = S.comps.indexOf(i);
    const isC = ci >= 0;
    if (!isT && !isC && !showAll) return;

    let col = s.saturated ? '#6b5a3a' : 'rgba(139,152,169,0.55)';
    let lw = 1, r = rAp * 0.75, label = '';
    if (isT) { col = '#ff6b6b'; lw = 2.2; r = rAp; label = 'T'; }
    else if (isC) { col = '#3fd68a'; lw = 1.8; r = rAp * 0.9; label = 'C' + (ci + 1); }

    ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832);
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.stroke();

    if (isT || isC) {
      // sky annulus
      const rin = (parseFloat($('phIn').value) || 3) * (parseFloat($('phFwhm').value) || 4) * sc;
      const rout = (parseFloat($('phOut').value) || 5) * (parseFloat($('phFwhm').value) || 4) * sc;
      ctx.setLineDash([3, 4]); ctx.lineWidth = 1; ctx.strokeStyle = col + '';
      ctx.globalAlpha = 0.45;
      ctx.beginPath(); ctx.arc(x, y, rin, 0, 6.2832); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, rout, 0, 6.2832); ctx.stroke();
      ctx.globalAlpha = 1; ctx.setLineDash([]);
      ctx.fillStyle = col; ctx.font = '600 12px ui-monospace, monospace';
      ctx.fillText(label, x + r + 4, y + 4);
    }
    if (s.saturated) {
      ctx.strokeStyle = '#ffb454'; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x - r * 0.7, y - r * 0.7); ctx.lineTo(x + r * 0.7, y + r * 0.7);
      ctx.stroke();
    }
  });
  $('zoomLbl').textContent = S.zoom.toFixed(1) + '×';
}

let dragging = false, dragStart = null, moved = 0;
cv.addEventListener('pointerdown', e => {
  dragging = true; moved = 0;
  dragStart = { x: e.offsetX, y: e.offsetY, px: S.panX, py: S.panY };
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove', e => {
  if (!dragging) return;
  const dx = e.offsetX - dragStart.x, dy = e.offsetY - dragStart.y;
  moved = Math.max(moved, Math.abs(dx) + Math.abs(dy));
  S.panX = dragStart.px + dx; S.panY = dragStart.py + dy;
  draw();
});
cv.addEventListener('pointerup', e => {
  dragging = false;
  if (moved < 5) clickStar(e.offsetX, e.offsetY);
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.18 : 1 / 1.18;
  const nz = Math.min(24, Math.max(1, S.zoom * f));
  const [ix, iy] = toImage(e.offsetX, e.offsetY);
  S.zoom = nz;
  const sc = (cv.width / S.shape[1]) * S.zoom;
  S.panX = e.offsetX - ix * sc;
  S.panY = e.offsetY - iy * sc;
  draw();
}, { passive: false });

function clickStar(cx, cy) {
  if (!S.sources.length) return;
  const [ix, iy] = toImage(cx, cy);
  let best = -1, bd = 1e9;
  S.sources.forEach((s, i) => {
    const d = Math.hypot(s.x - ix, s.y - iy);
    if (d < bd) { bd = d; best = i; }
  });
  const sc = (cv.width / S.shape[1]) * S.zoom;
  if (bd * sc > 26) return;                       // clicked empty sky
  const s = S.sources[best];

  if (S.mode === 'target') {
    if (s.saturated && !confirm('That star is saturated — its photometry will be unusable. Select it anyway?')) return;
    S.target = best;
    S.comps = S.comps.filter(c => c !== best);
  } else if (S.mode === 'comp') {
    if (best === S.target) { toast('That star is the target.'); return; }
    if (s.saturated && !confirm('That star is saturated. Use it as a comparison anyway?')) return;
    if (!S.comps.includes(best)) S.comps.push(best);
  } else {
    if (best === S.target) S.target = null;
    S.comps = S.comps.filter(c => c !== best);
    delete S.compMags[best];
  }
  draw(); renderPicks();
}

$('modeBtns').querySelectorAll('button').forEach(b => {
  b.onclick = () => {
    S.mode = b.dataset.mode;
    $('modeBtns').querySelectorAll('button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
  };
});
$('btnZoomIn').onclick = () => { S.zoom = Math.min(24, S.zoom * 1.4); draw(); };
$('btnZoomOut').onclick = () => { S.zoom = Math.max(1, S.zoom / 1.4); draw(); };
$('btnZoomFit').onclick = zoomFit;
$('showAll').onchange = draw;
$('btnClearPick').onclick = () => { S.target = null; S.comps = []; S.compMags = {}; draw(); renderPicks(); };
$('btnAutoPick').onclick = () => {
  const h = S.shape[0], w = S.shape[1];
  let bi = -1, bd = 1e9;
  S.sources.forEach((s, i) => {
    const d = Math.hypot(s.x - w / 2, s.y - h / 2);
    if (d < bd && !s.saturated) { bd = d; bi = i; }
  });
  S.target = bi >= 0 ? bi : null;
  S.comps = S.sources.map((s, i) => i)
    .filter(i => i !== S.target && !S.sources[i].saturated).slice(0, 5);
  draw(); renderPicks();
  toast('Picked the brightest unsaturated stars. Check that none of them is itself variable.');
};
['phAp', 'phIn', 'phOut', 'phFwhm'].forEach(id => $(id).addEventListener('input', draw));
$('stretch').onchange = async () => {
  const idx = Math.max(0, (parseInt($('refFrame').value) || 1) - 1);
  try { await loadPreview(idx); draw(); } catch (e) { }
};
window.addEventListener('resize', () => { if (S.img && S.zoom === 1) zoomFit(); });

function renderPicks() {
  const out = $('pickOut');
  if (S.target === null && !S.comps.length) { out.innerHTML = ''; return; }

  let html = `<div class="tablewrap"><table><thead><tr>
    <th>role</th><th>#</th><th>x</th><th>y</th><th>peak (ADU)</th>
    <th>rel. flux</th><th>catalog V mag</th></tr></thead><tbody>`;
  const rows = [];
  if (S.target !== null) rows.push([S.target, 'target']);
  S.comps.forEach((c, i) => rows.push([c, 'C' + (i + 1)]));
  const maxf = Math.max(...rows.map(([i]) => S.sources[i].flux));
  rows.forEach(([i, role]) => {
    const s = S.sources[i];
    const isT = role === 'target';
    html += `<tr class="${isT ? 'is-target' : 'is-comp'}">
      <td><span class="tag ${isT ? 't' : 'c'}">${role}</span></td>
      <td class="num">${i}</td>
      <td class="num">${fmt(s.x, 1)}</td><td class="num">${fmt(s.y, 1)}</td>
      <td class="num">${fmt(s.peak, 0)}${s.saturated ? ' ⚠' : ''}</td>
      <td class="num">${fmt(s.flux / maxf, 3)}</td>
      <td>${isT ? '<span style="color:var(--dim)">—</span>' :
      `<input type="number" step="0.01" class="compmag" data-i="${i}" style="width:88px"
                 value="${S.compMags[i] != null ? S.compMags[i] : ''}" placeholder="optional">`}</td></tr>`;
  });
  html += `</tbody></table></div>`;

  html += noticeHTML('info',
    'Catalog V magnitudes are optional for a light curve but REQUIRED for a distance — ' +
    'without them the photometry is differential only, and a differential magnitude has ' +
    'no absolute scale. Get them from AAVSO\'s Variable Star Plotter (a chart for this ' +
    'field lists comparison-star magnitudes) or from APASS. Enter at least one.');

  if (S.target === null) html += noticeHTML('warning', 'No target selected.');
  if (!S.comps.length) html += noticeHTML('warning',
    'No comparison stars selected. At least one is required — and three to five, of ' +
    'similar brightness to the target, gives the cleanest result.');

  html += `<div class="row" style="margin-top:12px">
    <button class="primary" id="btnToPhot" ${(S.target === null || !S.comps.length) ? 'disabled' : ''}>
      Continue to photometry</button></div>`;
  out.innerHTML = html;

  out.querySelectorAll('.compmag').forEach(inp => {
    inp.oninput = () => {
      const v = parseFloat(inp.value);
      const i = parseInt(inp.dataset.i);
      if (isNaN(v)) delete S.compMags[i]; else S.compMags[i] = v;
    };
  });
  const b = $('btnToPhot');
  if (b) b.onclick = () => {
    markDone(2);
    $('hint2').textContent = `target #${S.target} · ${S.comps.length} comparison stars`;
    openStep(3);
  };
}

/* ================================================== step 3 ============== */
$('btnPhot').onclick = async () => {
  if (S.target === null || !S.comps.length) { toast('Select a target and comparisons first.', 'err'); return; }
  const btn = $('btnPhot'); btn.disabled = true;
  try {
    await post(`/api/session/${S.sid}/photometry`, {
      target: S.target, comps: S.comps, comp_mags: S.compMags,
      channel: $('channel').value,
      fwhm: parseFloat($('phFwhm').value),
      ap_factor: parseFloat($('phAp').value),
      ann_in_factor: parseFloat($('phIn').value),
      ann_out_factor: parseFloat($('phOut').value),
      gain: parseFloat($('phGain').value) || 1,
      read_noise: parseFloat($('phRead').value) || 0,
      track: $('phTrack').checked, global_align: $('phAlign').checked,
    });
    await waitJob($('prog3'), 'measuring');
    await showPhot();
  } catch (e) { toast('Photometry failed: ' + e.message, 'err', 11000); }
  btn.disabled = false;
};

async function showPhot() {
  const r = await api(`/api/session/${S.sid}/photometry`);
  S.calibrated = !!r.calibrated;
  const out = $('photOut');

  if (!r.n_good) {
    out.innerHTML = noticeHTML('critical', r.rejection_note ||
      'No usable measurements were produced.') +
      (r.bitdepth_note ? noticeHTML('warning', r.bitdepth_note) : '') +
      (r.failures && r.failures.length ? `<details><summary>Per-frame failures</summary><div class="dbody">
        <ul>${r.failures.map(f => `<li>${esc(f.file)}: ${esc(f.reason)}</li>`).join('')}</ul></div></details>` : '');
    return;
  }

  let html = `<div class="cards">
    <div class="card hero"><div class="k">points measured</div><div class="v">${r.n_good}</div>
      <div class="n">of ${r.n_total} frames</div></div>
    <div class="card"><div class="k">scatter</div><div class="v">${fmt(r.scatter_mmag, 1)}<small> mmag</small></div>
      <div class="n">total variation in Δm</div></div>
    <div class="card"><div class="k">precision</div><div class="v">${fmt(r.median_sigma_mmag, 1)}<small> mmag</small></div>
      <div class="n">median per-point error</div></div>
    <div class="card"><div class="k">amplitude / noise</div><div class="v">${fmt(r.scatter_mmag / r.median_sigma_mmag, 0)}</div>
      <div class="n">is the star really varying?</div></div>
    <div class="card"><div class="k">seeing</div><div class="v">${fmt(r.median_fwhm_px, 2)}<small> px</small></div>
      <div class="n">median FWHM</div></div>
    <div class="card"><div class="k">field drift</div><div class="v">${fmt(r.drift_px, 1)}<small> px</small></div>
      <div class="n">tracked and removed</div></div>
  </div>`;

  if (r.calibrated) {
    html += noticeHTML('ok',
      `Absolutely calibrated: zero point ${fmt(r.zeropoint, 4)} ± ${fmt(r.zeropoint_sigma, 4)} mag ` +
      `from ${r.n_calibrators} comparison star(s) with catalog magnitudes. A distance is reachable.`);
    if (r.zeropoint_note) html += `<p class="hint">${esc(r.zeropoint_note)}</p>`;
  } else {
    html += noticeHTML('warning',
      'Differential only — no catalog magnitudes were supplied, so there is no absolute scale. ' +
      'You can still measure the period. For a distance, go back to step 2 and enter a catalog V ' +
      'magnitude for at least one comparison star, or type the mean apparent V directly in step 5.');
  }
  if (r.rejection_note) html += noticeHTML('warning', r.rejection_note);
  if (r.bitdepth_note) html += noticeHTML('warning', r.bitdepth_note);
  if (r.edge_note) html += noticeHTML('info', r.edge_note);

  const ratio = r.scatter_mmag / r.median_sigma_mmag;
  if (ratio < 3) html += noticeHTML('critical',
    `The total scatter (${fmt(r.scatter_mmag, 1)} mmag) is only ${fmt(ratio, 1)}× the per-point ` +
    `error. Any variation is buried in noise — longer exposures, a larger aperture, or better ` +
    `comparison stars are needed before a period means anything.`);

  html += `<div class="tablewrap"><table><thead><tr>
    <th>star</th><th>role</th><th>median flux</th><th>median S/N</th>
    <th>raw scatter</th><th>check scatter</th></tr></thead><tbody>`;
  r.stars.filter(s => s.role !== 'unused').forEach(s => {
    const flag = s.role === 'comparison' && s.check_rms_mmag && s.check_rms_mmag > 25;
    html += `<tr class="${s.role === 'target' ? 'is-target' : 'is-comp'}">
      <td class="num">${s.star}</td>
      <td><span class="tag ${s.role === 'target' ? 't' : 'c'}">${s.role}</span></td>
      <td class="num">${commas(s.median_flux, 0)}</td>
      <td class="num">${fmt(s.median_snr, 0)}</td>
      <td class="num">${fmt(s.rms_mmag, 1)} mmag</td>
      <td class="num" ${flag ? 'style="color:var(--warn)"' : ''}>
        ${s.check_rms_mmag == null ? '—' : fmt(s.check_rms_mmag, 1) + ' mmag' + (flag ? ' ⚠' : '')}</td></tr>`;
  });
  html += `</tbody></table></div>
    <p class="hint">“Check scatter” is each comparison star measured against the <em>other</em>
      comparisons. It should be small and similar for all of them — a comparison star with a
      much larger check scatter is variable, or is contaminated by a neighbour, and should be
      deselected. The target's large value is the signal you are after.</p>`;

  const bad = r.stars.filter(s => s.role === 'comparison' && s.check_rms_mmag > 25);
  if (bad.length) html += noticeHTML('warning',
    `Comparison star(s) ${bad.map(s => s.star).join(', ')} show a check scatter above 25 mmag. ` +
    `Go back to step 2, remove them, and re-run — a variable “comparison” star injects its own ` +
    `period into your light curve.`);

  html += `<div class="plot-grid">
    ${plotBlock('rawcurve', 'Differential light curve',
    'Target minus comparison ensemble. Cloud, haze and airmass have already divided out.')}
    ${plotBlock('diagnostics', 'Observing conditions',
    'When the light curve looks odd, the explanation is usually in one of these four panels.')}
    ${plotBlock('field', 'Aperture layout', 'Where the apertures and sky annuli actually sat.')}
  </div>`;

  html += `<div class="row" style="margin-top:12px">
    <button class="primary" id="btnToPeriod">Continue to period analysis</button></div>`;

  out.innerHTML = html;
  $('btnToPeriod').onclick = () => {
    markDone(3);
    $('hint3').textContent = `${r.n_good} points · ${fmt(r.median_sigma_mmag, 1)} mmag precision` +
      (r.calibrated ? ' · calibrated' : ' · differential only');
    openStep(4);
  };
}

/* ================================================== step 4 ============== */
async function runPeriod(force) {
  const btn = $('btnPeriod'); btn.disabled = true;
  try {
    const body = {
      p_min: parseFloat($('pMin').value), p_max: parseFloat($('pMax').value),
      nharm: parseInt($('nHarm').value), bootstrap: parseInt($('nBoot').value),
      time_system: $('timeSys').value,
      detrend_order: parseInt($('detrend').value),
      ra: $('raIn').value.trim() || null, dec: $('decIn').value.trim() || null,
    };
    if (force) body.force_period = force;
    await post(`/api/session/${S.sid}/period`, body);
    await waitJob($('prog4'), 'analysing');
    await showPeriod();
  } catch (e) { toast('Period analysis failed: ' + e.message, 'err', 11000); }
  btn.disabled = false;
}
$('btnPeriod').onclick = () => runPeriod(null);

async function showPeriod() {
  const r = await api(`/api/session/${S.sid}/period`);
  S.period = r;
  const a = r.assess;
  const out = $('periodOut');

  let html = `<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:14px 0 4px">
    <span class="verdict ${a.verdict}">${a.verdict}</span>
    <span class="hint" style="margin:0">${esc(a.verdict_note || '')}</span></div>`;

  html += `<div class="cards">
    <div class="card hero"><div class="k">period</div>
      <div class="v">${fmt(r.period_hours, 4)}<small> h</small></div>
      <div class="n">${fmt(r.period_days, 7)} d ± ${sci(r.sigma_period_days, 1)}</div></div>
    <div class="card"><div class="k">precision</div><div class="v">${fmt(r.rel_precision_pct, 2)}<small>%</small></div>
      <div class="n">${esc(String(r.sigma_driver || '').slice(0, 34))}</div></div>
    <div class="card"><div class="k">amplitude</div><div class="v">${fmt(r.amplitude_p2p_mag, 3)}<small> mag</small></div>
      <div class="n">peak to peak</div></div>
    <div class="card"><div class="k">cycles covered</div><div class="v">${fmt(r.cycles, 2)}</div>
      <div class="n">over ${fmt(r.span_hours, 2)} h</div></div>
    <div class="card"><div class="k">false alarm prob.</div><div class="v" style="font-size:16px">${sci(r.fap, 1)}</div>
      <div class="n">LS power ${fmt(r.power, 3)}</div></div>
    <div class="card"><div class="k">residual scatter</div><div class="v">${fmt(r.residual_rms_mmag, 1)}<small> mmag</small></div>
      <div class="n">after the fit</div></div>
  </div>`;

  if (r.catalog_period) {
    const dp = r.catalog_diff_pct;
    const lvl = dp < 1 ? 'ok' : (dp < 5 ? 'warning' : 'critical');
    html += noticeHTML(lvl,
      `Published period is ${r.catalog_period} d; yours differs by ${fmt(dp, 2)}%. ` +
      (dp < 1 ? 'That is a genuine independent confirmation.'
        : 'Look hard at the alias candidates below before trusting your value over the catalog.'));
  }

  (a.flags || []).forEach(f => { html += noticeHTML(f.level, f.text); });

  /* --- alias candidates --- */
  if (r.aliases && r.aliases.ambiguous) {
    html += `<h4 style="margin:16px 0 6px;font-size:13.5px">Competing periods the data cannot rule out</h4>
      <p class="hint" style="margin-top:0">Each of these phases your data almost as well as the winner.
      Clicking one re-runs the analysis locked to that period, so you can compare the folded curves —
      the correct period gives the tightest, most physically sensible curve shape.</p>`;
    r.aliases.candidates.forEach(c => {
      const cls = 'candidate' + (c.is_best ? ' best' : '') + (c.matches_catalog ? ' cat' : '');
      html += `<div class="${cls}">
        <span class="p">${fmt(c.period, 7)} d</span>
        <span class="p" style="color:var(--muted)">${fmt(c.period * 24, 4)} h</span>
        <span class="rel">${esc(c.relation)}${c.is_best ? ' — currently used' : ''}
          ${c.matches_catalog ? ' · <b style="color:var(--purple)">matches the catalog</b>' : ''}</span>
        <span style="font-family:var(--mono);font-size:11.5px;color:var(--dim)">
          power −${fmt(c.rel_deficit * 100, 2)}%</span>
        ${c.is_best ? '' : `<button class="small" onclick="window.__usePeriod(${c.period})">use this</button>`}</div>`;
    });
  }

  /* --- uncertainty breakdown --- */
  html += `<details><summary>How the period uncertainty was arrived at</summary><div class="dbody">
    <p class="hint" style="margin-top:0">${esc(r.sigma_rationale || '')}</p>
    <div class="tablewrap"><table><thead><tr><th>method</th><th>σ(P), days</th><th>as %</th><th>note</th></tr></thead><tbody>`;
  (r.sigma_estimates || []).forEach(e => {
    html += `<tr><td>${esc(e.method)}</td><td class="num">${sci(e.sigma, 2)}</td>
      <td class="num">${fmt(e.sigma / r.period_days * 100, 3)}%</td>
      <td style="text-align:left;color:var(--dim);white-space:normal">${esc(e.note)}</td></tr>`;
  });
  html += `</tbody></table></div>
    <div class="kv" style="margin-top:10px">
      <span class="k">frequency resolution 1/T</span><span class="v">${fmt(r.rayleigh_df, 4)} cycles/day</span>
      <span class="k">implied period resolution P²/T</span><span class="v">${sci(r.rayleigh_period, 2)} d</span>
      <span class="k">PDM independent estimate</span><span class="v">${fmt(r.pdm_period, 7)} d (θ = ${fmt(r.pdm_theta, 4)})</span>
      <span class="k">PDM vs Lomb-Scargle</span><span class="v">${fmt(r.pdm_vs_ls_pct, 3)}% apart</span>
      <span class="k">time system</span><span class="v">${esc(r.time_label)} (${fmt(r.time_correction_s, 1)} s applied)</span>
      <span class="k">epoch of maximum</span><span class="v">${fmt(r.t_max, 6)}</span>
    </div></div></details>`;

  /* --- modes --- */
  if (r.modes && r.modes.length) {
    html += `<details><summary>Pulsation modes found (${r.modes.filter(m => m.significant).length} significant)</summary>
      <div class="dbody"><p class="hint" style="margin-top:0">
      V0756 CrA is classed HADS(<b>B</b>) — a double-mode pulsator — so a second independent
      frequency is expected. Each mode is fitted and subtracted before searching for the next.</p>
      <div class="tablewrap"><table><thead><tr><th>mode</th><th>period (d)</th><th>freq (c/d)</th>
        <th>amplitude</th><th>S/N</th><th>significant</th></tr></thead><tbody>`;
    r.modes.forEach(m => {
      html += `<tr><td class="num">${m.mode}</td><td class="num">${fmt(m.period, 7)}</td>
        <td class="num">${fmt(m.freq, 5)}</td><td class="num">${fmt(m.amp_mmag, 1)} mmag</td>
        <td class="num">${fmt(m.snr, 1)}</td>
        <td>${m.significant ? '<span style="color:var(--good)">yes</span>'
        : '<span style="color:var(--dim)">no</span>'}</td></tr>`;
    });
    html += `</tbody></table></div>`;
    (r.mode_ratios || []).forEach(mr => {
      html += noticeHTML(mr.confidence === 'firm' ? 'ok' : 'info',
        `Period ratio ${fmt(mr.ratio, 4)} between modes ${mr.modes.join(' and ')}` +
        ` (${mr.confidence}). ${mr.interpretation}`);
    });
    html += `</div></details>`;
  }

  html += `<div class="plot-grid">
    ${plotBlock('periodogram', 'Lomb–Scargle periodogram',
    'Power against trial frequency. The lower panel zooms in; the arrow marks 1/T, the intrinsic peak width.')}
    ${plotBlock('folded', 'Phase-folded light curve',
    'The real test of a period: fold on it and a correct period collapses every cycle onto one clean curve.')}
    ${S.calibrated ? plotBlock('foldedcal', 'Folded, calibrated to apparent V',
      'Same fold on the absolute magnitude scale — this is what the distance calculation reads ⟨V⟩ from.') : ''}
    ${plotBlock('pdm', 'Phase dispersion minimisation',
    'An independent estimator that assumes nothing about curve shape. Its minimum should sit on the same period.')}
    ${plotBlock('lightcurve', 'Light curve with the fitted model', 'Harmonic fit overlaid on the measurements.')}
    ${r.sigma_estimates && r.sigma_estimates.some(e => e.method.includes('bootstrap'))
      ? plotBlock('bootstrap', 'Bootstrap period distribution',
        'Re-fitting resampled noise, keeping the observing window. Its width is one of the error estimates.') : ''}
  </div>`;

  html += `<div class="row" style="margin-top:12px">
    <button class="primary" id="btnToDist">Continue to distance</button></div>`;

  out.innerHTML = html;
  markDone(4);
  $('hint4').textContent = `P = ${fmt(r.period_hours, 4)} h ± ${fmt(r.rel_precision_pct, 2)}% · ${a.verdict}`;

  if (r.mean_mag_v != null) {
    $('meanMag').placeholder = fmt(r.mean_mag_v, 3) + ' (measured)';
    $('meanMagErr').placeholder = fmt(r.sigma_mean_mag_v, 3);
  }
  $('btnToDist').onclick = () => openStep(5);
}

window.__usePeriod = (p) => {
  toast(`Re-running locked to P = ${p.toFixed(7)} d…`);
  runPeriod(p);
};

/* ================================================== step 5 ============== */
$('periodSource').onchange = () => {
  $('manualPeriodWrap').style.display = $('periodSource').value === 'manual' ? 'flex' : 'none';
};

$('btnEbv').onclick = async () => {
  const b = $('btnEbv'); b.disabled = true; b.textContent = 'querying…';
  try {
    const r = await post(`/api/session/${S.sid}/lookup/ebv`,
      { ra: $('raIn').value.trim(), dec: $('decIn').value.trim() });
    if (r.ok) {
      $('ebv').value = r.ebv.toFixed(4);
      toast(`E(B−V) = ${r.ebv.toFixed(4)} (A_V = ${r.a_v.toFixed(3)}) — ${r.reference}. ${r.note}`, 'ok', 11000);
    } else toast('Dust-map query failed: ' + r.error, 'err', 9000);
  } catch (e) { toast('Dust-map query failed: ' + e.message, 'err', 9000); }
  b.disabled = false; b.textContent = 'Fetch from dust map';
};

$('btnGaia').onclick = async () => {
  const b = $('btnGaia'); b.disabled = true; b.textContent = 'querying Gaia…';
  try {
    const r = await post(`/api/session/${S.sid}/lookup/gaia`,
      { ra: $('raIn').value.trim(), dec: $('decIn').value.trim() });
    if (r.ok) {
      toast(`Gaia DR3 ${r.source_id}: parallax ${fmt(r.parallax_mas, 4)} ± ${fmt(r.parallax_error_mas, 4)} mas` +
        (r.distance_pc ? ` → ${commas(r.distance_pc, 0)} pc` : ''), 'ok', 9000);
      if (S.period) await calcDistance();
    } else toast('Gaia query failed: ' + r.error, 'err', 9000);
  } catch (e) { toast('Gaia query failed: ' + e.message, 'err', 9000); }
  b.disabled = false; b.textContent = 'Check against Gaia parallax';
};

async function calcDistance() {
  const body = {
    relation: $('relation').value,
    period_source: $('periodSource').value,
    manual_period: parseFloat($('manualPeriod').value) || null,
    mean_mag: $('meanMag').value !== '' ? parseFloat($('meanMag').value) : null,
    sigma_mean_mag: $('meanMagErr').value !== '' ? parseFloat($('meanMagErr').value) : null,
    ebv: $('ebv').value !== '' ? parseFloat($('ebv').value) : null,
    sigma_ext: $('ebvErr').value !== '' ? parseFloat($('ebvErr').value) : null,
    teff: $('teff').value !== '' ? parseFloat($('teff').value) : null,
  };
  const r = await post(`/api/session/${S.sid}/distance`, body);
  const d = r.distance, ab = r.absolute, ex = r.extinction, pr = r.properties;
  const out = $('distOut');

  let html = `<div class="cards">
    <div class="card hero"><div class="k">distance</div>
      <div class="v">${commas(d.distance_pc, 0)}<small> pc</small></div>
      <div class="n">${commas(d.distance_pc_lo, 0)} – ${commas(d.distance_pc_hi, 0)} pc (1σ)</div></div>
    <div class="card hero"><div class="k">in light years</div>
      <div class="v">${commas(d.distance_ly, 0)}<small> ly</small></div>
      <div class="n">± ${commas(d.sigma_ly, 0)} ly (${fmt(d.error_budget.relative_distance_pct, 1)}%)</div></div>
    <div class="card"><div class="k">absolute magnitude</div><div class="v">${fmt(ab.M, 3)}</div>
      <div class="n">± ${fmt(ab.sigma_M, 3)} from the P–L relation</div></div>
    <div class="card"><div class="k">distance modulus</div><div class="v">${fmt(d.mu_0, 3)}</div>
      <div class="n">± ${fmt(d.sigma_mu, 3)} mag, extinction-corrected</div></div>
    <div class="card"><div class="k">extinction A_V</div><div class="v">${fmt(ex.a_v, 3)}</div>
      <div class="n">E(B−V) = ${fmt(ex.ebv, 4)}</div></div>
    <div class="card"><div class="k">implied parallax</div><div class="v">${fmt(d.parallax_mas, 4)}<small> mas</small></div>
      <div class="n">for comparison with Gaia</div></div>
  </div>`;

  html += `<div class="eq">M_V = ${ab.slope} × log₁₀(${fmt(r.period, 7)}) ${ab.intercept >= 0 ? '+' : '−'} ${Math.abs(ab.intercept)} = ${fmt(ab.M, 3)} ± ${fmt(ab.sigma_M, 3)}</div>
    <div class="eq">μ₀ = ⟨V⟩ − M_V − A_V = ${fmt(r.mean_mag, 3)} − ${fmt(ab.M, 3)} − ${fmt(ex.a_v, 3)} = ${fmt(d.mu_0, 3)} ± ${fmt(d.sigma_mu, 3)}</div>
    <div class="eq">d = 10^(μ₀/5 + 1) = ${commas(d.distance_pc, 1)} pc = ${commas(d.distance_ly, 0)} light years</div>`;

  html += `<div class="kv" style="margin:12px 0">
    <span class="k">period used</span><span class="v">${fmt(r.period, 7)} d — ${esc(r.period_source)}</span>
    <span class="k">mean magnitude</span><span class="v">${fmt(r.mean_mag, 3)} ± ${fmt(r.sigma_mean_mag, 3)}</span>
    <span class="k">source of ⟨V⟩</span><span class="v" style="font-family:var(--sans)">${esc(r.mean_mag_source)}</span>
    <span class="k">relation</span><span class="v" style="font-family:var(--sans)">${esc(ab.relation_label)}</span>
    <span class="k">extinction</span><span class="v" style="font-family:var(--sans)">${esc(ex.source)}</span>
  </div>`;

  if (r.quality_warning) html += noticeHTML('warning', r.quality_warning);
  if (r.fundamental_note) html += noticeHTML('info', r.fundamental_note);
  if (ab.warning) html += noticeHTML('warning', ab.warning);
  if (ex.a_v === 0) html += noticeHTML('warning',
    'No extinction applied. Dust makes the star look fainter, so this distance is an ' +
    'over-estimate — press “Fetch from dust map” for the reddening along this sight line.');

  html += `<details><summary>Error budget — what actually limits this distance</summary><div class="dbody">
    <div class="tablewrap"><table><thead><tr><th>contribution</th><th>σ (mag)</th><th>share of variance</th></tr></thead><tbody>`;
  const tot = Math.pow(d.sigma_mu, 2);
  const items = [
    ['mean apparent magnitude', d.error_budget.mean_magnitude],
    ['absolute magnitude (P–L relation)', d.error_budget.absolute_magnitude],
    ['extinction', d.error_budget.extinction],
  ];
  items.forEach(([k, v]) => {
    html += `<tr><td>${k}</td><td class="num">${fmt(v, 4)}</td>
      <td class="num">${tot > 0 ? fmt(100 * v * v / tot, 1) : '—'}%</td></tr>`;
  });
  html += `<tr><td><b>total</b></td><td class="num"><b>${fmt(d.sigma_mu, 4)}</b></td><td class="num">100%</td></tr>
    </tbody></table></div>
    <p class="hint">Within the P–L term: ${fmt(ab.terms.from_period, 4)} mag comes from your period error,
      ${fmt(ab.terms.from_slope, 4)} from the relation's slope, ${fmt(ab.terms.from_intercept, 4)} from its
      zero point, and ${fmt(ab.terms.intrinsic_scatter, 4)} is the relation's intrinsic scatter — an
      irreducible floor set by the fact that δ Scuti stars of the same period are not all identical.</p>
    </div></details>`;

  if (pr) {
    html += `<details><summary>Derived stellar properties</summary><div class="dbody">
      <div class="kv">
        <span class="k">bolometric magnitude</span><span class="v">${fmt(pr.m_bol, 3)}</span>
        <span class="k">luminosity</span><span class="v">${fmt(pr.luminosity_lsun, 1)} ± ${fmt(pr.sigma_luminosity_lsun, 1)} L☉</span>
        ${pr.radius_rsun ? `<span class="k">radius</span><span class="v">${fmt(pr.radius_rsun, 2)} ± ${fmt(pr.sigma_radius_rsun, 2)} R☉</span>
        <span class="k">mean density</span><span class="v">${fmt(pr.density_rho_sun, 4)} ρ☉</span>
        <span class="k">mass (from Q = ${pr.q_assumed} d)</span><span class="v">${fmt(pr.mass_msun, 2)} M☉</span>` :
        `<span class="k">radius &amp; mass</span><span class="v" style="font-family:var(--sans)">enter a temperature above to unlock</span>`}
      </div><p class="hint">${esc(pr.caveat)}</p></div></details>`;
  }

  if (r.gaia_comparison) {
    const g = r.gaia_comparison;
    const lvl = g.n_sigma < 2 ? 'ok' : (g.n_sigma < 3 ? 'warning' : 'critical');
    html += `<h4 style="margin:16px 0 6px;font-size:13.5px">Independent check: Gaia DR3 parallax</h4>`;
    html += noticeHTML(lvl,
      `Your P–L distance ${commas(g.distance_pl_pc, 0)} ± ${commas(g.sigma_pl_pc, 0)} pc vs Gaia's ` +
      `${commas(g.distance_gaia_pc, 0)} ± ${commas(g.sigma_gaia_pc, 0)} pc — a difference of ` +
      `${fmt(g.percent_difference, 1)}%, or ${fmt(g.n_sigma, 2)}σ. ${g.verdict}`);
    if (r.gaia && r.gaia.warning) html += noticeHTML('warning', r.gaia.warning);
    if (r.gaia && r.gaia.note) html += `<p class="hint">${esc(r.gaia.note)}</p>`;
  } else {
    html += `<p class="hint">Press “Check against Gaia parallax” for a fully independent
      geometric distance to the same star — the best possible validation of this measurement.</p>`;
  }

  html += `<div class="plot-grid">
    ${plotBlock('distance', 'Distance and error budget',
    'Left: what limits the precision. Right: the distance interval, with Gaia alongside if queried.')}
    ${plotBlock('plrelation', 'Where this star sits on the period–luminosity relation',
    'The star plotted against the published relations; the shaded band is the intrinsic scatter.')}
  </div>`;

  out.innerHTML = html;
  markDone(5);
  $('hint5').textContent = `${commas(d.distance_pc, 0)} pc = ${commas(d.distance_ly, 0)} ly ± ${fmt(d.error_budget.relative_distance_pct, 1)}%`;
}

$('btnDistance').onclick = async () => {
  const b = $('btnDistance'); b.disabled = true;
  try { await calcDistance(); }
  catch (e) { toast(e.message, 'err', 13000); }
  b.disabled = false;
};

const dl = (path) => window.open(`/api/session/${S.sid}${path}`, '_blank');
$('btnCsv').onclick = () => dl('/export/lightcurve.csv');
$('btnJson').onclick = () => dl('/export/report.json');
$('btnZip').onclick = () => dl('/export/bundle.zip');

$('btnHelp').onclick = () => {
  const p = $('helpPanel');
  p.style.display = p.style.display === 'none' ? 'block' : 'none';
  if (p.style.display === 'block') p.scrollIntoView({ behavior: 'smooth' });
};
$('btnReset').onclick = () => location.reload();

boot().catch(e => toast('Could not start a session: ' + e.message, 'err', 15000));
