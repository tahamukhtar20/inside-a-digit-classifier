import { analyse, loadModel } from "./model.js";

const $ = (id) => document.getElementById(id);
const pad = $("pad");
const padCtx = pad.getContext("2d", { willReadFrequently: true });
const heat = $("heat");
const heatCtx = heat.getContext("2d");

const state = {
  model: null,
  pixels: new Float32Array(784), // what the model sees: 28x28, 1 = ink
  chosen: null, // digit picked by clicking a bar; null = follow the prediction
  method: "smoothgrad",
  result: null,
  samples: null,
};

const NOTES = {
  smoothgrad: "The gradient averaged over 24 noisy copies of your drawing. Same idea as plain saliency, with most of the speckle gone.",
  saliency: "How much each pixel's brightness would change the score. Works on empty space too: it shows where adding ink would help or hurt.",
  gradTimesInput: "The gradient multiplied by your ink, so only pixels you drew can get credit or blame.",
  gradcam: "The second layer's feature maps, weighted by how much they matter for the score, upsampled onto the image. Coarser, but shows the regions the network relies on.",
};

// ---------- drawing ----------

function resetPad() {
  padCtx.fillStyle = "#fff";
  padCtx.fillRect(0, 0, pad.width, pad.height);
}

let drawing = false;
let last = null;
const pos = (e) => {
  const r = pad.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * pad.width, ((e.clientY - r.top) / r.height) * pad.height];
};

pad.addEventListener("pointerdown", (e) => {
  drawing = true;
  pad.setPointerCapture(e.pointerId);
  last = pos(e);
  stroke(last, last);
});
pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = pos(e);
  stroke(last, p);
  last = p;
});
const stop = () => (drawing = false);
pad.addEventListener("pointerup", stop);
pad.addEventListener("pointercancel", stop);

function stroke(a, b) {
  padCtx.strokeStyle = "#111";
  padCtx.lineWidth = 20;
  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  padCtx.beginPath();
  padCtx.moveTo(a[0], a[1]);
  padCtx.lineTo(b[0], b[1]);
  padCtx.stroke();
  schedule();
}

$("clear").addEventListener("click", () => {
  resetPad();
  state.pixels.fill(0);
  state.chosen = null;
  run();
});

$("sample").addEventListener("click", async () => {
  if (!state.samples) state.samples = await (await fetch("model/samples.json")).json();
  const s = state.samples[Math.floor(Math.random() * state.samples.length)];
  // Show the MNIST digit on the pad (ink on paper), scaled up.
  const img = padCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = 255 - s.pixels[i];
    img.data.set([v, v, v, 255], i * 4);
  }
  const tmp = Object.assign(document.createElement("canvas"), { width: 28, height: 28 });
  tmp.getContext("2d").putImageData(img, 0, 0);
  resetPad();
  padCtx.imageSmoothingEnabled = true;
  padCtx.drawImage(tmp, 0, 0, pad.width, pad.height);
  state.chosen = null;
  schedule();
});

// ---------- preprocessing (MNIST style) ----------

/** Crop to the ink, scale to fit 20x20, centre by centre of mass in 28x28. */
function preprocess() {
  const { width: w, height: h } = pad;
  const data = padCtx.getImageData(0, 0, w, h).data;
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[(y * w + x) * 4] < 200) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  const out = new Float32Array(784);
  if (maxX < 0) return out;

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const sw = Math.max(1, Math.round(bw * scale));
  const sh = Math.max(1, Math.round(bh * scale));

  // Inverted copy (white ink on black) scaled into a small canvas.
  const small = Object.assign(document.createElement("canvas"), { width: sw, height: sh });
  const sctx = small.getContext("2d", { willReadFrequently: true });
  sctx.filter = "invert(1)";
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(pad, minX, minY, bw, bh, 0, 0, sw, sh);
  const sd = sctx.getImageData(0, 0, sw, sh).data;

  let mass = 0, cx = 0, cy = 0;
  const vals = new Float32Array(sw * sh);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const v = sd[(y * sw + x) * 4] / 255;
      vals[y * sw + x] = v;
      mass += v;
      cx += v * x;
      cy += v * y;
    }
  }
  const ox = Math.round(14 - cx / mass - 0.5);
  const oy = Math.round(14 - cy / mass - 0.5);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const tx = x + ox, ty = y + oy;
      if (tx >= 0 && tx < 28 && ty >= 0 && ty < 28) out[ty * 28 + tx] = vals[y * sw + x];
    }
  }
  return out;
}

// ---------- running the model ----------

let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    state.pixels = preprocess();
    run();
  });
}

// SmoothGrad is computed once drawing has paused for a moment.
let smoothTimer = null;
function run() {
  if (!state.model) return;
  clearTimeout(smoothTimer);
  analyseNow(false);
  if (state.result && state.method === "smoothgrad") smoothTimer = setTimeout(() => analyseNow(true), 180);
}

function analyseNow(smooth) {
  const empty = state.pixels.every((v) => v === 0);
  if (empty) {
    state.result = null;
    renderEmpty();
    return;
  }
  const versus = $("versus").value === "" ? null : Number($("versus").value);
  // Guess the target from the last prediction; redo only if the prediction changed.
  let target = state.chosen ?? state.predicted ?? 0;
  let r = analyse(state.model, state.pixels, target, versus, { smooth });
  const predicted = argmax(r.probs);
  if (state.chosen === null && predicted !== target) {
    target = predicted;
    r = analyse(state.model, state.pixels, target, versus, { smooth });
  }
  state.predicted = predicted;
  state.result = r;
  render(predicted, target, versus);
}

const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);

// ---------- rendering ----------

const bars = $("bars");
const barEls = [];
for (let d = 0; d < 10; d++) {
  const li = document.createElement("li");
  li.innerHTML = `<button class="bar" type="button" aria-pressed="false"><span class="bar-digit">${d}</span><span class="bar-track"><span class="bar-fill"></span></span><span class="bar-pct">0%</span></button>`;
  const btn = li.firstElementChild;
  btn.addEventListener("click", () => {
    state.chosen = state.chosen === d ? null : d;
    run();
  });
  bars.append(li);
  barEls.push(btn);
}

document.querySelectorAll("[data-method]").forEach((b) =>
  b.addEventListener("click", () => {
    state.method = b.dataset.method;
    document.querySelectorAll("[data-method]").forEach((o) => o.setAttribute("aria-selected", String(o === b)));
    run();
  }),
);
$("versus").addEventListener("change", run);

function renderEmpty() {
  barEls.forEach((b) => {
    b.classList.remove("top");
    b.setAttribute("aria-pressed", "false");
    b.querySelector(".bar-fill").style.width = "0";
    b.querySelector(".bar-pct").textContent = "–";
  });
  $("question").innerHTML = "Draw a digit to begin";
  heatCtx.fillStyle = "#fff";
  heatCtx.fillRect(0, 0, heat.width, heat.height);
  $("legend").innerHTML = "";
  $("methodNote").textContent = NOTES[state.method];
  clearMaps();
}

function render(predicted, target, versus) {
  const r = state.result;
  barEls.forEach((b, d) => {
    b.classList.toggle("top", d === predicted);
    b.setAttribute("aria-pressed", String(d === target && state.chosen !== null));
    b.querySelector(".bar-fill").style.width = `${(r.probs[d] * 100).toFixed(1)}%`;
    b.querySelector(".bar-pct").textContent = `${(r.probs[d] * 100).toFixed(r.probs[d] > 0.995 ? 0 : 1)}%`;
  });

  const q = versus !== null && versus !== target ? `Why <b>${target}</b> rather than <b>${versus}</b>?` : `Why <b>${target}</b>?`;
  $("question").innerHTML = target === predicted ? `It says ${target}. ${q}` : q;
  $("methodNote").textContent = NOTES[state.method];
  // While SmoothGrad is pending, show plain saliency so the map never lags behind the drawing.
  const values = r[state.method] ?? r.saliency;
  drawHeat(values, state.method === "gradcam", target, versus);
  drawMaps(r);
}

/** Digit in grey, attribution on top: pink pushes towards the answer, blue pushes away. */
function drawHeat(values, positiveOnly, target, versus) {
  let max = 1e-9;
  for (const v of values) max = Math.max(max, Math.abs(v));
  const img = heatCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const base = 255 - state.pixels[i] * 90; // drawn digit as light grey
    const a = Math.min(1, Math.abs(values[i]) / max) ** 0.75;
    const color = values[i] >= 0 ? (positiveOnly ? [255, 160, 60] : [255, 80, 140]) : [40, 120, 255];
    img.data.set([base * (1 - a) + color[0] * a, base * (1 - a) + color[1] * a, base * (1 - a) + color[2] * a, 255], i * 4);
  }
  const tmp = Object.assign(document.createElement("canvas"), { width: 28, height: 28 });
  tmp.getContext("2d").putImageData(img, 0, 0);
  heatCtx.imageSmoothingEnabled = false;
  heatCtx.drawImage(tmp, 0, 0, heat.width, heat.height);

  const toward = versus !== null && versus !== target ? `${target} over ${versus}` : `${target}`;
  $("legend").innerHTML = positiveOnly
    ? `<span><i class="swatch" style="background:#ffa03c"></i>Regions that support ${toward}</span>`
    : `<span><i class="swatch" style="background:#ff508c"></i>Pushes towards ${toward}</span><span><i class="swatch" style="background:#2878ff"></i>Pushes away</span>`;
}

// Feature maps and hidden units.
const maps1 = Array.from({ length: 8 }, () => mapCanvas($("maps1"), 28));
const maps2 = Array.from({ length: 16 }, () => mapCanvas($("maps2"), 14));
const units = Array.from({ length: 64 }, () => {
  const el = document.createElement("span");
  el.className = "unit";
  $("units").append(el);
  return el;
});

function mapCanvas(parent, size) {
  const c = Object.assign(document.createElement("canvas"), { width: size, height: size });
  parent.append(c);
  return c;
}

function drawChannel(canvas, data, size, channels, ch) {
  let max = 1e-9;
  for (let i = 0; i < size * size; i++) max = Math.max(max, data[i * channels + ch]);
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const t = data[i * channels + ch] / max;
    // paper -> ink, so strong activations read like pencil marks
    const v = 255 - t * 238;
    img.data.set([v, v, v, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
}

function drawMaps(r) {
  maps1.forEach((c, ch) => drawChannel(c, r.conv1, 28, 8, ch));
  maps2.forEach((c, ch) => drawChannel(c, r.conv2, 14, 16, ch));
  const max = Math.max(1e-9, ...r.hidden);
  units.forEach((u, i) => {
    const t = r.hidden[i] / max;
    u.style.background = t > 0 ? `color-mix(in srgb, #b9a6ff ${Math.round(t * 100)}%, #fff)` : "#fff";
    u.title = `Unit ${i + 1}: ${r.hidden[i].toFixed(2)}`;
  });
}

function clearMaps() {
  [...maps1, ...maps2].forEach((c) => {
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, c.width, c.height);
  });
  units.forEach((u) => (u.style.background = "#fff"));
}

// ---------- start ----------

resetPad();
renderEmpty();
loadModel("model/weights.json")
  .then((model) => {
    state.model = model;
    $("status").textContent = "Draw a digit, 0 to 9";
    if (model.meta?.test_accuracy) {
      $("accuracy").textContent = `It gets ${(model.meta.test_accuracy * 100).toFixed(1)}% of the 10,000 MNIST test digits right.`;
    }
    run();
  })
  .catch(() => ($("status").textContent = "The model failed to load."));
