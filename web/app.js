import { analyse, circuit, loadModel, noMasks, readProbes, SIZES } from "./model.js";

const $ = (id) => document.getElementById(id);
const pad = $("pad");
const padCtx = pad.getContext("2d", { willReadFrequently: true });
const heat = $("heat");
const heatCtx = heat.getContext("2d");

const state = {
  model: null,
  insights: null, // feature visualisations, dataset examples, probes (training/analysis.py)
  pixels: new Float32Array(784), // what the model sees: 28x28, 1 = ink
  chosen: null, // digit picked by clicking a bar; null = follow the prediction
  predicted: null,
  method: "smoothgrad",
  masks: noMasks(),
  result: null,
  baseline: null, // prediction with nothing switched off, when something is
  circuit: null,
  selected: null, // { layer, index } in the inspector
  samples: null,
};

const LAYER_NAME = { conv1: "Layer 1 · filter", conv2: "Layer 2 · filter", hidden: "Hidden unit" };
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
  const tmp = grayCanvas(Uint8ClampedArray.from(s.pixels, (v) => 255 - v));
  resetPad();
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
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
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

const ablatedCount = () => Object.values(state.masks).reduce((n, m) => n + m.filter((v) => v === 0).length, 0);
const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
const versusValue = () => ($("versus").value === "" ? null : Number($("versus").value));

// The cheap parts update on every frame; SmoothGrad, the circuit and the probes once drawing pauses.
let idleTimer = null;
function run() {
  if (!state.model) return;
  clearTimeout(idleTimer);
  if (state.pixels.every((v) => v === 0)) {
    state.result = state.circuit = null;
    renderEmpty();
    return;
  }
  analyseNow(false);
  idleTimer = setTimeout(() => {
    if (state.method === "smoothgrad") analyseNow(true);
    const versus = versusValue();
    state.circuit = circuit(state.model, state.pixels, state.target, versus, state.masks);
    renderCircuit();
    renderProbes();
    renderInspector();
  }, 180);
}

function analyseNow(smooth) {
  const versus = versusValue();
  // Guess the target from the last prediction; redo only if the prediction changed.
  let target = state.chosen ?? state.predicted ?? 0;
  let r = analyse(state.model, state.pixels, target, versus, { smooth, masks: state.masks });
  const predicted = argmax(r.probs);
  if (state.chosen === null && predicted !== target) {
    target = predicted;
    r = analyse(state.model, state.pixels, target, versus, { smooth, masks: state.masks });
  }
  state.predicted = predicted;
  state.target = target;
  state.result = r;
  state.baseline = ablatedCount() ? analyse(state.model, state.pixels, target, versus).probs : null;
  render(predicted, target, versus);
}

// ---------- predictions and attributions ----------

const barEls = [];
for (let d = 0; d < 10; d++) {
  const li = document.createElement("li");
  li.innerHTML = `<button class="bar" type="button" aria-pressed="false"><span class="bar-digit">${d}</span><span class="bar-track"><span class="bar-fill"></span></span><span class="bar-pct">–</span></button>`;
  const btn = li.firstElementChild;
  btn.addEventListener("click", () => {
    state.chosen = state.chosen === d ? null : d;
    run();
  });
  $("bars").append(li);
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
  $("question").textContent = "Draw a digit to begin";
  heatCtx.fillStyle = "#fff";
  heatCtx.fillRect(0, 0, heat.width, heat.height);
  $("legend").innerHTML = "";
  $("methodNote").textContent = NOTES[state.method];
  $("ablationNote").hidden = true;
  mapEls.forEach(({ canvas }) => blank(canvas));
  unitEls.forEach((u) => (u.style.background = "#fff"));
  $("circuitSvg").innerHTML = `<text x="380" y="140" text-anchor="middle" class="svg-empty">Draw a digit to see its circuit</text>`;
  $("circuitSvg").setAttribute("viewBox", "0 0 760 280");
  renderProbes();
  renderInspector();
  updateAblationBar();
}

const pct = (p) => `${(p * 100).toFixed(p > 0.995 ? 0 : 1)}%`;

function render(predicted, target, versus) {
  const r = state.result;
  barEls.forEach((b, d) => {
    b.classList.toggle("top", d === predicted);
    b.setAttribute("aria-pressed", String(d === target && state.chosen !== null));
    b.querySelector(".bar-fill").style.width = `${(r.probs[d] * 100).toFixed(1)}%`;
    b.querySelector(".bar-pct").textContent = pct(r.probs[d]);
  });

  const note = $("ablationNote");
  if (state.baseline) {
    const b = argmax(state.baseline);
    note.hidden = false;
    note.innerHTML = `With ${plural(ablatedCount(), "unit")} switched off it says <b>${predicted}</b> (${pct(r.probs[predicted])}). With everything on: <b>${b}</b> (${pct(state.baseline[b])}).`;
  } else note.hidden = true;

  const q = versus !== null && versus !== target ? `Why <b>${target}</b> rather than <b>${versus}</b>?` : `Why <b>${target}</b>?`;
  $("question").innerHTML = target === predicted ? `It says ${target}. ${q}` : q;
  $("methodNote").textContent = NOTES[state.method];
  // While SmoothGrad is pending, show plain saliency so the map never lags behind the drawing.
  drawHeat(r[state.method] ?? r.saliency, state.method === "gradcam", target, versus);
  drawMaps(r);
  updateAblationBar();
}

const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

/** Digit in grey, attribution on top: pink pushes towards the answer, blue pushes away. */
function drawHeat(values, positiveOnly, target, versus) {
  let max = 1e-9;
  for (const v of values) max = Math.max(max, Math.abs(v));
  const img = heatCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const base = 255 - state.pixels[i] * 90;
    const a = Math.min(1, Math.abs(values[i]) / max) ** 0.75;
    const color = values[i] >= 0 ? (positiveOnly ? [255, 160, 60] : [255, 80, 140]) : [40, 120, 255];
    img.data.set([0, 1, 2].map((k) => base * (1 - a) + color[k] * a).concat(255), i * 4);
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

// ---------- feature maps, units, ablation ----------

function grayCanvas(values, size = 28) {
  const c = Object.assign(document.createElement("canvas"), { width: size, height: size });
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) img.data.set([values[i], values[i], values[i], 255], i * 4);
  ctx.putImageData(img, 0, 0);
  return c;
}

function blank(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

/** One channel of a channels-last activation as ink on paper (strong = dark). */
function channelPixels(data, size, channels, ch) {
  let max = 1e-9;
  for (let i = 0; i < size * size; i++) max = Math.max(max, data[i * channels + ch]);
  return Uint8ClampedArray.from({ length: size * size }, (_, i) => 255 - (data[i * channels + ch] / max) * 238);
}

const mapEls = [];
function makeMap(parent, layer, index, size) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "map";
  btn.setAttribute("aria-label", `${LAYER_NAME[layer]} ${index + 1}`);
  const canvas = Object.assign(document.createElement("canvas"), { width: size, height: size });
  btn.append(canvas);
  btn.addEventListener("click", () => select(layer, index));
  parent.append(btn);
  mapEls.push({ btn, canvas, layer, index, size });
}
for (let i = 0; i < 8; i++) makeMap($("maps1"), "conv1", i, 28);
for (let i = 0; i < 16; i++) makeMap($("maps2"), "conv2", i, 14);

const unitEls = Array.from({ length: 64 }, (_, i) => {
  const el = document.createElement("button");
  el.type = "button";
  el.className = "unit";
  el.setAttribute("aria-label", `Hidden unit ${i + 1}`);
  el.addEventListener("click", () => select("hidden", i));
  $("units").append(el);
  return el;
});

function drawMaps(r) {
  for (const { canvas, layer, index, size } of mapEls) {
    const px = channelPixels(r[layer], size, SIZES[layer], index);
    canvas.getContext("2d").drawImage(grayCanvas(px, size), 0, 0);
  }
  const max = Math.max(1e-9, ...r.hidden);
  unitEls.forEach((u, i) => {
    const t = r.hidden[i] / max;
    u.style.background = t > 0 ? `color-mix(in srgb, #b9a6ff ${Math.round(t * 100)}%, #fff)` : "#fff";
    u.title = `Hidden unit ${i + 1}: ${r.hidden[i].toFixed(2)}`;
  });
}

function updateAblationBar() {
  const n = ablatedCount();
  $("ablationBar").hidden = n === 0;
  $("ablationCount").textContent = `${plural(n, "unit")} switched off`;
  mapEls.forEach(({ btn, layer, index }) => {
    btn.classList.toggle("off", state.masks[layer][index] === 0);
    btn.classList.toggle("selected", state.selected?.layer === layer && state.selected?.index === index);
  });
  unitEls.forEach((u, i) => {
    u.classList.toggle("off", state.masks.hidden[i] === 0);
    u.classList.toggle("selected", state.selected?.layer === "hidden" && state.selected?.index === i);
  });
}

$("resetAblation").addEventListener("click", () => {
  state.masks = noMasks();
  run();
});

// ---------- inspector ----------

function select(layer, index) {
  state.selected = { layer, index };
  updateAblationBar();
  renderInspector();
  $("inspector").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

$("inspClose").addEventListener("click", () => {
  state.selected = null;
  updateAblationBar();
  renderInspector();
});

$("inspToggle").addEventListener("click", () => {
  const { layer, index } = state.selected;
  state.masks[layer][index] = state.masks[layer][index] === 0 ? 1 : 0;
  run();
  renderInspector();
});

const fromB64 = (s) => Uint8ClampedArray.from(atob(s), (c) => c.charCodeAt(0));

function renderInspector() {
  const sel = state.selected;
  $("inspector").hidden = !sel;
  if (!sel) return;
  const { layer, index } = sel;
  const off = state.masks[layer][index] === 0;
  $("inspTitle").textContent = `${LAYER_NAME[layer]} ${index + 1}${off ? " (switched off)" : ""}`;
  $("inspToggle").textContent = off ? "Switch back on" : "Switch off";

  const now = $("inspNow");
  if (layer !== "hidden" && state.result) {
    const size = layer === "conv1" ? 28 : 14;
    now.width = now.height = size;
    now.getContext("2d").drawImage(grayCanvas(channelPixels(state.result[layer], size, SIZES[layer], index), size), 0, 0);
    now.parentElement.hidden = false;
  } else now.parentElement.hidden = true;

  if (state.insights) {
    const vis = fromB64(state.insights.featvis[layer][index]).map((v) => 255 - v);
    $("inspVis").getContext("2d").drawImage(grayCanvas(vis), 0, 0);
    $("inspExamples").replaceChildren(
      ...state.insights.examples[layer][index].map((s) => {
        const c = grayCanvas(fromB64(s).map((v) => 255 - v));
        return c;
      }),
    );
  }

  const parts = [];
  if (state.result) {
    if (layer === "hidden") parts.push(`Activation on your drawing: ${state.result.hidden[index].toFixed(2)}.`);
    else {
      const size = layer === "conv1" ? 28 : 14;
      let m = 0;
      for (let i = 0; i < size * size; i++) m = Math.max(m, state.result[layer][i * SIZES[layer] + index]);
      parts.push(`Strongest response on your drawing: ${m.toFixed(2)}.`);
    }
  }
  if (layer === "hidden" && state.circuit) {
    const c = state.circuit.toScore[index];
    parts.push(`It ${c >= 0 ? "adds" : "takes away"} ${Math.abs(c).toFixed(2)} ${c >= 0 ? "to" : "from"} the score for ${state.target}.`);
  }
  parts.push(off ? "It's switched off: its output is fixed at zero everywhere." : "Switch it off to see what the network does without it.");
  $("inspNote").textContent = parts.join(" ");
}

// ---------- circuit ----------

const SVGNS = "http://www.w3.org/2000/svg";
const svgEl = (name, attrs, parent) => {
  const el = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  parent?.append(el);
  return el;
};

const topBy = (values, k, filter = () => true) =>
  values
    .map((v, i) => [i, v])
    .filter(([i, v]) => v !== 0 && filter(i))
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, k)
    .map(([i]) => i);

function renderCircuit() {
  const svg = $("circuitSvg");
  const c = state.circuit;
  if (!c || !state.result) return;
  svg.innerHTML = "";
  const versus = versusValue();
  $("circuitHint").textContent =
    versus !== null && versus !== state.target ? `The strongest paths into ${state.target} rather than ${versus}` : `The strongest paths into the score for ${state.target}`;

  const hidden = topBy([...c.toScore], 5);
  const conv2 = [...new Set(hidden.flatMap((j) => topBy(c.conv2ToHidden.map((row) => row[j]), 2)))];
  const conv1 = [...new Set(conv2.flatMap((ch) => topBy([...c.conv1ToConv2[ch]], 2)))];

  const W = 760, node = 56;
  const rows = Math.max(conv1.length, conv2.length, hidden.length, 1);
  const H = Math.max(240, rows * (node + 30) + 50);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const xs = [70, 280, 490, 690];
  const place = (list, col) => new Map(list.map((id, k) => [id, { x: xs[col], y: ((k + 0.5) * (H - 40)) / list.length + 20 }]));
  const P1 = place(conv1, 0), P2 = place(conv2, 1), PH = place(hidden, 2), PO = new Map([[0, { x: xs[3], y: H / 2 }]]);

  ["Layer 1", "Layer 2", "Hidden", "Output"].forEach((t, i) => svgEl("text", { x: xs[i], y: 14, "text-anchor": "middle", class: "svg-col" }, svg).append(t));

  const edges = [];
  conv2.forEach((ch) => conv1.forEach((i) => edges.push([P1.get(i), P2.get(ch), c.conv1ToConv2[ch][i], 0])));
  hidden.forEach((j) => conv2.forEach((ch) => edges.push([P2.get(ch), PH.get(j), c.conv2ToHidden[ch][j], 1])));
  hidden.forEach((j) => edges.push([PH.get(j), PO.get(0), c.toScore[j], 2]));
  const maxPer = [0, 1, 2].map((lvl) => Math.max(1e-9, ...edges.filter((e) => e[3] === lvl).map((e) => Math.abs(e[2]))));
  const g = svgEl("g", {}, svg);
  for (const [a, b, v, lvl] of edges) {
    const w = Math.abs(v) / maxPer[lvl];
    if (w < 0.08) continue;
    const mx = (a.x + b.x) / 2;
    svgEl("path", {
      d: `M${a.x + node / 2},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x - node / 2},${b.y}`,
      class: `edge ${v >= 0 ? "pos" : "neg"}`,
      "stroke-width": (1 + w * 9).toFixed(1),
      "stroke-opacity": (0.25 + 0.7 * w).toFixed(2),
    }, g);
  }

  const imageNode = (p, layer, index, url, label) => {
    const a = svgEl("g", { class: "node", tabindex: 0, role: "button", "aria-label": `${LAYER_NAME[layer]} ${index + 1}` }, svg);
    svgEl("rect", { x: p.x - node / 2 - 3, y: p.y - node / 2 - 3, width: node + 6, height: node + 6, rx: 9, class: "node-frame" + (state.masks[layer][index] === 0 ? " off" : "") }, a);
    svgEl("image", { href: url, x: p.x - node / 2, y: p.y - node / 2, width: node, height: node, style: "image-rendering:pixelated" }, a);
    svgEl("text", { x: p.x, y: p.y + node / 2 + 16, "text-anchor": "middle", class: "svg-label" }, a).append(label);
    const open = () => select(layer, index);
    a.addEventListener("click", open);
    a.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && open());
  };
  const mapUrl = (layer, i) => {
    const size = layer === "conv1" ? 28 : 14;
    return grayCanvas(channelPixels(state.result[layer], size, SIZES[layer], i), size).toDataURL();
  };
  conv1.forEach((i) => imageNode(P1.get(i), "conv1", i, mapUrl("conv1", i), `filter ${i + 1}`));
  conv2.forEach((i) => imageNode(P2.get(i), "conv2", i, mapUrl("conv2", i), `filter ${i + 1}`));
  hidden.forEach((j) => {
    const url = state.insights ? grayCanvas(fromB64(state.insights.featvis.hidden[j]).map((v) => 255 - v)).toDataURL() : "";
    imageNode(PH.get(j), "hidden", j, url, `unit ${j + 1}`);
  });
  const o = PO.get(0);
  svgEl("rect", { x: o.x - 34, y: o.y - 34, width: 68, height: 68, rx: 12, class: "out-node" }, svg);
  svgEl("text", { x: o.x, y: o.y + 12, "text-anchor": "middle", class: "out-digit" }, svg).append(
    versus !== null && versus !== state.target ? `${state.target}−${versus}` : String(state.target),
  );
}

// ---------- probes ----------

const CONCEPTS = [
  { key: "digit", name: "Digit", metric: "accuracy" },
  { key: "loop", name: "Closed loop", metric: "accuracy" },
  { key: "slant", name: "Slant", metric: "R²" },
  { key: "thickness", name: "Stroke thickness", metric: "R²" },
];
const LAYER_LABELS = ["Input", "Layer 1", "Layer 2", "Hidden"];

function renderProbeChart() {
  const p = state.insights?.probes;
  if (!p) return;
  $("probeChart").innerHTML = CONCEPTS.map(({ key, name, metric }) => {
    const bars = p.scores[key]
      .map((s, i) => {
        const v = Math.max(0, s);
        return `<div class="pbar"><span class="pbar-label">${LAYER_LABELS[i]}</span><span class="pbar-track"><span class="pbar-fill" style="width:${(v * 100).toFixed(1)}%"></span></span><span class="pbar-val">${metric === "accuracy" ? `${(v * 100).toFixed(1)}%` : v.toFixed(2)}</span></div>`;
      })
      .join("");
    return `<div class="pgroup"><p class="pgroup-title">${name} <small>${metric}</small></p>${bars}</div>`;
  }).join("");

  // One-line reading of the chart, from the actual numbers.
  const [, l1, l2, hid] = p.scores.thickness;
  const [inp, , , dHid] = p.scores.digit;
  const el = $("probeTakeaway");
  el.hidden = false;
  el.innerHTML =
    `The digit gets easier to read out with depth (${(inp * 100).toFixed(0)}% from raw pixels, ${(dHid * 100).toFixed(0)}% from the hidden layer). ` +
    (hid < Math.max(l1, l2) - 0.2
      ? `Stroke thickness goes the other way: it's easy to read from the convolution layers (R² ${Math.max(l1, l2).toFixed(2)}) but mostly gone by the hidden layer (R² ${hid.toFixed(2)}). The network keeps what it needs to name the digit and throws away style.`
      : `Style properties such as thickness stay readable throughout.`);
}

// Ground truth for the concepts, computed from the drawing like training/analysis.py does.
function hasLoop(px) {
  const ink = px.map((v) => (v > 0.35 ? 1 : 0));
  const seen = new Uint8Array(784);
  const q = [];
  for (let i = 0; i < 784; i++) {
    const y = Math.floor(i / 28), x = i % 28;
    if ((y === 0 || y === 27 || x === 0 || x === 27) && !ink[i]) {
      seen[i] = 1;
      q.push(i);
    }
  }
  while (q.length) {
    const i = q.pop();
    const y = Math.floor(i / 28), x = i % 28;
    for (const [ny, nx] of [[y + 1, x], [y - 1, x], [y, x + 1], [y, x - 1]]) {
      const j = ny * 28 + nx;
      if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28 && !ink[j] && !seen[j]) {
        seen[j] = 1;
        q.push(j);
      }
    }
  }
  let holes = 0;
  for (let i = 0; i < 784; i++) if (!ink[i] && !seen[i]) holes++;
  return holes >= 4 ? 1 : 0;
}

function slant(px) {
  let m = 1e-9, my = 0, mx = 0;
  for (let i = 0; i < 784; i++) {
    m += px[i];
    my += px[i] * Math.floor(i / 28);
    mx += px[i] * (i % 28);
  }
  my /= m;
  mx /= m;
  let cov = 0, vy = 1e-9;
  for (let i = 0; i < 784; i++) {
    const dy = Math.floor(i / 28) - my, dx = (i % 28) - mx;
    cov += px[i] * dx * dy;
    vy += px[i] * dy * dy;
  }
  return -cov / m / (vy / m);
}

function thickness(px) {
  const ink = (y, x) => y >= 0 && y < 28 && x >= 0 && x < 28 && px[y * 28 + x] > 0.35;
  let area = 0, edge = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      if (!ink(y, x)) continue;
      area++;
      if (!(ink(y - 1, x) && ink(y + 1, x) && ink(y, x - 1) && ink(y, x + 1))) edge++;
    }
  }
  return (2 * area) / Math.max(edge, 1);
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const fmt = {
  digit: (z) => {
    const e = z.map((v) => Math.exp(v - Math.max(...z)));
    const s = e.reduce((a, b) => a + b, 0);
    const d = argmax(z);
    return `${d} <small>${Math.round((e[d] / s) * 100)}%</small>`;
  },
  loop: (z) => {
    const p = sigmoid(z[0]);
    return `${p >= 0.5 ? "yes" : "no"} <small>${Math.round((p >= 0.5 ? p : 1 - p) * 100)}%</small>`;
  },
  slant: (z) => `${z[0] >= 0 ? "+" : ""}${z[0].toFixed(2)}`,
  thickness: (z) => `${z[0].toFixed(1)} px`,
};

function renderProbes() {
  const p = state.insights?.probes;
  const table = $("probeTable");
  if (!p) return;
  if (!state.result) {
    table.innerHTML = `<tr><td class="probe-empty">Draw a digit to read its layers.</td></tr>`;
    return;
  }
  const px = state.pixels;
  const truth = {
    digit: `${state.predicted} <small>network's answer</small>`,
    loop: hasLoop(px) ? "yes" : "no",
    slant: `${slant(px) >= 0 ? "+" : ""}${slant(px).toFixed(2)}`,
    thickness: `${thickness(px).toFixed(1)} px`,
  };
  const reads = p.layers.map((layer) => readProbes(p.weights[layer], state.result.features[layer]));
  table.innerHTML =
    `<thead><tr><th></th><th>Actual</th>${LAYER_LABELS.map((l) => `<th>${l}</th>`).join("")}</tr></thead><tbody>` +
    CONCEPTS.map(({ key, name }) => `<tr><th>${name}</th><td class="truth">${truth[key]}</td>${reads.map((r) => `<td>${fmt[key](r[key])}</td>`).join("")}</tr>`).join("") +
    "</tbody>";
}

// ---------- start ----------

resetPad();
renderEmpty();
Promise.all([loadModel("model/weights.json"), fetch("model/insights.json").then((r) => (r.ok ? r.json() : null))])
  .then(([model, insights]) => {
    state.model = model;
    state.insights = insights;
    $("status").textContent = "Draw a digit, 0 to 9";
    if (model.meta?.test_accuracy) {
      $("accuracy").textContent = `It gets ${(model.meta.test_accuracy * 100).toFixed(1)}% of the 10,000 MNIST test digits right.`;
    }
    renderProbeChart();
    run();
  })
  .catch(() => ($("status").textContent = "The model failed to load."));
