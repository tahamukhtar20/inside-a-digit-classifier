// The guided journey: draw, pick a way in, walk through the network step by step,
// and only then see the answer. All numbers on screen come from the real model.

import { analyse, circuit, loadModel, noMasks, readProbes } from "./model.js";
import { icon } from "./icons.js";
import {
  argmax, channelPixels, describeRegion, drawRGBA, fromB64, grayCanvas, hasLoop, heatPixels,
  inputPixels, Pad, pct, plural, slant, thickness, topBy,
} from "./common.js";

const $ = (id) => document.getElementById(id);
const stage = $("stage");
const SVGNS = "http://www.w3.org/2000/svg";

// ---------- loading ----------

const ready = Promise.all([
  loadModel("model/weights.json"),
  fetch("model/insights.json").then((r) => r.json()),
  fetch("model/samples.json").then((r) => r.json()),
]).then(([model, insights, samples]) => Object.assign(app, { model, insights, samples }));

const app = { model: null, insights: null, samples: null, data: null, journey: null, step: 0, masks: null };

// The drawing pad lives across screens so the drawing survives going back.
const padCanvas = Object.assign(document.createElement("canvas"), { width: 280, height: 280, className: "pad" });
padCanvas.setAttribute("aria-label", "Drawing area: draw a digit with your mouse, finger or pen");
const pad = new Pad(padCanvas, () => onInk());

// ---------- small helpers ----------

const el = (tag, attrs = {}, html = "") => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (html) e.innerHTML = html;
  return e;
};

const svg = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent?.append(e);
  return e;
};

const curve = (x1, y1, x2, y2) => {
  const mx = (x1 + x2) / 2;
  return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
};

const url = (canvas) => canvas.toDataURL();
const digitURL = () => url(grayCanvas(inputPixels(app.data.px)));
const mapURL = (layer, ch) => {
  const size = layer === "conv1" ? 28 : 14;
  return url(grayCanvas(channelPixels(app.data.r[layer], size, layer === "conv1" ? 8 : 16, ch), size));
};
const hasVis = (j) => {
  const b = app.insights.featvis.hidden[j];
  return b && new Set(fromB64(b)).size > 8;
};
const featURL = (layer, i) => url(grayCanvas(fromB64(app.insights.featvis[layer][i]).map((v) => 255 - v)));

function image(parent, href, x, y, size, cls = "", delay = 0) {
  const g = svg("g", { class: `svg-pop ${cls}`, style: `animation-delay:${delay}s` }, parent);
  svg("rect", { x: x - 3, y: y - 3, width: size + 6, height: size + 6, rx: 8, class: `svg-frame ${cls.includes("star") ? "star" : ""}` }, g);
  svg("image", { href, x, y, width: size, height: size, preserveAspectRatio: "none", style: "image-rendering:pixelated" }, g);
  return g;
}

function flowLine(parent, x1, y1, x2, y2, { width = 2, kind = "", speed = 1.1, delay = 0 } = {}) {
  return svg("path", {
    d: curve(x1, y1, x2, y2),
    class: `flow ${kind}`,
    "stroke-width": width.toFixed(1),
    style: `animation-duration:${speed}s; animation-delay:-${delay}s`,
  }, parent);
}

function particles(parent, d, n, dur = 1.6) {
  for (let k = 0; k < n; k++) {
    const c = svg("circle", { r: 3.2, class: "particle" }, parent);
    svg("animateMotion", { dur: `${dur}s`, repeatCount: "indefinite", begin: `${(-k * dur) / n}s`, path: d }, c);
  }
}

function countUp(node, to, ms = 1200) {
  const t0 = performance.now();
  const tick = (t) => {
    const k = Math.min(1, (t - t0) / ms);
    node.textContent = Math.round(to * (1 - (1 - k) ** 3)).toLocaleString("en");
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const later = (fn, ms = 60) => setTimeout(fn, ms);

function show(screen, { dots = null } = {}) {
  const old = stage.firstElementChild;
  const swap = () => {
    stage.replaceChildren(screen);
    screen.classList.add("screen");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  if (old) {
    old.classList.add("leaving");
    setTimeout(swap, 200);
  } else swap();
  renderDots(dots);
}

function renderDots(dots) {
  const ol = $("dots");
  ol.replaceChildren();
  if (!dots) return;
  for (let i = 0; i < dots.total; i++) {
    const li = el("li", { class: i < dots.at ? "done" : i === dots.at ? "now" : "" });
    ol.append(li);
  }
}

function showMini(on) {
  $("mini").hidden = !on;
  if (on) {
    const c = $("miniCanvas");
    c.getContext("2d").drawImage(grayCanvas(inputPixels(app.data.px)), 0, 0);
  }
}
$("redraw").addEventListener("click", () => drawScreen());

// ---------- 1. draw ----------

let doneBtn = null;
function onInk() {
  if (doneBtn) doneBtn.disabled = !pad.inked;
  document.querySelector(".ghost")?.classList.toggle("gone", pad.inked);
}

function drawScreen() {
  showMini(false);
  const s = el("section", { class: "draw-wrap" });
  s.innerHTML = `
    <div class="reveal-in">
      <h1>Draw a digit. Then look inside the network reading it.</h1>
      <p class="say">A small neural network will read your drawing. Before it tells you what it thinks, you'll follow
      what happens inside it, step by step. The answer comes last.</p>
      <div class="draw-actions">
        <button class="btn btn-big" id="done" type="button" disabled>Done, take me inside ${icon("arrow-right")}</button>
        <button class="linkish" id="example" type="button">${icon("shuffle", 16)} Use an example</button>
      </div>
    </div>
    <div class="pad-col"><div class="pad-box"></div><button class="linkish pad-clear" id="clearPad" type="button">${icon("eraser", 16)} Clear</button></div>`;
  const box = s.querySelector(".pad-box");
  box.append(padCanvas);
  const ghost = svg("svg", { class: `ghost ${pad.inked ? "gone" : ""}`, viewBox: "0 0 280 280", "aria-hidden": "true" }, box);
  svg("path", { d: "M80,80 C120,62 170,60 205,78 C185,130 160,175 140,215" }, ghost);
  doneBtn = s.querySelector("#done");
  doneBtn.disabled = !pad.inked;
  doneBtn.addEventListener("click", start);
  s.querySelector("#clearPad").addEventListener("click", () => pad.clear());
  s.querySelector("#example").addEventListener("click", async () => {
    await ready;
    pad.loadDigit(app.samples[Math.floor(Math.random() * app.samples.length)].pixels);
  });
  show(s);
}

// ---------- analysis of the drawing (done once, then shared by every journey) ----------

async function start() {
  doneBtn.disabled = true;
  doneBtn.textContent = "Reading your drawing…";
  doneBtn.classList.add("busy");
  await ready;
  const px = pad.pixels();
  const first = analyse(app.model, px, 0);
  const predicted = argmax(first.probs);
  const r = analyse(app.model, px, predicted, null, { smooth: true });
  const p = app.insights.probes;
  app.data = {
    px,
    predicted,
    r,
    circ: circuit(app.model, px, predicted),
    reads: p.layers.map((layer) => readProbes(p.weights[layer], r.features[layer])),
    truth: { loop: hasLoop(px), slant: slant(px), thickness: thickness(px) },
  };
  app.masks = noMasks();
  doneBtn.innerHTML = `Done, take me inside ${icon("arrow-right")}`;
  doneBtn.classList.remove("busy");
  chooseScreen();
}

// ---------- 2. choose a way in ----------

const JOURNEYS = {
  signal: { title: "Follow the signal", icon: "activity", steps: [signal1, signal2, signal3, signal4, signal5] },
  mind: { title: "Read its mind", icon: "brain", steps: [mind1, mind2, mind3, mind4] },
  break: { title: "Try to break it", icon: "zap", steps: [break1, break2] },
};

function chooseScreen(exclude = null) {
  showMini(true);
  const s = el("section");
  s.innerHTML = `
    <div class="reveal-in">
      <h1>How do you want to look inside?</h1>
    </div>
    <div class="paths">
      ${pathCard("signal", "activity", "Follow the signal", "Watch your pixels turn into strokes, then parts, then features, layer by layer, until it decides.", "5 steps")}
      ${pathCard("mind", "brain", "Read its mind", "Ask what each layer knows about your drawing before it answers: loops, slant, thickness, the digit.", "4 steps")}
      ${pathCard("break", "zap", "Try to break it", "Find the few units its decision rests on and switch them off. Can you change its mind?", "2 steps + a challenge")}
    </div>`;
  s.querySelectorAll(".path").forEach((b) => b.addEventListener("click", () => startJourney(b.dataset.j)));
  if (exclude) s.querySelector(`[data-j="${exclude}"]`)?.classList.add("tried");
  show(s);
}

function pathCard(key, ico, title, text, len) {
  const icons = {
    signal: `<svg viewBox="0 0 200 90" aria-hidden="true"><rect x="12" y="25" width="40" height="40" rx="6" fill="#fff" stroke="#111" stroke-width="2.5"/>
      ${[20, 45, 70].map((y) => `<path class="flow" d="M52,45 C80,45 80,${y} 104,${y}" stroke-width="2"/><rect x="104" y="${y - 9}" width="18" height="18" rx="4" fill="#fff" stroke="#111" stroke-width="2"/><path class="flow hot" d="M122,${y} C150,${y} 150,45 168,45" stroke-width="2"/>`).join("")}
      <circle cx="178" cy="45" r="11" fill="#ffd23f" stroke="#111" stroke-width="2.5"/></svg>`,
    mind: `<svg viewBox="0 0 200 90" aria-hidden="true"><circle cx="58" cy="45" r="26" fill="#fff" stroke="#111" stroke-width="2.5"/><circle cx="58" cy="45" r="11" fill="none" stroke="#b9a6ff" stroke-width="5"/>
      ${[0, 1, 2].map((k) => `<rect x="104" y="${16 + k * 22}" width="78" height="12" rx="5" fill="#fff" stroke="#111" stroke-width="2"/><rect x="106" y="${18 + k * 22}" width="${30 + k * 22}" height="8" rx="3" fill="#7ae7c7"><animate attributeName="width" values="4;${30 + k * 22};${30 + k * 22}" dur="2.4s" repeatCount="indefinite"/></rect>`).join("")}</svg>`,
    break: `<svg viewBox="0 0 200 90" aria-hidden="true">${[18, 45, 72].map((y, k) => `<path class="flow ${k === 1 ? "dead" : "hot"}" d="M40,${y} C90,${y} 110,45 156,45" stroke-width="${k === 1 ? 2 : 3}"/><circle cx="30" cy="${y}" r="9" fill="${k === 1 ? "#ffb3c7" : "#fff"}" stroke="#111" stroke-width="2"/>`).join("")}
      <circle cx="168" cy="45" r="13" fill="#ffd23f" stroke="#111" stroke-width="2.5"/><path d="M96,26 l8,16 h-9 l8,18" fill="none" stroke="#111" stroke-width="3" stroke-linejoin="round"/></svg>`,
  };
  return `<button class="path path-${key}" data-j="${key}" type="button">${icons[key]}<h3>${icon(ico, 20)} ${title}</h3><p>${text}</p><span class="go">${len} ${icon("arrow-right", 16)}</span></button>`;
}

// ---------- 3. steps ----------

function startJourney(key) {
  app.journey = key;
  app.step = 0;
  app.masks = noMasks();
  stepScreen();
}

function stepScreen() {
  const j = JOURNEYS[app.journey];
  const total = j.steps.length + 1;
  const s = el("section", { class: "step" });
  const { title, say, visual, how, next } = j.steps[app.step]();
  s.innerHTML = `
    <div class="reveal-in">
      <h2 class="step-title">${title}</h2><p class="say">${say}</p></div>`;
  const v = el("div", { class: "visual" });
  v.append(visual);
  s.append(v);
  if (how) s.append(el("details", { class: "how" }, `<summary>${icon("info", 16)} How is this computed?</summary><p>${how}</p>`));
  const nav = el("div", { class: "nav" });
  const back = el("button", { class: "linkish", type: "button" }, `${icon("arrow-left", 16)} ${app.step === 0 ? "Choose another way" : "Back"}`);
  back.addEventListener("click", () => (app.step === 0 ? chooseScreen() : (app.step--, stepScreen())));
  const fwd = el("button", { class: "btn btn-big", type: "button" }, `${next ?? (app.step === j.steps.length - 1 ? "See its answer" : "Next")} ${icon("arrow-right")}`);
  fwd.addEventListener("click", () => (app.step < j.steps.length - 1 ? (app.step++, stepScreen()) : revealScreen()));
  nav.append(back, fwd);
  s.append(nav);
  show(s, { dots: { total, at: app.step } });
}

// ----- Follow the signal -----

function signal1() {
  const { px } = app.data;
  const inked = px.filter((v) => v > 0.1).length;
  const W = 640, H = 260;
  const box = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "flow-svg", role: "img", "aria-label": "Your drawing becoming a 28 by 28 grid" });
  const raw = url(padCanvas);
  image(box, raw, 20, 30, 200);
  for (let k = 0; k < 5; k++) flowLine(box, 226, 70 + k * 30, 336, 70 + k * 30, { width: 2, delay: k * 0.2 });
  const g = svg("g", {}, box);
  const cell = 7.6, ox = 360, oy = 24;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = px[y * 28 + x];
      const shade = Math.round(255 - v * 238);
      svg("rect", {
        x: ox + x * cell, y: oy + y * cell, width: cell - 0.6, height: cell - 0.6, rx: 1.2,
        fill: `rgb(${shade},${shade},${shade})`, class: "svg-pop",
        style: `animation-delay:${(y * 0.035 + x * 0.004).toFixed(3)}s`,
      }, g);
    }
  }
  const t = svg("text", { x: ox + 14 * cell, y: H - 4, "text-anchor": "middle", class: "svg-big" }, box);
  const n = svg("tspan", { class: "counter" }, t);
  n.textContent = "0";
  t.append(" numbers");
  later(() => countUp(n, 784, 1400), 300);
  return {
    title: "First, your drawing becomes numbers",
    say: `The network never sees your strokes. It sees a 28 × 28 grid of brightness values, <b>784 numbers</b>, and <b>${inked}</b> of them carry ink.`,
    visual: box,
    how: "The drawing is cropped to the ink, scaled so its longer side is 20 pixels, and centred by its centre of mass on a 28 × 28 grid, the same way the MNIST training digits were prepared.",
  };
}

function strongest(layer, n) {
  const size = layer === "conv1" ? 28 : 14, ch = layer === "conv1" ? 8 : 16;
  const tot = Array.from({ length: ch }, (_, c) => {
    let s = 0;
    for (let i = 0; i < size * size; i++) s += app.data.r[layer][i * ch + c];
    return s;
  });
  return { order: topBy(tot, n), tot };
}

function signal2() {
  const W = 640, H = 290;
  const box = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "flow-svg", role: "img", "aria-label": "Layer 1: eight feature maps" });
  const { order } = strongest("conv1", 8);
  const best = order[0];
  const mx = (c) => 14 + c * 78;
  for (let c = 0; c < 8; c++) {
    flowLine(box, 320, 94, mx(c) + 33, 172, { width: c === best ? 3.5 : 1.6, kind: c === best ? "hot" : "", delay: c * 0.13 });
  }
  image(box, digitURL(), 285, 12, 70);
  for (let c = 0; c < 8; c++) {
    image(box, mapURL("conv1", c), mx(c), 178, 66, c === best ? "star" : "", 0.25 + c * 0.09);
    svg("text", { x: mx(c) + 33, y: 268, "text-anchor": "middle", class: "svg-cap" }, box).append(`filter ${c + 1}`);
  }
  return {
    title: "Layer 1 looks for strokes",
    say: `Eight small filters slide across the grid. Each one looks for a single simple pattern, an edge or a stroke in one direction, and produces its own map: dark where it found it. On your drawing, <b>filter ${best + 1}</b> fires the most.`,
    visual: box,
    how: "Each filter is a 3 × 3 grid of learned weights. At every position it multiplies the 9 pixels underneath by its weights and sums them, then negative results are set to zero (ReLU). Darker means a stronger response.",
  };
}

function signal3() {
  const W = 640, H = 270;
  const box = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "flow-svg", role: "img", "aria-label": "Layer 2: sixteen feature maps built from layer 1" });
  const p1 = (c) => ({ x: 34 + c * 74, y: 12 });
  const p2 = (c) => ({ x: 8 + c * 39.5, y: 200 });
  const { conv1ToConv2 } = app.data.circ;
  const pairs = [];
  conv1ToConv2.forEach((row, c2) => row.forEach((v, c1) => pairs.push([c1, c2, v])));
  pairs.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
  const top = pairs.slice(0, 16);
  const max = Math.abs(top[0]?.[2] ?? 1) || 1;
  top.forEach(([c1, c2, v], k) => {
    const a = p1(c1), b = p2(c2);
    svg("path", {
      d: `M${a.x + 26},${a.y + 58} C${a.x + 26},${a.y + 130} ${b.x + 16},${b.y - 70} ${b.x + 16},${b.y - 4}`,
      class: `flow ${v >= 0 ? "hot" : "cold"}`,
      "stroke-width": (0.8 + (Math.abs(v) / max) * 4).toFixed(1),
      style: `animation-delay:-${k * 0.07}s`,
    }, box);
  });
  for (let c = 0; c < 8; c++) image(box, mapURL("conv1", c), p1(c).x, p1(c).y, 52);
  const { order } = strongest("conv2", 16);
  for (let c = 0; c < 16; c++) image(box, mapURL("conv2", c), p2(c).x, p2(c).y, 32, c === order[0] ? "star" : "", 0.3 + c * 0.05);
  svg("text", { x: 320, y: 262, "text-anchor": "middle", class: "svg-cap" }, box).append("16 layer-2 maps, 14 × 14");
  return {
    title: "Layer 2 combines strokes into parts",
    say: `The second layer reads all eight stroke maps at once and combines them into <b>16 maps of bigger shapes</b>: corners, curves, bars. The picture is now half the size, 14 × 14, so each pixel covers more of your digit. The lines show the strongest connections on your drawing: <b>pink adds</b>, <b>blue suppresses</b>.`,
    visual: box,
    how: "Before layer 2, each map is shrunk by keeping the largest value in every 2 × 2 block (max-pooling). Each layer-2 filter has 3 × 3 weights for each of the 8 input maps. Line thickness is the gradient of the layer-2 map's total activation with respect to the layer-1 map, times that map's activation.",
  };
}

function signal4() {
  const { hidden } = app.data.r;
  const W = 640, H = 350;
  const box = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "flow-svg", role: "img", "aria-label": "The 64 hidden units lighting up" });
  const on = hidden.filter((v) => v > 0).length;
  const rank = topBy(hidden, 64);
  const max = Math.max(1e-9, ...hidden);
  const p2 = (c) => ({ x: 10 + c * 39.5, y: 10 });
  const star = rank.find(hasVis) ?? rank[0];
  const pu = (j) => ({ x: 170 + (j % 8) * 30, y: 120 + Math.floor(j / 8) * 28 });
  const { conv2ToHidden } = app.data.circ;
  rank.slice(0, 6).forEach((j, k) => {
    topBy(conv2ToHidden.map((row) => row[j]), 2).forEach((c) => {
      const a = p2(c), b = pu(j);
      svg("path", {
        d: `M${a.x + 15},${a.y + 34} C${a.x + 15},${a.y + 90} ${b.x},${b.y - 60} ${b.x},${b.y - 10}`,
        class: `flow hot`, "stroke-width": j === star ? 3 : 1.6, style: `animation-delay:-${k * 0.2}s`,
      }, box);
    });
  });
  for (let c = 0; c < 16; c++) image(box, mapURL("conv2", c), p2(c).x, p2(c).y, 30);
  rank.forEach((j, k) => {
    const t = hidden[j] / max;
    const p = pu(j);
    svg("circle", {
      cx: p.x, cy: p.y, r: 3.5 + t * 8.5,
      fill: t > 0 ? `color-mix(in srgb, #b9a6ff ${Math.round(20 + t * 80)}%, #fff)` : "#fff",
      class: "unit-dot svg-pop", style: `animation-delay:${(0.2 + k * 0.025).toFixed(3)}s`,
    }, box);
  });
  svg("text", { x: 275, y: H - 6, "text-anchor": "middle", class: "svg-cap" }, box).append("64 hidden units (bigger = stronger)");
  flowLine(box, pu(star).x + 12, pu(star).y, 486, 205, { width: 2.5, kind: "hot" });
  image(box, featURL("hidden", star), 492, 160, 90, "star", 1.4);
  svg("text", { x: 537, y: 272, "text-anchor": "middle", class: "svg-cap" }, box).append(`unit ${star + 1} likes this`);
  return {
    title: "64 features summarise what it saw",
    say: `All 16 maps are flattened and fed into <b>64 hidden units</b>, each a yes-or-no-ish detector for some combination of parts. <b>${on} of 64</b> switch on for your drawing. The biggest dots are the strongest. One of the strongest, <b>unit ${star + 1}</b>, responds most to inputs like the one on the right.`,
    visual: box,
    how: "Each hidden unit is a weighted sum of all 784 pooled layer-2 values (7 × 7 × 16) plus a bias, passed through ReLU. The image on the right is a feature visualisation: an input optimised by gradient ascent to excite that unit as much as possible.",
  };
}

function signal5() {
  const { hidden, logits } = app.data.r;
  const fc2 = app.model.fc2; // [64 * 10], row-major [unit][digit]
  const W = 640, H = 330;
  const box = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "flow-svg", role: "img", "aria-label": "Hidden units voting for the ten digits" });
  const units = topBy(hidden, 8, hasVis);
  const pu = (k) => ({ x: 70, y: 30 + k * 36 });
  // Outputs in a scrambled order, so the glowing one's position doesn't give the digit away.
  const order = [3, 8, 1, 6, 0, 9, 4, 2, 7, 5];
  const slot = (d) => (order.indexOf(d) + app.data.predicted * 3) % 10;
  const po = (d) => ({ x: 520, y: 22 + slot(d) * 30 });
  const votes = [];
  units.forEach((j, k) => {
    for (let d = 0; d < 10; d++) votes.push([k, d, hidden[j] * fc2[j * 10 + d]]);
  });
  votes.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
  const maxV = Math.abs(votes[0][2]) || 1;
  const winner = argmax(logits);
  votes.slice(0, 34).forEach(([k, d, v], i) => {
    const a = pu(k), b = po(d);
    const line = flowLine(box, a.x + 16, a.y, b.x - 14, b.y, { width: 0.7 + (Math.abs(v) / maxV) * 4.5, kind: v >= 0 ? "hot" : "cold", delay: i * 0.05 });
    if (d === winner && v > 0 && i < 12) particles(box, line.getAttribute("d"), 2, 1.4 + i * 0.05);
  });
  units.forEach((j, k) => {
    const p = pu(k);
    image(box, featURL("hidden", j), p.x - 16, p.y - 16, 32, "", k * 0.06);
  });
  svg("text", { x: 70, y: H - 4, "text-anchor": "middle", class: "svg-cap" }, box).append("its strongest features");
  for (let d = 0; d < 10; d++) {
    const p = po(d);
    svg("circle", { cx: p.x, cy: p.y, r: 12, class: `out-node svg-pop ${d === winner ? "hot" : ""}`, style: `animation-delay:${0.4 + d * 0.05}s` }, box);
    svg("text", { x: p.x, y: p.y + 5.5, "text-anchor": "middle", class: "out-q" }, box).append("?");
  }
  svg("text", { x: 520, y: H - 4, "text-anchor": "middle", class: "svg-cap" }, box).append("ten digits, names hidden");
  return {
    title: "Every feature votes, and one digit wins",
    say: `Finally, each of the 64 features casts a weighted vote for each of the ten digits. Here are the strongest votes from its 8 strongest features: <b>pink for</b>, <b>blue against</b>. One digit collects far more than the rest, the glowing one. We've hidden the labels until the end.`,
    visual: box,
    how: "Each digit's score is the sum over hidden units of activation × weight, plus a bias. A softmax turns the ten scores into probabilities. Line thickness here is activation × weight for each unit-digit pair.",
    next: "See which one it picked",
  };
}

// ----- Read its mind -----

const LAYERS = ["Input", "Layer 1", "Layer 2", "Hidden"];

function holePixels(px) {
  const ink = px.map((v) => (v > 0.35 ? 1 : 0));
  const seen = new Uint8Array(784);
  const q = [];
  for (let i = 0; i < 784; i++) {
    const y = Math.floor(i / 28), x = i % 28;
    if ((y === 0 || y === 27 || x === 0 || x === 27) && !ink[i]) (seen[i] = 1), q.push(i);
  }
  while (q.length) {
    const i = q.pop(), y = Math.floor(i / 28), x = i % 28;
    for (const [ny, nx] of [[y + 1, x], [y - 1, x], [y, x + 1], [y, x - 1]]) {
      const j = ny * 28 + nx;
      if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28 && !ink[j] && !seen[j]) (seen[j] = 1), q.push(j);
    }
  }
  return [...Array(784).keys()].filter((i) => !ink[i] && !seen[i]);
}

function digitStack(overlay) {
  const wrap = el("div", { class: "digit-stack" });
  const c = el("canvas", { width: 28, height: 28, class: "digit-view" });
  c.getContext("2d").drawImage(grayCanvas(inputPixels(app.data.px)), 0, 0);
  wrap.append(c);
  if (overlay) {
    const o = svg("svg", { viewBox: "0 0 28 28" }, wrap);
    overlay(o);
  }
  return wrap;
}

function meters(rows) {
  const m = el("div", { class: "meters" });
  rows.forEach((row, i) => {
    const r = el("div", { class: "meter" });
    r.innerHTML = `<span class="meter-name">${LAYERS[i]}</span><span class="meter-track">${row.fill != null ? `<span class="meter-fill ${row.cls ?? ""}"></span>` : ""}${row.mark != null ? `<span class="meter-mark"></span>` : ""}${row.truth != null ? `<span class="meter-mark" style="left:${row.truth * 100}%;background:#ff508c;opacity:.7"></span>` : ""}</span><span class="meter-val">${row.label}</span>`;
    m.append(r);
    later(() => {
      if (row.fill != null) r.querySelector(".meter-fill").style.width = `${row.fill * 100}%`;
      if (row.mark != null) r.querySelector(".meter-mark").style.left = `calc(${row.mark * 100}% - 1.5px)`;
    }, 250 + i * 220);
  });
  return m;
}

function mindLayout(left, right) {
  const g = el("div", { class: "mind" });
  g.append(left, right);
  return g;
}

function mind1() {
  const { reads, truth, px } = app.data;
  const yes = truth.loop === 1;
  const probs = reads.map((r) => 1 / (1 + Math.exp(-r.loop[0])));
  const sure = probs.map((p) => (yes ? p : 1 - p));
  const holes = holePixels(px);
  const left = el("div");
  left.append(
    digitStack((o) => holes.forEach((i, k) => svg("rect", { x: i % 28, y: Math.floor(i / 28), width: 1, height: 1, fill: "#ffd23f", class: "svg-pop", style: `animation-delay:${0.3 + k * 0.004}s` }, o))),
    el("span", { class: "truth-chip" }, yes ? `${icon("check", 15)} Closed loop` : `${icon("x", 15)} No closed loop`),
  );
  const right = meters(sure.map((s) => ({ fill: s, cls: s >= 0.5 ? "good" : "bad", label: `${Math.round(s * 100)}% ${s >= 0.5 ? '<span class="tick">✓</span>' : '<span class="tick">✗</span>'}` })));
  const firstSure = sure.findIndex((s) => s >= 0.9);
  return {
    title: "Does it notice a closed loop?",
    say: `${yes ? "Your digit has a closed loop (highlighted)." : "Your digit has no closed loop."} The network was never told what a loop is. But we trained a simple <b>probe</b> on each layer to read the answer straight out of its activations. ${firstSure > 0 ? `From raw pixels the probe is ${Math.round(sure[0] * 100)}% sure; from <b>${LAYERS[firstSure].toLowerCase()}</b>, ${Math.round(sure[firstSure] * 100)}%.` : firstSure === 0 ? `Here even the raw pixels are enough (${Math.round(sure[0] * 100)}%).` : "On this drawing no layer is confident; loops are easier to read on neater digits."}`,
    visual: mindLayout(left, right),
    how: "A probe is a logistic regression trained on 12,000 MNIST digits' activations at one layer, with the label computed from the image (a background region fully enclosed by ink). On 3,000 held-out digits it's right 78% of the time from raw pixels and 93% from layer 2.",
  };
}

function scaleOf(key) {
  const [lo, hi] = app.insights.probes.ranges[key];
  return (v) => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
}

function mind2() {
  const { reads, truth } = app.data;
  const s = truth.slant;
  const word = Math.abs(s) < 0.12 ? "stands upright" : s > 0 ? "leans right" : "leans left";
  const to01 = scaleOf("slant");
  const angle = (Math.atan(s) * 180) / Math.PI;
  const left = el("div");
  left.append(
    digitStack((o) => {
      const line = svg("line", { x1: 14, y1: 2, x2: 14, y2: 26, stroke: "#ff508c", "stroke-width": 1.2, "stroke-linecap": "round", style: "transition: transform 1.2s cubic-bezier(.2,.8,.2,1); transform-origin: 14px 14px" }, o);
      later(() => (line.style.transform = `rotate(${angle}deg)`), 300);
    }),
    el("span", { class: "truth-chip" }, `${word === "stands upright" ? "Upright" : word === "leans right" ? "Leans right" : "Leans left"} · ${s >= 0 ? "+" : ""}${s.toFixed(2)}`),
  );
  const right = meters(reads.map((r) => ({ mark: to01(r.slant[0]), truth: to01(s), label: `${r.slant[0] >= 0 ? "+" : ""}${r.slant[0].toFixed(2)}` })));
  const err = reads.map((r) => Math.abs(r.slant[0] - s));
  const best = argmax(err.map((e) => -e));
  return {
    title: "Can it tell which way your digit leans?",
    say: `Your digit ${word}. Each bar shows where a layer's probe puts the slant (black) against the real value (pink). Slant isn't needed to name a digit, yet the convolution layers track it closely. Closest here: <b>${LAYERS[best].toLowerCase()}</b>.`,
    visual: mindLayout(left, right),
    how: "Slant is the horizontal shift per row of the ink's centre of mass. The probes are ridge regressions on each layer's activations; on held-out digits they explain 81% of the variance from raw pixels and 96% from layer 2 (R²).",
  };
}

function mind3() {
  const { reads, truth } = app.data;
  const to01 = scaleOf("thickness");
  const scores = app.insights.probes.scores.thickness;
  const left = el("div");
  left.append(digitStack(), el("span", { class: "truth-chip" }, `Strokes ≈ ${truth.thickness.toFixed(1)} px`));
  const right = meters(reads.map((r) => ({ mark: to01(r.thickness[0]), truth: to01(truth.thickness), label: `${r.thickness[0].toFixed(1)} px` })));
  const forgets = scores[3] < Math.max(scores[1], scores[2]) - 0.2;
  return {
    title: "Does it care how thick your pen was?",
    say: forgets
      ? `Early on, yes: layers 1 and 2 track stroke thickness well (R² ${Math.max(scores[1], scores[2]).toFixed(2)} across 3,000 test digits). By the hidden layer it has mostly <b>thrown it away</b> (R² ${scores[3].toFixed(2)}). Thickness doesn't help name a digit, so the network stops carrying it.`
      : `Every layer keeps track of stroke thickness, even though it doesn't help name the digit.`,
    visual: mindLayout(left, right),
    how: "Thickness is 2 × inked area ÷ perimeter of the ink (in pixels of the 28 × 28 grid). The probes are ridge regressions; the R² values come from 3,000 held-out MNIST digits, not just your drawing.",
  };
}

function ring(p, label, caption, delay) {
  const r = 46, c = 2 * Math.PI * r;
  const d = el("div", { class: "ring" });
  d.innerHTML = `<svg viewBox="0 0 120 120"><circle class="ring-track" cx="60" cy="60" r="${r}"/><circle class="ring-fill" cx="60" cy="60" r="${r}" stroke-dasharray="${c}" stroke-dashoffset="${c}"/><text class="ring-label" x="60" y="68" text-anchor="middle">${label}</text></svg><p>${caption}</p>`;
  later(() => (d.querySelector(".ring-fill").style.strokeDashoffset = `${c * (1 - p)}`), delay);
  return d;
}

function mind4() {
  const { reads } = app.data;
  const conf = reads.map((r) => {
    const z = r.digit, m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    return { p: e[argmax(z)] / e.reduce((a, b) => a + b, 0), d: argmax(z) };
  });
  const rings = el("div", { class: "rings" });
  conf.forEach(({ p }, i) => rings.append(ring(p, "?", `${LAYERS[i]} · ${Math.round(p * 100)}%`, 250 + i * 250)));
  const agree = conf.slice(1).every((c) => c.d === conf[3].d);
  return {
    title: "And which digit is it?",
    say: `The last question is the one it was trained for. Each ring shows how confident a probe on that layer is about its pick; we've hidden the pick itself. From raw pixels it's <b>${Math.round(conf[0].p * 100)}%</b> sure; from the hidden layer, <b>${Math.round(conf[3].p * 100)}%</b>. ${agree ? "From layer 1 on, every layer points to the same digit." : "The layers don't all agree, which makes this one interesting."}`,
    visual: rings,
    how: "Ten-way logistic regressions on each layer's activations. On held-out digits they're right 87.6% of the time from raw pixels, 94.5% from layer 1, 97.3% from layer 2 and 97.9% from the hidden layer.",
  };
}

// ----- Try to break it -----

function topSupport() {
  const { toScore, conv2ToHidden } = app.data.circ;
  const hidden = topBy(toScore, 6, (j) => toScore[j] > 0 && hasVis(j));
  const chScore = Array.from({ length: 16 }, (_, c) => hidden.reduce((s, j) => s + Math.max(0, conv2ToHidden[c][j]), 0));
  const conv2 = topBy(chScore, 4);
  return { hidden, conv2, chScore };
}

function break1() {
  const { toScore } = app.data.circ;
  const { hidden } = topSupport();
  const W = 640, H = 300;
  const box = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "flow-svg", role: "img", "aria-label": "The units its decision rests on" });
  const total = toScore.reduce((s, v) => s + Math.max(0, v), 0) || 1;
  const max = toScore[hidden[0]] || 1;
  const pos = (k) => ({ x: 90, y: 30 + k * 44 });
  hidden.forEach((j, k) => {
    const p = pos(k);
    const line = flowLine(box, p.x + 24, p.y + 4, 500, 150, { width: 1.5 + (toScore[j] / max) * 9, kind: "hot", delay: k * 0.15, speed: 1.6 - (toScore[j] / max) * 0.8 });
    particles(box, line.getAttribute("d"), Math.max(1, Math.round((toScore[j] / max) * 4)), 1.8 - (toScore[j] / max) * 0.7);
    image(box, featURL("hidden", j), p.x - 20, p.y - 16, 40, k === 0 ? "star" : "", k * 0.08);
    svg("text", { x: p.x - 34, y: p.y + 9, "text-anchor": "end", class: "svg-cap" }, box).append(`unit ${j + 1}`);
    svg("text", { x: p.x + 40, y: p.y - 6, class: "svg-cap" }, box).append(`${Math.round((toScore[j] / total) * 100)}%`);
  });
  svg("circle", { cx: 520, cy: 150, r: 34, class: "out-node hot svg-pop" }, box);
  svg("text", { x: 520, y: 158, "text-anchor": "middle", class: "out-q", style: "font-size:24px" }, box).append("?");
  svg("text", { x: 520, y: 205, "text-anchor": "middle", class: "svg-cap" }, box).append("its answer");
  const share = Math.round((hidden.reduce((s, j) => s + toScore[j], 0) / total) * 100);
  return {
    title: "Its answer rests on a handful of units",
    say: `Of the 64 hidden units, these <b>six</b> push hardest towards its answer. Together they carry <b>${share}%</b> of the total push; the thickest line, unit ${hidden[0] + 1}, alone carries ${Math.round((toScore[hidden[0]] / total) * 100)}%. Next, you get to switch them off.`,
    visual: box,
    how: "Each unit's push is its activation × (its output weight for the answer digit). Percentages are shares of the total positive push from all 64 units. The images are feature visualisations of each unit.",
  };
}

function break2() {
  const { predicted } = app.data;
  const { hidden, conv2, chScore } = topSupport();
  const wrap = el("div", { class: "break" });
  const left = el("div");
  const right = el("div", { class: "conf" });
  const maxH = app.data.circ.toScore[hidden[0]] || 1;
  const maxC = chScore[conv2[0]] || 1;

  const make = (layer, i, img, weight, label) => {
    const b = el("button", { class: "switch", type: "button", "aria-pressed": "false" });
    b.innerHTML = `<canvas width="28" height="28"></canvas><span class="who">${label}</span><span class="weight"><span style="width:${Math.round(weight * 100)}%"></span></span><span class="toggle" aria-hidden="true"></span>`;
    b.querySelector("canvas").getContext("2d").drawImage(img, 0, 0, 28, 28);
    const sync = () => {
      const off = app.masks[layer][i] === 0;
      b.classList.toggle("off", off);
      b.setAttribute("aria-pressed", String(off));
      b.setAttribute("aria-label", `${label}: ${off ? "off" : "on"}`);
    };
    b.addEventListener("click", () => {
      app.masks[layer][i] = app.masks[layer][i] === 0 ? 1 : 0;
      sync();
      update();
    });
    sync();
    return b;
  };

  left.append(el("p", { class: "group-label" }, "Hidden units it leans on"));
  const g1 = el("div", { class: "switches" });
  hidden.forEach((j) => g1.append(make("hidden", j, grayCanvas(fromB64(app.insights.featvis.hidden[j]).map((v) => 255 - v)), app.data.circ.toScore[j] / maxH, `unit ${j + 1}`)));
  left.append(g1, el("p", { class: "group-label" }, "Layer-2 filters feeding them"));
  const g2 = el("div", { class: "switches" });
  conv2.forEach((c) => g2.append(make("conv2", c, grayCanvas(channelPixels(app.data.r.conv2, 14, 16, c), 14), chScore[c] / maxC, `filter ${c + 1}`)));
  left.append(g2);

  const R = 70, C = 2 * Math.PI * R;
  right.innerHTML = `<svg viewBox="0 0 180 180"><circle class="ring-track" cx="90" cy="90" r="${R}"/><circle class="ring-fill" cx="90" cy="90" r="${R}" stroke-dasharray="${C}" stroke-dashoffset="0" style="stroke:#ffd23f;stroke-width:14"/><text class="conf-label" x="90" y="88" text-anchor="middle">100%</text><text class="conf-sub" x="90" y="108" text-anchor="middle">sure of its</text><text class="conf-sub" x="90" y="123" text-anchor="middle">first answer</text></svg><span class="status" id="brkStatus">${icon("check", 15)} Same answer as before</span>`;
  wrap.append(left, right);

  const update = () => {
    const probs = analyse(app.model, app.data.px, predicted, null, { masks: app.masks }).probs;
    const p = probs[predicted];
    right.querySelector(".ring-fill").style.strokeDashoffset = `${C * (1 - p)}`;
    right.querySelector(".conf-label").textContent = `${Math.round(p * 100)}%`;
    const changed = argmax(probs) !== predicted;
    const st = right.querySelector("#brkStatus");
    st.innerHTML = changed ? `${icon("zap", 15)} It changed its mind!` : `${icon("check", 15)} Same answer as before`;
    st.classList.toggle("changed", changed);
    app.broken = changed;
  };
  later(update, 50);
  return {
    title: "Now switch them off",
    say: `Click a card to switch that unit off. The ring shows how sure it still is of its <b>original</b> answer. See if you can make it <b>change its mind</b>. If it holds, try the layer-2 filters too.`,
    visual: wrap,
    how: "Switching a unit off multiplies its output by zero everywhere (an ablation). The network is re-run in your browser on every click. The bars under each card show how much that unit pushed towards the original answer.",
    next: "Reveal its answer",
  };
}

// ---------- 4. reveal ----------

function revealScreen() {
  const j = JOURNEYS[app.journey];
  const ablated = Object.values(app.masks).some((m) => m.some((v) => v === 0));
  const final = ablated ? analyse(app.model, app.data.px, app.data.predicted, null, { masks: app.masks }) : app.data.r;
  const said = argmax(final.probs);
  const s = el("section");
  s.innerHTML = `<div class="drum"><span></span><span></span><span></span></div>`;
  show(s, { dots: { total: j.steps.length + 1, at: j.steps.length } });

  setTimeout(() => {
    const a = el("div", { class: "answer" });
    const big = el("div", { class: "big-digit", "aria-live": "polite" }, String(said));
    const info = el("div");
    const bars = [...final.probs]
      .map((p, d) => [d, p])
      .sort((x, y) => y[1] - x[1])
      .slice(0, 4)
      .map(([d, p], k) => `<div class="mini-bar ${k === 0 ? "top" : ""}"><b>${d}</b><span class="t"><span class="f" data-w="${(p * 100).toFixed(1)}"></span></span><span>${pct(p)}</span></div>`)
      .join("");
    const heat = el("canvas", { width: 120, height: 120 });
    const region = describeRegion(app.data.r.smoothgrad, app.data.px);
    let extra = "";
    if (ablated) {
      const orig = app.data.predicted;
      extra = `<div class="compare"><div>Untouched: ${orig} (${pct(app.data.r.probs[orig])})</div><div>With your switches: ${said} (${pct(final.probs[said])})</div></div>
        <p class="say">${said !== orig ? "<b>You changed its mind.</b> Those few units were carrying the decision." : `It held on to <b>${orig}</b>. Its knowledge is spread across many units, so knocking out a few isn't enough.`}</p>`;
    }
    info.innerHTML = `<h2 class="step-title">It's ${said === 8 ? "an" : "a"} ${said}, ${Math.round(final.probs[said] * 100)}% sure.</h2>
      <div class="mini-bars" style="margin-top:1rem">${bars}</div>${extra}
      <div class="why"><div></div><p><b>Why${ablated && said !== app.data.predicted ? ` its original answer, ${app.data.predicted}` : ""}?</b> It leaned most on <b>${region.where}</b> of your drawing (pink). Blue marks places where more ink would have argued against ${app.data.predicted}.</p></div>`;
    info.querySelector(".why div").append(heat);
    drawRGBA(heat, heatPixels(app.data.r.smoothgrad, app.data.px));
    a.append(big, info);

    const others = Object.keys(JOURNEYS).filter((k) => k !== app.journey);
    const next = el("div", { class: "next-ways" });
    others.forEach((k) => {
      const b = el("button", { class: "btn", type: "button" }, `${icon(JOURNEYS[k].icon)} ${JOURNEYS[k].title}`);
      b.addEventListener("click", () => startJourney(k));
      next.append(b);
    });
    const again = el("button", { class: "btn btn-alt", type: "button" }, `${icon("pencil")} Draw another`);
    again.addEventListener("click", () => {
      pad.clear();
      drawScreen();
    });
    next.append(again);

    s.replaceChildren(a, next);
    later(() => s.querySelectorAll(".mini-bar .f").forEach((f) => (f.style.width = `${f.dataset.w}%`)), 400);
  }, 1300);
}

// ---------- go ----------

document.querySelector(".jbrand").insertAdjacentHTML("afterbegin", icon("scan-eye", 18) + " ");
$("redraw").innerHTML = `${icon("pen-line", 13)} Redraw`;

drawScreen();
