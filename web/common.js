// The drawing pad, MNIST-style preprocessing, small canvas helpers and the concept measures
// used by the probes.

export const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
export const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
export const pct = (p) => `${(p * 100).toFixed(p > 0.995 ? 0 : 1)}%`;
export const fromB64 = (s) => Uint8ClampedArray.from(atob(s), (c) => c.charCodeAt(0));

/** Indices of the k largest values by magnitude (non-zero only). */
export const topBy = (values, k, filter = () => true) =>
  [...values]
    .map((v, i) => [i, v])
    .filter(([i, v]) => v !== 0 && filter(i))
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, k)
    .map(([i]) => i);

// ---------- drawing pad ----------

export class Pad {
  constructor(canvas, onChange) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { willReadFrequently: true });
    this.onChange = onChange;
    this.inked = false;
    this.reset();
    let drawing = false;
    let last = null;
    const pos = (e) => {
      const r = canvas.getBoundingClientRect();
      return [((e.clientX - r.left) / r.width) * canvas.width, ((e.clientY - r.top) / r.height) * canvas.height];
    };
    canvas.addEventListener("pointerdown", (e) => {
      drawing = true;
      canvas.setPointerCapture(e.pointerId);
      last = pos(e);
      this.stroke(last, last);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!drawing) return;
      const p = pos(e);
      this.stroke(last, p);
      last = p;
    });
    const stop = () => (drawing = false);
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
  }

  reset() {
    this.ctx.fillStyle = "#fff";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.inked = false;
  }

  clear() {
    this.reset();
    this.onChange?.();
  }

  stroke(a, b) {
    const { ctx } = this;
    ctx.strokeStyle = "#111";
    ctx.lineWidth = this.canvas.width / 14;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    this.inked = true;
    this.onChange?.();
  }

  /** Paints a 28x28 MNIST digit (0-255, white on black) onto the pad as ink on paper. */
  loadDigit(pixels) {
    this.reset();
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.drawImage(grayCanvas(Uint8ClampedArray.from(pixels, (v) => 255 - v)), 0, 0, this.canvas.width, this.canvas.height);
    this.inked = true;
    this.onChange?.();
  }

  /** What the model sees: crop to the ink, fit into 20x20, centre by centre of mass in 28x28. */
  pixels() {
    return preprocess(this.canvas);
  }
}

export function preprocess(canvas) {
  const { width: w, height: h } = canvas;
  const data = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
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
  sctx.drawImage(canvas, minX, minY, bw, bh, 0, 0, sw, sh);
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

// ---------- small canvases ----------

/** A size x size canvas from grey values (0 = black, 255 = white). */
export function grayCanvas(values, size = 28) {
  const c = Object.assign(document.createElement("canvas"), { width: size, height: size });
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) img.data.set([values[i], values[i], values[i], 255], i * 4);
  ctx.putImageData(img, 0, 0);
  return c;
}

/** One channel of a channels-last activation as ink on paper (strong = dark). */
export function channelPixels(data, size, channels, ch) {
  let max = 1e-9;
  for (let i = 0; i < size * size; i++) max = Math.max(max, data[i * channels + ch]);
  return Uint8ClampedArray.from({ length: size * size }, (_, i) => 255 - (data[i * channels + ch] / max) * 238);
}

/** The model's 28x28 input (1 = ink) as ink on paper. */
export const inputPixels = (px) => Uint8ClampedArray.from(px, (v) => 255 - v * 238);

/** An attribution map over the faint digit: pink pushes towards the answer, blue away. */
export function heatPixels(values, digit, positiveOnly = false) {
  let max = 1e-9;
  for (const v of values) max = Math.max(max, Math.abs(v));
  const out = new Uint8ClampedArray(784 * 4);
  for (let i = 0; i < 784; i++) {
    const base = 255 - digit[i] * 90;
    const a = Math.min(1, Math.abs(values[i]) / max) ** 0.75;
    const color = values[i] >= 0 ? (positiveOnly ? [255, 160, 60] : [255, 80, 140]) : [40, 120, 255];
    out.set([0, 1, 2].map((k) => base * (1 - a) + color[k] * a).concat(255), i * 4);
  }
  return out;
}

export function drawRGBA(canvas, rgba, size = 28) {
  const tmp = Object.assign(document.createElement("canvas"), { width: size, height: size });
  tmp.getContext("2d").putImageData(new ImageData(rgba, size, size), 0, 0);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

// ---------- concepts (same definitions as training/analysis.py) ----------

export function hasLoop(px) {
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

/** Horizontal shift per row going up: positive leans right. */
export function slant(px) {
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

/** Average stroke width in pixels: 2 x area / perimeter of the inked region. */
export function thickness(px) {
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

/** Where on the drawing an attribution map puts most of its positive weight, in words. */
export function describeRegion(values, digit) {
  const rows = ["top", "middle", "bottom"];
  const cols = ["left", "centre", "right"];
  const mass = Array(9).fill(0);
  let total = 0;
  for (let i = 0; i < 784; i++) {
    const v = Math.max(0, values[i]) * (0.3 + digit[i]);
    const y = Math.min(2, Math.floor((Math.floor(i / 28) - 2) / 8));
    const x = Math.min(2, Math.floor(((i % 28) - 2) / 8));
    mass[Math.max(0, y) * 3 + Math.max(0, x)] += v;
    total += v;
  }
  const best = argmax(mass);
  const share = mass[best] / (total || 1);
  const r = rows[Math.floor(best / 3)];
  const c = cols[best % 3];
  const where = c === "centre" ? (r === "middle" ? "the centre" : `the ${r}`) : r === "middle" ? `the ${c} side` : `the ${r} ${c}`;
  return { where, share };
}
