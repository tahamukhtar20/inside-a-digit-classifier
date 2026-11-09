// The CNN from training/train.py, rebuilt with TensorFlow.js ops so gradients are available
// for attributions, and so units can be switched off (ablation). `tf` is the global from the
// TensorFlow.js script tag.

/* global tf */

export const SIZES = { conv1: 8, conv2: 16, hidden: 64 };

export async function loadModel(url) {
  const raw = await (await fetch(url)).json();
  const W = {};
  for (const layer of ["conv1", "conv2", "fc1", "fc2"]) {
    W[layer] = {
      kernel: tf.tensor(raw[layer].kernel.data, raw[layer].kernel.shape),
      bias: tf.tensor(raw[layer].bias.data, raw[layer].bias.shape),
    };
  }
  return { W, meta: raw.meta, fc1: raw.fc1.kernel.data, fc2: raw.fc2.kernel.data };
}

/** All-ones masks, i.e. nothing switched off. */
export const noMasks = () => Object.fromEntries(Object.entries(SIZES).map(([k, n]) => [k, new Float32Array(n).fill(1)]));

const conv = (x, l) => tf.relu(tf.add(tf.conv2d(x, l.kernel, 1, "same"), l.bias));
const pool = (x) => tf.maxPool(x, 2, 2, "valid");

/** Input -> both convolution outputs (after ReLU and ablation, before pooling). */
function trunk(W, M, x) {
  const a1 = tf.mul(conv(x, W.conv1), M.conv1);
  const p1 = pool(a1);
  const a2 = tf.mul(conv(p1, W.conv2), M.conv2);
  return { a1, p1, a2 };
}

/** Second convolution's output -> hidden units and logits. */
function head(W, M, a2) {
  const p2 = pool(a2);
  const flat = tf.reshape(p2, [-1, 7 * 7 * 16]);
  const hidden = tf.mul(tf.relu(tf.add(tf.matMul(flat, W.fc1.kernel), W.fc1.bias)), M.hidden);
  const logits = tf.add(tf.matMul(hidden, W.fc2.kernel), W.fc2.bias);
  return { p2: flat, hidden, logits };
}

const scoreWeights = (target, versus) => {
  const w = new Float32Array(10);
  w[target] = 1;
  if (versus !== null && versus !== target) w[versus] = -1;
  return w;
};

/**
 * Runs the model on a 28x28 image (Float32Array, 0 = background, 1 = ink) and explains the
 * score of `target` (minus the score of `versus`, if given), with `masks` switching units off.
 */
export function analyse(model, pixels, target, versus = null, { smooth = false, masks = noMasks() } = {}) {
  const { W } = model;
  return tf.tidy(() => {
    const M = { conv1: tf.tensor(masks.conv1), conv2: tf.tensor(masks.conv2), hidden: tf.tensor(masks.hidden) };
    const x = tf.tensor(pixels, [1, 28, 28, 1]);
    const { a1, p1, a2 } = trunk(W, M, x);
    const { p2, hidden, logits } = head(W, M, a2);
    const w = tf.tensor(scoreWeights(target, versus), [1, 10]); // broadcasts over any batch size
    const score = (l) => tf.sum(tf.mul(l, w));

    // Saliency: how the score changes with each input pixel.
    const gradOf = tf.grad((inp) => score(head(W, M, trunk(W, M, inp).a2).logits));
    const gradInput = gradOf(x);
    // SmoothGrad: average the gradient over noisy copies of the input to cancel out the noise.
    // It costs 24 extra passes, so the page asks for it only when drawing pauses.
    let smoothGrad = null;
    if (smooth) {
      const copies = 24;
      const noise = tf.randomNormal([copies, 28, 28, 1], 0, 0.15, "float32", 7); // fixed seed: stable maps
      smoothGrad = tf.mean(gradOf(tf.add(tf.tile(x, [copies, 1, 1, 1]), noise)), 0);
    }
    // Grad-CAM: weight each second-layer feature map by its average gradient.
    const gradA2 = tf.grad((a) => score(head(W, M, a).logits))(a2);
    const cam = tf.image.resizeBilinear(tf.relu(tf.sum(tf.mul(a2, tf.mean(gradA2, [1, 2], true)), 3, true)), [28, 28]);

    return {
      probs: tf.softmax(logits).dataSync().slice(),
      logits: logits.dataSync().slice(),
      saliency: gradInput.dataSync().slice(),
      smoothgrad: smoothGrad ? smoothGrad.dataSync().slice() : null,
      gradTimesInput: tf.mul(gradInput, x).dataSync().slice(),
      gradcam: cam.dataSync().slice(),
      conv1: a1.dataSync().slice(), // 28*28*8, channels last
      conv2: a2.dataSync().slice(), // 14*14*16, channels last
      hidden: hidden.dataSync().slice(), // 64
      // pooled features, flattened (h, w, c), for the probes
      features: { input: pixels, conv1: tf.reshape(p1, [-1]).dataSync().slice(), conv2: p2.dataSync().slice(), hidden: hidden.dataSync().slice() },
    };
  });
}

/**
 * The circuit behind the current score, as linear contributions (activation x weight):
 *   hidden unit j        -> score:          h_j * (w2[j, target] - w2[j, versus])
 *   conv-2 channel c     -> hidden unit j:  sum over positions of pooled activation x fc1 weight
 *   conv-1 channel i     -> conv-2 channel c: gradient x activation, summed over positions
 */
export function circuit(model, pixels, target, versus = null, masks = noMasks()) {
  const { W } = model;
  return tf.tidy(() => {
    const M = { conv1: tf.tensor(masks.conv1), conv2: tf.tensor(masks.conv2), hidden: tf.tensor(masks.hidden) };
    const x = tf.tensor(pixels, [1, 28, 28, 1]);
    const { p1, a2 } = trunk(W, M, x);
    const { p2, hidden } = head(W, M, a2);

    const sw = tf.tensor(scoreWeights(target, versus), [10, 1]);
    const toScore = tf.mul(tf.reshape(hidden, [64]), tf.reshape(tf.matMul(W.fc2.kernel, sw), [64])); // [64]

    // [784] x [784, 64] -> [49, 16, 64] -> sum positions -> [16, 64]
    const conv2ToHidden = tf.sum(tf.reshape(tf.mul(tf.reshape(p2, [784, 1]), W.fc1.kernel), [49, 16, 64]), 0);

    // For each conv-2 channel, grad of its total activation with respect to p1, times p1.
    const conv1ToConv2 = [];
    for (let c = 0; c < 16; c++) {
      const g = tf.grad((p) => tf.sum(tf.mul(conv(p, W.conv2), M.conv2).slice([0, 0, 0, c], [1, 14, 14, 1])))(p1);
      conv1ToConv2.push(tf.sum(tf.mul(g, p1), [0, 1, 2]).dataSync().slice()); // [8]
    }
    return {
      toScore: toScore.dataSync().slice(),
      conv2ToHidden: conv2ToHidden.arraySync(), // [16][64]
      conv1ToConv2, // [16][8]
    };
  });
}

/** Linear probe readouts for one layer's features. */
export function readProbes(probeWeights, features) {
  const out = {};
  for (const [concept, { shape, w, b }] of Object.entries(probeWeights)) {
    const [n, k] = shape;
    const z = b.slice();
    for (let f = 0; f < n; f++) {
      const v = features[f];
      if (v === 0) continue;
      for (let o = 0; o < k; o++) z[o] += v * w[f * k + o];
    }
    out[concept] = z;
  }
  return out;
}
