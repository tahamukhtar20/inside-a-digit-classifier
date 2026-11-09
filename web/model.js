// The CNN from training/train.py, rebuilt with TensorFlow.js ops so gradients are available
// for attributions. `tf` is the global from the TensorFlow.js script tag.

/* global tf */

export async function loadModel(url) {
  const raw = await (await fetch(url)).json();
  const W = {};
  for (const layer of ["conv1", "conv2", "fc1", "fc2"]) {
    W[layer] = {
      kernel: tf.tensor(raw[layer].kernel.data, raw[layer].kernel.shape),
      bias: tf.tensor(raw[layer].bias.data, raw[layer].bias.shape),
    };
  }
  return { W, meta: raw.meta };
}

const conv = (x, l) => tf.relu(tf.add(tf.conv2d(x, l.kernel, 1, "same"), l.bias));
const pool = (x) => tf.maxPool(x, 2, 2, "valid");

/** Input [1,28,28,1] -> both convolution outputs (after ReLU, before pooling). */
function trunk(W, x) {
  const a1 = conv(x, W.conv1);
  const a2 = conv(pool(a1), W.conv2);
  return { a1, a2 };
}

/** Second convolution's output -> hidden units and logits. */
function head(W, a2) {
  const flat = tf.reshape(pool(a2), [-1, 7 * 7 * 16]);
  const hidden = tf.relu(tf.add(tf.matMul(flat, W.fc1.kernel), W.fc1.bias));
  const logits = tf.add(tf.matMul(hidden, W.fc2.kernel), W.fc2.bias);
  return { hidden, logits };
}

/**
 * Runs the model on a 28x28 image (Float32Array, 0 = background, 1 = ink) and explains the
 * score of `target` (minus the score of `versus`, if given).
 */
export function analyse(model, pixels, target, versus = null, { smooth = false } = {}) {
  const { W } = model;
  // The score is summed over the batch, so each copy's gradient is independent of the others.
  return tf.tidy(() => {
    const x = tf.tensor(pixels, [1, 28, 28, 1]);
    const { a1, a2 } = trunk(W, x);
    const { hidden, logits } = head(W, a2);
    const probs = tf.softmax(logits);

    const weights = new Float32Array(10);
    weights[target] = 1;
    if (versus !== null && versus !== target) weights[versus] = -1;
    const w = tf.tensor(weights, [1, 10]); // broadcasts over any batch size
    const score = (l) => tf.sum(tf.mul(l, w));

    // Saliency: how the score changes with each input pixel.
    const gradOf = tf.grad((inp) => score(head(W, trunk(W, inp).a2).logits));
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
    const gradA2 = tf.grad((a) => score(head(W, a).logits))(a2);
    const channelWeights = tf.mean(gradA2, [1, 2], true);
    let cam = tf.relu(tf.sum(tf.mul(a2, channelWeights), 3, true));
    cam = tf.image.resizeBilinear(cam, [28, 28]);

    return {
      probs: probs.dataSync().slice(),
      logits: logits.dataSync().slice(),
      saliency: gradInput.dataSync().slice(),
      smoothgrad: smoothGrad ? smoothGrad.dataSync().slice() : null,
      gradTimesInput: tf.mul(gradInput, x).dataSync().slice(),
      gradcam: cam.dataSync().slice(),
      conv1: a1.dataSync().slice(), // 28*28*8, channels last
      conv2: a2.dataSync().slice(), // 14*14*16, channels last
      hidden: hidden.dataSync().slice(), // 64
    };
  });
}
