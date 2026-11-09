# Inside a Digit Classifier

Draw a digit and watch a small convolutional network read it, live, in your browser. As you draw, the page shows the network's prediction, where it looks (SmoothGrad, saliency, gradient × input and Grad-CAM), what every filter in both convolution layers responds to, and the activations of the 64 hidden units. Pick any digit, or compare two ("why 3 rather than 8?"), and the explanation follows.

Everything runs client-side with [TensorFlow.js](https://www.tensorflow.org/js). There's no server, and nothing you draw leaves the browser.

## How it works

**Model.** Two 3 × 3 convolution layers (8 and 16 filters, each followed by ReLU and 2 × 2 max-pooling), a 64-unit hidden layer and 10 outputs, about 52,000 weights. It's trained on MNIST with random rotations (±15°), scaling (0.85 to 1.15) and shifts (±3 px), because drawings on a canvas are messier than the dataset. It reaches **98.9%** test accuracy.

**Attributions.** All of them explain the score of the chosen digit, or the difference between two digits' scores when a comparison is selected.

| Method | What it shows |
|---|---|
| Saliency | Gradient of the score with respect to each input pixel |
| SmoothGrad | Saliency averaged over 24 noisy copies of the input, which removes most of the noise; computed once drawing pauses |
| Gradient × input | Saliency multiplied by the drawing, so only drawn pixels get credit or blame |
| Grad-CAM | Second-layer feature maps weighted by their mean gradient, then upsampled to 28 × 28 |

**Preprocessing.** The drawing is cropped to its ink, scaled to fit 20 × 20 and centred by centre of mass on a 28 × 28 grid, matching how MNIST was prepared.

## Project layout

```
training/train.py           Trains the CNN (PyTorch) and exports weights to web/model/weights.json
training/export_samples.py  Exports 100 MNIST test digits for the "Random MNIST digit" button
web/index.html, style.css   The page
web/model.js                The network rebuilt from TensorFlow.js ops, with gradients for attribution
web/app.js                  Drawing, preprocessing and rendering
```

## Running locally

```bash
python -m http.server 8000 -d web      # then open http://localhost:8000
```

To retrain (CPU, a few minutes; downloads MNIST on first run):

```bash
pip install -r training/requirements.txt
python training/train.py
cd training && python export_samples.py
```

## Deploying

The `web/` folder is a static site. It's set up for Cloudflare Workers static assets (`wrangler.jsonc`): run `npx wrangler deploy`.
