# Inside a Digit Classifier

Live at [digits.tahamukhtar.com](https://digits.tahamukhtar.com).

Draw a digit, then take a guided tour through the small convolutional network reading it. Pick one of three ways in, step through it one idea at a time, and only see its answer at the end:

- Follow the signal. Your pixels become 784 numbers, then strokes (layer 1), parts (layer 2), 64 features, and finally votes for ten digits, with animated flows showing the strongest connections on your drawing.
- Read its mind. Probes ask what each layer knows before it answers: is there a closed loop, which way does it lean, how thick is the pen, which digit is it.
- Try to break it. See the handful of units its decision rests on, switch them off, and find out whether you changed its mind.

The answer screen ends with a SmoothGrad map of where it looked.

Everything runs client-side with [TensorFlow.js](https://www.tensorflow.org/js). There's no server, and nothing you draw leaves the browser.

## How it works

### Model

Two 3 × 3 convolution layers (8 and 16 filters, each followed by ReLU and 2 × 2 max-pooling), a 64-unit hidden layer and 10 outputs, about 52,000 weights. It's trained on MNIST with random rotations (±15°), scaling (0.85 to 1.15) and shifts (±3 px), because drawings on a canvas are messier than the dataset. It reaches 98.9% test accuracy.

### Where it looked

SmoothGrad: the gradient of the answer's score with respect to each pixel, averaged over 24 noisy copies of the drawing so the map isn't speckled.

### Circuits

Hidden unit to score: activation × output weight for the answer digit. Layer-2 filter to hidden unit: pooled activations × first dense weights, summed over positions. Layer-1 to layer-2 filter: gradient of the layer-2 filter's total activation × layer-1 activation, summed over positions. The tour draws the strongest of these links.

### Probes

Logistic regression (digit, closed loop) and ridge regression (slant, stroke thickness) on frozen activations of 12,000 MNIST training digits, scored on 3,000 test digits. Digit accuracy climbs from 87.6% on raw pixels to 97.9% at the hidden layer. Stroke thickness goes the other way: R² 0.93 from layer 1 but 0.43 from the hidden layer, so the network throws away style it doesn't need.

### Feature visualisation

Gradient ascent on a blank image with jitter and an L2 penalty, 300 steps per unit.

### Preprocessing

The drawing is cropped to its ink, scaled to fit 20 × 20 and centred by centre of mass on a 28 × 28 grid, matching how MNIST was prepared.

## Project layout

```
training/train.py           Trains the CNN (PyTorch) and exports weights to web/model/weights.json
training/export_samples.py  Exports 100 MNIST test digits for the "Use an example" button
training/analysis.py        Feature visualisation and linear probes -> web/model/insights.json
web/index.html, journey.js  The guided tour (journey.css on top of style.css)
web/common.js               Drawing pad, preprocessing, concept measures, small canvas helpers
web/model.js                The network rebuilt from TensorFlow.js ops: attributions, ablation masks, circuits, probe readouts
web/icons.js                The few Lucide icons the page uses, inlined
```

## Running locally

```bash
python -m http.server 8000 -d web      # then open http://localhost:8000
```

To retrain (CPU, a few minutes; downloads MNIST on first run):

```bash
pip install -r training/requirements.txt
python training/train.py
cd training && python export_samples.py && python analysis.py
```

## Deploying

The `web/` folder is a static site. It's set up for Cloudflare Workers static assets (`wrangler.jsonc`): run `npx wrangler deploy`.
