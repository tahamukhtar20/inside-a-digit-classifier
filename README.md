# Inside a Digit Classifier

Draw a digit, then take a guided tour through the small convolutional network reading it. Pick one of three ways in, step through it one idea at a time, guess what it will say, and only then see its answer:

- **Follow the signal.** Your pixels become 784 numbers, then strokes (layer 1), parts (layer 2), 64 features, and finally votes for ten digits, with animated flows showing the strongest connections on your drawing.
- **Read its mind.** Probes ask what each layer knows before it answers: is there a closed loop, which way does it lean, how thick is the pen, which digit is it.
- **Try to break it.** See the handful of units its decision rests on, switch them off, and find out whether you changed its mind.

For free exploration there's also a **full lab** (`lab.html`) with everything on one page:

- **Where it looks.** SmoothGrad, saliency, gradient × input and Grad-CAM, for any digit or for the difference between two ("why 3 rather than 8?").
- **Inside the network.** Every filter in both convolution layers and all 64 hidden units, updating as you draw. Click any of them to see the input that excites it most and the MNIST digits that do.
- **Ablation.** Switch any filter or unit off and watch the prediction change.
- **The circuit behind the answer.** The strongest paths from first-layer filters through second-layer filters and hidden units to the score, as activation × weight contributions.
- **What each layer knows.** Linear probes read the digit and three things the network was never trained on (a closed loop, slant, stroke thickness) out of each layer, on the test set and on your drawing.

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

**Circuits.** Hidden unit to score: activation × (output weight for the digit − output weight for the comparison digit). Layer-2 filter to hidden unit: pooled activations × first dense weights, summed over positions. Layer-1 to layer-2 filter: gradient of the layer-2 filter's total activation × layer-1 activation, summed over positions. The graph keeps the top 5 hidden units, their top 2 inputs each, and so on.

**Probes.** Logistic regression (digit, closed loop) and ridge regression (slant, stroke thickness) on frozen activations of 12,000 MNIST training digits, scored on 3,000 test digits. Digit accuracy climbs from 87.6% on raw pixels to 97.9% at the hidden layer. Stroke thickness goes the other way: R² 0.93 from layer 1 but 0.43 from the hidden layer, so the network throws away style it doesn't need.

**Feature visualisation.** Gradient ascent on a blank image with jitter and an L2 penalty, 300 steps per unit.

**Preprocessing.** The drawing is cropped to its ink, scaled to fit 20 × 20 and centred by centre of mass on a 28 × 28 grid, matching how MNIST was prepared.

## Project layout

```
training/train.py           Trains the CNN (PyTorch) and exports weights to web/model/weights.json
training/export_samples.py  Exports 100 MNIST test digits for the "Random MNIST digit" button
training/analysis.py        Feature visualisation, top dataset examples and linear probes -> web/model/insights.json
web/index.html, journey.js  The guided tour (journey.css on top of style.css)
web/lab.html, lab.js        The full lab
web/common.js               Drawing pad, preprocessing, concept measures, small canvas helpers
web/model.js                The network rebuilt from TensorFlow.js ops: attributions, ablation masks, circuits, probe readouts
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
