# Simple MTG Keypoint Regression

**Synthetic Dataset + Heatmap-Based Corner Detection**

Train a lightweight heatmap regression model to localize Magic: The Gathering card corners in real-time using only artwork and synthetic transformations.

[📘 View Training Notebook on Kaggle](https://www.kaggle.com/code/jaketurner616/mtg-keypoint-regression-heatmap)

<p align="center">
  <img src="/docs/epoch_03.jpg" width="400"/>
  <img src="/docs/epoch_30.jpg" width="400"/>
</p>
<p align="center">
  <b>Left:</b> Early prediction (Epoch 3) — poor alignment<br>
  <b>Right:</b> Late prediction (Epoch 30) — close match to ground truth<br><br>
  🟢 = Ground truth corner | 🔴 = Model prediction via soft-argmax
</p>

---

## 🧭 Project Workflow

This repo covers the entire pipeline:

### 🔧 1. **Synthetic Dataset Generation**

* Uses real Scryfall card art with equal sampling of black, white, and borderless cards
* Applies controlled augmentations (rotation, perspective, scaling)
* Randomly composites onto photo/textured backgrounds
* Records true 4-corner annotations in pixel space
* Output format:

```json
{
  "filename": "00042_Llanowar_Elves.jpg",
  "card_name": "Llanowar Elves",
  "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

> ✅ All assets are locally generated, reproducible, and extensible

---

### 🧠 2. **Model Training (Heatmap + SoftArgmax)**

* Model type: Convolutional neural network with dual outputs

  * `heatmaps` – 4-channel spatial probability maps
  * `coords` – XY corner coordinates via differentiable soft-argmax
* Backbone: **MobileNetV2** (ImageNet pretrained)
* Loss: MSE over heatmaps (softargmax is used for decoding, not supervised directly)
* Metrics: Pixel distance between predicted and ground-truth corners
* Visualized per-epoch outputs are saved to disk

> 💡 Optimized for TensorFlow\.js WASM-compatible deployment

---

### 🌐 3. **Web Deployment (TF.js Export)**

* Keras model saved as `.keras` format with custom `SoftArgmax` layer
* Converted using `tensorflowjs_converter` to `tfjs_graph_model`
* Outputs ZIP archive: `web_model.zip` for frontend use
* Compatible with browser-based MTG scanners, ROI extractors, or mobile inference

---

## 🗂️ Repository Layout

```
.
├── dataset.py                             # Synthetic image + annotation generator
├── backgrounds/                           # Natural backgrounds used for compositing
├── dataset/                               # Output image directory
├── dataset/annotations.json               # Ground truth corner positions
├── unique-artwork-*.json                  # MTG card art scraped from Scryfall
├── mtg-keypoint-regression-heatmap.ipynb  # Training notebook (Kaggle-ready)
```

---

## ⚙️ Technical Specs

### 🧪 Dataset Summary

| Parameter     | Value                                             |
| ------------- | ------------------------------------------------- |
| Image size    | 1024×1024                                         |
| Format        | `.jpg`                                            |
| Labels        | 4 corner points in pixel space                    |
| Dataset size  | \~5,000 samples (expandable)                      |
| Augmentations | Affine, perspective, rotation, background overlay |

---

### 🚀 Training Hardware

| Property      | Value                  |
| ------------- | ---------------------- |
| GPU           | Tesla P100 (16GB VRAM) |
| TF XLA        | Enabled                |
| cuDNN Version | 9.3.0                  |
| Training Time | \~30 mins (35 epochs)  |

---

## License

All dataset generation code is released under the [GNU GPL v3.0](LICENSE) LICENSE. Dependencies may have different Licensing.