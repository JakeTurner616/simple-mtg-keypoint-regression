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

## Project Overview

This repository covers the full pipeline for:

1. Generating synthetic training data from real MTG card art
2. Training a keypoint regression model using heatmaps + soft-argmax
3. Exporting a compact TensorFlow\.js model for web/mobile inference

---

## 1. Synthetic Dataset Generation

* Sources card images from the Scryfall API (black, white, and borderless frame parity enforced)
* Applies controlled perspective, affine, and rotation transforms
* Cards are composited over randomized photographic or textured backgrounds
* Annotations include the 4 corner coordinates in 1024×1024 pixel space

**Output example:**

```json
{
  "filename": "00042_Llanowar_Elves.jpg",
  "card_name": "Llanowar Elves",
  "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

All assets are procedurally generated and can be reproduced or extended with additional transformation rules or art sources.

---

## 2. Model Training

### Workflow

* Backbone: `MobileNetV2` (pretrained, lightweight)
* Output: 4-channel heatmap (56×56) + decoded corner coordinates
* Decoder: Differentiable soft-argmax (implemented as a Keras layer)
* Loss: MSE over heatmaps only (coordinate output is derived, not directly supervised)
* Training: \~35 epochs on Kaggle P100; early stopping and simplified callbacks

### Differences from Previous Workflow for mobile target:

| Aspect             | Old Workflow                               | New Workflow                                 |
| ------------------ | ------------------------------------------ | -------------------------------------------- |
| Backbone           | ResNet50 with multi-scale skip connections | MobileNetV2 (single-output, smaller model)   |
| Heatmap resolution | 112×112                                    | 56×56                                        |
| Sigma              | 2.0 (broader peaks)                        | 1.5 (sharper localization)                   |
| Loss config        | Combined loss with heatmap + coordinates   | Only heatmap loss (coordinates inferred)     |
| Callbacks          | TQDM, ReduceLROnPlateau, visual debugger   | EarlyStopping only                           |
| Export flow        | `.keras` → SavedModel → TFJS               | Direct `.keras` export with `model.export()` |
| TFJS compatibility | Manual wrapping of soft-argmax for export  | Fully WASM-compatible model at export time   |

---

## 3. Web Deployment

* Model exported as a TFJS graph model using `tensorflowjs_converter`
* Target format: `tfjs_graph_model` (WASM-safe)
* Export directory is zipped as `web_model.zip` for use in the frontend
* Optimized for use with `@tensorflow/tfjs-backend-wasm` or `webgl` runtime backends

---

## Repository Structure

```
.
├── dataset.py                             # Synthetic dataset generator
├── backgrounds/                           # Real-world composite backgrounds
├── dataset/                               # Generated images + annotations
│   └── annotations.json                   # 4-corner annotations
├── unique-artwork-*.json                  # Scryfall data source
├── mtg-keypoint-regression-heatmap.ipynb  # Training notebook
```

---

## Dataset Summary

| Attribute     | Value                             |
| ------------- | --------------------------------- |
| Image size    | 1024×1024                         |
| Format        | `.jpg`                            |
| Labels        | 4-point corner annotations (x, y) |
| Sample count  | \~5,000 (configurable)            |
| Augmentations | Perspective, affine, rotation     |
| Backgrounds   | Random photos/textures            |

---

## Training Environment

| Spec         | Value                  |
| ------------ | ---------------------- |
| Hardware     | Tesla P100 (16GB VRAM) |
| Image input  | 224×224                |
| Heatmap size | 56×56                  |
| Epochs       | 35                     |
| Batch size   | 16                     |
| TFJS runtime | WASM + WebGL           |

---

## License

All dataset generation and model code is licensed under [GNU GPL v3.0](LICENSE).
Please consult individual dependencies for their respective licenses. Card artwork is pulled via [Scryfall](https://scryfall.com/docs/api).
