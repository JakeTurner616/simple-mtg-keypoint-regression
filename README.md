# MTG Keypoint Regression – Dataset + Model Training Pipeline

[View the full model training notebook on Kaggle.](https://www.kaggle.com/code/jaketurner616/mtg-keypoint-regression-heatmap) Or run the project locally.


This project generates synthetic training data using MTG artwork, applies controlled distortions, and trains a TensorFlow-based corner regression model. The final model is exported to TensorFlow.js for real-time web inference for MTG ROI extraction on edge devices.

<p align="center"> <img src="/docs/epoch_03.jpg" alt="Early epoch prediction with poor alignment" width="400"/> <img src="/docs/epoch_30.jpg" alt="Late epoch prediction with precise corners" width="400"/> </p> <p align="center"> <b>Left:</b> Early prediction (Epoch 3) &nbsp;&nbsp; | &nbsp;&nbsp; <b>Right:</b> Refined prediction (Epoch 30) </p>

---

## Repository Structure

```
.
├── dataset.py                             # Script to create synthetic dataset + labels
├── backgrounds/                           # Directory of background images for synthetic dataset
├── dataset/                               # Output directory for synthetic dataset
├── dataset/annotations.json               # Output annotation file with 4 corner keypoints per image
├── unique-artwork-*.json                  # Scryfall artwork dump for synthetic dataset
├── mtg-keypoint-regression-heatmap.ipynb  # Exported notebook for heatmap-based training using Kaggle GPU
```

---

## Overview

### 1. Dataset Generation (`dataset.py`)

* **Downloads card images** by performing a set amount of 'cycles' to add equally diverse border styles.
* **Applies perspective & affine transforms** to simulate real-world camera angle perspective shifts
* **Composites each card** onto random photographic and texture backgrounds
* **Annotates true corner positions** of the distorted card in the output image

Each synthetic image is saved as a `.jpg`, and its 4 corner positions are recorded in `annotations.json`.

### Output format:

```json
{
  "filename": "00042_Llanowar_Elves.jpg",
  "card_name": "Llanowar Elves",
  "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

> Images are 1024×1024 with randomized placement, scale, and angle. All data is stored locally and can be extended with new backgrounds or distortion rules.

---

### 2. Model Training (`mtg-keypoint-regression-heatmap.ipynb`)

* Loads the dataset and builds a dual-output model:

  * `heatmaps`: 4 heatmap channels (one per corner)
  * `coords`: predicted coordinates via soft-argmax
* Based on a pretrained ResNet-50 backbone (ImageNet weights)
* Uses MSE loss on heatmaps with soft-argmax refinement for coordinate accuracy
* Metrics include average corner error in pixels

> Visualizations are saved per epoch showing both ground-truth and predicted corners.

* Web Export

* Loads the `.keras` model with the custom `SoftArgmax` layer
* Converts it into a TFJS `graph_model` for browser inference
* Outputs to a zip archive (`web_model.zip`) containing:

  ```
  web_model/
    model.json
    group1-shard1of1.bin
  ```

> This model can be used on the web for ROI card extraction.

---

## Dataset Details

| Parameter         | Value                                           |
| ----------------- | ----------------------------------------------- |
| Image size        | 1024×1024                                       |
| Output format     | `.jpg`                                          |
| Label format      | 4 corner points in pixel space                  |
| Number of samples | \~5,000 (customizable)                          |
| Augmentations     | Rotation, Affine, Perspective, Background Blend |

---

## Use in Production

The model created from this workflow  can be used to:

* Detect the precise location of a real MTG card in a frame
* Normalize the region of interest for descriptor extraction

---

## License

All dataset generation code is released under the [GNU GPL v3.0](LICENSE) LICENSE.
