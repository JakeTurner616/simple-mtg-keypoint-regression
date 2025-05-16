import json
import random
import time
import os
import cv2
import numpy as np
import requests
import string
from pathlib import Path

# Configuration
REPEAT_CYCLES = 250
DELAY_BETWEEN_DOWNLOADS = 0.2
OUTPUT_DIM = 1024
OUTPUT_DIR = "dataset"
ANNOTATION_FILE = os.path.join(OUTPUT_DIR, "annotations.json")
BACKGROUND_DIR = "backgrounds"
SCYFALL_JSON_PATH = "unique-artwork-20250415090518.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
annotations = []

def sanitize_filename(name):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c if c in valid_chars else '_' for c in name)

with open(SCYFALL_JSON_PATH, "r", encoding="utf-8") as f:
    cards = json.load(f)

usable_cards = [
    card for card in cards
    if not card.get("is_funny")
    and card.get("layout") not in ("token", "emblem", "art_series", "double_faced_token")
    and card.get("image_uris")
    and (card["image_uris"].get("png") or card["image_uris"].get("normal"))
    and not card.get("card_faces")
    and any(v == "legal" for v in card.get("legalities", {}).values())
]

# Load backgrounds
bg_paths = list(Path(BACKGROUND_DIR).glob("*.jpg")) + list(Path(BACKGROUND_DIR).glob("*.png"))
backgrounds = []
for p in bg_paths:
    b = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if b is None or b.ndim != 3 or b.shape[0] < OUTPUT_DIM or b.shape[1] < OUTPUT_DIM:
        continue
    if b.shape[2] == 4:
        b = cv2.cvtColor(b, cv2.COLOR_BGRA2BGR)
    backgrounds.append(b)
if not backgrounds:
    raise RuntimeError("No valid backgrounds found.")

def random_background_crop(bg_img):
    h, w = bg_img.shape[:2]
    x = random.randint(0, w - OUTPUT_DIM)
    y = random.randint(0, h - OUTPUT_DIM)
    crop = bg_img[y:y + OUTPUT_DIM, x:x + OUTPUT_DIM]
    return crop.astype(np.uint8).copy()

def apply_random_transform(card_img, orig_pts):
    try:
        h, w = card_img.shape[:2]
        pad = int(max(h, w) * 0.25)
        padded = cv2.copyMakeBorder(card_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        transform_type = random.choices(["rotate", "affine", "perspective"], weights=[0.6, 0.2, 0.2])[0]
        transformed_pts = orig_pts.copy() + pad

        if transform_type == "rotate":
            angle = random.uniform(-180, 180)
            M = cv2.getRotationMatrix2D((padded.shape[1] / 2, padded.shape[0] / 2), angle, 1.0)
            padded = cv2.warpAffine(padded, M, (padded.shape[1], padded.shape[0]), borderValue=(0, 0, 0, 0))
            transformed_pts = cv2.transform(np.array([transformed_pts], dtype=np.float32), M)[0]

        elif transform_type == "affine":
            shift = int(min(w, h) * 0.08)
            pts1 = np.float32([[pad, pad], [pad + w, pad], [pad, pad + h]])
            pts2 = np.float32([
                [pad + random.uniform(-shift, shift), pad + random.uniform(-shift, shift)],
                [pad + w + random.uniform(-shift, shift), pad + random.uniform(-shift, shift)],
                [pad + random.uniform(-shift, shift), pad + h + random.uniform(-shift, shift)]
            ])
            M = cv2.getAffineTransform(pts1, pts2)
            padded = cv2.warpAffine(padded, M, (padded.shape[1], padded.shape[0]), borderValue=(0, 0, 0, 0))
            transformed_pts = cv2.transform(np.array([transformed_pts], dtype=np.float32), M)[0]

        elif transform_type == "perspective":
            margin = int(min(w, h) * 0.05)
            pts1 = np.float32([
                [pad, pad],
                [pad + w, pad],
                [pad, pad + h],
                [pad + w, pad + h]
            ])
            pts2 = np.float32([
                [pad + random.uniform(0, margin), pad + random.uniform(0, margin)],
                [pad + w - random.uniform(0, margin), pad + random.uniform(0, margin)],
                [pad + random.uniform(0, margin), pad + h - random.uniform(0, margin)],
                [pad + w - random.uniform(0, margin), pad + h - random.uniform(0, margin)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            padded = cv2.warpPerspective(padded, M, (padded.shape[1], padded.shape[0]), borderValue=(0, 0, 0, 0))
            transformed_pts = cv2.perspectiveTransform(np.array([transformed_pts], dtype=np.float32), M)[0]

        return padded, transformed_pts
    except Exception as e:
        print(f"[!] Transformation error: {e}")
        return None, None

def composite_card(card_img, bg_img, filename, card_name):
    if card_img.ndim == 2:
        card_img = cv2.cvtColor(card_img, cv2.COLOR_GRAY2BGRA)
    elif card_img.shape[2] == 3:
        card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2BGRA)

    scale_factor = random.uniform(0.60, 0.85)
    card_img = cv2.resize(card_img, (int(card_img.shape[1] * scale_factor), int(card_img.shape[0] * scale_factor)))

    orig_pts = np.array([
        [0, 0],
        [card_img.shape[1] - 1, 0],
        [card_img.shape[1] - 1, card_img.shape[0] - 1],
        [0, card_img.shape[0] - 1]
    ], dtype=np.float32)

    card_img, transformed_pts = apply_random_transform(card_img, orig_pts)
    if card_img is None or transformed_pts is None:
        raise ValueError("Transformation failed")

    new_h, new_w = card_img.shape[:2]
    if new_w > OUTPUT_DIM or new_h > OUTPUT_DIM:
        fit_scale = min(OUTPUT_DIM / new_w, OUTPUT_DIM / new_h)
        card_img = cv2.resize(card_img, (int(new_w * fit_scale), int(new_h * fit_scale)))
        transformed_pts *= fit_scale
        new_h, new_w = card_img.shape[:2]

    alpha = card_img[:, :, 3]
    card_rgb = card_img[:, :, :3]
    mask = alpha.astype(np.uint8)
    inv_mask = cv2.bitwise_not(mask)

    canvas = random_background_crop(random.choice(backgrounds))
    x_offset = (OUTPUT_DIM - new_w) // 2
    y_offset = (OUTPUT_DIM - new_h) // 2

    roi = canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    if roi.shape[:2] != mask.shape:
        raise ValueError("ROI and mask shape mismatch")

    fg = cv2.bitwise_and(card_rgb.astype(np.uint8), card_rgb.astype(np.uint8), mask=mask)
    bg_roi = cv2.bitwise_and(roi.astype(np.uint8), roi.astype(np.uint8), mask=inv_mask)
    composite = cv2.add(fg, bg_roi)
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = composite

    corners = [[int(x + x_offset), int(y + y_offset)] for x, y in transformed_pts]
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), canvas)

    annotations.append({
        "filename": filename,
        "card_name": card_name,
        "corners": corners
    })

# Color group matchers
color_groups = {
    "white": lambda c: c == ["W"],
    "blue": lambda c: c == ["U"],
    "black": lambda c: c == ["B"],
    "red": lambda c: c == ["R"],
    "green": lambda c: c == ["G"],
    "colorless": lambda c: not c,
    "multicolor": lambda c: len(c) > 1
}
border_colors = ["black", "white", "borderless"]

# Generate dataset
counter = 0
for cycle in range(REPEAT_CYCLES):
    for border in border_colors:
        for color_name, color_filter in color_groups.items():
            matches = [
                c for c in usable_cards
                if c.get("border_color") == border and color_filter(c.get("color_identity", []))
            ]
            if not matches:
                continue

            card = random.choice(matches)
            try:
                url = card["image_uris"].get("png") or card["image_uris"].get("normal")
                response = requests.get(url, timeout=10, stream=True)
                content = response.content
                if len(content) > 5_000_000:
                    continue
                img_array = np.asarray(bytearray(content), dtype=np.uint8)
                card_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                if card_img is None or card_img.size == 0 or card_img.shape[0] < 100:
                    continue

                filename = f"{counter:05d}_{sanitize_filename(card['name'])[:25]}.jpg"
                composite_card(card_img, random.choice(backgrounds), filename, card['name'])
                counter += 1
                time.sleep(DELAY_BETWEEN_DOWNLOADS)
            except Exception as e:
                print(f"[!] Failed on '{card['name']}': {e}")

# Save final annotations
with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2)