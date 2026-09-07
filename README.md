# AIMONK — Multi-Label Image Attribute Classification

## Problem Statement
Given a dataset of images where each image can carry up to four independent binary attributes (`Attr1`–`Attr4`), build a model that predicts all four attributes for a new image. Two things make this harder than standard multi-label classification:
- Some attribute labels are missing (`NA`) rather than 0/1 — a naive loss function would either crash or learn the wrong signal by treating missing labels as negatives.
- The attributes are imbalanced — some appear far less often than others, so a standard loss biases the model toward always predicting the majority class.

## What This Project Solves
- **Missing labels:** Implemented a custom `MaskedBCEWithLogitsLoss` — `NA` values are mapped to `-1` at load time, and during training a binary mask zeroes out the gradient contribution from any `-1` label, so the model only ever learns from confirmed annotations.
- **Class imbalance:** Computed a per-attribute negative-to-positive ratio and passed it as `pos_weight` into the loss, forcing the optimizer to weight rare-attribute mistakes more heavily.
- **Data quality:** Cleaned inconsistent label strings (e.g. `"1mobilenet"` → `1`) and filtered the dataframe down to only images that actually exist on disk before training (dataset had 972 usable images after this sync step).

## Approach
- **Backbone:** ResNet50 pretrained on ImageNet.
- **Fine-tuning:** Replaced the final fully connected layer with a 4-neuron output (one logit per attribute) and trained end-to-end at a low learning rate (`1e-4`, Adam) to preserve the pretrained features.
- **Data pipeline:** Custom PyTorch `Dataset`/`DataLoader`, images resized to 224x224, normalized with ImageNet stats, batch size 8.

## Tech Stack
Python, PyTorch, torchvision, pandas, Pillow, matplotlib, NumPy — developed and run on Google Colab (GPU runtime).

## Results
Trained for 20 epochs. Average training loss dropped from **0.594 to 0.043**, with the loss curve saved as a plot in the notebook.

## How to Run
This notebook was built for Google Colab and expects your data in Google Drive:
1. Open `AIMONK_Assignment.ipynb` in Colab.
2. Place your dataset in Google Drive as:
   ```
   MyDrive/Multilabel/labels.txt
   MyDrive/Multilabel/images/
   ```
3. Run all cells top to bottom — the first cell mounts your Drive.
4. Trained weights are saved as `aimonk_model_weights.pth`; the final cell exposes a `predict_image(image_path)` function for inference on a new image.

To run locally instead of Colab, install dependencies with `pip install -r requirements.txt` and update `FOLDER_PATH`/`IMAGE_DIR` to a local dataset path.

## Why This Matters
Masked-loss + pos_weight is a reusable pattern anywhere labels are partially missing and imbalanced — e.g. product-attribute tagging, content moderation flags, or medical findings where not every image is annotated for every condition.
