# Reproduction Guide

[← Back to project](../README.md)

## Two ways to explore

**Review the recorded experiment:** open [the notebook](../Convolutional%20Autoencoder.ipynb) or read the README figures. The saved outputs are already included; no dataset or ML environment is needed to inspect them.

**Run a corrected experiment:** supply your own authorized dataset, prepare the environment, fix preprocessing, and retrain. The repository does not contain the original images or checkpoint, so the original results cannot currently be independently reproduced from this checkout alone.

## Environment

The notebook records Python 3.9.21. Exact package versions were not saved. The following is a suggested starting environment, not a validated lockfile:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy matplotlib pillow tqdm polars jupyterlab
jupyter lab
```

`polars` is imported by the original notebook but is not used in the model pipeline. Google Colab supplies the `google.colab` module; a local environment does not. When running locally, skip the Drive mount cell and replace all `/content/drive/...` paths, including the initial sample-image cell and checkpoint save/load paths.

## Dataset layout

The notebook expects training images inside class-like subdirectories and test images directly inside a separate folder:

```text
data/
├── crop_images/
│   ├── category_a/
│   │   ├── image_001.jpeg
│   │   └── image_002.jpeg
│   └── category_b/
│       └── image_003.jpeg
└── test_crop_image/
    ├── sample_001.jpeg
    └── sample_002.jpeg
```

Folder labels are not used by the reconstruction loss. The original loader accepts only `.jpeg` training filenames and attempts to open most test-directory entries as images. For a new run, filter supported file extensions explicitly in both branches and sort paths. Record the dataset source and license, and verify split independence before reporting test performance.

## Corrections before training

Convert each image explicitly to RGB, resize to 224 × 224, and normalize to `[0, 1]`. Then reorder axes:

```python
# image_array: normalized NumPy array, shape (N, 224, 224, 3)
images = torch.from_numpy(image_array).float()
images = images.permute(0, 3, 1, 2).contiguous()
assert images.shape[1:] == (3, 224, 224)
```

Use the same transform for every split. Avoid `reshape` for channel conversion. This changes the experiment and requires new training; the existing saved figures remain evidence of the original implementation.

Track a sample-weighted training loss. This fragment replaces the inner training loop after defining `model`, `data_loader_train`, `optimizer`, and `loss_fn`:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.train()
loss_sum = 0.0
sample_count = 0
for batch in data_loader_train:
    batch = batch.to(device)
    optimizer.zero_grad()
    prediction = model(batch)
    loss = loss_fn(prediction, batch)
    loss.backward()
    optimizer.step()
    loss_sum += loss.item() * batch.size(0)
    sample_count += batch.size(0)
epoch_mae = loss_sum / sample_count
```

Use a separate validation loader in evaluation mode to select a checkpoint. Set random seeds before model initialization and loader creation, and record environment versions and split filenames. These steps improve traceability; they do not guarantee bit-for-bit equivalence across hardware.

## Inference and visualization

The following fragment assumes an instantiated model, an available checkpoint from the same architecture and preprocessing, and a corrected CHW test tensor:

```python
model.load_state_dict(torch.load('rgb_test.pt', map_location='cpu', weights_only=True))
model = model.cpu().eval()
with torch.inference_mode():
    prediction = model(test_data[0:1].cpu())[0]

plt.imshow(prediction.permute(1, 2, 0).numpy())
plt.axis('off')
```

Use a PyTorch version supporting `weights_only=True`, as in the original notebook. Evaluate the full test set rather than relying on a few selected images, and keep the test set separate from checkpoint selection.

## Regenerate repository visuals

From the repository root:

```bash
python3 scripts/export_notebook_assets.py
```

This standard-library script extracts four embedded PNG figures, exports all 50 recorded losses to [CSV](../assets/training-loss.csv), and redraws the loss chart as SVG. It does not train the model, generate predictions, or require the original dataset. The [hero graphic](../assets/hero.svg) is a separate hand-authored architecture illustration.
