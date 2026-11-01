# Experiment Review

[← Back to project](../README.md)

## Research question

Can a small convolutional encoder–decoder learn to reconstruct crop imagery from a spatial feature representation?

The saved notebook provides evidence of learning on the training objective and three qualitative examples from the designated test array. It does not establish compression efficiency, robust generalization, or production readiness.

## Evidence and scope

This review uses the notebook source, saved console logs, model summary, and four embedded figures. No training or inference was rerun: neither the dataset nor the saved `rgb_test.pt` checkpoint is present.

| Evidence | Observation | Scope |
| :--- | :--- | :--- |
| Dataset shape output | 211 training and 46 test images | Counts are recorded; split construction and overlap cannot be checked |
| Training logs | Last-batch MAE generally decreases over 50 epochs | Not an epoch average; no validation curve |
| Inference output | Feature shape `256 × 56 × 56`, output `3 × 224 × 224` | Matches the declared architecture |
| Three saved comparisons | Recognizable scene structure, blurred details, grid patterns | Selected examples, not an aggregate test evaluation |

## Architecture audit

For one image, the network follows:

`3 × 224 × 224 → 64 × 112 × 112 → 256 × 56 × 56 → 64 × 112 × 112 → 3 × 224 × 224`

The encoder applies ReLU only after the first convolution. The decoder applies ReLU after its first transposed convolution and sigmoid at the output. The sigmoid range is consistent with inputs normalized by dividing pixel values by 255.

| Layer | Trainable parameters, including bias |
| :--- | ---: |
| Encoder convolution 1 | 1,792 |
| Encoder convolution 2 | 147,712 |
| Decoder transposed convolution 1 | 262,208 |
| Decoder transposed convolution 2 | 3,715 |
| **Total** | **415,427** |

The encoder reduces spatial area by 16×, but increases channel count by approximately 85.33×. The representation therefore expands from 150,528 to 802,816 elements. It may learn useful features, but this experiment does not demonstrate dimensionality reduction by total feature count or image-file compression. File size would also depend on precision, quantization, and coding.

## Training-loss interpretation

The training function overwrites `loss` for every batch and records it after the inner loop. With 211 images and batch size 16, there are 14 batches: 13 full batches and one batch of 3. Only that final batch contributes to each plotted point.

- Epoch 1: **0.1865366**.
- Lowest logged value: **0.0417591**, at epoch **43**.
- Epoch 50: **0.0670119**.

The overall decrease is consistent with learning the reconstruction objective. Fluctuations are partly confounded by the changing composition of the final shuffled batch. The notebook's statement that the lowest loss occurs at the end is inaccurate. The code saves the final model; it does not select a checkpoint by validation performance.

A sample-weighted mean over all training batches would provide a more useful epoch statistic. A separate validation split is needed for checkpoint selection and learning-curve comparisons; the test set should remain reserved for final evaluation.

## Critical issue: tensor layout

Images are initially stored as `(N, H, W, C)`. The notebook converts them to `(N, C, H, W)` with `torch.reshape`. This changes shape while keeping the flattened element order; it does not move the RGB axis into the channel position.

Consequently, convolutional filters operate on incorrectly arranged color/spatial values. The inference visualization reshapes tensors back to HWC, which can make the input appear normal again while concealing the internal layout error. Visually plausible output does not resolve that error.

Use an axis permutation instead:

```python
# Given a normalized NHWC NumPy array:
images = torch.from_numpy(image_array).float().permute(0, 3, 1, 2).contiguous()

# Given one CHW prediction:
display_image = prediction.detach().cpu().permute(1, 2, 0).numpy()
```

Apply the correction consistently to training, test preprocessing, and visualization, then **retrain**. Existing recorded losses and weights describe the original layout and cannot be presented as results from the corrected workflow.

## Qualitative findings

The vegetation example retains the broad arrangement of people and foliage, with softened boundaries. The grain example preserves warm tones and larger stem structures but loses texture. The field example retains the sky, horizon, and field regions, while showing conspicuous grid lines.

These observations support a limited claim: the saved outputs resemble the displayed inputs at a broad visual level. They do not quantify fidelity. Grid artifacts are visible, but their cause is not isolated; attributing them solely to transposed convolutions would require a controlled comparison after fixing tensor layout.

## Reproducibility gaps

| Gap | Consequence | Recommended action |
| :--- | :--- | :--- |
| Dataset source/license absent | Provenance and redistribution rights are unknown | Record the actual source, permitted use, and dataset version |
| Absolute Drive paths | Notebook cannot run unchanged elsewhere | Centralize paths in one configuration cell |
| Unsorted directory enumeration | Sample indices may identify different images on rerun | Sort filenames and retain a split manifest |
| No explicit RGB conversion | Grayscale/RGBA files may break shape assumptions | Use `Image.open(path).convert('RGB')` |
| No random seeds or environment lock | Exact results cannot be reproduced reliably | Record seeds, versions, hardware, and deterministic settings |
| No full test metrics | Generalization is unmeasured | Report aggregate and per-image errors on a verified separate split |
| No explicit training-device transfer | GPU use is not established by the training code | Move both model and batches to the selected device |
| No inference-mode context | Inference unnecessarily builds autograd graphs | Use `torch.inference_mode()` |

Colab metadata mentions a T4 GPU, but the training code never explicitly moves the model or tensors to CUDA. The saved run should not be described as confirmed GPU training. The inference section explicitly moves the model to CPU.

## Prioritized follow-up

1. **Restore correct image semantics.** Fix layout, enforce RGB, sort file loading, and verify shapes and display round trips.
2. **Make evaluation reliable.** Establish documented train/validation/test splits, seed the run, average losses by sample count, and save the best validation checkpoint.
3. **Measure reconstruction quality.** Report full-test MAE, PSNR, and SSIM with a defined data range, plus representative successes and failures.
4. **Test the representation.** Reduce latent channels or add a smaller bottleneck and compare fidelity against representation size.
5. **Investigate artifacts.** Compare decoder variants under the same corrected data pipeline and evaluation protocol.

The strongest portfolio story here is an implemented reconstruction experiment accompanied by a clear understanding of its evidence, mistakes, and next research steps.
