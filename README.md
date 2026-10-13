# Fractional-Kernel Brain Tumour Segmentation

**A MATLAB research prototype for binary whole-tumour segmentation from prepared 3D FLAIR MRI volumes.**

MATLAB · NIfTI · Fractional-order-inspired filtering · Adaptive thresholding · Clustering · Optional GPU convolution

> **Research use only.** This is an experimental segmentation pipeline, not a clinically validated diagnostic system. It does not classify a tumour as benign or malignant.

## Problem and approach

This project explores how classical numerical image processing can produce a whole-tumour mask without training a neural network. It combines fractional-order-inspired kernels, directional convolution sweeps, a difference-of-diffusion-style feature, adaptive thresholding, and clustering-based refinement.

The historical repository/function names refer to mesh-free ADI. However, the current `sweep` helper performs explicit sequential convolution updates along the three axes; it does not implement an implicit linear-system solve. Treat this as the implementation described below, not as a verified reproduction of a paper's ADI solver.

## Pipeline

```text
Prepared 3D FLAIR image + matching label volume
                  |
       Robust intensity normalization
                  |
     Fractional-kernel directional sweeps
          at two iteration counts
                  |
       Absolute difference feature (DoD)
                  |
      Blockwise adaptive Otsu thresholding
                  |
       FCM refinement, or k-means fallback
                  |
  Morphological cleanup + largest 3D component
                  |
       Binary whole-tumour mask + metrics
```

### Implemented stages

1. Load the image and segmentation label using MATLAB's NIfTI functions.
2. Convert every nonzero ground-truth label to a single whole-tumour foreground class.
3. Normalize image intensities using the 1st and 99th percentiles within an intensity-based brain mask.
4. Construct one-dimensional kernels and apply them along the x, y, and z axes.
5. Form `abs(u2 - u1)` and normalize it as a DoD feature.
6. Combine local and global Otsu thresholds to select candidate voxels.
7. Cluster candidate features `[FLAIR intensity, DoD]`, choosing the cluster with the highest mean DoD.
8. Apply morphological cleanup and retain the largest 26-connected component.
9. Compare the predicted mask with the binary reference mask.

Three-plane fusion is implemented but disabled by default. GPU arrays are optional for the convolution portion; thresholding and clustering run after data is gathered back to the CPU.

## Repository contents

| File | Purpose |
|---|---|
| `meshfree_ADI3D_paper.m` | Single-subject segmentation function and helper routines |
| `run_ADI3D.m` | Batch execution, metric export, plots, and summaries |
| `ADI_metrics.csv` | Existing result artifact; not a substitute for a documented evaluation protocol |
| `*.png` | Existing metric visualizations |
| `3d-video.mp4` | Existing visualization artifact |

MRI volumes, reference labels, and dataset-preparation scripts are not included. The current batch script does not necessarily regenerate every plot already committed to the repository.

## Requirements

- MATLAB with support for the functions used in the scripts, including `niftiread` and `exportgraphics`.
- **Image Processing Toolbox** for NIfTI/image-processing and morphology operations.
- **Statistics and Machine Learning Toolbox** for k-means fallback and batch plotting/statistical functions such as `ksdensity` and `boxplot`.
- **Fuzzy Logic Toolbox**, optionally, for `fcm`. The script uses the legacy options-vector calling style; check compatibility with your MATLAB release.
- **Parallel Computing Toolbox** and a supported GPU only if you enable GPU execution.

A tested MATLAB/toolbox version matrix has not been recorded. Start with CPU execution if GPU support is unavailable.

## Prepare the data

Use a directory containing already-prepared, spatially aligned image/label pairs:

```text
data/
├── imagesTr/
│   ├── case_001.nii
│   └── case_002.nii
└── labelsTr/
    ├── case_001.nii
    └── case_002.nii
```

Important input assumptions:

- Each image must be a **single 3D FLAIR volume**, not an unprocessed 4D stack of MRI modalities.
- Each label volume must have the same dimensions and alignment as its image.
- Images and labels are paired by their **separately sorted filenames**, not by a validated subject-ID manifest. Check their ordering before running.
- Prefer one uncompressed `.nii` file per subject in the current implementation. The function combines overlapping `*.nii*` and `*.nii.gz*` searches, which can duplicate compressed files.
- Obtain data under its own access and licensing terms. Do not put identifiable patient data into a public repository.

## Run one subject

```bash
git clone https://github.com/thinkbin006/fractional-calculous-based-mesh-free-brain-tumor-segmentation.git
cd fractional-calculous-based-mesh-free-brain-tumor-segmentation
```

Open MATLAB in the repository directory, then run:

```matlab
rootDir = '/absolute/path/to/prepared-data';
rng(0);  % Make stochastic clustering easier to reproduce.

R = meshfree_ADI3D_paper(1, rootDir, false);  % CPU execution
disp(R.metrics);

sliceIndex = round(size(R.mask, 3) / 2);
figure;
imshow(R.mask(:, :, sliceIndex));
title('Predicted whole-tumour mask');
```

Use `true` as the third argument only when GPU support is available. The actual function name is `meshfree_ADI3D_paper`; the `_fast` name in the source's usage comment is not a separate supplied function.

### Return value

- `R.mask`: logical 3D predicted mask.
- `R.metrics`: `Dice`, `Jacc`, `Sens`, `Spec`, `Acc`, and `HD95`.

The `HD95` field is produced by the repository's custom helper and needs validation before use as a standard surface-distance metric. See the evaluation caveats below.

## Run a batch

Before running `run_ADI3D.m`:

1. Replace its hard-coded `rootDir = 'K:\explo_parv'` with your own dataset path.
2. Change the function call's third argument from `true` to `false` for CPU-only execution.
3. Review `N = min(369, numel(imgs))`; the current script processes at most 369 entries.
4. Check pairing, file-discovery, and metric-summary caveats below.

Then run:

```matlab
rng(0);
run_ADI3D;
```

Outputs are written to `<rootDir>/results_ADI_fast/`:

- `ADI_metrics.csv`.
- `ADI_metrics.xlsx`, if the export succeeds.
- `Dice_Histogram.png` and `Dice_Box_Violin.png`, when plotting completes.

Failed cases are caught and represented by missing metric values. Predicted masks are returned for single-subject calls but are not automatically saved by the batch runner.

## Main parameters

These are currently edited inside `meshfree_ADI3D_paper.m`:

| Parameter | Default | Role |
|---|---:|---|
| `alpha` | `1.40` | Fractional-order-inspired kernel parameter |
| `lambda` | `0.35` | Directional update strength |
| `t1`, `t2` | `4`, `12` | Iteration counts used to construct the difference feature |
| `neigh` | `3` | Kernel half-width |
| `fusePlanes` | `false` | Optional axial/coronal/sagittal fusion |
| `resizeTo` | `[]` | Keep original dimensions unless a target size is supplied |

These are implementation defaults, not clinically or statistically validated optimal settings.
