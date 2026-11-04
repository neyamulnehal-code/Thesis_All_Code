# SMGF-Net + MS-CA-ECA: code for brain tumor MRI classification

Code behind the paper's experiments on the BRISC2025 four-class brain tumor MRI dataset (glioma, meningioma, no tumor, pituitary). The notebooks are Kaggle notebooks with all outputs cleared. Run them top to bottom on a Kaggle GPU session.

## Contents

| Notebook | What it does | Paper |
|---|---|---|
| `01_stage1_SMGF-Net_centralized_uncertainty_noise.ipynb` | Centralized training of SMGF-Net + MS-CA-ECA (seed 42 and the multi-seed runs), the single-run component, attention and training-pipeline ablations, internal test evaluation, external validation, XAI, error analysis, robustness, statistics, calibration, multi-scale uncertainty, noise-induced uncertainty. | Results: overall performance, ablations, calibration, uncertainty, XAI, external validation |
| `02_rerun1_baselines_paired_tests_seed42.ipynb` | One baseline set trained from scratch with the same recipe as SMGF-Net. Exact McNemar test and paired-bootstrap accuracy CI against each baseline, plus parameters, FLOPs, latency and file size. Seed 42. | Matched-protocol baseline comparison |
| `03_rerun1_baselines_paired_tests_seed13.ipynb` | Same as 02, seed 13. | Second seed of the matched comparison |
| `04_rerun2_ablation_plain_backbone_seeds42-13-77.ipynb` | Plain-backbone baseline (spatial branch only) and the full model retrained under the ablation protocol for seeds 42, 13 and 77, plus the four removal arms. McNemar tests against the freshly trained full model. | Same-procedure three-seed repeat of the architecture ablation |
| `05_stage2_federated_IID_FedAvg_FedProx_DP.ipynb` | Federated learning with 5 simulated hospitals on IID splits: FedAvg, FedProx, DP-FedAvg (DP-SGD, Opacus), a GroupNorm control, and post-hoc diagnostics. | Federated and DP results, IID; rerun without AMEF |
| `06_stage2_federated_nonIID_FedAvg_FedProx_DP.ipynb` | The same pipeline on label-skewed splits (Dirichlet, alpha = 0.3), with per-client privacy accounting and an epsilon ablation (1, 3 and 20). | Federated and DP results, non-IID; rerun without AMEF |
| `07_eval_checkpoint_analyses.ipynb` | Analyses that need predictions but no training: external accuracy on images that are and are not in BRISC2025 (bootstrap CIs), random-saliency baseline for deletion/insertion, weight-randomization check, attribution overlap with the BRISC2025 tumor masks, and a check of which preprocessing reproduces the reported accuracy. Loads the seed-42 checkpoint written by notebook 01, or retrains it once if none is attached. | Clean-subset external accuracy, random-saliency, randomization and mask-overlap tables |
| `08_phash_neardup_and_param_counts.ipynb` | CPU-only checks: perceptual-hash (pHash) near-duplicates between the BRISC2025 test and training pools, inside the training pool, and between each external set and BRISC2025 (including those that are not byte-identical); four-class parameter counts of MobileNetV4-Small and RepViT-M1. | Leakage control; baseline parameter counts |

Suggested reading order: 01, then 02-04 (additional reruns), then 05-06 (federated and DP), then 07-08.

## Data

- **BRISC2025** (classification task), attached as a Kaggle dataset. Every notebook expects it at
  `/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task` with `train` and `test` folders.
- **External validation sets** (notebooks 01 and 07): the external validation cell lists four public Mendeley brain tumor MRI collections (`mendeley3`, `mendeley5`, `mendeley_epic_cscr`, `mendeley2`) and skips any that are not attached. The paper uses two of them, Dataset 1 and Dataset 2, so only those two were attached for the reported results. Attach the collections you want as Kaggle datasets and update the paths in the external validation cell to match your own dataset names.

Datasets are not redistributed here. Follow each dataset's own license and terms of use.

## Environment

- Kaggle GPU session (CUDA, mixed precision).
- Python 3 with PyTorch, torchvision, `timm`, `scikit-learn`, `scikit-image`, `thop` (notebooks 01-04, 07-08) and `lime`, Opacus (notebooks 05-06). Each notebook installs what it needs in its first cells.
- Internet must be on for the first cell of each notebook (pip install).

## Reproducing

1. Attach BRISC2025 (and, for notebooks 01 and 07, the external sets) to a new Kaggle notebook.
2. Upload a notebook, turn on a GPU, and run all cells in order.
3. Results, figures and CSV/NPZ artifacts are written to `/kaggle/working`.

Training recipe shared by notebooks 01-04: 50 epochs, early stopping (patience 10), CAWL+ loss, Mixup, EMA weights, flip test-time augmentation, fixed seeds.

Notes on runtime:
- Notebooks 02 and 03 train each baseline for up to 50 epochs. They save results after every model and skip finished models, so they can be split across sessions by editing `RUN_MODELS` and setting `RESUME_DIR`.
- Notebook 04 defaults to seeds 42, 13 and 77 and runs the removal arms (18 runs). For a quicker check set `SEED_LIST = [42]` and `RUN_REMOVAL_ARMS = False`.
- Notebooks 05 and 06 run 50 communication rounds with 2 local epochs per client. `USE_AMEF = False` in both, so the federated runs use the same preprocessing as the centralized notebook.

## Reproducibility notes

- Seeds are fixed in each notebook, but GPU nondeterminism means results can differ between runs and between GPU types. In the paper, repeats of seed 42 gave 93.47% and 93.87% against 96.28% for the original run, so a rerun of notebook 01 may not reproduce 96.28%.
- The checkpoint and predictions of the original seed-42 run (96.28%) were not kept. The analyses in notebook 07 therefore use a model retrained with the same recipe and seed (94.17% single-pass, 93.87% with flip-TTA).
- The federated and DP tables marked "Earlier Run, With AMEF" in the paper came from earlier versions of notebooks 05-06 with AMEF on. The notebooks here have `USE_AMEF = False` and reproduce the rerun tables.
- The federated notebooks use seed 42 for the client partition; the DP runs reuse the baseline partition.
- DP settings: per-sample clipping norm 1.0, delta 1e-5, RDP accountant, target epsilon 8 for the main run. The noise multiplier is solved from the sampling rate and step count of the executed data loader.

## License

Code is released under the MIT License (see `LICENSE`).

## Contact

Neyamul Islam, Khwaja Yunus Ali University (KYAU). neyamul.nehal@gmail.com
