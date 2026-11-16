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

Training recipe shared by notebooks 01-04: see "Review-response revision" below (60 epochs, patience 8, CAWL+ loss, Mixup + CutMix, flip test-time augmentation chosen on validation).

Notes on runtime:
- Notebooks 02 and 03 train each baseline for up to 60 epochs (early stopping, patience 8). They save results after every model and skip finished models, so they can be split across sessions by editing `RUN_MODELS`, `SEED_LIST` and setting `RESUME_DIR` to a folder holding `baseline_results.csv` and `baseline_predictions.npz`. `TIME_BUDGET_H` (11 h) stops new runs before the 12 h Kaggle limit.
- Notebook 04 defaults to seeds 42, 13 and 77 and runs 15 arms (full model, plain backbone, branch/fusion/attention removals, MS-CA-ECA parts, loss terms, lesion head): 45 runs. For a quicker check set `SEED_LIST = [42]`. It resumes from `plain_backbone_results.csv` and `plain_backbone_predictions.npz` through `RESUME_DIR`.
- Notebooks 05 and 06 run 60 communication rounds with 2 local epochs per client at seeds 42, 13 and 77. Each finished federated run is saved to `/kaggle/working/fl_cache` as a `.pt` file. The notebooks stop at `TIME_BUDGET_H` (10.5 h) with a message; upload the `.pt` files as a Kaggle dataset, attach it, set `FL_RESUME_DIR` (and `EXPECT_RESUMED_RUNS` to the number of saved runs) and run again. Full 05 takes about 24 runs, full 06 about 27, over 2-3 sessions each. `USE_AMEF = False` in both.

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


## Review-response revision (SMGF_Net_7 setup)

Notebooks 01-07 share one setup, the profile "final" of the SMGF_Net_7 notebook: 288 px aspect-preserving resize, robust normalisation, lesion-guided SMGF-Net + MS-CA-ECA,
Mixup + CutMix, source-balanced sampling, near-duplicate cleaning (thumbnail correlation 0.98) and a grouped 10 % validation split, 60 epochs, early-stopping patience 8
counted from epoch 30, checkpoint selection on validation loss. The federated notebooks (05/06) use the same preprocessing, cleaning and model (lesion head off: clients have
no masks) with their own client optimiser.

Run order: 08 (pHash pairs and threshold calibration) -> 01 -> 07 (with `ab_final_seed42.pth` attached as a Kaggle dataset or after 01) -> 02/03/04 (seeds 42,13,77,7,2024 and 101..505) -> 05/06 -> 09
(paired tests across seeds, Holm correction, McNemar, minimum detectable effect).

- Frozen recipe (`frozen_recipe.json`) and a validation-only choice of the inference mode (flip-TTA vs single pass); every run saves `<run>_predictions.npz`.
- 01: before/after preprocessing figure, 10-seed resumable multi-seed run with t-intervals and MDE, matched vs unmatched accuracy, bounded temperature fit, temperature scaling on unmatched external subsets (fitted T, 1, 0.438), `XAI_LARGE` sample sizes.
- 05/06: 60 rounds, 3 seeds (42, 13, 77), centralised hyper-parameters, GroupNorm FedAvg/FedProx for all seeds, convergence diagnostic, DP tuning on 12-round proxy runs (batches larger than the smallest client's data are skipped), clipping-only and centralised controls (centralised DP-SGD uses a physical batch of 48 to fit GPU memory), one table `dp_consolidated_table.csv`. 06 adds an epsilon ablation (1, 3, 8, 20). `RUN_MU_SWEEP`, `DP_RUN_SECOND_CONFIG` and `DP_TUNED_EXTRA_EPS` are off by default to save GPU time.
- 08: nearest training image for every test image, pair inspection, pHash threshold calibration (false-positive rate and detection of edited copies), external-set overlap counts, parameter counts. Runs on CPU.
- 09: aggregation, no training. Attach `multiseed_results.csv` (01), `baseline_results.csv` + `baseline_predictions.npz` (02/03) and `plain_backbone_results.csv` + `plain_backbone_predictions.npz` (04); the most complete copy of each is used and the log prints its row count.
- 07: checkpoint accuracy check, matched/unmatched test accuracy, temperature scaling on unmatched external subsets, mask alignment overlay with a perfect-map control, larger XAI samples including errors.
- All nine notebooks have been run on Kaggle in this setup. Results are summarised in the review-response report; the stated limits are federated runs at a fixed 60 rounds (most not converged), a single-seed DP study, and ablation differences below the minimum detectable effect at three seeds.
