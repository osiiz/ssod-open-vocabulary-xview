# Semi-supervised detection with open-vocabulary detectors on xView

[Galego](README.md) · **English** · [Español](README.es.md)

Code for the Bachelor's Thesis (TFG) on **semi-supervised object detection (SSOD)** on satellite
imagery (the **xView** dataset), integrating the frozen open-vocabulary detectors **Grounding
DINO** and **Rex-Omni** as a complementary source of pseudo-labels over a **Faster R-CNN** base
detector.

The whole flow is defined as a **reproducible pipeline with [DVC](https://dvc.org)** in
`dvc.yaml`, parameterized in `params.yaml`. This repository contains the **code** and the **result metrics** (small JSON files); the data, the
bulky intermediate outputs and the model weights are not included, and are obtained as described
in [Section 2](#2-get-the-data).

---

## Pipeline overview

The work is organized into the following phases, each defined as one or more DVC stages:

1. **Preprocessing** — convert xView (GeoJSON) to COCO format, group the 60 original classes into
   9 macro-categories, validate boxes, stratified 70/10/20 split, and tile the images into
   700×700 px crops.
2. **Base detector** — train a Faster R-CNN (ResNet-50 + FPN v2) on the labeled subset (lower
   bound, *Faster10*) and on the full `train` split (upper bound).
3. **Open-vocabulary inference** — Grounding DINO and Rex-Omni over the unlabeled images, with
   multi-prompt *ensembling* and aggregation (*union-find* + Borda count).
4. **Pseudo-label selection** — per-source selection rules and exclusion zones.
5. **SSOD re-training** — Faster R-CNN with the combined pseudo-labels (`FT`, `GD`, `RO` and
   their combinations).
6. **Evaluation** — COCO metrics on `test` and a final comparison table.

---

## Requirements

**Hardware**

- GNU/Linux with an **NVIDIA GPU** supporting CUDA 11.8 (required for training and inference).
- **Disk**: the distributed DVC store is **~62 GB** compressed; plan for several tens of GB more
  for the full materialization of data and outputs.

**Software**

- **git** (to clone) and **Docker** with the **NVIDIA Container Toolkit**.
- Internet access on first run: the Grounding DINO and Rex-Omni weights are downloaded
  automatically from HuggingFace.

---

## 1. Get the code

```bash
git clone https://github.com/osiiz/ssod-open-vocabulary-xview.git
cd ssod-open-vocabulary-xview
```

---

## 2. Get the data

There are **two ways**. **Way B (DVC store) is recommended**: it avoids both the manual xView
download and re-running the pipeline. Either way, **get the data before creating the
environment**, so you can mount it afterwards.

### Way A — original xView download

1. Register and download the xView training set (DIUx xView 2018 Detection Challenge) at
   **https://xviewdataset.org/**.
2. Place the data inside the `xView/` folder (at the repo root), keeping the original structure:
   ```
   xView/
   ├── train_images/            # training .tif images (downloaded)
   ├── xView_train.geojson      # original annotations (downloaded)
   ├── xView_classes.json       # already included in this repo (60 original classes)
   └── xView_macro_classes.json # already included (mapping to the 9 macro-categories)
   ```
   The base path is defined in `params.yaml` (`xview.extracted_data_path: "xView"`). Since
   `xView/` lives inside the repo, it is mounted automatically with the container. With this way
   you must **run the pipeline** (Section 4) to generate the outputs.

### Way B — precomputed DVC store (recommended)

A compressed **DVC store** (`dvcstore.zip`, ~62 GB) is distributed, containing the data **and**
the intermediate pipeline outputs (only the best checkpoint of each training is kept). Download
it from OneDrive:

**https://nubeusc-my.sharepoint.com/:f:/g/personal/lois_fraga_rai_usc_es/IgCr699L1RBuS575_dAzNMYtAZivxxWq-VCx9-yUrzWWLEA?e=iaQW3S**

and unzip it into a local folder (its content is the directory `tfg_lois_ssod-vocabulario-aberto/`):

```bash
unzip dvcstore.zip -d dvc_store      # creates dvc_store/tfg_lois_ssod-vocabulario-aberto/
#  (if you don't have 'unzip': python -m zipfile -e dvcstore.zip dvc_store)
```

Materialization (`dvc pull`) is done **inside the environment** (Section 3), since DVC is
installed there.

---

## 3. Create the environment and materialize the data

Build the Docker image from the repo root:

```bash
docker build -f docker/Dockerfile -t tfg_ssod:latest .
```

Launch the container mounting the repo at `/workspace` and, **if you use Way B**, the DVC store at
`/dvcstore` (the path the repo's DVC configuration already expects):

```bash
docker run -it --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/dvc_store":/dvcstore \
  tfg_ssod:latest /bin/bash
```

With **Way A** the `xView/` folder is already inside the mounted repo at `/workspace`, so you can
omit the `-v "$(pwd)/dvc_store":/dvcstore` mount.

Once inside the container (the conda environment activates by itself), with **Way B** materialize
data and outputs:

```bash
dvc pull
```

The DVC `remote` already points to `/dvcstore/tfg_lois_ssod-vocabulario-aberto`, so **no**
`dvc remote modify` is needed. After `dvc pull`, `xView/`, `results/` and the rest of the outputs
are populated exactly as they were used in the work, without re-running anything (you can pull
only a part with `dvc pull <file.dvc>` or `dvc pull <stage_name>`).

---

## 4. Reproduce the pipeline

With the data in place (Way A) or already materialized (Way B):

```bash
dvc status         # outdated stages
dvc repro          # run whatever is needed to bring everything up to date
```

- A **GPU** is required; **Rex-Omni** inference (an autoregressive model) is the most expensive
  stage.
- To pin the GPU: `CUDA_VISIBLE_DEVICES=0 dvc repro`.
- To reproduce a single stage: `dvc repro <stage_name>` (listed in `dvc.yaml`).
- All parameters (seeds, thresholds, epochs, etc.) live in `params.yaml`.

> With **Way B** everything is materialized, so `dvc repro` re-runs nothing expensive. The two
> figure stages (`generate_charts*`) are `frozen` (they need the full `detection_results.json`
> dumps, not distributed); `dvc repro` skips them. The figures and the comparison table are already
> in the repo (`docs/charts*/`, `results/ssod/comparison_table.csv`).

---

## 5. Results

The final evaluations on `test` (COCO protocol) are written to:

```
results/inference_test_ssod_baseline/metrics.json   # lower bound (Faster10, 10 % labeled)
results/inference_test/metrics.json                 # upper bound (full train)
results/inference_test_ssod_pe/<EXP>/metrics.json   # each SSOD combination
```

where `<EXP>` ∈ {`FT`, `GD`, `RO`, `FT_GD`, `FT_RO`, `FT_GD_RO`, …}. The
`generate_pe_comparison_table` stage gathers all metrics and produces the comparison table and
bar chart under `docs/`. The headline metric is **AP50**.

---

## 6. Tests

```bash
pytest -v tests
```

---

## 7. Repository structure

```
src/             source code (preprocessing, training, inference, ssod, utils)
scripts/         auxiliary and experiment-runner scripts
configs/         model and prompt configuration (prompt sets)
vendor/Rex-Omni/ Rex-Omni wrapper used at inference (third-party code)
tests/           pytest tests
docker/          Dockerfile and entrypoint
dvc.yaml         reproducible pipeline definition
params.yaml      parameters of all stages
environment.yml  conda environment dependencies
```

---

## 8. Configuration

- **`params.yaml`** — single configuration entry point, with one section per phase: `xview`,
  `preprocessing`, `supervised_curve`, `dino`, `rexomni`, `training`, `training_ssod_baseline`,
  `training_ssod_pe`, `pe_policy_ft`, `ssod`, `evaluation`, etc.
- **`configs/models/`** — definition of the base Faster R-CNN (ResNet-50 + FPN v2 with custom
  anchors adapted to the object sizes in xView).
- **`configs/prompts/`** — YAML files with the open-vocabulary prompts; the main one is
  `single_term_prompts.yaml` (five *prompt sets*, one term per class).

---

## 9. Notes and troubleshooting

- The **Grounding DINO** (`IDEA-Research/grounding-dino-base`) and **Rex-Omni**
  (`IDEA-Research/Rex-Omni`) weights are downloaded from HuggingFace on first run; they are not
  included in the repository (requires internet and HuggingFace cache space).
- **Rex-Omni** is vendored under `vendor/Rex-Omni/` and installed in editable mode (needed for the
  score-extraction logic); `environment.yml` does this automatically.
- The CUDA version bundled with PyTorch (11.8) is independent of the system CUDA driver and
  compatible with it; you do not need to install CUDA separately.
- If `unzip` is not installed, use `python -m zipfile -e dvcstore.zip <path>` (the environment
  ships Python).
