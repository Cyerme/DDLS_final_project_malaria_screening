# Malaria Thin‑Smear Screening (MobileNetV2, calibrated) — DDLS final project

This repository contains a reproducible pipeline and a small web app for binary screening of malaria thin‑smear cell crops (**Parasitized** vs **Uninfected**) using **MobileNetV2** with post‑hoc **temperature calibration**, a validation‑selected **operating threshold** (specificity ≥ 0.95 target), **TTA‑based abstention**, **MSP‑based OOD flagging**, and **Grad‑CAM** overlays. The notebook trains the model end‑to‑end and writes all artefacts to `results/`. The web app serves calibrated predictions and explanations using those artefacts; it is a local demo for decision support only.

> **Safety note:** This code is for research and education; it is **not** a medical device or a diagnostic product.

---

## Repository layout

```
DDLS_final_project_malaria_screening/
├── results/                         # Generated artefacts (created by the notebook)
│   ├── abstention_test_summary.json
│   ├── abstention_val_summary.json
│   ├── coverage_accuracy_curve.png
│   ├── ood_metrics_test.json
│   ├── operating_point_val.json     # temperature_T, threshold, target specificity, summary
│   ├── reliability_diagram_test.png
│   ├── reliability_diagram_val.png
│   ├── roc_test.png
│   ├── test_metrics.json
│   ├── trainlog_finetune_tail.csv
│   └── trainlog_head.csv
├── webapp/
│   ├── static/
│   │   └── app.js                   # Front‑end logic
│   ├── index.html                   # Simple UI
│   └── server.py                    # FastAPI backend (loads artefacts from results/)
├── main.ipynb                       # End‑to‑end training, calibration, evaluation
└── requirements.txt
```

---

## Method at a glance

- **Preprocessing:** pad‑to‑square using a border‑ring probe (constant near‑black borders; else reflect); bicubic to **128×128**; ImageNet normalisation.
- **Backbone:** MobileNetV2 with a single‑logit head.
- **Training:** head warm‑up (frozen backbone) then fine‑tune the last **N=12** backbone blocks; AdamW with differential learning rates; BCEWithLogits with label smoothing.
- **Calibration:** single temperature \(T\) fitted on validation by minimising NLL; probabilities use \(\sigma(z/T)\).
- **Operating point:** threshold chosen on validation to maximise sensitivity subject to **specificity ≥ 0.95**; held fixed for test and the web app.
- **Uncertainty & abstention:** mild TTA (N=8) gives mean/std of \(p\); abstain if \(|\bar p-0.5| < \delta\) or \(\sigma_p > \sigma_{\rm thr}\) (picked on validation for a target coverage).
- **OOD:** Maximum Softmax Probability \( \max(p,1-p) \); \(\tau\) set to the 5th percentile of ID MSP on validation (≈95% TPR on ID).
- **Interpretability:** Grad‑CAM from the last conv block; overlays at \(\alpha=0.35\).
- **Figures and logs:** saved under `results/` and consumed by the web app.

---

## Initiation

1. Add the project folder to your VS Code workspace.  
2. Download the dataset:
   - Go to [this page](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)
   - Download the publicly available **"NLM-Falciparum&Uninfected-Thin-193Patients"** dataset, provided courtesy of the U.S. National Library of Medicine
   - Unzip the downloaded archive into a convenient location (for example, `C:\Projects\cell images`)

3. Open a terminal in VS Code (PowerShell recommended) and run the following commands in order:

> **Note:** Replace the path in the first command with the location where you unzipped `cell images`.

```powershell
# 1) Go to the project data folder
cd "C:\[path to where you unzipped 'cell images']"

# 2) Create a Python 3.11 virtual environment
py -3.11 -m venv .venv

# 3) Activate the environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# 4) Upgrade pip
python -m pip install --upgrade pip

# 5) Install PyTorch (CUDA 12.1 build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6) Install the packages
pip install numpy pandas pillow scikit-learn matplotlib tqdm jupyter ipykernel fastapi uvicorn==0.30.* starlette python-multipart

# 7) Register the Jupyter kernel
python -m ipykernel install --user --name ddls-local --display-name "Python 3.11 (.venv) DDLS"
```

4. In VS Code, select the Jupyter kernel: **Python 3.11 (.venv) DDLS**.

5. Change all instances of **`[path_placeholder]`** in the notebook to the path where you unzipped **`cell images`**.

---

## Reproducing the results

### 1) Run the notebook

- Open **`main.ipynb`** in VS Code; choose the **Python 3.11 (.venv) DDLS** kernel.  
- Run the notebook from top to bottom. The notebook performs:
  - dataset indexing and group‑aware splits,
  - preprocessing parity checks,
  - head warm‑up and tail fine‑tuning,
  - temperature fitting and operating‑point selection on validation,
  - final test metrics and figures,
  - TTA‑based abstention selection and summaries,
  - MSP OOD thresholding and test metrics,
  - Grad‑CAM panels.
- On success, among many more, the following files appear in **`results/`**:
  - `operating_point_val.json` — `temperature_T`, threshold, and a compact summary,
  - `reliability_diagram_val.png` and `reliability_diagram_test.png`,
  - `roc_test.png`, `test_metrics.json`,
  - `abstention_val_summary.json`, `abstention_test_summary.json`, `coverage_accuracy_curve.png`,
  - `ood_metrics_test.json`,
  - `trainlog_head.csv`, `trainlog_finetune_tail.csv`.

### 2) Launch the local web app

From the folder `webapp/` in repository root:

```powershell
# Activate the environment if not already active
.\.venv\Scripts\Activate.ps1

# Start the server (serves results/ figures too)
python server.py
```

Then open **http://127.0.0.1:8000** in your browser.  
Pick one or more images; click **Run analysis**. Each card shows: calibrated \(p\), decision at the validation threshold, abstention/OOD flags, the preprocessed input, and a Grad‑CAM overlay. A “Download JSON” link gives the raw response for audit.

> The server prints probabilities are calibrated by `temperature_T` from `operating_point_val.json`.

---

## Notes on paths and parity

- The dataset is **not** part of the repository; point the notebook to your local **`cell images`** folder by replacing `[path_placeholder]`.  
- The web app loads weights and thresholds from `results/` and applies the **same preprocessing** and **same temperature** as the notebook; decisions match notebook behaviour.  
- Abstention and OOD flags are computed exactly as in the notebook (TTA with N=8; MSP with \(\tau\) chosen on validation).

---

## Troubleshooting
 
- **CPU only:** the code runs on CPU; it prints `device: cpu`. Training will be slower; the web app remains usable.  
- **Pillow resampling import error:** upgrade `Pillow` (the repo pins a recent version); the server includes a small shim for older versions.  
- **Missing artefacts at server start:** run the notebook to regenerate `results/`; `server.py` raises a clear error if, for example, `operating_point_val.json` is absent.

---

## Credits

- Dataset: NIH/NLM — **“NLM‑Falciparum & Uninfected — Thin — 193 Patients.”**  
- Backbone: PyTorch **torchvision.models.mobilenet_v2**.  
- Grad‑CAM: implemented via hooks on the last conv block.

---

## Licence and use

Code is provided for research and education. Do **not** use this repository to make clinical decisions.