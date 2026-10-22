# Big Data: Predictive Maintenance Analysis

This project explores the **AI4I 2020 Predictive Maintenance Dataset**. Its Python workflow inspects and analyzes the data, visualizes temperature differences, and trains a Random Forest classifier to predict machine failures.

## Project structure

```text
.
├── data/
│   ├── ai4i2020.csv       # Project dataset
│   └── README.md          # Dataset source and attribution
└── python/
    ├── download_data.py   # Downloads the dataset from UCI
    ├── inspect_data.py    # Runs the analysis and model evaluation
    └── requirements.txt   # Python dependencies
```

## Dataset

The project uses the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from the UCI Machine Learning Repository.

- DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Local file: `data/ai4i2020.csv`

The dataset is included in this repository. To download a fresh copy, see [Downloading the dataset](#downloading-the-dataset).

## Requirements

- Python 3.10 or later
- pip

The project dependencies are listed in `python/requirements.txt`: pandas, Matplotlib, and scikit-learn.

## Setup and run

Run these commands from the repository root. The following example uses PowerShell on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r python\requirements.txt
python python\inspect_data.py
```

On macOS or Linux, create and activate the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r python/requirements.txt
python python/inspect_data.py
```

The script reads `data/ai4i2020.csv` relative to the repository, so it can be run from any working directory. It prints data previews, dimensions, data types, missing-value and duplicate checks, failure statistics, temperature differences, and average torque by machine type. It then displays a temperature-difference histogram and evaluates a Random Forest model.

The model uses air temperature, process temperature, rotational speed, torque, and tool wear to predict the `Machine failure` label. It uses a stratified 80/20 train-test split and compares accuracy, precision, and recall against a baseline that always predicts no failure. A confusion matrix is also displayed. The charts open in interactive windows; close each window to continue the analysis.

## Downloading the dataset

If the CSV is missing or you want to replace it with a fresh download, run:

```powershell
python python\download_data.py
```

The script fetches the dataset archive from UCI and saves `data/ai4i2020.csv`. An alternate destination can be specified with `--output`:

```powershell
python python\download_data.py --output path\to\ai4i2020.csv
```

The download requires an internet connection.
