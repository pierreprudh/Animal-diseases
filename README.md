# Animal Condition Classifier

End-to-end workflow for predicting whether an animal's symptoms indicate a dangerous health condition using Kaggle's [Animal Condition dataset](https://www.kaggle.com/datasets/gracehephzibahm/animal-disease?resource=download).

## Project layout

- `Animal_condition.ipynb` – reproducible notebook that downloads the dataset, explores it, trains/evaluates models, and saves the champion pipeline plus metrics.
- `data/raw/` – cached CSV downloaded via `kagglehub` (create the folder before dropping the file).
- `models/` – serialized pipelines (e.g., `animal_condition_log_reg.joblib`).
- `reports/` – evaluation artifacts such as `log_reg_metrics.json`.
- `requirements.txt` – pinned dependencies for running the notebook.

## Environment setup

1. Install Python 3.13 with [pyenv](https://github.com/pyenv/pyenv) :
   ```bash
   pyenv install --list          # optional view of available versions
   pyenv install 3.13.0
   pyenv virtualenv 3.13.0 animalcondenv
   pyenv local animalcondenv     # sets `.python-version`
   ```
2. Install dependencies :
   ```bash
   pip install -r requirements.txt
   ```

## Dataset download

The notebook expects `data/raw/animal_condition.csv`.

- **Automatic (preferred) :** the first notebook cell uses `kagglehub` to pull `gracehephzibahm/animal-disease` and caches the CSV locally. Ensure your Kaggle API token is configured if prompted.
- **Manual fallback :**
  1. Download the dataset zip from Kaggle.
  2. Extract `animal_condition.csv`.
  3. Move it under `data/raw/`:
     ```bash
     mkdir -p data/raw
     mv /path/to/animal_condition.csv data/raw/
     ```

## Running the workflow

1. Launch Jupyter (or VS Code, JupyterLab, etc.) :
   ```bash
   jupyter notebook Animal_condition.ipynb
   ```
2. Execute each cell sequentially. The notebook will :
   - explore missing values and target imbalance,
   - engineer the number of reported symptoms per record,
   - compare logistic regression vs. random forest via cross-validation,
   - fit/evaluate the best pipeline on a held-out stratified test set,
   - save the fitted pipeline and metrics for reuse.

Outputs land in :

- `models/animal_condition_log_reg.joblib`
- `reports/log_reg_metrics.json`

## Making predictions from Python

The final notebook cell exposes a `predict_condition` helper that mirrors the training preprocessing steps. Once you've executed the notebook, call it from the same kernel (or copy the helper into your own module) to score new cases :

```python
payload = {
    "AnimalName": "Buffalo",
    "Symptoms1": "Fever",
    "Symptoms2": "Diarrhea",
    "Symptoms3": "Weight loss",
    "Symptoms4": "Weakness",
    "Symptoms5": "Pain",
}

predict_condition(payload)
```

Alternatively, load `models/animal_condition_log_reg.joblib` with `joblib.load` and feed it the canonical feature frame returned by `prepare_features_from_raw` from the notebook.

## Troubleshooting

- **Cannot reach Kaggle :** download the CSV manually and drop it into `data/raw/` as described above. The rest of the notebook will continue to work offline.
- **Missing dependencies :** rerun `pip install -r requirements.txt` inside your `animalcondenv` virtual environment.
- **Old artifacts :** delete the contents of `models/` and `reports/` before rerunning if you want a clean slate.
