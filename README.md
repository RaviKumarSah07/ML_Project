# Network Intrusion Detection System

This repository contains a complete binary network intrusion detection project built with the UNSW-NB15 dataset. It compares an Artificial Neural Network and a Random Forest implemented from scratch, then retrains the models with a smaller feature set for use in a Flask web application.

The models classify each network flow as either **Normal (0)** or **Attack (1)**. Ready-made machine-learning classifiers are not used. The ANN, decision trees, Random Forest, preprocessing, predictions, and evaluation metrics are implemented in the project source code with NumPy and pandas.

## Why there are two folders

The two folders serve different but connected purposes:

| Folder | Purpose |
|---|---|
| [`ml_project`](ml_project/) | The main academic experiment. It uses the broad UNSW-NB15 feature set, transforms 42 usable raw predictors into 79 model inputs, and compares the from-scratch ANN and Random Forest. |
| [`packet_retraining`](packet_retraining/) | The application-compatible experiment. It retrains the same model types using eight flow fields that can be entered manually or approximated from packets, and contains the Flask web application. |

The full academic model cannot be used directly with ordinary captured packets because many UNSW-NB15 columns were created by the original Argus, Bro, and custom feature-generation process. The web application therefore uses a separately retrained model with these eight obtainable fields:

- `proto`
- `dur`
- `spkts`
- `dpkts`
- `sbytes`
- `dbytes`
- `sttl`
- `dttl`

`proto` becomes 16 one-hot encoded columns: the 15 most common protocols and an `__OTHER__` category. The remaining seven fields are numeric, so the eight raw fields become 23 model inputs.

## Repository structure

```text
ML_Project/
├── README.md
├── ml_project/
│   ├── main.py
│   ├── train.py
│   ├── preprocessing.py
│   ├── ann.py
│   ├── random_forest.py
│   ├── evaluate.py
│   ├── README.md
│   ├── data/
│   └── saved_artifacts/
└── packet_retraining/
    ├── train.py
    ├── packet_preprocessing.py
    ├── ann.py
    ├── random_forest.py
    ├── evaluate.py
    ├── requirements.txt
    ├── README.md
    ├── saved_artifacts/
    └── web_app/
        ├── app.py
        ├── model_service.py
        ├── capture.py
        ├── flow.py
        ├── test_app.py
        ├── templates/
        └── static/
```

## Main experiment

The main experiment uses the provided UNSW-NB15 training and testing CSV files.

- Training records: 175,341
- Held-out testing records: 82,332
- Raw usable predictors: 42
- Processed model inputs: 79
- Models: ANN and Random Forest implemented from scratch
- Selected model: ANN, based on the higher held-out test F1-score

| Metric | ANN | Random Forest |
|---|---:|---:|
| Accuracy | 85.63% | 85.29% |
| Attack precision | 79.82% | 79.36% |
| Attack recall | 98.91% | 99.06% |
| F1-score | 88.34% | 88.12% |
| Normal recall | 69.35% | 68.43% |

See [`ml_project/README.md`](ml_project/README.md) for preprocessing details, model configurations, results, and artifact loading.

## Application-compatible model

The reduced-feature pipeline selects its deployment model using F1-score on a stratified validation split. The ANN is selected with a validation F1-score of 95.32%, narrowly above the Random Forest score of 95.27%.

On the held-out testing CSV, the selected reduced-feature ANN records:

- Accuracy: 80.91%
- Attack precision: 74.35%
- Attack recall: 99.73%
- F1-score: 85.19%
- Normal recall: 57.85%

The model strongly favors detecting attacks, which also creates many false alerts for normal traffic. PCAP and live predictions are experimental because the application's packet-to-flow conversion approximates, but does not exactly reproduce, the original UNSW-NB15 feature-generation process.

See [`packet_retraining/README.md`](packet_retraining/README.md) for the complete application guide.

## Run the main ML experiment

From the repository root:

```powershell
cd ml_project
python -m pip install numpy pandas matplotlib
python main.py
```

The two dataset files must be present under `ml_project/data`.

## Retrain the application model

```powershell
cd packet_retraining
python -m pip install -r requirements.txt
python train.py
```

The retraining script reads the dataset from `../ml_project/data` and saves its own artifacts under `packet_retraining/saved_artifacts`.

## Run the Flask web application

```powershell
cd packet_retraining
python -m web_app.app
```

Open <http://127.0.0.1:5000> in a browser.

The application supports:

- Manual eight-field prediction
- PCAP upload and analysis
- Experimental live network monitoring

Manual prediction works with the Python requirements. PCAP and live modes also require [Wireshark with TShark](https://www.wireshark.org/download.html). Live capture on Windows commonly requires [Npcap](https://npcap.com/).

## Run the web application tests

```powershell
cd packet_retraining
python -m unittest web_app.test_app -v
```

## Important limitation

The saved scores measure performance on UNSW-NB15 CSV records. They do not prove accuracy on a new live network. Validate the system with correctly labeled traffic from the intended environment before considering operational use.
