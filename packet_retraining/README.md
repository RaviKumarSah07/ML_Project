# Packet-feature model retraining

This folder retrains the existing from-scratch ANN and Random Forest on eight
UNSW-NB15 flow fields and provides a local Flask service for manual, PCAP, and
live analysis. The original `../ml_project` source files and saved models
remain separate.

## Folder structure

```text
project/
├── ml_project/
│   ├── main.py
│   ├── train.py
│   ├── preprocessing.py
│   ├── ann.py
│   ├── random_forest.py
│   ├── evaluate.py
│   ├── README.md
│   ├── data/
│   │   ├── UNSW_NB15_training-set.csv
│   │   └── UNSW_NB15_testing-set.csv
│   └── saved_artifacts/
│       ├── ann_weights.pkl
│       ├── random_forest.pkl
│       ├── preprocessor.pkl
│       ├── evaluation_results.json
│       ├── model_comparison.png
│       └── selected_model.txt
└── packet_retraining/
    ├── ann.py
    ├── random_forest.py
    ├── evaluate.py
    ├── packet_preprocessing.py
    ├── train.py
    ├── requirements.txt
    ├── README.md
    ├── dashboard_preview.png
    ├── dashboard_alert_preview.png
    ├── web_app/
    │   ├── __init__.py
    │   ├── app.py
    │   ├── capture.py
    │   ├── flow.py
    │   ├── model_service.py
    │   ├── test_app.py
    │   ├── templates/
    │   │   └── index.html
    │   └── static/
    │       ├── favicon.svg
    │       ├── hero-network.png
    │       ├── style.css
    │       └── app.js
    └── saved_artifacts/
        ├── ann_weights.pkl
        ├── random_forest.pkl
        ├── preprocessor.pkl
        ├── feature_schema.json
        ├── validation_results.json
        ├── evaluation_results.json
        ├── model_comparison.png
        └── selected_model.txt
```

## What each file does

- `packet_preprocessing.py` accepts the eight fields, keeps the 15 most common
  training protocols as individual categories, groups other protocols, applies
  `log1p` to duration and counts, and scales numeric values using only the
  training split. It accepts unlabeled rows for later prediction.
- `ann.py` implements the neural network using NumPy.
- `random_forest.py` implements decision trees and the forest using NumPy.
- `evaluate.py` calculates classification metrics and draws the comparison chart.
- `train.py` splits the training CSV, fits preprocessing, trains both models,
  chooses the winner by validation F1, and then evaluates both on the held-out
  test CSV. No ready-made classifier is used.

## Input fields

| Field | Intended capture calculation |
|---|---|
| `proto` | Network or transport protocol |
| `dur` | Time between first and last packet of a flow, in seconds |
| `spkts`, `dpkts` | Packet counts in the first-seen direction and reverse direction |
| `sbytes`, `dbytes` | Byte counts in those two directions |
| `sttl`, `dttl` | TTL or hop-limit from the first packet in each direction |

The capture adapter is implemented in `web_app/flow.py` and `web_app/capture.py`.
UNSW-NB15 used Argus, Bro, and custom algorithms to create its CSV flow fields.
Without the corresponding raw PCAP and matching flow records, we cannot prove
that a new packet extractor will calculate exactly the same flow boundaries or
byte lengths. These models are evaluated on UNSW-NB15 CSV records only; their
accuracy on local traffic is unknown.

## Run

From this folder:

```powershell
python -m pip install numpy pandas matplotlib
python train.py
```

`train.py` reads the two CSV files from `../ml_project/data` and writes only to
`packet_retraining/saved_artifacts`.

## Results

The stratified validation split is 10% of the original training CSV. The
preprocessor is fit on the remaining 90%. Selection uses validation F1. The
original testing CSV is used only after selection.

| Metric | ANN validation | Forest validation | ANN test | Forest test |
|---|---:|---:|---:|---:|
| Accuracy | 0.9332 | 0.9324 | 0.8091 | 0.8080 |
| Attack precision | 0.9118 | 0.9097 | 0.7435 | 0.7414 |
| Attack recall | 0.9984 | 1.0000 | 0.9973 | 1.0000 |
| F1 | **0.9532** | 0.9527 | 0.8519 | 0.8515 |
| Normal recall | 0.7943 | 0.7884 | 0.5785 | 0.5728 |

**Selected model: ANN.** The validation F1 margin is small. On the held-out
test set, the ANN classified 15,596 of 37,000 normal flows as attacks. The
test score is much lower than the validation score, so these results should
not be presented as proven live-network performance.

The files under `saved_artifacts` contain both trained models for the academic
comparison. The web app reads `selected_model.txt` and loads only that model
together with `preprocessor.pkl`.

## Run the local web service

From `packet_retraining`:

```powershell
python -m pip install -r requirements.txt
python -m web_app.app
```

Open `http://127.0.0.1:5000` in your browser. The service listens on your own
computer only. The Manual flow tab works with the Python dependencies above.
It also has two examples taken from the held-out test CSV for demonstrating
the normal and attack visuals.

PCAP upload and live monitoring require [Wireshark with TShark](https://www.wireshark.org/download.html).
On Windows, install [Npcap](https://wiki.wireshark.org/CaptureSetup/CaptureSupport)
for live packet capture. The app looks for `tshark` on PATH and in the usual
Wireshark installation folders. If installed elsewhere, set `TSHARK_PATH` to
the full path of `tshark.exe` before starting the app:

```powershell
$env:TSHARK_PATH = "C:\Program Files\Wireshark\tshark.exe"
python -m web_app.app
```

In the Live monitor tab, choose a network interface and click Start monitoring.
The app does not capture traffic before that click. Click Stop capture to end
the session. Capture permissions depend on the local Wireshark/Npcap setup.

### Web files in simple terms

- `web_app/app.py` creates the Flask server and its routes. Flask routes play
  the same role as Express routes in Node.js. The routes serve the page and
  JSON endpoints for prediction, PCAP upload, and live capture controls.
- `web_app/model_service.py` loads the model named by `selected_model.txt` and
  the saved preprocessor. It scores one flow at a time.
- `web_app/capture.py` starts TShark, reads packet fields, and manages capture
  state. TShark decodes packets; the flow calculations are in our Python code.
- `web_app/flow.py` groups packets by source/destination address, ports, and
  protocol, then calculates the eight model input fields.
- `web_app/templates/index.html` is the page structure. Flask renders it as a
  template, similar to sending an HTML view from an Express route.
- `web_app/static/style.css` creates the restrained editorial dashboard and
  focused visual alerts. `hero-network.png` is the original network artwork.
- `web_app/static/app.js` switches tabs, calls the JSON routes, and updates
  the page without reloading.
- `web_app/test_app.py` checks manual prediction and flow processing without
  capturing real traffic.

There is no database model or separate `urls.py` in this Flask app. Prediction
uses the saved ML artifacts, and routes are declared in `app.py`.

### Capture behavior and limits

The first packet of a connection sets the source direction. The reverse
direction supplies `dpkts`, `dbytes`, and `dttl`. IPv4 bytes use `ip.len`;
IPv6 bytes use `ipv6.plen + 40`. Flows are scored after three idle seconds or
15 seconds of activity, then a continuing connection starts a new window.
Non-IP packets are ignored. These rules are approximations of the original
UNSW-NB15 flow-generation process, so PCAP and live predictions are marked
experimental in the interface.

Uploads are limited to 20 MB, 200,000 displayed IP packets, and 10,000 flows.
The app deletes the temporary uploaded file after analysis and keeps recent
results in memory for the current server session.

To run the web checks:

```powershell
python -m unittest web_app.test_app -v
```
