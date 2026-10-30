import atexit
import hmac
import json
import secrets
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from .capture import LiveCapture, analyze_pcap, find_tshark, list_interfaces
from .model_service import ModelService


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "saved_artifacts"
CSRF_TOKEN = secrets.token_urlsafe(32)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
model_service = ModelService(ARTIFACTS_DIR)
capture = LiveCapture(model_service, find_tshark())
atexit.register(capture.stop)
test_metrics = json.loads((ARTIFACTS_DIR / "evaluation_results.json").read_text(encoding="utf-8"))


@app.before_request
def check_token():
    if request.method == "POST":
        supplied = request.headers.get("X-CSRF-Token", "")
        if not hmac.compare_digest(supplied, CSRF_TOKEN):
            return jsonify({"error": "Refresh the page and try again"}), 403


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({"error": "Capture file must be 20 MB or smaller"}), 413


@app.get("/")
def index():
    return render_template(
        "index.html",
        csrf_token=CSRF_TOKEN,
        model_name=model_service.model_name,
        model_metrics=test_metrics[model_service.model_name],
    )


@app.get("/api/config")
def config():
    interfaces = []
    interface_error = None
    if capture.tshark:
        try:
            interfaces = list_interfaces(capture.tshark)
        except (OSError, RuntimeError) as error:
            interface_error = str(error)
    return jsonify({
        "model": model_service.model_name,
        "tshark_available": bool(capture.tshark),
        "interfaces": interfaces,
        "interface_error": interface_error,
    })


@app.get("/api/status")
def status():
    return jsonify(capture.status())


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return jsonify({"error": "Provide the eight flow fields"}), 400
    try:
        result = model_service.predict(fields, "manual")
    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400
    capture.record_event(result)
    return jsonify(result)


@app.post("/api/pcap")
def upload_pcap():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a PCAP or PCAPNG file"}), 400
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in {".pcap", ".pcapng"}:
        return jsonify({"error": "File must end in .pcap or .pcapng"}), 400
    if not capture.tshark:
        return jsonify({"error": "TShark is required for PCAP analysis"}), 503
    try:
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / f"capture{suffix}"
            uploaded.save(file_path)
            summary = analyze_pcap(
                capture.tshark, file_path, model_service, capture.record_event
            )
    except (OSError, RuntimeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(summary)


@app.post("/api/live/start")
def start_live():
    payload = request.get_json(silent=True) or {}
    interface = str(payload.get("interface", ""))
    try:
        capture.start(interface)
    except (OSError, RuntimeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(capture.status())


@app.post("/api/live/stop")
def stop_live():
    try:
        capture.stop()
    except (OSError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(capture.status())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
