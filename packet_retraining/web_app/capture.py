import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from .flow import FlowAggregator, TSHARK_FIELDS, parse_tshark_line


MAX_PACKETS = 200000
MAX_FLOWS = 10000


def find_tshark():
    configured = os.environ.get("TSHARK_PATH")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("tshark")
    if found:
        return found
    if sys.platform == "win32":
        for directory in ("C:/Program Files/Wireshark", "C:/Program Files (x86)/Wireshark"):
            candidate = Path(directory) / "tshark.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def tshark_fields_command(tshark, source, value):
    command = [tshark, "-n", "-l", "-T", "fields", "-E", "occurrence=f"]
    for field in TSHARK_FIELDS:
        command.extend(["-e", field])
    command.extend(["-Y", "ip or ipv6"])
    if source == "file":
        command.extend(["-r", str(value)])
    else:
        command.extend(["-i", str(value)])
    return command


def list_interfaces(tshark):
    if not tshark:
        return []
    completed = subprocess.run(
        [tshark, "-D"], capture_output=True, text=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Could not list capture interfaces")
    interfaces = []
    for line in completed.stdout.splitlines():
        match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if match:
            interfaces.append({"id": match.group(1), "name": match.group(2)})
    return interfaces


def analyze_pcap(tshark, file_path, model_service, record_event):
    if not tshark:
        raise RuntimeError("TShark is not installed. Install Wireshark with TShark to analyze PCAP files.")
    aggregator = FlowAggregator()
    results = []
    packets = 0
    ignored = 0
    normal = 0
    attacks = 0
    flows = 0
    command = tshark_fields_command(tshark, "file", file_path)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        for line in process.stdout:
            packets += 1
            if packets > MAX_PACKETS:
                raise ValueError(f"Capture exceeds the {MAX_PACKETS:,}-packet limit")
            packet = parse_tshark_line(line)
            if packet is None:
                ignored += 1
                continue
            for flow in aggregator.add(packet):
                flows += 1
                if flows > MAX_FLOWS:
                    raise ValueError(f"Capture exceeds the {MAX_FLOWS:,}-flow limit")
                result = model_service.predict(flow.features(), "pcap", flow.endpoints())
                attacks += int(result["attack"])
                normal += int(not result["attack"])
                record_event(result)
                if len(results) < 100 or result["attack"]:
                    results.append(result)
        if process.wait(timeout=10) != 0:
            raise RuntimeError("TShark could not read this capture file")
        for flow in aggregator.finish():
            flows += 1
            if flows > MAX_FLOWS:
                raise ValueError(f"Capture exceeds the {MAX_FLOWS:,}-flow limit")
            result = model_service.predict(flow.features(), "pcap", flow.endpoints())
            attacks += int(result["attack"])
            normal += int(not result["attack"])
            record_event(result)
            if len(results) < 100 or result["attack"]:
                results.append(result)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    results.sort(key=lambda item: (item["attack"], item["score"]), reverse=True)
    return {
        "packets": packets,
        "ignored": ignored,
        "flows": flows,
        "attacks": attacks,
        "normal": normal,
        "results": results[:100],
    }


class LiveCapture:
    def __init__(self, model_service, tshark):
        self.model_service = model_service
        self.tshark = tshark
        self.lock = threading.Lock()
        self.events = deque(maxlen=80)
        self.state = "idle"
        self.error = None
        self.interface = None
        self.process = None
        self.reader = None
        self.sweeper = None
        self.stop_event = threading.Event()
        self.aggregator = FlowAggregator()
        self.packets = 0
        self.total = 0
        self.attacks = 0
        self.normal = 0

    def record_event(self, result):
        with self.lock:
            self.events.appendleft(result)
            self.total += 1
            self.attacks += int(result["attack"])
            self.normal += int(not result["attack"])

    def status(self):
        with self.lock:
            return {
                "state": self.state,
                "error": self.error,
                "interface": self.interface,
                "packets": self.packets,
                "total": self.total,
                "attacks": self.attacks,
                "normal": self.normal,
                "events": list(self.events),
                "tshark_available": bool(self.tshark),
            }

    def start(self, interface):
        if not self.tshark:
            raise RuntimeError("TShark is not installed")
        valid_ids = {item["id"] for item in list_interfaces(self.tshark)}
        if interface not in valid_ids:
            raise ValueError("Choose a valid capture interface")
        with self.lock:
            if self.state == "running":
                raise ValueError("Live capture is already running")
            self.stop_event = threading.Event()
            self.aggregator = FlowAggregator()
            self.interface = interface
            self.error = None
            self.process = subprocess.Popen(
                tshark_fields_command(self.tshark, "live", interface),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            self.state = "running"
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.sweeper = threading.Thread(target=self._sweep, daemon=True)
            self.reader.start()
            self.sweeper.start()

    def _score_flows(self, flows):
        for flow in flows:
            try:
                result = self.model_service.predict(flow.features(), "live", flow.endpoints())
                self.record_event(result)
            except (ValueError, KeyError) as error:
                with self.lock:
                    self.error = str(error)

    def _read(self):
        try:
            for line in self.process.stdout:
                packet = parse_tshark_line(line)
                if packet is None:
                    continue
                with self.lock:
                    self.packets += 1
                    finished = self.aggregator.add(packet)
                self._score_flows(finished)
        finally:
            with self.lock:
                finished = self.aggregator.finish()
                stopped_by_user = self.stop_event.is_set()
                if not stopped_by_user:
                    self.state = "error"
                    self.error = "Capture stopped. Check the selected interface and capture permissions."
                else:
                    self.state = "stopped"
            self._score_flows(finished)

    def _sweep(self):
        while not self.stop_event.wait(1):
            with self.lock:
                if self.state != "running":
                    return
                finished = self.aggregator.expire(time.time())
            self._score_flows(finished)

    def stop(self):
        with self.lock:
            if self.state != "running":
                return
            self.stop_event.set()
            process = self.process
            reader = self.reader
        if process.poll() is None:
            process.terminate()
        reader.join(timeout=5)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
