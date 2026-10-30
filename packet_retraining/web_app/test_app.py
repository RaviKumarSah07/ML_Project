import unittest
from unittest.mock import MagicMock, patch

from .app import CSRF_TOKEN, app, model_service
from .capture import analyze_pcap
from .flow import FlowAggregator, parse_tshark_line


class WebAppTests(unittest.TestCase):
    def test_manual_prediction(self):
        client = app.test_client()
        fields = {
            "proto": "tcp", "dur": 0.121478, "spkts": 6, "dpkts": 4,
            "sbytes": 258, "dbytes": 172, "sttl": 252, "dttl": 254,
        }
        response = client.post(
            "/api/predict", json={"fields": fields},
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json["label"], {"Attack", "Normal"})
        self.assertGreaterEqual(response.json["score"], 0)
        self.assertLessEqual(response.json["score"], 1)

    def test_packet_flow_directions(self):
        forward = "1.0\t10.0.0.1\t10.0.0.2\t\t\t6\t\t60\t\t64\t\t50000\t443\t\t"
        reverse = "1.1\t10.0.0.2\t10.0.0.1\t\t\t6\t\t52\t\t63\t\t443\t50000\t\t"
        aggregator = FlowAggregator()
        aggregator.add(parse_tshark_line(forward))
        aggregator.add(parse_tshark_line(reverse))
        flows = aggregator.finish()
        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0].features()["spkts"], 1)
        self.assertEqual(flows[0].features()["dpkts"], 1)
        self.assertEqual(flows[0].features()["sbytes"], 60)
        self.assertEqual(flows[0].features()["dbytes"], 52)

    def test_pcap_processing_with_packet_lines(self):
        forward = "1.0\t10.0.0.1\t10.0.0.2\t\t\t6\t\t60\t\t64\t\t50000\t443\t\t\n"
        reverse = "1.1\t10.0.0.2\t10.0.0.1\t\t\t6\t\t52\t\t63\t\t443\t50000\t\t\n"
        fake_process = MagicMock()
        fake_process.stdout = [forward, reverse]
        fake_process.wait.return_value = 0
        fake_process.poll.return_value = 0
        events = []
        with patch("web_app.capture.subprocess.Popen", return_value=fake_process):
            summary = analyze_pcap("fake-tshark", "fake.pcap", model_service, events.append)
        self.assertEqual(summary["packets"], 2)
        self.assertEqual(summary["flows"], 1)
        self.assertEqual(summary["attacks"] + summary["normal"], 1)
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
