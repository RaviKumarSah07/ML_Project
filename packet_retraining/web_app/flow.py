from dataclasses import dataclass


TSHARK_FIELDS = [
    "frame.time_epoch", "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
    "ip.proto", "ipv6.nxt", "ip.len", "ipv6.plen", "ip.ttl", "ipv6.hlim",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
]

PROTOCOL_NAMES = {
    1: "icmp", 4: "ipv4", 6: "tcp", 17: "udp", 41: "ipv6",
    47: "gre", 50: "esp", 51: "ah", 58: "icmpv6", 89: "ospf",
    132: "sctp",
}


def _number(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Packet:
    timestamp: float
    source: str
    destination: str
    source_port: int
    destination_port: int
    protocol: str
    byte_count: int
    ttl: int


def parse_tshark_line(line):
    values = line.rstrip("\r\n").split("\t")
    if len(values) != len(TSHARK_FIELDS):
        return None
    try:
        timestamp = float(values[0])
    except ValueError:
        return None

    source = values[1] or values[3]
    destination = values[2] or values[4]
    if not source or not destination:
        return None

    protocol_number = _number(values[5] or values[6], -1)
    if values[11] or values[12]:
        protocol = "tcp"
        source_port = _number(values[11])
        destination_port = _number(values[12])
    elif values[13] or values[14]:
        protocol = "udp"
        source_port = _number(values[13])
        destination_port = _number(values[14])
    else:
        protocol = PROTOCOL_NAMES.get(protocol_number, str(protocol_number))
        source_port = 0
        destination_port = 0

    if values[7]:
        byte_count = _number(values[7], -1)
        ttl = _number(values[9])
    elif values[8]:
        byte_count = _number(values[8], -1) + 40
        ttl = _number(values[10])
    else:
        return None
    if byte_count <= 0:
        return None

    return Packet(
        timestamp, source, destination, source_port, destination_port,
        protocol, byte_count, ttl,
    )


@dataclass
class Flow:
    source: str
    destination: str
    source_port: int
    destination_port: int
    protocol: str
    first_seen: float
    last_seen: float
    source_packets: int = 0
    destination_packets: int = 0
    source_bytes: int = 0
    destination_bytes: int = 0
    source_ttl: int = 0
    destination_ttl: int = 0

    def add(self, packet, forward):
        self.last_seen = max(self.last_seen, packet.timestamp)
        if forward:
            if self.source_packets == 0:
                self.source_ttl = packet.ttl
            self.source_packets += 1
            self.source_bytes += packet.byte_count
        else:
            if self.destination_packets == 0:
                self.destination_ttl = packet.ttl
            self.destination_packets += 1
            self.destination_bytes += packet.byte_count

    def features(self):
        return {
            "proto": self.protocol,
            "dur": round(max(0.0, self.last_seen - self.first_seen), 6),
            "spkts": self.source_packets,
            "dpkts": self.destination_packets,
            "sbytes": self.source_bytes,
            "dbytes": self.destination_bytes,
            "sttl": self.source_ttl,
            "dttl": self.destination_ttl,
        }

    def endpoints(self):
        return {
            "source": self.source,
            "destination": self.destination,
            "source_port": self.source_port,
            "destination_port": self.destination_port,
        }


class FlowAggregator:
    def __init__(self, idle_seconds=3, max_age_seconds=15, max_active=5000):
        self.idle_seconds = idle_seconds
        self.max_age_seconds = max_age_seconds
        self.max_active = max_active
        self.flows = {}
        self.last_sweep = float("-inf")

    def add(self, packet):
        finished = []
        if packet.timestamp - self.last_sweep >= 1:
            finished = self.expire(packet.timestamp)
            self.last_sweep = packet.timestamp

        key = (
            packet.source, packet.destination, packet.source_port,
            packet.destination_port, packet.protocol,
        )
        reverse = (
            packet.destination, packet.source, packet.destination_port,
            packet.source_port, packet.protocol,
        )
        if key in self.flows:
            flow = self.flows[key]
            forward = True
        elif reverse in self.flows:
            flow = self.flows[reverse]
            forward = False
        else:
            flow = Flow(
                packet.source, packet.destination, packet.source_port,
                packet.destination_port, packet.protocol,
                packet.timestamp, packet.timestamp,
            )
            self.flows[key] = flow
            forward = True
        flow.add(packet, forward)

        if len(self.flows) > self.max_active:
            oldest_key = min(self.flows, key=lambda item: self.flows[item].last_seen)
            finished.append(self.flows.pop(oldest_key))
        return finished

    def expire(self, now):
        finished = []
        for key, flow in list(self.flows.items()):
            if now - flow.last_seen >= self.idle_seconds or now - flow.first_seen >= self.max_age_seconds:
                finished.append(self.flows.pop(key))
        return finished

    def finish(self):
        finished = list(self.flows.values())
        self.flows.clear()
        return finished
