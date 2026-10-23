"""Writes small synthetic files in NSL-KDD format (for smoke tests only; NOT real data)."""
import os, sys
import numpy as np
from src.data import FEATURES

SERVICES = ["http", "ftp", "smtp", "domain_u", "private", "telnet", "other", "ecr_i"]
FLAGS = ["SF", "S0", "REJ", "RSTO"]
TRAIN_ATT = ["neptune", "smurf", "ipsweep", "guess_passwd", "buffer_overflow"]
TEST_ONLY = ["mailbomb", "snmpguess"]


def make(path, n, attacks, rng):
    classes = ["normal"] + attacks
    rows = []
    for _ in range(n):
        k = rng.integers(len(classes)); base = rng.normal(0, 1, 38) + k * 1.2
        row = []
        num = iter(np.abs(base))
        for f in FEATURES:
            if f == "protocol_type": row.append(["tcp", "udp", "icmp"][k % 3])
            elif f == "service": row.append(SERVICES[(k + rng.integers(3)) % len(SERVICES)])
            elif f == "flag": row.append(FLAGS[k % len(FLAGS)])
            else: row.append(round(next(num), 3))
        rows.append(",".join(map(str, row)) + f",{classes[k]},15")
    open(path, "w").write("\n".join(rows) + "\n")


def main(out):
    rng = np.random.default_rng(0); os.makedirs(out, exist_ok=True)
    make(os.path.join(out, "KDDTrain+.txt"), 3000, TRAIN_ATT, rng)
    make(os.path.join(out, "KDDTest+.txt"), 1000, TRAIN_ATT + TEST_ONLY, rng)


if __name__ == "__main__":
    main(sys.argv[1])
