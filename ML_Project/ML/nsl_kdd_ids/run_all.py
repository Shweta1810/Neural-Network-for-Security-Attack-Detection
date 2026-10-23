"""One command: train/evaluate, then build the write-up PDF and slide deck.
python run_all.py --team "Shweta (PES1UG24AM404), Pranav P (PES1UG24AM387)" --team-no 3 [--quick]"""
import argparse, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data"); p.add_argument("--team", default="Shweta (PES1UG24AM404), Pranav P (PES1UG24AM387)")
p.add_argument("--team-no", default="3"); p.add_argument("--quick", action="store_true", help="2 seeds, 20 curve epochs, no baselines")
a = p.parse_args()
train = [sys.executable, "-m", "src.train", "--data-dir", a.data_dir, "--out-dir", "results"]
if a.quick: train += ["--seeds", "2", "--curve-epochs", "20", "--skip-baselines"]
for cmd in (train,
            [sys.executable, "-m", "src.make_report", "--team", a.team, "--team-no", a.team_no],
            [sys.executable, "-m", "src.make_slides", "--team", a.team, "--team-no", a.team_no]):
    subprocess.run(cmd, check=True)
print("\nDone. See results/ and docs/")
