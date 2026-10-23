# Neural Network for Security Attack Detection (NSL-KDD)

UE24CS352A Machine Learning – Mini-Project. A from-scratch NumPy neural network
(**122 → 500 → 24**, sigmoid hidden layer, softmax output) that classifies network connections
from the NSL-KDD dataset into normal / 22 known attack types / "other" (unseen attacks), and
reports both attack-type accuracy and normal-vs-abnormal accuracy.

Team no.: **3** — Shweta (PES1UG24AM404), Pranav P (PES1UG24AM387)

## Repository layout
```
data/            put KDDTrain+.txt and KDDTest+.txt here (see data/README.md)
src/data.py      loading, one-hot encoding, standardisation, label construction
src/model.py     NumPy neural network, L2-regularised mini-batch gradient descent
src/metrics.py   accuracy / confusion metrics
src/train.py     multi-seed training, epoch curves, baselines, figures, results.json
src/make_report.py   builds docs/Project_Writeup.pdf (2 pages) from results
src/make_slides.py   builds docs/Project_Slides.pptx from results
tests/           unit + smoke tests (use synthetic data, not real data)
run_all.py       one command that does everything
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download `KDDTrain+.txt` and `KDDTest+.txt` into `data/` (sources listed in `data/README.md`).

## Run
```bash
python run_all.py --team "Shweta (PES1UG24AM404), Pranav P (PES1UG24AM387)" --team-no 3
```
This trains 5 seeds (about a few minutes on a laptop CPU), then writes:
- `results/results.json`, `results/fig_*.png`
- `docs/Project_Writeup.pdf` (the 2-page write-up)
- `docs/Project_Slides.pptx` (review deck)

Useful options: `python run_all.py --quick` (fast check), or individual steps:
```bash
python -m src.train --data-dir data --out-dir results --seeds 5 --epochs 10
python -m src.make_report --team "Shweta (PES1UG24AM404), Pranav P (PES1UG24AM387)" --team-no 3
python -m src.make_slides --team "Shweta (PES1UG24AM404), Pranav P (PES1UG24AM387)" --team-no 3
```
Hyper-parameters (defaults follow the reference paper): hidden 500, lr 0.5, batch 1000, 10 epochs, L2 1e-4.

## Tests
```bash
python -m pytest -q tests        # or: python -c "from tests.test_smoke import *; test_gradient_check()"
```

## Method summary
1. One-hot encode `protocol_type`, `service`, `flag` (41 → 122 columns); standardise with training statistics.
2. Labels: normal + 22 training attacks + one `other` unit (test attacks unseen in training map here).
3. Train 122-500-24 network with cross-entropy + L2, mini-batch size 1000, 10 epochs, several seeds.
4. Evaluate attack-type accuracy and normal-vs-abnormal accuracy; compare with logistic regression and random forest.

## Reference
Tavallaee, Bagheri, Lu, Ghorbani. *A detailed analysis of the KDD CUP 99 data set*. IEEE CISDA 2009.
