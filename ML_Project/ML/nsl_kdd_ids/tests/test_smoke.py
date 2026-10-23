import json, os, subprocess, sys
import numpy as np
from src.model import NeuralNet
from src.data import load_dataset
from tests.make_synthetic import main as make_synth


def test_gradient_check():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(20, 6)), rng.integers(0, 3, 20)
    net = NeuralNet(6, 5, 3, seed=1)
    # one full-batch step with small lr must reduce the loss
    before = net.loss(X, y); net.fit(X, y, epochs=1, lr=0.1, batch_size=20, lam=0.0)
    assert net.loss(X, y) < before


def test_pipeline(tmp_path):
    make_synth(str(tmp_path / "d"))
    d = load_dataset(str(tmp_path / "d/KDDTrain+.txt"), str(tmp_path / "d/KDDTest+.txt"))
    assert d["X_train"].shape[1] == d["X_test"].shape[1]
    assert d["classes"][0] == "normal" and d["classes"][-1] == "other"
    net = NeuralNet(d["X_train"].shape[1], 50, len(d["classes"])).fit(d["X_train"], d["y_train"], 10, 0.5, 200)
    assert (net.predict(d["X_train"]) == d["y_train"]).mean() > 0.5
