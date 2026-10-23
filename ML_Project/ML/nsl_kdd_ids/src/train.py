"""Train + evaluate the 122-500-24 network on NSL-KDD; write results.json and figures.

Usage: python -m src.train --data-dir data --out-dir results
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import load_dataset, category_of
from .model import NeuralNet
from .metrics import multiclass_acc, binary_acc, binary_confusion


def evaluate(net, d):
    ptr, pte = net.predict(d["X_train"]), net.predict(d["X_test"])
    n = d["normal_idx"]
    return dict(
        train_multi=multiclass_acc(d["y_train"], ptr), test_multi=multiclass_acc(d["y_test"], pte),
        train_bin=binary_acc(d["y_train"], ptr, n), test_bin=binary_acc(d["y_test"], pte, n),
    ), pte


def run(args):
    t0 = time.time()
    d = load_dataset(os.path.join(args.data_dir, "KDDTrain+.txt"), os.path.join(args.data_dir, "KDDTest+.txt"))
    n_in, n_out = d["X_train"].shape[1], len(d["classes"])
    print(f"inputs={n_in} outputs={n_out} train={len(d['y_train'])} test={len(d['y_test'])}")
    os.makedirs(args.out_dir, exist_ok=True)
    R = dict(n_inputs=n_in, n_outputs=n_out, n_hidden=args.hidden, lr=args.lr, batch_size=args.batch,
             epochs=args.epochs, l2=args.l2, n_train=len(d["y_train"]), n_test=len(d["y_test"]))

    # class distribution per main category (Table 3 of the paper)
    def dist(labels):
        out = {}
        for l in labels:
            c = category_of(l); out[c] = out.get(c, 0) + 1
        return out
    R["dist_train"], R["dist_test"] = dist(d["raw_train_labels"]), dist(d["raw_test_labels"])
    R["n_unseen_test_attack_types"] = int(len(set(d["raw_test_labels"]) - set(d["raw_train_labels"])))

    # 1) several seeds at the paper's setting (initialisation sensitivity)
    runs, best_pred = [], None
    for s in range(args.seeds):
        net = NeuralNet(n_in, args.hidden, n_out, seed=s).fit(
            d["X_train"], d["y_train"], args.epochs, args.lr, args.batch, args.l2, seed=s)
        m, pte = evaluate(net, d); m["seed"] = s; runs.append(m)
        print(m)
        if best_pred is None or m["test_bin"] > max(r["test_bin"] for r in runs[:-1] + [dict(test_bin=-1)]):
            best_pred = pte
    R["runs"] = runs
    for k in ("train_multi", "test_multi", "train_bin", "test_bin"):
        v = np.array([r[k] for r in runs]); R[k + "_mean"], R[k + "_std"] = float(v.mean()), float(v.std())
        R[k + "_min"], R[k + "_max"] = float(v.min()), float(v.max())

    # detailed analysis for the best seed
    nrm = d["normal_idx"]
    R["confusion_binary"] = binary_confusion(d["y_test"], best_pred, nrm)
    cats = np.array([category_of(l) for l in d["raw_test_labels"]])
    per_cat = {}
    for c in sorted(set(cats)):
        mk = cats == c
        pred_norm = (best_pred[mk] == nrm)
        per_cat[c] = dict(n=int(mk.sum()),
                          predicted_normal=float(pred_norm.mean()),
                          detected=float(1 - pred_norm.mean()))
    R["per_category_test"] = per_cat
    seen = np.isin(d["raw_test_labels"], list(set(d["raw_train_labels"])))
    att = d["y_test"] != nrm
    R["unseen_attack_detection"] = float((best_pred[att & ~seen] != nrm).mean()) if (att & ~seen).any() else None
    R["seen_attack_detection"] = float((best_pred[att & seen] != nrm).mean()) if (att & seen).any() else None

    # 2) accuracy-vs-epoch curves (Figure 2 of the paper), seed 0, longer run
    hist = dict(train_multi=[], test_multi=[], train_bin=[], test_bin=[])
    def cb(ep, net):
        m, _ = evaluate(net, d)
        for k in hist: hist[k].append(m[k])
    NeuralNet(n_in, args.hidden, n_out, seed=0).fit(
        d["X_train"], d["y_train"], args.curve_epochs, args.lr, args.batch, args.l2, seed=0, on_epoch=cb)
    R["curve"] = hist

    # 3) baselines (binary normal-vs-attack)
    if not args.skip_baselines:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        ytr_b, yte_b = (d["y_train"] != nrm).astype(int), (d["y_test"] != nrm).astype(int)
        R["baselines"] = {}
        for name, clf in [("Logistic Regression", LogisticRegression(max_iter=300)),
                          ("Random Forest", RandomForestClassifier(100, n_jobs=-1, random_state=0))]:
            clf.fit(d["X_train"], ytr_b)
            R["baselines"][name] = dict(test_bin=float((clf.predict(d["X_test"]) == yte_b).mean()))
            print(name, R["baselines"][name])
    R["runtime_sec"] = time.time() - t0
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(R, f, indent=2)
    make_figures(R, d, best_pred, args.out_dir)
    print(f"saved results to {args.out_dir} in {R['runtime_sec']:.0f}s")


def make_figures(R, d, pred, out):
    ep = np.arange(1, len(R["curve"]["train_multi"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=200)
    for a, kind, title in [(ax[0], "multi", "Attack-type accuracy (24 classes)"),
                           (ax[1], "bin", "Normal vs abnormal accuracy")]:
        a.plot(ep, R["curve"]["train_" + kind], label="train", color="#1f77b4")
        a.plot(ep, R["curve"]["test_" + kind], label="test", color="#d62728")
        a.axvline(R["epochs"], ls=":", color="gray", lw=0.8)
        a.set_title(title, fontsize=8); a.set_xlabel("epoch", fontsize=7); a.set_ylabel("accuracy", fontsize=7)
        a.tick_params(labelsize=7); a.legend(fontsize=7); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_curves.png")); plt.close(fig)

    cats = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
    w = 0.38; x = np.arange(len(cats))
    ax.bar(x - w/2, [R["dist_train"].get(c, 0) for c in cats], w, label="train", color="#1f77b4")
    ax.bar(x + w/2, [R["dist_test"].get(c, 0) for c in cats], w, label="test", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=7); ax.tick_params(labelsize=7)
    ax.set_yscale("log"); ax.set_ylabel("records (log)", fontsize=7); ax.legend(fontsize=7)
    ax.set_title("Class distribution", fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_distribution.png")); plt.close(fig)

    c = R["confusion_binary"]; M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=200)
    ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:,}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred normal", "pred attack"], fontsize=7)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true normal", "true attack"], fontsize=7)
    ax.set_title("Test confusion (best seed)", fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_confusion.png")); plt.close(fig)

    pc = R["per_category_test"]; names = [k for k in pc if k != "Normal"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
    ax.bar(names, [pc[k]["detected"] for k in names], color="#2ca02c")
    ax.set_ylim(0, 1); ax.set_ylabel("detection rate", fontsize=7); ax.tick_params(labelsize=7)
    ax.set_title("Attack detection by category (test)", fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_detection.png")); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data"); p.add_argument("--out-dir", default="results")
    p.add_argument("--hidden", type=int, default=500); p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=1000); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--l2", type=float, default=1e-4); p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--curve-epochs", type=int, default=50); p.add_argument("--skip-baselines", action="store_true")
    run(p.parse_args())
