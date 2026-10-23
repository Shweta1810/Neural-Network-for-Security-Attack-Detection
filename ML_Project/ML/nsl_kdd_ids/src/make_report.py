"""Builds the 2-page PDF write-up from results/results.json + figures.
Usage: python -m src.make_report --team "Shweta (PES1UG24AM404), Pranav P (PES1UG24AM387)" --team-no 3"""
import argparse, json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle


def pct(x, d=1): return f"{100 * x:.{d}f}%"


def build(a):
    R = json.load(open(os.path.join(a.results, "results.json")))
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=10.6, spaceAfter=3, alignment=4)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=10.5, leading=12, spaceBefore=4, spaceAfter=2)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=14, leading=16, spaceAfter=2)
    sub = ParagraphStyle("s", parent=ss["BodyText"], fontSize=8.5, alignment=1, textColor=colors.grey)
    cap = ParagraphStyle("c", parent=ss["BodyText"], fontSize=7.5, leading=9, alignment=1, textColor=colors.grey)
    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=.6*inch, rightMargin=.6*inch,
                            topMargin=.5*inch, bottomMargin=.5*inch,
                            title="Neural Network for Security Attack Detection", author=a.team)
    W = A4[0] - 1.2 * inch
    cm, c = R["confusion_binary"], R["per_category_test"]
    S = []
    S += [Paragraph("Using a Neural Network for Security Attack Detection (NSL-KDD)", title),
          Paragraph(f"UE24CS352A Machine Learning – Mini-Project | Team {a.team_no} | {a.team}", sub), Spacer(1, 4)]

    S += [Paragraph("1. Problem Statement", h), Paragraph(
        "Software systems are frequent targets of cyber attacks, and attackers keep changing their behaviour, "
        "which makes signature-based intrusion detection brittle. The task is to build a neural network that "
        "reads the recorded parameters of a network connection and (i) predicts which attack type it belongs "
        "to, and (ii) decides whether the behaviour is normal or abnormal, including attacks that never "
        "appeared in the training data.", body)]

    S += [Paragraph("2. Dataset", h), Paragraph(
        f"We use the <b>NSL-KDD</b> dataset, an improved version of KDD Cup 99 with no redundant training records "
        f"and no duplicate test records. Each record has 41 features (basic TCP/IP, content and traffic features); "
        f"protocol_type, service and flag are categorical. Training set: <b>{R['n_train']:,}</b> records (normal + 22 "
        f"attacks); test set: <b>{R['n_test']:,}</b> records, containing <b>{R['n_unseen_test_attack_types']}</b> attack "
        f"types absent from training. Attacks fall into DoS, Probe, R2L and U2R. Records per class:", body)]
    order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    t = Table([["Class"] + order, ["Train"] + [f"{R['dist_train'].get(k, 0):,}" for k in order],
               ["Test"] + [f"{R['dist_test'].get(k, 0):,}" for k in order]], colWidths=[W / 6] * 6)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    S += [t, Spacer(1, 3)]

    S += [Paragraph("3. Approach", h), Paragraph(
        f"<b>Preprocessing.</b> Categorical features are one-hot encoded, giving <b>{R['n_inputs']}</b> input columns; "
        f"all columns are standardised to zero mean and unit variance using training statistics only. Labels become "
        f"a one-hot vector of <b>{R['n_outputs']}</b> outputs: normal, the 22 training attacks, and one extra "
        f"<i>other</i> unit reserved for unseen attacks. <b>Model.</b> A fully-connected network "
        f"{R['n_inputs']}–{R['n_hidden']}–{R['n_outputs']} with sigmoid hidden units and a softmax output, trained by "
        f"L2-regularised mini-batch gradient descent on cross-entropy (learning rate {R['lr']}, batch size "
        f"{R['batch_size']}, {R['epochs']} epochs, λ={R['l2']}). <b>Evaluation.</b> (a) attack-type accuracy over all "
        f"classes; (b) normal-vs-abnormal accuracy, where any non-normal prediction counts as abnormal. Because the "
        f"paper notes sensitivity to random initialisation, we repeat training with {len(R['runs'])} seeds.", body)]

    S += [Paragraph("4. Implementation Overview", h), Paragraph(
        "Everything is written in Python. <font face='Courier'>src/data.py</font> loads and preprocesses NSL-KDD; "
        "<font face='Courier'>src/model.py</font> is a from-scratch NumPy network (forward pass, back-propagation, "
        "mini-batch SGD); <font face='Courier'>src/metrics.py</font> holds the two accuracy measures and the confusion "
        "counts; <font face='Courier'>src/train.py</font> runs the seeds, the accuracy-vs-epoch curves, logistic "
        "regression and random-forest baselines (scikit-learn), and saves figures and "
        "<font face='Courier'>results.json</font>. This report and the slides are generated from that file, so all "
        "numbers here come directly from the code run. Unit tests are in <font face='Courier'>tests/</font>.", body)]

    S += [Paragraph("5. Results", h)]
    rows = [["Metric", "Train", "Test"],
            [f"Attack-type accuracy ({len(R['runs'])} seeds, mean ± std)",
             f"{pct(R['train_multi_mean'])} ± {pct(R['train_multi_std'])}", f"{pct(R['test_multi_mean'])} ± {pct(R['test_multi_std'])}"],
            ["Normal vs abnormal accuracy (mean ± std)",
             f"{pct(R['train_bin_mean'])} ± {pct(R['train_bin_std'])}", f"{pct(R['test_bin_mean'])} ± {pct(R['test_bin_std'])}"],
            ["Normal vs abnormal accuracy (min – max over seeds)", "",
             f"{pct(R['test_bin_min'])} – {pct(R['test_bin_max'])}"]]
    for k, v in R.get("baselines", {}).items():
        rows.append([f"Baseline: {k} (binary)", "", pct(v["test_bin"])])
    t = Table(rows, colWidths=[W * .56, W * .22, W * .22])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    S += [t, Spacer(1, 3)]
    S += [Image(os.path.join(a.results, "fig_curves.png"), width=W * .92, height=W * .92 * 2.6 / 7.2),
          Paragraph(f"Figure 1: accuracy vs epoch (seed 0, {len(R['curve']['train_bin'])} epochs; dotted line = {R['epochs']} epochs used in the final model).", cap)]
    imgs = Table([[Image(os.path.join(a.results, "fig_confusion.png"), width=W * .44, height=W * .44 * 2.4 / 3.0),
                   Image(os.path.join(a.results, "fig_detection.png"), width=W * .44, height=W * .44 * 2.4 / 3.4)]])
    S += [imgs, Paragraph("Figure 2: binary confusion matrix and per-category detection rate on the test set (best seed).", cap)]

    unseen, seen = R.get("unseen_attack_detection"), R.get("seen_attack_detection")
    gap = R["train_multi_mean"] - R["test_multi_mean"]
    S += [Paragraph("6. Conclusions", h), Paragraph(
        f"The network fits the training data well ({pct(R['train_multi_mean'])} attack-type accuracy) but reaches "
        f"{pct(R['test_multi_mean'])} on the test set, a gap of {100 * gap:.1f} points. Part of this gap is expected: the test set "
        f"contains {R['n_unseen_test_attack_types']} attack types never seen in training, which can only map to the <i>other</i> unit "
        f"or to a wrong class. Judged only on normal vs abnormal, test accuracy is {pct(R['test_bin_mean'])} "
        f"(false-alarm rate {pct(cm['false_alarm_rate'])}, attack recall {pct(cm['recall'])}, F1 {cm['f1']:.3f} for the best seed). "
        + (f"Detection of attacks seen in training is {pct(seen)} versus {pct(unseen)} for unseen attacks. " if unseen is not None and seen is not None else "")
        + f"Across seeds the test binary accuracy ranges from {pct(R['test_bin_min'])} to {pct(R['test_bin_max'])}, "
        f"so results depend on random initialisation and should be reported over several runs. Limitations: the test set "
        f"distribution differs from training, U2R/R2L are rare in training, and false alarms would still need "
        f"human review. Future work: class weighting or resampling for rare classes, early stopping, and additional datasets.", body)]
    S += [Paragraph("<b>References.</b> Tavallaee et al., <i>A detailed analysis of the KDD CUP 99 data set</i>, IEEE CISDA 2009; "
                    "Buczak &amp; Guven, IEEE Commun. Surveys &amp; Tutorials 18(2), 2016; LeCun, Bengio &amp; Hinton, <i>Deep learning</i>, Nature 2015.", body)]
    doc.build(S)
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results"); p.add_argument("--out", default="docs/Project_Writeup.pdf")
    p.add_argument("--team", default="Shweta (PES1UG24AM404), Pranav P (PES1UG24AM387)"); p.add_argument("--team-no", default="3")
    os.makedirs("docs", exist_ok=True); build(p.parse_args())
