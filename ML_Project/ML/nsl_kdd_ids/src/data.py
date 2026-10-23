"""NSL-KDD loading and preprocessing (one-hot + standardisation), as in the paper."""
import numpy as np
import pandas as pd

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Table 2 of the paper: attack -> main class
CATEGORY = {"normal": "Normal"}
for cls, names in {
    "DoS": "back land neptune pod smurf teardrop mailbomb apache2 processtable udpstorm",
    "Probe": "ipsweep nmap portsweep satan mscan saint",
    "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail "
           "named snmpgetattack snmpguess xlock xsnoop worm",
    "U2R": "buffer_overflow loadmodule perl rootkit httptunnel ps sqlattack xterm",
}.items():
    for n in names.split():
        CATEGORY[n] = cls
OTHER = "other"  # catch-all output unit for attacks unseen in training


def load_raw(path):
    df = pd.read_csv(path, header=None, names=FEATURES + ["label", "difficulty"])
    df["label"] = df["label"].str.strip().str.lower()
    return df


def load_dataset(train_path, test_path):
    tr, te = load_raw(train_path), load_raw(test_path)

    # one-hot with categories taken from the union so both sets share the same columns
    full = pd.concat([tr[FEATURES], te[FEATURES]], keys=["tr", "te"])
    for c in CATEGORICAL:
        full[c] = pd.Categorical(full[c])
    full = pd.get_dummies(full, columns=CATEGORICAL, dtype=float)
    Xtr, Xte = full.loc["tr"].to_numpy(float), full.loc["te"].to_numpy(float)
    names = list(full.columns)

    # zero mean / unit variance, statistics from the training set only
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    # labels: normal + training attacks + one "other" unit for unseen attacks
    classes = sorted(tr["label"].unique(), key=lambda s: (s != "normal", s)) + [OTHER]
    idx = {c: i for i, c in enumerate(classes)}
    ytr = tr["label"].map(idx).to_numpy()
    yte = te["label"].map(lambda s: idx.get(s, idx[OTHER])).to_numpy()
    return dict(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte, classes=classes,
        normal_idx=idx["normal"], feature_names=names,
        raw_train_labels=tr["label"].to_numpy(), raw_test_labels=te["label"].to_numpy(),
    )


def category_of(label):
    return CATEGORY.get(label, "Other/Unknown")
