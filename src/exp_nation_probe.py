"""RQ: how recoverable is Nation identity from a single provision, and does redaction remove it?

Labels are corpus metadata, so this probe requires no annotation. Ablation terms for M3 are
fitted on the training fold only.
"""
import json, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import load, bootstrap_ci, SEED
import masking

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(rows, level, folds=5):
    y = np.array([r["nation"] for r in rows])
    X_raw = [r["text"] for r in rows]
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    preds = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X_raw, y):
        cues = set()
        if level >= 3:
            # union over Nations, so the mask itself carries no label information
            fitted = masking.fit_discriminative_terms([rows[i] for i in tr])
            cues = set().union(*fitted.values()) if fitted else set()
        Xtr = [masking.apply(X_raw[i], level, cues) for i in tr]
        Xte = [masking.apply(X_raw[i], level, cues) for i in te]
        clf = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        clf.fit(Xtr, y[tr])
        preds[te] = clf.predict(Xte)
    acc = bootstrap_ci(accuracy_score, y, preds)
    mf1 = bootstrap_ci(lambda a, b: f1_score(a, b, average="macro", zero_division=0), y, preds)
    return {"level": f"M{level}", "accuracy": acc, "macro_f1": mf1,
            "chance": float(max(np.bincount(np.unique(y, return_inverse=True)[1])) / len(y))}


def main():
    rows = load()
    print(f"{len(rows)} provisions, {len(set(r['nation'] for r in rows))} Nations")
    out = []
    for lvl in range(4):
        r = run(rows, lvl)
        out.append(r)
        a, alo, ahi = r["accuracy"]; m, mlo, mhi = r["macro_f1"]
        print(f"  {r['level']}  acc {a:.3f} [{alo:.3f},{ahi:.3f}]   macroF1 {m:.3f} [{mlo:.3f},{mhi:.3f}]",
              flush=True)
    print(f"  majority-class baseline: {out[0]['chance']:.3f}")
    json.dump(out, open(ROOT / "results/nation_probe.json", "w"), indent=1)


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    main()
