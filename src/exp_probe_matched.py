"""Random-split counterpart of the document-held-out Nation probe.

Runs on the identical provisions, Nations, and classifier, with training folds matched in
size to the document folds, so the only difference between the two numbers is whether a
test provision's source document was seen during training.
"""
import collections, json, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import load, bootstrap_ci
from exp_nation_probe_lodo import doc_folds
import masking

ROOT = pathlib.Path(__file__).resolve().parents[1]
K = 4


def main():
    rows = load()
    ndoc = collections.defaultdict(set)
    for r in rows:
        ndoc[r["nation"]].add(r["doc_id"])
    keep = {n for n, s in ndoc.items() if len(s) >= 2}
    rows = [r for r in rows if r["nation"] in keep]
    y = np.array([r["nation"] for r in rows])
    print(f"{len(rows)} provisions, {len(keep)} Nations", flush=True)

    # match each random fold's training size to the corresponding document fold
    sizes = [len(tr) for tr, _ in doc_folds(rows, k=K)]
    folds = list(StratifiedKFold(K, shuffle=True, random_state=17).split(np.zeros(len(y)), y))
    print(f"train sizes  doc-folds {sizes}  random {[len(tr) for tr, _ in folds]}", flush=True)

    rng = np.random.default_rng(17)
    out = {}
    for level in (0, 2, 3):
        preds = np.empty(len(y), dtype=object)
        for (tr, te), n in zip(folds, sizes):
            tr = tr[rng.permutation(len(tr))[:n]]
            cues = set()
            if level >= 3:
                fitted = masking.fit_discriminative_terms([rows[i] for i in tr])
                cues = set().union(*fitted.values()) if fitted else set()
            Xtr = [masking.apply(rows[i]["text"], level, cues) for i in tr]
            Xte = [masking.apply(rows[i]["text"], level, cues) for i in te]
            m = make_pipeline(
                TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
            m.fit(Xtr, y[tr])
            preds[te] = m.predict(Xte)
        a = bootstrap_ci(accuracy_score, y, preds)
        f = bootstrap_ci(lambda p, q: f1_score(p, q, average="macro", zero_division=0), y, preds)
        tag = {0: "Mzero", 2: "Mtwo", 3: "Mthree"}[level]
        out[f"Rand{tag}Acc"], out[f"Rand{tag}AccLo"], out[f"Rand{tag}AccHi"] = a
        out[f"Rand{tag}F"], out[f"Rand{tag}Flo"], out[f"Rand{tag}Fhi"] = f
        print(f"  M{level}  acc {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]  "
              f"macroF1 {f[0]:.3f} [{f[1]:.3f},{f[2]:.3f}]", flush=True)

    json.dump(out, open(ROOT / "results/probe_matched.json", "w"), indent=1)


if __name__ == "__main__":
    main()
