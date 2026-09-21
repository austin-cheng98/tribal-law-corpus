"""Nation probe with documents held out, so no test provision shares a source document
with any training provision. Distinguishes drafting style from document-level artefacts.
"""
import collections, json, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import load, bootstrap_ci
import masking

ROOT = pathlib.Path(__file__).resolve().parents[1]


def doc_folds(rows, k=4, seed=17):
    """Split documents within each Nation, so every Nation is present in every training fold."""
    rng = np.random.default_rng(seed)
    assign = {}
    bynat = collections.defaultdict(list)
    for r in rows:
        bynat[r["nation"]].append(r["doc_id"])
    for nat, docs in bynat.items():
        d = sorted(set(docs))
        rng.shuffle(d)
        for i, doc in enumerate(d):
            assign[doc] = i % k
    fold_of = np.array([assign[r["doc_id"]] for r in rows])
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(k)]


def main():
    rows = load()
    ndoc = collections.defaultdict(set)
    for r in rows:
        ndoc[r["nation"]].add(r["doc_id"])
    keep = {n for n, s in ndoc.items() if len(s) >= 2}
    rows = [r for r in rows if r["nation"] in keep]
    y = np.array([r["nation"] for r in rows])
    g = np.array([r["doc_id"] for r in rows])
    print(f"{len(rows)} provisions, {len(keep)} multi-document Nations, "
          f"{len(set(g))} documents", flush=True)
    folds = doc_folds(rows, k=4)
    out = []
    for level in (0, 2, 3):
        preds = np.empty(len(y), dtype=object)
        for tr, te in folds:
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
        out.append({"level": f"M{level}", "accuracy": a, "macro_f1": f})
        print(f"  M{level}  acc {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]  "
              f"macroF1 {f[0]:.3f} [{f[1]:.3f},{f[2]:.3f}]", flush=True)
    chance = max(collections.Counter(y).values()) / len(y)
    print(f"  majority baseline {chance:.3f}")

    # permutation null: identical pipeline on shuffled Nation labels
    rng = np.random.default_rng(99)
    nulls = []
    for _ in range(10):
        yp = rng.permutation(y)
        pr = np.empty(len(y), dtype=object)
        for tr, te in folds:
            m = make_pipeline(
                TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
            m.fit([rows[i]["text"] for i in tr], yp[tr])
            pr[te] = m.predict([rows[i]["text"] for i in te])
        nulls.append((accuracy_score(yp, pr),
                      f1_score(yp, pr, average="macro", zero_division=0)))
    na = np.array([x[0] for x in nulls]); nf = np.array([x[1] for x in nulls])
    print(f"  permutation null: acc {na.mean():.3f}+-{na.std():.3f}  "
          f"macroF1 {nf.mean():.3f}+-{nf.std():.3f}")
    json.dump({"results": out, "majority": chance, "n": len(rows), "n_nations": len(keep),
               "perm_null_acc": [float(na.mean()), float(na.std())],
               "perm_null_f1": [float(nf.mean()), float(nf.std())]},
              open(ROOT / "results/nation_probe_lodo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
