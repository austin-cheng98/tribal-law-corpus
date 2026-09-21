"""At what unit does the document-held-out probe clear its permutation null?

The reported interval on that figure resamples provisions, which treats provisions inside a
Nation as independent observations. This refits the M0 level, keeps the predictions, and
rebuilds the interval resampling documents and then Nations, the units the claim is made at.
"""
import collections, json, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import load
from exp_nation_probe_lodo import doc_folds

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOT, SEED = 2000, 20260101
MACRO = lambda a, b: f1_score(a, b, average="macro", zero_division=0)


def cluster_ci(y, pred, unit):
    """Percentile interval resampling whole clusters with replacement."""
    idx = collections.defaultdict(list)
    for i, u in enumerate(unit):
        idx[u].append(i)
    keys = sorted(idx)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(BOOT):
        sel = [i for k in rng.choice(keys, len(keys), replace=True) for i in idx[k]]
        vals.append(MACRO(y[sel], pred[sel]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main():
    rows = load()
    nd = collections.defaultdict(set)
    for r in rows:
        nd[r["nation"]].add(r["doc_id"])
    rows = [r for r in rows if len(nd[r["nation"]]) >= 2]
    y = np.array([r["nation"] for r in rows])
    pred = np.empty(len(y), dtype=object)
    for tr, te in doc_folds(rows, k=4):
        m = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        m.fit([rows[i]["text"] for i in tr], y[tr])
        pred[te] = m.predict([rows[i]["text"] for i in te])

    point = MACRO(y, pred)
    null = json.load(open(ROOT / "results/nation_probe_lodo.json"))["perm_null_f1"][0]
    out = {"point": point, "null": null, "n": len(y),
           "provision": cluster_ci(y, pred, list(range(len(y)))),
           "document": cluster_ci(y, pred, [r["doc_id"] for r in rows]),
           "nation": cluster_ci(y, pred, [r["nation"] for r in rows])}
    for u in ("provision", "document", "nation"):
        lo, hi = out[u]
        out[u + "_clears_null"] = lo > null
        print(f"{u:10s} [{lo:.3f}, {hi:.3f}]  clears null {null:.3f}: "
              f"{'yes' if lo > null else 'NO'}")
    print(f"point {point:.4f}")
    json.dump(out, open(ROOT / "results/probe_cluster.json", "w"), indent=1)


if __name__ == "__main__":
    main()
