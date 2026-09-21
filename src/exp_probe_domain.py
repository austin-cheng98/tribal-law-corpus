"""Within-domain replication of the split contrast.

Holding documents out could depress the Nation probe simply because a Nation's held-out
document covers a different legal domain than its training documents. Restricting
both regimes to a single domain removes that explanation.
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_N, K = 40, 4
F1 = lambda a, b: f1_score(a, b, average="macro", zero_division=0)


def evaluate(rows, y, folds, sizes, rng):
    pred = np.empty(len(y), dtype=object)
    for (tr, te), n in zip(folds, sizes):
        if n is not None and len(tr) > n:
            tr = tr[rng.permutation(len(tr))[:n]]
        m = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        m.fit([rows[i]["text"] for i in tr], y[tr])
        pred[te] = m.predict([rows[i]["text"] for i in te])
    return {"accuracy": bootstrap_ci(accuracy_score, y, pred),
            "macro_f1": bootstrap_ci(F1, y, pred)}


def main():
    allrows = load()
    out = {}
    for dom in ("criminal", "civil"):
        rs = [r for r in allrows if r["domain"] == dom]
        nd = collections.defaultdict(collections.Counter)
        for r in rs:
            nd[r["nation"]][r["doc_id"]] += 1
        keep = {n for n, c in nd.items() if len(c) >= 2 and sum(c.values()) >= MIN_N}
        rows = [r for r in rs if r["nation"] in keep]
        if len(keep) < 3:
            continue
        y = np.array([r["nation"] for r in rows])
        print(f"{dom}: {len(rows)} provisions, {len(keep)} Nations, "
              f"{len({r['doc_id'] for r in rows})} documents", flush=True)
        dfolds = doc_folds(rows, k=K)
        sizes = [len(tr) for tr, _ in dfolds]
        rfolds = list(StratifiedKFold(K, shuffle=True, random_state=17)
                      .split(np.zeros(len(y)), y))
        rng = np.random.default_rng(17)
        res = {"random": evaluate(rows, y, rfolds, sizes, rng),
               "doc_heldout": evaluate(rows, y, dfolds, [None] * K, rng),
               "n": len(rows), "n_nations": len(keep),
               "majority": max(collections.Counter(y).values()) / len(y)}
        rng2 = np.random.default_rng(99)
        nulls = []
        for _ in range(10):
            yp = rng2.permutation(y)
            p = np.empty(len(y), dtype=object)
            for tr, te in dfolds:
                m = make_pipeline(
                    TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                    LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
                m.fit([rows[i]["text"] for i in tr], yp[tr])
                p[te] = m.predict([rows[i]["text"] for i in te])
            nulls.append(F1(yp, p))
        res["perm_null_f1"] = [float(np.mean(nulls)), float(np.std(nulls))]
        for k in ("random", "doc_heldout"):
            f = res[k]["macro_f1"]
            print(f"   {k:12s} macroF1 {f[0]:.3f} [{f[1]:.3f},{f[2]:.3f}]", flush=True)
        print(f"   permutation null {res['perm_null_f1'][0]:.3f}"
              f"+-{res['perm_null_f1'][1]:.3f}  "
              f"ratio {res['random']['macro_f1'][0]/res['doc_heldout']['macro_f1'][0]:.1f}x",
              flush=True)
        out[dom] = res
    json.dump(out, open(ROOT / "results/probe_domain.json", "w"), indent=1)


if __name__ == "__main__":
    main()
