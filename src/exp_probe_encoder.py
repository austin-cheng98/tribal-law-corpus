"""Nation probe with a fine-tuned legal-domain encoder, under both split protocols.

Checks that the random-split gap is a property of the evaluation design rather than of
the bag-of-ngrams classifier used elsewhere.
"""
import collections, json, pathlib, sys, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import load, bootstrap_ci
from encoder import MODEL, fit_predict
from exp_nation_probe_lodo import doc_folds

ROOT = pathlib.Path(__file__).resolve().parents[1]
K = 4


def run(rows, y, folds, labels, sizes, rng):
    preds = np.empty(len(y), dtype=object)
    for f, ((tr, te), n) in enumerate(zip(folds, sizes)):
        if n is not None and len(tr) > n:
            tr = tr[rng.permutation(len(tr))[:n]]
        t0 = time.time()
        preds[te] = fit_predict([rows[i]["text"] for i in tr], y[tr],
                                [rows[i]["text"] for i in te], labels, seed=f)
        print(f"    fold {f}  train {len(tr)}  test {len(te)}  {time.time()-t0:.0f}s",
              flush=True)
    a = bootstrap_ci(accuracy_score, y, preds)
    fl = bootstrap_ci(lambda p, q: f1_score(p, q, average="macro", zero_division=0), y, preds)
    return {"accuracy": a, "macro_f1": fl}


def main():
    rows = load()
    ndoc = collections.defaultdict(set)
    for r in rows:
        ndoc[r["nation"]].add(r["doc_id"])
    keep = {n for n, s in ndoc.items() if len(s) >= 2}
    rows = [r for r in rows if r["nation"] in keep]
    y = np.array([r["nation"] for r in rows])
    labels = sorted(keep)
    print(f"{MODEL} on {len(rows)} provisions, {len(keep)} Nations", flush=True)

    dfolds = doc_folds(rows, k=K)
    sizes = [len(tr) for tr, _ in dfolds]
    rfolds = list(StratifiedKFold(K, shuffle=True, random_state=17).split(np.zeros(len(y)), y))
    rng = np.random.default_rng(17)

    out = {}
    for name, folds, sz in (("random", rfolds, sizes), ("doc_heldout", dfolds, None)):
        print(f"  {name}", flush=True)
        out[name] = run(rows, y, folds, labels, sz or [None] * K, rng)
        r = out[name]
        print(f"  {name}: acc {r['accuracy'][0]:.3f} "
              f"[{r['accuracy'][1]:.3f},{r['accuracy'][2]:.3f}]  "
              f"macroF1 {r['macro_f1'][0]:.3f} "
              f"[{r['macro_f1'][1]:.3f},{r['macro_f1'][2]:.3f}]", flush=True)
    out["model"] = MODEL
    json.dump(out, open(ROOT / "results/probe_encoder.json", "w"), indent=1)


if __name__ == "__main__":
    main()
