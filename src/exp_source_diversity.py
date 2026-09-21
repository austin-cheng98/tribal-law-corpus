"""What predicts how much a random split inflates: nesting, or source diversity?

Runs the random / document-held-out ladder for four labels on one fixed row set, one fold
construction and one classifier, so the labels differ only in their relation to the source
document. Theil's U measures nesting; documents per label value measures how many distinct
sources a label value is realized in. Only the second orders the distortion.
"""
import argparse, collections, json, math, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from exp_label_nesting import theil_u
from exp_nation_probe_lodo import doc_folds
from data import corpus_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED, NULL_REPS, BOOT = 20260101, 200, 2000
MACRO = lambda y, p: f1_score(y, p, average="macro", zero_division=0)


def fit(rows, y, folds):
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        m = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        m.fit([rows[i]["text"] for i in tr], y[tr])
        pred[te] = m.predict([rows[i]["text"] for i in te])
    return pred


def nation_boot(y, pred, nats, rng):
    """Resample Nations, not provisions: the unit the contrasts are claimed at."""
    idx = collections.defaultdict(list)
    for i, n in enumerate(nats):
        idx[n].append(i)
    keys, out = sorted(idx), []
    for _ in range(BOOT):
        sel = [i for k in rng.choice(keys, len(keys), replace=True) for i in idx[k]]
        out.append(MACRO(y[sel], pred[sel]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def ladder(rows, y, name, rng):
    y = np.array(y)
    nats = [r["nation"] for r in rows]
    docs = [r["doc_id"] for r in rows]
    rnd = fit(rows, y, list(StratifiedKFold(4, shuffle=True, random_state=17).split(rows, y)))
    dh = fit(rows, y, doc_folds(rows, k=4))
    fr, fd = MACRO(y, rnd), MACRO(y, dh)

    per = collections.defaultdict(set)
    for d, v in zip(docs, y):
        per[v].add(d)
    ndoc = [len(s) for s in per.values()]

    u = theil_u(docs, list(y))
    null = [theil_u(docs, [y[i] for i in rng.permutation(len(y))]) for _ in range(NULL_REPS)]
    mu = float(np.mean(null))
    u_adj = (u - mu) / (1 - mu) if mu < 1 else 0.0

    # is the gap real at the Nation level?
    byn = collections.defaultdict(lambda: [[], [], []])
    for i, n in enumerate(nats):
        byn[n][0].append(y[i]); byn[n][1].append(rnd[i]); byn[n][2].append(dh[i])
    d = np.array([MACRO(np.array(a), np.array(b)) - MACRO(np.array(a), np.array(c))
                  for a, b, c in byn.values()])
    flip = (d * rng.choice([-1.0, 1.0], size=(20000, len(d)))).mean(axis=1)
    p = float((1 + np.sum(np.abs(flip) >= abs(d.mean()))) / 20001)

    res = {"label": name, "n": len(y), "n_classes": len(set(y)), "n_docs": len(set(docs)),
           "u": u, "u_adj": u_adj, "docs_per_value_mean": float(np.mean(ndoc)),
           "docs_per_value_min": int(min(ndoc)),
           "random_f1": fr, "doc_heldout_f1": fd, "ratio": fr / fd if fd else float("nan"),
           "gap": fr - fd, "gap_p": p,
           "random_ci": nation_boot(y, rnd, nats, rng),
           "doc_heldout_ci": nation_boot(y, dh, nats, rng)}
    print(f"{name:20s} n={len(y):5d} K={len(set(y)):3d}  U_adj {u_adj:.3f}  "
          f"docs/value {np.mean(ndoc):5.1f} (min {min(ndoc)})  "
          f"random {fr:.3f}  doc-held-out {fd:.3f}  ratio {fr/fd:.2f}x  p={p:.4g}", flush=True)
    return res


def main(out):
    rows = [json.loads(l) for l in open(corpus_path())]
    nd = collections.defaultdict(set)
    for r in rows:
        nd[r["nation"]].add(r["doc_id"])
    keep = {n for n, s in nd.items() if len(s) >= 2}
    base = [r for r in rows if r["nation"] in keep]
    print(f"{len(base)} provisions, {len(keep)} multi-document Nations, "
          f"{len({r['doc_id'] for r in base})} documents\n", flush=True)

    rng = np.random.default_rng(SEED)
    res = [ladder(base, [r["nation"] for r in base], "Nation", rng),
           ladder(base, [r["domain"] for r in base], "legal domain", rng),
           ladder(base, [(r["modality"] or ["NONE"])[0] for r in base], "deontic modality", rng)]

    g = json.load(open(ROOT / "results/gold.json"))["gold"]
    sub = [r for r in base if g.get(r["provision_id"])]
    keep2 = {n for n, c in collections.Counter(r["nation"] for r in sub).items() if c >= 10}
    sub = [r for r in sub if r["nation"] in keep2]
    res.append(ladder(sub, [g[r["provision_id"]] for r in sub], "provision function", rng))

    json.dump({"results": res, "n_base": len(base), "n_nations": len(keep)},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/source_diversity.json"))
    main(ap.parse_args().out)
