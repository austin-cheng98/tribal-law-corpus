"""Does starving a label's source diversity create the distortion? A within-corpus test.

The four-label comparison in exp_source_diversity.py is observational: the labels differ
in many ways besides how many documents realize each value. Here document diversity is
set directly. For a label that shows no distortion at full diversity, each value is
restricted to m documents while the class set and the provisions per value are held
fixed, so m is the only quantity that moves.
"""
import argparse, collections, json, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import corpus_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED, FOLDS, REPEATS, N_PERM = 20260101, 2, 50, 20
MS = [2, 3, 5, 10, 20]
MACRO = lambda y, p: f1_score(y, p, average="macro", zero_division=0)
LABELS = {"legal domain": (lambda r: r["domain"], 120),
          "deontic modality": (lambda r: (r["modality"] or ["NONE"])[0], 40)}


def value_doc_folds(rows, y, k, rng):
    """Document-disjoint folds that keep every label value in every training fold."""
    byval = collections.defaultdict(set)
    for r, v in zip(rows, y):
        byval[v].add(r["doc_id"])
    assign = {}
    for v in sorted(byval):                      # round-robin within each value
        d = sorted(byval[v])
        for i, j in enumerate(rng.permutation(len(d))):
            assign[d[j]] = i % k
    fold = np.array([assign[r["doc_id"]] for r in rows])
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


def fit(rows, y, folds):
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        if len(tr) == 0 or len(te) == 0:
            continue
        m = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        m.fit([rows[i]["text"] for i in tr], y[tr])
        pred[te] = m.predict([rows[i]["text"] for i in te])
    return pred


def draw(base, get, m, target, rng):
    """m documents per label value, `target` provisions per value: only m varies."""
    need = -(-target // m)
    byval = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in base:
        byval[get(r)][r["doc_id"]].append(r)
    out = []
    for v, docs in byval.items():
        ok = sorted(d for d, rs in docs.items() if len(rs) >= need)
        if len(ok) < m:
            return None
        pick = [ok[i] for i in rng.permutation(len(ok))[:m]]
        pool = [r for d in pick for r in docs[d]]
        if len(pool) < target:
            return None
        out += [pool[i] for i in rng.permutation(len(pool))[:target]]
    return out


def point(base, get, m, target, rng, perm):
    rs, ds, unseen = [], [], []
    for _ in range(REPEATS):
        sub = draw(base, get, m, target, rng)
        if sub is None:
            return None
        y = np.array([get(r) for r in sub])
        for _ in range(20):
            vf = value_doc_folds(sub, y, FOLDS, rng)
            if all(not set(y[te]) - set(y[tr]) for tr, te in vf):
                break
        sf = list(StratifiedKFold(FOLDS, shuffle=True, random_state=17).split(sub, y))
        rs.append(MACRO(y, fit(sub, y, sf)))
        ds.append(MACRO(y, fit(sub, y, vf)))
        unseen.append(float(np.mean([len(set(y[te]) - set(y[tr])) for tr, te in vf])))
    per = [a / b for a, b in zip(rs, ds) if b]
    row = {"m": m, "n": len(sub), "n_values": len(set(y)), "n_docs": len({r["doc_id"] for r in sub}),
           "random_f1": float(np.mean(rs)), "random_sd": float(np.std(rs)),
           "doc_heldout_f1": float(np.mean(ds)), "doc_heldout_sd": float(np.std(ds)),
           "unseen_values": float(np.mean(unseen)), "repeats": REPEATS,
           "ratio_mean": float(np.mean(per)), "ratio_draws": [float(x) for x in per],
           "ratio_lo": float(np.percentile(per, 2.5)), "ratio_hi": float(np.percentile(per, 97.5))}
    row["ratio"] = row["random_f1"] / row["doc_heldout_f1"] if row["doc_heldout_f1"] else None
    d = np.array(rs) - np.array(ds)
    flip = (d * rng.choice([-1.0, 1.0], size=(20000, len(d)))).mean(axis=1)
    row["gap"] = float(d.mean())
    row["gap_p"] = float((1 + np.sum(np.abs(flip) >= abs(d.mean()))) / 20001)
    if perm:
        sub = draw(base, get, m, target, rng)
        y = np.array([get(r) for r in sub])
        q = [MACRO(y, fit(sub, np.array(list(rng.permutation(y))),
                          value_doc_folds(sub, y, FOLDS, rng))) for _ in range(N_PERM)]
        row["perm_null_f1"] = float(np.mean(q))
        row["perm_null_sd"] = float(np.std(q))
    return row


def main(out):
    rows = [json.loads(l) for l in open(corpus_path())]
    nd = collections.defaultdict(set)
    for r in rows:
        nd[r["nation"]].add(r["doc_id"])
    keep = {n for n, s in nd.items() if len(s) >= 2}
    base = [r for r in rows if r["nation"] in keep]
    print(f"{len(base)} provisions, {len({r['doc_id'] for r in base})} documents\n", flush=True)

    rng = np.random.default_rng(SEED)
    res = {}
    for lab, (get, target) in LABELS.items():
        print(f"--- {lab} (target {target} provisions per value) ---", flush=True)
        curve = []
        for m in MS:
            p = point(base, get, m, target, rng, perm=(m in (MS[0], MS[-1])))
            if p is None:
                print(f"  m={m:3d}  not feasible", flush=True)
                continue
            curve.append(p)
            print(f"  m={m:3d}  n={p['n']:4d} over {p['n_docs']:3d} docs  "
                  f"random {p['random_f1']:.3f}  doc-held-out {p['doc_heldout_f1']:.3f}  "
                  f"ratio {p['ratio']:.2f}x  p={p['gap_p']:.4g}"
                  + (f"  null {p['perm_null_f1']:.3f}" if "perm_null_f1" in p else ""),
                  flush=True)
        x = np.log([p["m"] for p in curve for _ in p["ratio_draws"]])
        yy = np.log([r for p in curve for r in p["ratio_draws"]])
        b = np.polyfit(x, yy, 1)[0]
        null = [np.polyfit(x, rng.permutation(yy), 1)[0] for _ in range(20000)]
        pv = float((1 + np.sum(np.abs(null) >= abs(b))) / 20001)
        print(f"  trend: slope of log inflation on log documents per value = {b:+.3f} "
              f"(permutation p = {pv:.4g}, {len(x)} draws)", flush=True)
        res[lab] = {"curve": curve, "log_slope": float(b), "log_slope_p": pv,
                    "n_draws": int(len(x))}
    json.dump({"seed": SEED, "folds": FOLDS, "repeats": REPEATS, "targets":
               {k: v[1] for k, v in LABELS.items()}, "curves": res}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/label_dose.json"))
    main(ap.parse_args().out)
