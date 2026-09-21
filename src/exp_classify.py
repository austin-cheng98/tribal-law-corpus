"""Provision-function classification under three nested training regimes.

The test set is held fixed within each fold; only the composition of the training set
differs between regimes, so the gap between them isolates the effect of having seen the
test Nation and its documents. Evaluation is on human-verified labels only.
"""
import argparse, collections, json, pathlib, sys, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import SEED, corpus_path, load
import masking, labeling_functions as lf

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_TEST, REPS = 15, 20000
MACRO = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
REGIMES = ["R1_random", "R1d_doc_heldout", "R2_lono"]


def regimes(rows, held_out, test_ids, test_docs, rng):
    """R1 sees the Nation and its documents; R1d sees the Nation only; R2 sees neither.
    All three are matched in size to R2, so the contrast is composition alone."""
    pool = [r for r in rows if r["provision_id"] not in test_ids]
    r2 = [r for r in pool if r["nation"] != held_out]
    r1d = [r for r in pool if r["doc_id"] not in test_docs]
    n = len(r2)
    pick = lambda xs: [xs[i] for i in rng.permutation(len(xs))[:n]]
    return {"R1_random": pick(pool), "R1d_doc_heldout": pick(r1d), "R2_lono": r2}


def featurise(train, test, level):
    cues = set()
    if level >= 3:
        fitted = masking.fit_discriminative_terms(train)
        cues = set().union(*fitted.values()) if fitted else set()
    return ([masking.apply(r["text"], level, cues) for r in train],
            [masking.apply(r["text"], level, cues) for r in test])


def svm():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000),
        LinearSVC(C=0.5, class_weight="balanced"))


def logreg():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))


def cluster_bootstrap(items, key, reps=5000, seed=SEED):
    """Resample Nations, not provisions: items from one Nation are not independent."""
    rng = np.random.default_rng(seed)
    bynat = collections.defaultdict(list)
    for it in items:
        bynat[it["nation"]].append(it)
    nats = sorted(bynat)
    vals = []
    for _ in range(reps):
        draw = [x for n in rng.choice(nats, len(nats), replace=True) for x in bynat[n]]
        vals.append(MACRO([d["gold"] for d in draw], [d[key] for d in draw]))
    point = MACRO([d["gold"] for d in items], [d[key] for d in items])
    return [float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def signflip(per_nation_a, per_nation_b, reps=REPS, seed=SEED):
    """Paired across Nations, which is the unit that regimes actually vary over."""
    d = np.array([per_nation_a[n] - per_nation_b[n] for n in sorted(per_nation_a)])
    rng = np.random.default_rng(seed)
    null = (d * rng.choice([-1.0, 1.0], size=(reps, len(d)))).mean(axis=1)
    return float(d.mean()), float((1 + np.sum(np.abs(null) >= abs(d.mean()))) / (reps + 1))


def main(corpus, label_key, gold_key, out, encoder_level, test_frac, min_test):
    rows = [r for r in load(corpus) if r.get(label_key)]
    gold = [r for r in rows if r.get(gold_key)]
    bynat = collections.defaultdict(list)
    for r in gold:
        bynat[r["nation"]].append(r)
    # When the training labels are the verified ones, testing on a Nation's whole verified
    # set would leave R1 with nothing of that Nation to see, collapsing the three regimes.
    # A fraction is held out instead, so the rest stays available to R1 and R1d.
    if test_frac < 1.0:
        sp = np.random.default_rng(SEED)
        cut = {}
        for n, v in bynat.items():
            v = [v[i] for i in sp.permutation(len(v))]
            cut[n] = v[:max(min_test, int(round(test_frac * len(v))))]
        bynat = cut
    folds = {n: v for n, v in sorted(bynat.items()) if len(v) >= min_test}
    print(f"{len(rows)} labelled | {len(gold)} verified | "
          f"{len(folds)} evaluation folds", flush=True)

    rng = np.random.default_rng(SEED)
    preds = collections.defaultdict(list)          # (model, regime, level) -> item records
    for nat, items in folds.items():
        t0 = time.time()
        test_ids = {r["provision_id"] for r in items}
        test_docs = {r["doc_id"] for r in items}
        regs = regimes(rows, nat, test_ids, test_docs, rng)
        base = [{"nation": nat, "provision_id": r["provision_id"], "gold": r[gold_key]}
                for r in items]
        for level in range(4):
            for name, mk in (("tfidf_svm", svm), ("tfidf_lr", logreg)):
                for reg, tr in regs.items():
                    Xtr, Xte = featurise(tr, items, level)
                    p = mk().fit(Xtr, [r[label_key] for r in tr]).predict(Xte)
                    preds[(name, reg, f"M{level}")] += [
                        dict(b, pred=v) for b, v in zip(base, p)]
        if encoder_level is not None:
            from encoder import MODEL, fit_predict
            labels = sorted({r[label_key] for r in rows} | {r[gold_key] for r in gold})
            for reg, tr in regs.items():
                Xtr, Xte = featurise(tr, items, encoder_level)
                p = fit_predict(Xtr, [r[label_key] for r in tr], Xte, labels)
                preds[(MODEL, reg, f"M{encoder_level}")] += [
                    dict(b, pred=v) for b, v in zip(base, p)]
        # training-free references
        rb = [lf.label_provision(r["text"], r.get("heading", ""))[0] or "PROCEDURE"
              for r in items]
        preds[("rules", "none", "M0")] += [dict(b, pred=v) for b, v in zip(base, rb)]
        nb = [r.get("nli_label") or "PROCEDURE" for r in items]
        preds[("nli_zeroshot", "none", "M0")] += [dict(b, pred=v) for b, v in zip(base, nb)]
        print(f"  {nat[:34]:34s} n={len(items):4d}  {time.time()-t0:.0f}s", flush=True)

    recs = []
    for (model, reg, lvl), items in preds.items():
        pn = {n: MACRO([d["gold"] for d in items if d["nation"] == n],
                       [d["pred"] for d in items if d["nation"] == n])
              for n in {d["nation"] for d in items}}
        recs.append({"model": model, "regime": reg, "level": lvl,
                     "macro_f1": cluster_bootstrap(items, "pred"),
                     "per_nation": pn})
    tests = []
    for model in {r["model"] for r in recs} - {"rules", "nli_zeroshot"}:
        for lvl in {r["level"] for r in recs if r["model"] == model}:
            g = {r["regime"]: r["per_nation"] for r in recs
                 if r["model"] == model and r["level"] == lvl}
            if set(REGIMES) <= set(g):
                for a, b in (("R1_random", "R2_lono"), ("R1_random", "R1d_doc_heldout"),
                             ("R1d_doc_heldout", "R2_lono")):
                    d, p = signflip(g[a], g[b])
                    tests.append({"model": model, "level": lvl, "a": a, "b": b,
                                  "delta": d, "p": p})
    json.dump({"results": recs, "tests": tests, "folds": {n: len(v) for n, v in folds.items()}},
              open(out, "w"), indent=1)

    get = lambda m, r, l: next((x["macro_f1"] for x in recs if x["model"] == m
                                and x["regime"] == r and x["level"] == l), [float("nan")] * 3)
    print(f"\n{'':22s} {'random':>8s} {'doc-held':>9s} {'LONO':>8s}   rand-LONO   p")
    for lvl in ["M0", "M1", "M2", "M3"]:
        for name in sorted({r["model"] for r in recs} - {"rules", "nli_zeroshot"}):
            a, b, c = (get(name, k, lvl)[0] for k in REGIMES)
            t = next((x for x in tests if x["model"] == name and x["level"] == lvl
                      and x["a"] == "R1_random" and x["b"] == "R2_lono"), None)
            if np.isnan(a):
                continue
            print(f"{lvl} {name[:18]:18s} {a:8.3f} {b:9.3f} {c:8.3f}   "
                  f"{a-c:+9.3f}   {t['p']:.4g}" if t else "")
    for m in ("rules", "nli_zeroshot"):
        v = get(m, "none", "M0")
        print(f"{m:22s} {v[0]:8.3f} [{v[1]:.3f},{v[2]:.3f}]  (no training data)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(corpus_path()))
    ap.add_argument("--label", default="silver")
    ap.add_argument("--gold", default="gold")
    ap.add_argument("--out", default=str(ROOT / "results/classify.json"))
    ap.add_argument("--encoder-level", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=1.0)
    ap.add_argument("--min-test", type=int, default=MIN_TEST)
    a = ap.parse_args()
    main(a.corpus, a.label, a.gold, a.out, a.encoder_level, a.test_frac, a.min_test)
