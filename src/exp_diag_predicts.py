"""Does the diagnostic predict how much a random split inflates?

Pairs each Nation's within-Nation document-probe accuracy (the diagnostic) against its
random-minus-LONO gap in provision-function classification (the inflation). Nation-level
permutation for significance; leave-one-out to show the correlation is not one Nation.
"""
import argparse, json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED, REPS = 20240501, 20000


def rank(x):
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    for v in set(x):                                   # average ties
        m = np.array(x) == v
        if m.sum() > 1:
            o[m] = o[m].mean()
    return o


def spearman(x, y):
    a, b = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else float("nan")


def main(doc_probe, classify, level, out):
    dp = json.load(open(doc_probe))["within_nation"]
    cl = json.load(open(classify))["results"]
    pick = lambda r: next((x["per_nation"] for x in cl if x["model"] == "tfidf_svm"
                           and x["regime"] == r and x["level"] == level), {})
    rnd, lono = pick("R1_random"), pick("R2_lono")
    nats = sorted(set(dp) & set(rnd) & set(lono))
    if len(nats) < 5:
        print(f"only {len(nats)} Nations have both measurements; not testing")
        return
    x = [dp[n] for n in nats]
    y = [rnd[n] - lono[n] for n in nats]
    rho = spearman(x, y)

    rng = np.random.default_rng(SEED)
    null = [spearman(x, list(rng.permutation(y))) for _ in range(REPS)]
    p = (1 + sum(abs(v) >= abs(rho) for v in null)) / (REPS + 1)
    loo = [spearman([x[i] for i in range(len(x)) if i != j],
                    [y[i] for i in range(len(y)) if i != j]) for j in range(len(nats))]

    res = {"level": level, "n_nations": len(nats), "rho": rho, "p": float(p),
           "loo_min": float(min(loo)), "loo_max": float(max(loo)),
           "pairs": {n: {"doc_probe": dp[n], "gap": rnd[n] - lono[n]} for n in nats}}
    json.dump(res, open(out, "w"), indent=1)
    print(f"n={len(nats)}  rho={rho:+.3f}  p={p:.4g}  leave-one-out [{min(loo):+.3f},{max(loo):+.3f}]")
    for n in sorted(nats, key=lambda k: dp[k]):
        print(f"  {n[:38]:38s} probe {dp[n]:.3f}   gap {rnd[n]-lono[n]:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-probe", default=str(ROOT / "results/doc_probe.json"))
    ap.add_argument("--classify", default=str(ROOT / "results/classify.json"))
    ap.add_argument("--level", default="M0")
    ap.add_argument("--out", default=str(ROOT / "results/diag_predicts.json"))
    a = ap.parse_args()
    main(a.doc_probe, a.classify, a.level, a.out)
