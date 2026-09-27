"""Rebuild every central interval and permutation test at the unit the claim is made at.

Provision-level resampling treats provisions inside a source document as independent
observations, which they are not. This refits the probe contrasts, then rebuilds their
intervals resampling documents and Nations; retests the dose-response slope with the dose
level as the exchangeable block rather than the overlapping draw; and adds a Nation-level
test of the template-adoption contrast, which was a provision-level Fisher test.
"""
import collections, itertools, json, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import load, corpus_path
from exp_nation_probe_lodo import doc_folds
from exp_template_adoption import borrowed, genre, OK_CLUSTER

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOT, PERM, SEED, K = 2000, 20000, 20260101, 4
SLOPE_BOOT = 5000
MACRO = lambda a, b: f1_score(a, b, average="macro", zero_division=0)


def fit(rows, y, folds, sizes, rng):
    pred = np.empty(len(y), dtype=object)
    for (tr, te), n in zip(folds, sizes):
        if n is not None and len(tr) > n:
            tr = tr[rng.permutation(len(tr))[:n]]
        m = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        m.fit([rows[i]["text"] for i in tr], y[tr])
        pred[te] = m.predict([rows[i]["text"] for i in te])
    return pred


def paired_ci(y, a, b, unit):
    """Resample whole clusters and rescore both regimes on the same draw."""
    idx = collections.defaultdict(list)
    for i, u in enumerate(unit):
        idx[u].append(i)
    keys = sorted(idx)
    rng = np.random.default_rng(SEED)
    ra, rb, rr = [], [], []
    for _ in range(BOOT):
        sel = [i for k in rng.choice(keys, len(keys), replace=True) for i in idx[k]]
        fa, fb = MACRO(y[sel], a[sel]), MACRO(y[sel], b[sel])
        ra.append(fa); rb.append(fb)
        if fb:
            rr.append(fa / fb)
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"n_clusters": len(keys), "random": q(ra), "doc_heldout": q(rb), "ratio": q(rr),
            "gap_p": float((1 + sum(1 for x, z in zip(ra, rb) if x <= z)) * 2 / (len(ra) + 1))}


def probe_units(rows, tag, out):
    y = np.array([r["nation"] for r in rows])
    dfolds = doc_folds(rows, k=K)
    sizes = [len(tr) for tr, _ in dfolds]
    rfolds = list(StratifiedKFold(K, shuffle=True, random_state=17).split(np.zeros(len(y)), y))
    a = fit(rows, y, rfolds, sizes, np.random.default_rng(17))
    b = fit(rows, y, dfolds, [None] * K, np.random.default_rng(17))
    res = {"n": len(y), "n_docs": len({r["doc_id"] for r in rows}),
           "n_nations": len(set(y)), "random": MACRO(y, a), "doc_heldout": MACRO(y, b)}
    res["ratio"] = res["random"] / res["doc_heldout"]
    for unit, key in (("provision", list(range(len(y)))),
                      ("document", [r["doc_id"] for r in rows]),
                      ("nation", list(y))):
        res[unit] = paired_ci(y, a, b, key)
    out[tag] = res
    print(f"{tag}: random {res['random']:.3f}  doc-held-out {res['doc_heldout']:.3f}  "
          f"ratio {res['ratio']:.2f}x")
    for unit in ("provision", "document", "nation"):
        c = res[unit]
        print(f"   {unit:9s} ({c['n_clusters']:3d})  ratio [{c['ratio'][0]:.2f}, "
              f"{c['ratio'][1]:.2f}]  gap p {c['gap_p']:.4g}")


def block_slope(curve_draws, label, out):
    """Slope of log inflation on log documents per value, permuted over dose levels.

    Draws at one dose level share documents, so they are not exchangeable with draws at
    another. Permuting whole levels respects that and caps resolution at 1/L!.
    """
    ms = sorted(curve_draws)
    blocks = [np.log(curve_draws[m]) for m in ms]
    x = np.log([m for m in ms for _ in curve_draws[m]])
    slope = float(np.polyfit(x, np.concatenate(blocks), 1)[0])
    null = []
    for p in itertools.permutations(range(len(ms))):
        null.append(np.polyfit(x, np.concatenate([blocks[i] for i in p]), 1)[0])
    null = np.array(null)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(SLOPE_BOOT):
        r = [b[rng.integers(0, len(b), len(b))] for b in blocks]
        boot.append(np.polyfit(x, np.concatenate(r), 1)[0])
    out[label] = {"levels": ms, "draws_per_level": [len(curve_draws[m]) for m in ms],
                  "slope": slope, "n_perm": len(null),
                  "p_block": float(np.mean(np.abs(null) >= abs(slope) - 1e-12)),
                  "p_floor": float(1.0 / len(null)),
                  "slope_ci": [float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))],
                  "slope_boot": SLOPE_BOOT,
                  "level_means": [float(np.mean(np.exp(b))) for b in blocks]}
    ci = out[label]["slope_ci"]
    print(f"{label}: slope {slope:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  block permutation "
          f"p = {out[label]['p_block']:.4g} over {len(null)} level orderings "
          f"(floor {out[label]['p_floor']:.4g})")


def adoption_nation(out):
    """Nation-paired counterpart of the provision-level Fisher test."""
    rows = [json.loads(l) for l in open(corpus_path())
            if json.loads(l).get("domain") in ("criminal", "civil")]
    res = {}
    for cond, sub in (("all", rows),
                      ("no_oklahoma", [r for r in rows if r["nation"] not in OK_CLUSTER]),
                      ("procedural_only", [r for r in rows if r["domain"] == "civil"
                                           or genre(r) == "procedural"])):
        cell = collections.defaultdict(lambda: [0, 0])
        for r in sub:
            c = cell[(r["nation"], r["domain"])]
            c[0] += bool(borrowed(r)); c[1] += 1
        cell = dict(cell)
        nats = sorted({n for n, _ in cell
                       if cell.get((n, "criminal"), [0, 0])[1] >= 20
                       and cell.get((n, "civil"), [0, 0])[1] >= 20})
        d = np.array([cell[(n, "civil")][0] / cell[(n, "civil")][1]
                      - cell[(n, "criminal")][0] / cell[(n, "criminal")][1] for n in nats])
        signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
        null = (signs * d).mean(axis=1)
        rng = np.random.default_rng(SEED)
        bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
        res[cond] = {"n_nations": len(nats), "delta": float(d.mean()),
                     "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     "p": float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-12)),
                     "p_floor": float(1.0 / len(null)), "n_positive": int((d > 0).sum())}
        r = res[cond]
        print(f"adoption {cond:16s} {len(nats)} Nations  delta {r['delta']:+.3f} "
              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  p {r['p']:.4g} "
              f"(floor {r['p_floor']:.4g})  {r['n_positive']}/{len(nats)} positive")
    out["adoption_nation"] = res


def docprobe_nation(out):
    """Interval over Nations for the within-Nation document probe, whose unit is the Nation."""
    d = json.load(open(ROOT / "results/docprobe_domain.json"))
    per = {n: v["score"] for n, v in d["per_nation"].items()}
    same = [v for n, v in per.items() if not d["per_nation"][n]["domains_differ"]]
    rng = np.random.default_rng(SEED)
    def ci(vals):
        vals = np.array(vals, dtype=float)
        bs = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(5000)]
        return {"n": len(vals), "mean": float(vals.mean()),
                "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    out["docprobe_nation"] = {"all": ci(list(per.values()))}
    if same:
        out["docprobe_nation"]["same_domain"] = ci(same)
    for k, v in out["docprobe_nation"].items():
        print(f"docprobe {k:12s} mean {v['mean']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] "
              f"over {v['n']} Nations")


def main():
    out = {"boot": BOOT, "seed": SEED}
    allrows = load()
    nd = collections.defaultdict(set)
    for r in allrows:
        nd[r["nation"]].add(r["doc_id"])
    keep = {n for n, s in nd.items() if len(s) >= 2}
    probe_units([r for r in allrows if r["nation"] in keep], "nation_probe", out)

    for dom in ("criminal", "civil"):
        rs = [r for r in allrows if r["domain"] == dom]
        c = collections.defaultdict(collections.Counter)
        for r in rs:
            c[r["nation"]][r["doc_id"]] += 1
        k = {n for n, v in c.items() if len(v) >= 2 and sum(v.values()) >= 40}
        probe_units([r for r in rs if r["nation"] in k], f"within_{dom}", out)

    dose = {}
    d = json.load(open(ROOT / "results/label_dose.json"))["curves"]
    for lab, v in d.items():
        block_slope({p["m"]: p["ratio_draws"] for p in v["curve"]}, lab, dose)
    sc = json.load(open(ROOT / "results/statecode_dose.json"))
    # below this dose a state can fall out of a training fold, so the ratio is inflated by a
    # missing class rather than by low diversity; the reported slope uses the covered levels
    keep = [p for p in sc["fixed_n"] if p["unseen_classes_group"] < 0.5]
    block_slope({p["k"]: p.get("ratio_draws") or [p["ratio"]] for p in keep},
                "state codes", dose)
    block_slope({p["k"]: p.get("ratio_draws") or [p["ratio"]] for p in sc["fixed_n"]},
                "state codes (all levels)", dose)
    strict = [p for p in sc["fixed_n"] if p["unseen_classes_group"] <= 0.0]
    block_slope({p["k"]: p.get("ratio_draws") or [p["ratio"]] for p in strict},
                "state codes (strict)", dose)
    out["dose_block"] = dose

    docprobe_nation(out)
    adoption_nation(out)
    json.dump(out, open(ROOT / "results/cluster_units.json", "w"), indent=1)
    print("\nwrote results/cluster_units.json")


if __name__ == "__main__":
    main()
