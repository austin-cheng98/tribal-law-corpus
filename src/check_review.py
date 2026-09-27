"""Checks a reviewer can run to attack the criterion and the diagnostic directly.

Five questions the headline numbers do not answer on their own. (1) A macro-F1 ratio is
bounded above for a label the model already predicts well, so the four labels are also
compared on the share of above-chance performance the hold-out destroys. (2) The criterion is
about label values, not labels, so every value is scored separately against the number of
documents realizing it. (3) The document fingerprint is reported per Nation against that
Nation's degenerate baseline, with and without name redaction, against a permutation null. (4) The dose design's folds are asserted document-disjoint, and nesting is measured
at every dose. (5) A Jensen-Shannon divergence between two samples of one distribution gives
the noise floor for the convergence contrast at the cell size actually used.
"""
import collections, json, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import corpus_path
from exp_label_nesting import theil_u
from exp_nation_probe_lodo import doc_folds
from exp_comparative import OK_CLUSTER

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED, NPERM, MIN_DOC = 20260101, 20, 20
MACRO = lambda y, p: f1_score(y, p, average="macro", zero_division=0)


def ranks(z):
    """Average ranks, so ties do not make the correlation depend on row order."""
    z = np.asarray(z, float)
    o = np.argsort(z, kind="mergesort")
    rk, i = np.empty(len(z)), 0
    while i < len(z):
        j = i
        while j + 1 < len(z) and z[o[j + 1]] == z[o[i]]:
            j += 1
        rk[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return rk


def spearman(a, b, reps=20000):
    ra, rb = ranks(a), ranks(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(SEED)
    null = [np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(reps)]
    return rho, float((1 + np.sum(np.abs(null) >= abs(rho))) / (reps + 1))


def model():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))


def fit(rows, y, folds):
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        if len(tr) == 0 or len(te) == 0 or len(set(y[tr])) < 2:
            continue
        pred[te] = model().fit([rows[i]["text"] for i in tr], y[tr]).predict(
            [rows[i]["text"] for i in te])
    return np.array([p if p is not None else "" for p in pred], dtype=object)


def both_regimes(rows, y):
    y = np.array(y)
    rnd = fit(rows, y, list(StratifiedKFold(4, shuffle=True, random_state=17).split(rows, y)))
    dh = fit(rows, y, doc_folds(rows, k=4))
    return y, rnd, dh


# ---------------------------------------------------------------- 1. chance-normalized
def chance_normalized(labels, out):
    """A ratio of macro-F1 is capped when the random arm is already near 1.

    The share of above-chance performance lost is not: it asks how much of what the model
    knew beyond guessing the hold-out takes away. Chance is the permuted-label score under
    the same regime, so each label is measured against its own floor.
    """
    res = {}
    for name, (rows, y) in labels.items():
        y, rnd, dh = both_regimes(rows, y)
        rng = np.random.default_rng(SEED)
        floor = float(np.mean([MACRO(y, fit(rows, np.array(list(rng.permutation(y))),
                                            doc_folds(rows, k=4))) for _ in range(NPERM)]))
        fr, fd = MACRO(y, rnd), MACRO(y, dh)
        ar, ad = accuracy_score(y, rnd), accuracy_score(y, dh)
        maj = collections.Counter(y).most_common(1)[0][1] / len(y)
        per = collections.defaultdict(set)
        for r, v in zip(rows, y):
            per[v].add(r["doc_id"])
        nd = sorted(len(s) for s in per.values())
        res[name] = {
            "n": len(y), "n_classes": len(set(y)),
            "docs_per_value_mean": float(np.mean(nd)),
            "docs_per_value_median": float(np.median(nd)),
            "docs_per_value_min": int(nd[0]),
            "values_at_two_docs": int(sum(1 for v in nd if v <= 2)),
            "random_f1": fr, "doc_heldout_f1": fd, "ratio": fr / fd if fd else None,
            "perm_null_f1": floor,
            "lost_share_f1": (fr - fd) / (fr - floor) if fr > floor else None,
            "random_acc": float(ar), "doc_heldout_acc": float(ad), "majority_acc": float(maj),
            "lost_share_acc": float((ar - ad) / (ar - maj)) if ar > maj else None}
        v = res[name]
        print(f"{name:20s} ratio {v['ratio']:.2f}x  "
              f"lost share of above-chance F1 {v['lost_share_f1']:.0%}  "
              f"accuracy {v['lost_share_acc']:.0%}  "
              f"docs/value mean {v['docs_per_value_mean']:.1f} median "
              f"{v['docs_per_value_median']:.0f}", flush=True)
    out["chance_normalized"] = res


# ------------------------------------------------------------------- 2. per label value
def per_value(labels, out):
    """Does the criterion hold within a label, across its own values?

    If source diversity is what matters, the values realized in few documents should lose
    the most, whatever label they belong to. This pools every value of every label and
    correlates its loss against its document count.
    """
    res, pooled = {}, []
    for name, (rows, y) in labels.items():
        y, rnd, dh = both_regimes(rows, y)
        docs = collections.defaultdict(set)
        for r, v in zip(rows, y):
            docs[v].add(r["doc_id"])
        rows_out = []
        for v in sorted(set(y)):
            a = f1_score(y == v, rnd == v, zero_division=0)
            b = f1_score(y == v, dh == v, zero_division=0)
            rows_out.append({"value": str(v), "n": int((y == v).sum()),
                             "n_docs": len(docs[v]), "random_f1": float(a),
                             "doc_heldout_f1": float(b),
                             "ratio": float(a / b) if b else None,
                             "drop": float(a - b)})
            pooled.append((len(docs[v]), float(a - b), name, str(v)))
        res[name] = rows_out
        for q in sorted(rows_out, key=lambda z: z["n_docs"]):
            print(f"  {name[:12]:12s} {q['value'][:30]:30s} docs={q['n_docs']:3d} "
                  f"n={q['n']:4d}  {q['random_f1']:.3f} -> {q['doc_heldout_f1']:.3f}  "
                  f"drop {q['drop']:+.3f}", flush=True)
    rho, pp = spearman(np.log([p[0] for p in pooled]), [p[1] for p in pooled])
    res["pooled"] = {"n_values": len(pooled), "spearman_docs_vs_drop": rho, "p": pp}
    print(f"\n  pooled over {len(pooled)} label values: Spearman(log documents, drop) "
          f"= {rho:+.3f}  p = {res['pooled']['p']:.4g}", flush=True)
    out["per_value"] = res


# ---------------------------------------------------------------------- 3. diagnostic
def diagnostic(rows, out):
    """The within-Nation document probe, per Nation, against its own degenerate baseline."""
    big = {d for d, c in collections.Counter(r["doc_id"] for r in rows).items()
           if c >= MIN_DOC}
    sub = [r for r in rows if r["doc_id"] in big]
    bynat = collections.defaultdict(list)
    for r in sub:
        bynat[r["nation"]].append(r)
    res = {}
    for nat, rs in sorted(bynat.items()):
        ds = {r["doc_id"] for r in rs}
        if len(ds) < 2:
            continue
        y = np.array([r["doc_id"] for r in rs])
        k = min(4, collections.Counter(y).most_common()[-1][1])
        if k < 2:
            continue
        # a probe that names the largest document for everything scores this
        share = collections.Counter(y).most_common(1)[0][1] / len(y)
        rec = {"n": len(rs), "n_docs": len(ds),
               "domains": sorted({r["domain"] for r in rs}),
               "degenerate_f1": float(2 * share / (share + 1) / len(ds))}
        for level in (0, 2):
            folds = list(StratifiedKFold(k, shuffle=True, random_state=17).split(rs, y))
            rec[f"M{level}"] = float(MACRO(y, fit_masked(rs, y, folds, level)))
        rng = np.random.default_rng(SEED)
        folds = list(StratifiedKFold(k, shuffle=True, random_state=17).split(rs, y))
        rec["perm_null"] = float(np.mean([
            MACRO(y, fit_masked(rs, np.array(list(rng.permutation(y))), folds, 0))
            for _ in range(NPERM)]))
        res[nat] = rec
        print(f"  {nat[:38]:38s} docs={rec['n_docs']}  M0 {rec['M0']:.3f}  "
              f"M2 {rec['M2']:.3f}  degenerate {rec['degenerate_f1']:.3f}  "
              f"null {rec['perm_null']:.3f}  "
              f"{'one domain' if len(rec['domains']) == 1 else 'split'}", flush=True)
    scores = [v["M0"] for v in res.values()]
    same = [v["M0"] for v in res.values() if len(v["domains"]) == 1]
    out["diagnostic"] = {
        "per_nation": res, "n_nations": len(res),
        "mean_M0": float(np.mean(scores)), "mean_M2": float(np.mean([v["M2"] for v in res.values()])),
        "same_domain_mean": float(np.mean(same)) if same else None,
        "at_or_below_degenerate": sorted(
            n for n, v in res.items() if v["M0"] <= v["degenerate_f1"] + 0.02),
        "above_null": sorted(n for n, v in res.items() if v["M0"] > v["perm_null"] + 0.05)}
    print(f"\n  mean M0 {out['diagnostic']['mean_M0']:.3f}  M2 "
          f"{out['diagnostic']['mean_M2']:.3f}; at or below degenerate: "
          f"{out['diagnostic']['at_or_below_degenerate']}", flush=True)


def fit_masked(rows, y, folds, level):
    import masking
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        cues = set()
        if level >= 3:
            f = masking.fit_discriminative_terms([rows[i] for i in tr])
            cues = set().union(*f.values()) if f else set()
        m = model().fit([masking.apply(rows[i]["text"], level, cues) for i in tr], y[tr])
        pred[te] = m.predict([masking.apply(rows[i]["text"], level, cues) for i in te])
    return np.array([p if p is not None else "" for p in pred], dtype=object)


# --------------------------------------------------- 4. the dose design's own assumptions
def dose_assumptions(rows, out):
    """Are the dose folds document-disjoint, and does nesting move with the dose?"""
    from exp_label_dose import value_doc_folds, draw, LABELS, MS
    nd = collections.defaultdict(set)
    for r in rows:
        nd[r["nation"]].add(r["doc_id"])
    base = [r for r in rows if len(nd[r["nation"]]) >= 2]
    res = {}
    for lab, (get, target) in LABELS.items():
        rng = np.random.default_rng(SEED)
        curve = []
        for m in MS:
            leaks, us = 0, []
            for _ in range(10):
                sub = draw(base, get, m, target, rng)
                if sub is None:
                    break
                y = np.array([get(r) for r in sub])
                for tr, te in value_doc_folds(sub, y, 2, rng):
                    if {sub[i]["doc_id"] for i in tr} & {sub[i]["doc_id"] for i in te}:
                        leaks += 1
                us.append(theil_u([r["doc_id"] for r in sub], list(y)))
            if not us:
                continue
            curve.append({"m": m, "doc_overlaps": leaks, "theil_u": float(np.mean(us))})
            print(f"  {lab[:16]:16s} m={m:3d}  train/test document overlaps {leaks}  "
                  f"Theil U {np.mean(us):.3f}", flush=True)
        res[lab] = curve
    out["dose_assumptions"] = res


# ------------------------------------------------------------------ 5. divergence floor
def divergence_floor(rows, out):
    """What divergence do two samples of ONE distribution give at the cell size used?

    The convergence contrast compares mean pairwise Jensen-Shannon divergence among criminal
    codes against the same among civil codes, on subsamples of a few dozen provisions. At that
    size two draws of a single distribution already diverge, and the floor depends on the
    distribution's shape, so each arm is given its own null from its own pooled class
    distribution. A measured divergence at its arm's floor says the Nations in that arm are
    indistinguishable from repeated samples of one distribution.
    """
    def jsd(p, q):
        p, q = np.asarray(p, float), np.asarray(q, float)
        p, q = p / p.sum(), q / q.sum()
        m = (p + q) / 2
        kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    obs = {(e["feature"], e["condition"]): e
           for e in json.load(open(ROOT / "results/comparative.json"))}
    feats = {"silver": lambda r: r.get("silver"),
             "modality_1": lambda r: (r["modality"] or ["NONE"])[0]}
    rng = np.random.default_rng(SEED)
    res = {"log_base": 2, "reps": 5000, "arms": {}}
    for (feat, cond), e in sorted(obs.items()):
        cell = e["subsample_n"]
        nats = set(e.get("nations") or [])
        for dom in ("criminal", "civil"):
            sel = [r for r in rows if r["domain"] == dom
                   and (not nats or r["nation"] in nats) and feats[feat](r)]
            cnt = collections.Counter(feats[feat](r) for r in sel)
            prob = np.array([cnt[c] for c in sorted(cnt)], float)
            prob /= prob.sum()
            d = [jsd(rng.multinomial(cell, prob) + 1e-12,
                     rng.multinomial(cell, prob) + 1e-12) for _ in range(5000)]
            measured = e[f"mean_jsd_{dom}"]
            key = f"{feat} | {cond} | {dom}"
            res["arms"][key] = {
                "cell": cell, "n_provisions": len(sel), "n_classes": len(cnt),
                "measured": measured, "floor_mean": float(np.mean(d)),
                "floor_p95": float(np.percentile(d, 95)),
                "at_or_below_floor": bool(measured <= np.mean(d)),
                "p_one_sided": float((1 + np.sum(np.array(d) >= measured)) / 5001)}
            v = res["arms"][key]
            print(f"  {key:56s} cell={cell:3d}  measured {measured:.4f}  "
                  f"floor {v['floor_mean']:.4f} (95th {v['floor_p95']:.4f})  "
                  f"{'AT OR BELOW FLOOR' if v['at_or_below_floor'] else 'above floor'}",
                  flush=True)
    out["divergence_floor"] = res


# ------------------------------------------------------------- 6. Oklahoma cluster removed
def no_oklahoma(rows, out):
    """The headline Nation contrast with the one jointly drafting cluster taken out."""
    nd = collections.defaultdict(set)
    for r in rows:
        nd[r["nation"]].add(r["doc_id"])
    res = {}
    for name, sel in (("all", rows),
                      ("no_oklahoma", [r for r in rows if r["nation"] not in OK_CLUSTER]),
                      ("no_near_duplicates", [r for r in rows if not r.get("xnation_dup")])):
        d = collections.defaultdict(set)
        for r in sel:
            d[r["nation"]].add(r["doc_id"])
        sub = [r for r in sel if len(d[r["nation"]]) >= 2]
        if len({r["nation"] for r in sub}) < 3:
            continue
        y, rnd, dh = both_regimes(sub, [r["nation"] for r in sub])
        fr, fd = MACRO(y, rnd), MACRO(y, dh)
        res[name] = {"n": len(sub), "n_nations": len(set(y)),
                     "n_docs": len({r["doc_id"] for r in sub}),
                     "random_f1": fr, "doc_heldout_f1": fd,
                     "ratio": fr / fd if fd else None}
        print(f"  {name:20s} n={len(sub):5d}  {len(set(y)):2d} Nations  "
              f"random {fr:.3f} -> doc-held-out {fd:.3f}  ratio {fr/fd:.2f}x", flush=True)
    out["no_oklahoma"] = res


# ---------------------------------------------------------------- 7. sub-baseline accuracy
def sibling_confusion(rows, out):
    """Why held-out accuracy falls under a majority-class predictor.

    Macro-F1 and accuracy disagree on the Nation label because the errors are not spread
    over the classes. Each condition is scored with and without class weighting, and the
    errors are traced to their destination: a Nation from the jointly drafting Oklahoma
    cluster, or anywhere else.
    """
    res = {}
    for name, sel in (("all", rows),
                      ("no_oklahoma", [r for r in rows if r["nation"] not in OK_CLUSTER]),
                      ("no_near_duplicates", [r for r in rows if not r.get("xnation_dup")])):
        d = collections.defaultdict(set)
        for r in sel:
            d[r["nation"]].add(r["doc_id"])
        sub = [r for r in sel if len(d[r["nation"]]) >= 2]
        if len({r["nation"] for r in sub}) < 3:
            continue
        y, rnd, dh = both_regimes(sub, [r["nation"] for r in sub])
        maj, majn = collections.Counter(y).most_common(1)[0]
        rec = {"n": len(sub), "n_nations": len(set(y)), "majority_nation": maj,
               "majority_acc": majn / len(y),
               "random_acc": accuracy_score(y, rnd), "heldout_acc": accuracy_score(y, dh),
               "heldout_macro_f1": MACRO(y, dh)}
        if name == "all":
            # unweighted, to rule out class_weight="balanced" as the cause
            un = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                               LogisticRegression(max_iter=2000, C=4.0))
            pu = np.empty(len(y), dtype=object)
            for tr, te in doc_folds(sub, k=4):
                pu[te] = un.fit([sub[i]["text"] for i in tr], y[tr]).predict(
                    [sub[i]["text"] for i in te])
            rec["heldout_acc_unweighted"] = accuracy_score(y, pu)
            others = len(set(y)) - 1
            err = [(t, p) for t, p in zip(y, dh) if t != p]
            in_cl = [(t, p) for t, p in err if t in OK_CLUSTER]
            rec["cluster_nations"] = sorted(OK_CLUSTER & set(y))
            rec["majority_errors"] = sum(t == maj for t, _ in err)
            rec["majority_errors_to_cluster"] = (
                sum(t == maj and p in OK_CLUSTER for t, p in err) / rec["majority_errors"])
            rec["chance_to_cluster"] = (len(OK_CLUSTER & set(y)) - 1) / others
            rec["cluster_errors_stay_in_cluster"] = (
                sum(p in OK_CLUSTER for _, p in in_cl) / len(in_cl)) if in_cl else None
        res[name] = rec
        print(f"  {name:20s} held-out acc {rec['heldout_acc']:.3f} vs majority "
              f"{rec['majority_acc']:.3f}  "
              f"({'above' if rec['heldout_acc'] > rec['majority_acc'] else 'BELOW'})", flush=True)
    a = res.get("all", {})
    if "majority_errors" in a:
        print(f"  {a['majority_nation']}: {a['majority_errors_to_cluster']:.0%} of "
              f"{a['majority_errors']} held-out errors name a cluster Nation, against "
              f"{a['chance_to_cluster']:.0%} by chance")
        print(f"  errors on cluster Nations stay in the cluster "
              f"{a['cluster_errors_stay_in_cluster']:.0%} of the time")
    out["sibling_confusion"] = res


# ------------------------------------------------- 8. does the fingerprint order the damage?
def fingerprint_association(out):
    """Post-processing: is the fingerprint, or source diversity, predictive within a label?

    The body claims recognizable sources are a precondition for inflation, not a predictor of
    its size, and that source diversity orders the damage across label values but not inside
    the Nation label. Both are checked here on the per-Nation results already computed: the
    fingerprint against the inflation drop, and documents per Nation against the same drop.
    """
    pn = out.get("diagnostic", {}).get("per_nation")
    pv = out.get("per_value", {}).get("Nation")
    if not pn or not pv:
        return
    drop = {q["value"]: q["drop"] for q in pv}
    docs = {q["value"]: q["n_docs"] for q in pv}
    shared = sorted(set(pn) & set(drop))
    rho, pp = spearman([pn[n]["M0"] for n in shared], [drop[n] for n in shared])
    res = {"n_nations": len(shared), "rho_finger_vs_drop": rho, "p": pp, "leave_one_out": {}}
    for n in out["diagnostic"]["at_or_below_degenerate"]:
        sub = [m for m in shared if m != n]
        r2, p2 = spearman([pn[m]["M0"] for m in sub], [drop[m] for m in sub])
        res["leave_one_out"][n] = {"rho": r2, "p": p2, "n_nations": len(sub)}

    # source diversity inside the Nation label, over every Nation the analysis set holds
    nats = sorted(docs)
    r3, p3 = spearman([docs[n] for n in nats], [drop[n] for n in nats])
    lo = sorted(n for n in nats if docs[n] <= 3)
    hi = max(nats, key=lambda n: docs[n])
    floor = set(out["diagnostic"]["at_or_below_degenerate"])
    res["within_label"] = {
        "n_nations": len(nats), "rho_docs_vs_drop": r3, "p": p3,
        "n_docs_total": sum(docs.values()), "n_at_three_or_fewer": len(lo),
        "drop_lo_min": min(drop[n] for n in lo), "drop_lo_max": max(drop[n] for n in lo),
        "drop_lo_min_excl_floor": min(drop[n] for n in lo if n not in floor),
        "drop_lo_median": float(np.median([drop[n] for n in lo if n not in floor])),
        "top_nation": hi, "top_docs": docs[hi], "top_drop": drop[hi]}
    w = res["within_label"]
    print(f"  Spearman(fingerprint, drop) = {rho:+.3f}  p = {pp:.4g}  "
          f"over {len(shared)} Nations")
    for n, v in res["leave_one_out"].items():
        print(f"    without {n}: {v['rho']:+.3f}  p = {v['p']:.3g}")
    print(f"  Spearman(documents, drop) inside the Nation label = {r3:+.3f}  p = {p3:.3g}  "
          f"over {len(nats)} Nations, {w['n_docs_total']} documents")
    print(f"  {w['n_at_three_or_fewer']} Nations at three documents or fewer, drop "
          f"{w['drop_lo_min']:.2f} to {w['drop_lo_max']:.2f} "
          f"({w['drop_lo_min_excl_floor']:.2f} to {w['drop_lo_max']:.2f}, median "
          f"{w['drop_lo_median']:.2f}, excluding the two at the fingerprint floor); "
          f"{w['top_nation']} at {w['top_docs']} documents drops {w['top_drop']:.2f}")
    out["fingerprint_association"] = res


def main():
    rows = [json.loads(l) for l in open(corpus_path())]
    nd = collections.defaultdict(set)
    for r in rows:
        nd[r["nation"]].add(r["doc_id"])
    base = [r for r in rows if len(nd[r["nation"]]) >= 2]
    gold = json.load(open(ROOT / "results/gold.json"))["gold"]
    gsub = [r for r in base if gold.get(r["provision_id"])]
    keep = {n for n, c in collections.Counter(r["nation"] for r in gsub).items() if c >= 10}
    gsub = [r for r in gsub if r["nation"] in keep]
    labels = {"Nation": (base, [r["nation"] for r in base]),
              "legal domain": (base, [r["domain"] for r in base]),
              "deontic modality": (base, [(r["modality"] or ["NONE"])[0] for r in base]),
              "provision function": (gsub, [gold[r["provision_id"]] for r in gsub])}
    out = {"seed": SEED, "n_base": len(base), "n_gold": len(gsub)}

    print("=== 1. share of above-chance performance lost ===", flush=True)
    chance_normalized(labels, out)
    print("\n=== 2. per label value ===", flush=True)
    per_value(labels, out)
    print("\n=== 3. document fingerprint per Nation ===", flush=True)
    diagnostic(rows, out)
    print("\n=== 4. dose design assumptions ===", flush=True)
    dose_assumptions(rows, out)
    print("\n=== 5. divergence noise floor ===", flush=True)
    divergence_floor(rows, out)
    print("\n=== 6. Nation contrast without the Oklahoma cluster ===", flush=True)
    no_oklahoma(rows, out)
    print("\n=== 7. where the held-out errors go ===", flush=True)
    sibling_confusion(rows, out)

    print("\n=== 8. does the fingerprint order the damage? ===", flush=True)
    fingerprint_association(out)

    json.dump(out, open(ROOT / "results/review_checks.json", "w"), indent=1)
    print("\nwrote results/review_checks.json")


if __name__ == "__main__":
    main()
