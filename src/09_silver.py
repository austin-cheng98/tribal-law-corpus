"""Combine rule-based and NLI labels into silver annotations and report their agreement.

Agreement here is between two automatic annotators. It is not human inter-annotator
agreement and is never reported as such.
"""
import collections, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import labeling_functions as lf

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)


def krippendorff_nominal(pairs):
    """Alpha for two coders over nominal categories."""
    cats = sorted({c for p in pairs for c in p})
    idx = {c: i for i, c in enumerate(cats)}
    n = len(pairs)
    obs = sum(1 for a, b in pairs if a != b) / n
    marg = collections.Counter(c for p in pairs for c in p)
    tot = 2 * n
    exp = 1 - sum((v / tot) ** 2 for v in marg.values())
    exp *= tot / (tot - 1)
    return 1 - obs / exp if exp else float("nan")


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/processed/corpus.jsonl")]
    nli = {}
    p = ROOT / "data/interim/nli_labels.jsonl"
    if p.exists():
        for l in open(p):
            d = json.loads(l)
            nli[d["provision_id"]] = d
    agree, out = [], []
    for r in rows:
        rule, nfired = lf.label_provision(r["text"], r.get("heading", ""))
        r["rule_label"] = rule
        r["rule_support"] = nfired
        r["modality"] = lf.modality(r["text"])
        r["xref"] = lf.xrefs(r["text"])
        d = nli.get(r["provision_id"])
        if d:
            r["nli_label"] = d["nli_label"]
            r["nli_conf"] = max(d["nli_probs"].values())
            if rule:
                agree.append((rule, d["nli_label"]))
        r["silver"] = rule or r.get("nli_label")
        r["silver_high_conf"] = bool(rule and d and rule == d["nli_label"])
        out.append(r)
    with open(ROOT / "data/processed/corpus_labeled.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    cov = sum(1 for r in out if r["silver"])
    print(f"{len(out)} provisions | silver coverage {cov} ({cov/len(out):.1%})")
    print("silver distribution:", dict(collections.Counter(r["silver"] for r in out).most_common()))
    if agree:
        raw = sum(1 for a, b in agree if a == b) / len(agree)
        print(f"rule-vs-NLI on {len(agree)} items: raw {raw:.3f}, Krippendorff alpha "
              f"{krippendorff_nominal(agree):.3f}  (automatic annotators, not human IAA)")
        print(f"both agree (high-confidence silver): {sum(1 for r in out if r['silver_high_conf'])}")


if __name__ == "__main__":
    main()
