"""Ingest the coded spreadsheet and report annotation reliability.

Repeated provisions are scored against each other for intra-annotator agreement and
then collapsed to their first showing. INSUFFICIENT_CONTEXT is recorded but excluded
from the gold label, since it marks an unjudgeable excerpt rather than a class.
"""
import argparse, collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import corpus_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = ["DEFINITION", "GRANT_OF_AUTHORITY", "SANCTION_REMEDY", "RIGHT_ENTITLEMENT",
           "PROHIBITION", "OBLIGATION", "PROCEDURE", "SCOPE_APPLICABILITY"]
SKIP = "INSUFFICIENT_CONTEXT"


def read_xlsx(path):
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True, data_only=True)["code"]
    rows = ws.iter_rows(min_row=2, values_only=True)
    return {int(r[0]): str(r[1]).strip() for r in rows if r[0] is not None and r[1]}


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    idx = {c: k for k, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main(xlsx, out):
    items = json.load(open(ROOT / "data/interim/gold_sample.json"))
    byi = {it["i"]: it for it in items}
    coded = read_xlsx(xlsx)
    bad = {v for v in coded.values() if v not in CLASSES + [SKIP]}
    if bad:
        sys.exit(f"unrecognised labels in the sheet: {sorted(bad)}")
    print(f"{len(coded)}/{len(items)} rows coded ({len(coded)/len(items):.0%})")

    # intra-annotator: the same provision shown twice, far apart, unmarked
    pos = collections.defaultdict(list)
    for it in items:
        pos[it["provision_id"]].append(it["i"])
    pairs = [(a, b) for v in pos.values() if len(v) == 2 for a, b in [sorted(v)]
             if a in coded and b in coded]
    blindp = [(a, b) for a, b in pairs if not byi[a]["suggested"] and not byi[b]["suggested"]]
    rel = {}
    for name, ps in (("all", pairs), ("blind_only", blindp)):
        if len(ps) >= 5:
            x = [coded[a] for a, _ in ps]
            y = [coded[b] for _, b in ps]
            rel[name] = {"n": len(ps), "raw": float(np.mean([p == q for p, q in zip(x, y)])),
                         "kappa": kappa(x, y)}
            r = rel[name]
            print(f"intra-annotator ({name}): n={r['n']}  raw {r['raw']:.3f}  "
                  f"kappa {r['kappa']:.3f}")

    # collapse repeats to the first showing, drop unjudgeable excerpts
    gold, skipped = {}, 0
    for it in sorted(items, key=lambda x: x["i"]):
        lab = coded.get(it["i"])
        if not lab or it["provision_id"] in gold:
            continue
        if lab == SKIP:
            skipped += 1
            continue
        gold[it["provision_id"]] = lab
    print(f"\ngold provisions {len(gold)}  |  marked unjudgeable {skipped} "
          f"({skipped/max(len(coded),1):.1%})")

    # how the two automatic annotators compare, split by whether a suggestion was shown
    agr = {}
    for name, sel in (("blind", lambda it: not it["suggested"]),
                      ("suggested", lambda it: it["suggested"])):
        sub = [it for it in items if sel(it) and coded.get(it["i"])
               and coded[it["i"]] != SKIP]
        rule = [it for it in sub if it.get("rule")]
        agr[name] = {
            "n": len(sub),
            "rule_n": len(rule),
            "rule": float(np.mean([it["rule"] == coded[it["i"]] for it in rule]))
                    if rule else None,
            "nli": float(np.mean([it["nli"] == coded[it["i"]] for it in sub])) if sub else None,
        }
        a = agr[name]
        print(f"human vs rules ({name}): {a['rule']:.3f} on {a['rule_n']}  |  "
              f"vs zero-shot: {a['nli']:.3f} on {a['n']}")
    accept = [it for it in items if it["suggested"] and coded.get(it["i"])]
    agr["accept_rate"] = float(np.mean([coded[it["i"]] == it["suggested"] for it in accept]))
    print(f"suggestion accept rate {agr['accept_rate']:.3f} on {len(accept)}  "
          f"(vs blind rule agreement {agr['blind']['rule']:.3f} -- the anchoring check)")

    dist = collections.Counter(gold.values())
    print("\ngold distribution:")
    for k, v in dist.most_common():
        print(f"  {v:4d}  {k}")

    json.dump({"gold": gold, "reliability": rel, "agreement": agr,
               "distribution": dict(dist), "n_coded": len(coded),
               "n_unjudgeable": skipped}, open(out, "w"), indent=1)

    # merge into the corpus
    src = corpus_path()
    rows = [json.loads(l) for l in open(src)]
    for r in rows:
        if r["provision_id"] in gold:
            r["gold"] = gold[r["provision_id"]]
    with open(src, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nmerged {sum('gold' in r for r in rows)} gold labels into "
          f"{src.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(ROOT / "annotation/provisions_to_code.xlsx"))
    ap.add_argument("--out", default=str(ROOT / "results/gold.json"))
    a = ap.parse_args()
    main(a.xlsx, a.out)
