"""Is the within-Nation document probe recognizing documents, or recognizing subject matter?

Some Nations publish their criminal and civil material as separate files, so telling those
two documents apart is partly a domain classification. This splits the per-Nation probe
scores by whether a Nation's documents share a dominant domain, which bounds how much of the
headline figure is subject matter rather than document style.
"""
import collections, json, pathlib, statistics as st, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import corpus_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_DOC = 20


def degenerate(k):
    """Macro-F1 of a single-class predictor over k balanced classes."""
    return (2 * (1 / k) / (1 / k + 1)) / k


def main():
    scores = json.load(open(ROOT / "results/doc_probe.json"))["within_nation"]
    rows = [json.loads(l) for l in open(corpus_path())]

    bydoc = collections.defaultdict(list)
    for r in rows:
        bydoc[(r["nation"], r["doc_id"])].append(r)
    byn = collections.defaultdict(dict)
    for (n, d), v in bydoc.items():
        if len(v) >= MIN_DOC:
            byn[n][d] = v

    split = {"differ": [], "same": []}
    detail = {}
    for n, sc in scores.items():
        doms = [collections.Counter(r["domain"] for r in v).most_common(1)[0][0]
                for v in byn[n].values()]
        key = "differ" if len(set(doms)) > 1 else "same"
        split[key].append(sc)
        detail[n] = {"score": sc, "n_docs": len(byn[n]), "domains_differ": key == "differ",
                     "degenerate": degenerate(len(byn[n]))}

    out = {"overall_mean": st.mean(scores.values()),
           "differ_mean": st.mean(split["differ"]), "differ_n": len(split["differ"]),
           "same_mean": st.mean(split["same"]), "same_n": len(split["same"]),
           "same_at_degenerate": sorted(n for n, d in detail.items()
                                        if not d["domains_differ"]
                                        and d["score"] <= d["degenerate"] + 0.07),
           "per_nation": detail}
    print(f"overall {out['overall_mean']:.3f} | documents differ in domain "
          f"{out['differ_mean']:.3f} (n={out['differ_n']}) | same domain "
          f"{out['same_mean']:.3f} (n={out['same_n']})")
    print("near the degenerate rate:", ", ".join(out["same_at_degenerate"]) or "none")
    json.dump(out, open(ROOT / "results/docprobe_domain.json", "w"), indent=1)


if __name__ == "__main__":
    main()
