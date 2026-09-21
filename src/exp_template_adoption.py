"""Does the civil side of the corpus carry more borrowed drafting than the criminal side?

The criminal and civil sides of an enacted code are not matched on subject matter, so this
records the composition of each, counts provisions that cite an external model instrument or
retain an unfilled template placeholder, and re-runs the reuse contrast with the criminal side
restricted to its procedural provisions.
"""
import collections, json, pathlib, re, sys
from scipy.stats import fisher_exact

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from exp_diffusion import shingles, find_pairs, paired_test
from data import corpus_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
OK_CLUSTER = {"Pawnee Nation of Oklahoma", "Sac & Fox Nation", "Kickapoo Tribe of Oklahoma",
              "Wyandotte Nation", "Iowa Tribe of Oklahoma", "Delaware Tribe of Indians"}

SUB = re.compile(r"OFFENS|CRIMES|CRIMINAL CODE|PENAL", re.I)
PRO = re.compile(r"PROCEDUR|RULES OF|PLEADING|DISCOVERY|APPELLATE", re.I)
EXTERNAL = re.compile(
    r"Federal Rules of (?:Civil|Criminal) Procedure|Federal Rules of Evidence"
    r"|Court of Indian Offenses|25 C\.?F\.?R", re.I)
PLACEHOLDER = re.compile(r"\[\s*(?:tribe|nation|name|insert|xxx)[^\]]{0,30}\]|_{4,}|\bXXX\b", re.I)


def genre(r):
    """Subject matter, read from the document and chapter headings rather than the text."""
    st = " > ".join(filter(None, [r.get("doc_title"), r.get("heading_path"),
                                  r.get("chapter_heading")]))
    s, p = bool(SUB.search(st)), bool(PRO.search(st))
    return "both" if s and p else "substantive" if s else "procedural" if p else "unmarked"


def borrowed(r):
    return bool(EXTERNAL.search(r["text"]) or PLACEHOLDER.search(r["text"]))


def main():
    rows = [json.loads(l) for l in open(corpus_path())
            if json.loads(l).get("domain") in ("criminal", "civil")]

    comp = {d: collections.Counter(genre(r) for r in rows if r["domain"] == d)
            for d in ("criminal", "civil")}
    for d, c in comp.items():
        n = sum(c.values())
        print(f"{d:9s} n={n}  " + "  ".join(f"{k} {v} ({v/n:.1%})" for k, v in c.most_common()))

    out = {"composition": {d: dict(c) for d, c in comp.items()}, "adoption": {}}
    for cond, sub in (("all", rows),
                      ("no_oklahoma", [r for r in rows if r["nation"] not in OK_CLUSTER]),
                      ("procedural_only", [r for r in rows if r["domain"] == "civil"
                                           or genre(r) == "procedural"])):
        tot = collections.Counter(r["domain"] for r in sub)
        hit = [r for r in sub if borrowed(r)]
        c = collections.Counter(r["domain"] for r in hit)
        odds, p = fisher_exact([[c["civil"], tot["civil"] - c["civil"]],
                                [c["criminal"], tot["criminal"] - c["criminal"]]])
        out["adoption"][cond] = {
            "civil_n": c["civil"], "civil_tot": tot["civil"],
            "criminal_n": c["criminal"], "criminal_tot": tot["criminal"],
            "civil_rate": c["civil"] / tot["civil"], "criminal_rate": c["criminal"] / tot["criminal"],
            "civil_nations": len({r["nation"] for r in hit if r["domain"] == "civil"}),
            "criminal_nations": len({r["nation"] for r in hit if r["domain"] == "criminal"}),
            "odds_ratio": float(odds), "p": float(p)}
        print(f"{cond:12s} civil {c['civil']}/{tot['civil']} ({c['civil']/tot['civil']:.2%})  "
              f"criminal {c['criminal']}/{tot['criminal']} ({c['criminal']/tot['criminal']:.2%})  "
              f"OR={odds:.1f} p={p:.2g}")

    sh = {r["provision_id"]: shingles(r["text"]) for r in rows}
    involved = {q for pr in find_pairs(sh, {r["provision_id"]: r["nation"] for r in rows})
                for q in pr}
    for cond, sub in (("all", rows),
                      ("procedural_only", [r for r in rows if r["domain"] == "civil"
                                           or genre(r) == "procedural"])):
        t = paired_test(sub, involved)
        out.setdefault("reuse", {})[cond] = {
            "delta": t["delta"], "ci": t["ci"], "p": t["p"], "n_nations": len(t["nations"]),
            "n_criminal": sum(1 for r in sub if r["domain"] == "criminal")}
        print(f"reuse {cond:16s} civil-criminal {t['delta']:+.3f} "
              f"[{t['ci'][0]:+.3f},{t['ci'][1]:+.3f}] p={t['p']:.4g} ({len(t['nations'])} Nations)")

    json.dump(out, open(ROOT / "results/template_adoption.json", "w"), indent=1)


if __name__ == "__main__":
    main()
