"""Structural assignment of provisions to criminal or civil procedure.

Domain is read off the enacted heading hierarchy, never off provision wording, so the
label is independent of the lexical features later used to model it. Provisions whose
enclosing structure names neither domain are left unassigned rather than guessed.
"""
import collections, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CRIM = re.compile(r"criminal\s+procedure|rules?\s+of\s+criminal|criminal\s+(?:code|actions?|offen\w+)"
                  r"|crimes?\s+and\s+offen|penal\s+code|\boffenses?\b|\bcrimes?\b|misdemeanor"
                  r"|arrest|sentencing|\bbail\b|criminal", re.I)
CIV = re.compile(r"civil\s+procedure|rules?\s+of\s+civil|civil\s+actions?|civil\s+code"
                 r"|small\s+claims", re.I)
# "civil rights" is a rights chapter, not civil procedure
NOT_CIV = re.compile(r"civil\s+rights?", re.I)


def assign(*fields):
    blob = " ".join(f or "" for f in fields)
    civ = bool(CIV.search(NOT_CIV.sub(" ", blob)))
    crim = bool(CRIM.search(blob))
    if civ and not crim:
        return "civil"
    if crim and not civ:
        return "criminal"
    return None


def main():
    src = ROOT / "data/interim/provisions_raw.jsonl"
    rows = [json.loads(l) for l in open(src)]
    by_src = collections.Counter()
    for r in rows:
        d = assign(r.get("heading_path"))
        if d:
            by_src["heading"] += 1
        else:
            d = assign(r.get("doc_title"))
            if d:
                by_src["doc_title"] += 1
        r["domain"] = d
        r["domain_src"] = ("heading" if assign(r.get("heading_path")) else
                           ("doc_title" if d else None))
    with open(src, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tot = collections.Counter(r["domain"] for r in rows)
    print(f"{len(rows)} provisions | {dict(tot)} | evidence {dict(by_src)}")
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["nation"]][r["domain"]] += 1
    ok = [(n, c) for n, c in by.items() if c["criminal"] >= 20 and c["civil"] >= 20]
    print(f"Nations with >=20 in both domains: {len(ok)}")
    for n, c in sorted(ok, key=lambda x: -(x[1]["criminal"] + x[1]["civil"])):
        print(f"   crim {c['criminal']:5d} / civ {c['civil']:5d}   {n[:52]}")


if __name__ == "__main__":
    main()
