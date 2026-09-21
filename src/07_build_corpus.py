"""Assemble the analysis corpus: filter to procedure domains, normalise metadata,
and flag cross-Nation near-duplicates (model-code diffusion, and a leakage risk for
leave-one-Nation-out evaluation)."""
import collections, hashlib, json, pathlib, re, sys, urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_WORDS, MAX_WORDS = 15, 600
MIN_PROV_PER_NATION = 60

CRIM = re.compile(r"criminal|offen[cs]e|arrest|arraign|bail|bond|sentenc|prosecut|misdemeanor"
                  r"|penal|jail|probation|parole|police|law\s*(and|&)\s*order", re.I)
CIV = re.compile(r"civil|complaint|summons|pleading|service of process|small claims|judgment"
                 r"|discovery|injunction|garnish|mandamus|replevin|damages|forcible entry", re.I)
NONPROC = re.compile(r"gaming|casino|tax|zoning|business licen|utilit|housing|employment|election"
                     r"|enrollment|membership|environment|water quality|fish|wildlife|liquor licen"
                     r"|cemetery|health|education|budget|procurement|corporation", re.I)


def norm_nation(n):
    n = re.sub(r",\s*(?:the\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$", "", n.strip())
    n = re.sub(r"\s+of the .*Reservation.*$", "", n, flags=re.I)
    n = re.sub(r"\s*\(.*?\)\s*", " ", n)
    return re.sub(r"\s+", " ", n).strip(" ,")


def shingle_hash(t, k=9):
    w = re.findall(r"[a-z]+", t.lower())
    if len(w) < k:
        return set()
    return {hashlib.blake2b(" ".join(w[i:i + k]).encode(), digest_size=8).digest()
            for i in range(len(w) - k + 1)}


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/interim/provisions_raw.jsonl")]
    print(f"raw provisions: {len(rows)}")

    keep = []
    for r in rows:
        ctx = " ".join(filter(None, [r.get("doc_title"), r.get("chapter_heading"),
                                     r.get("heading"), r.get("source_url")]))
        nw = len(r["text"].split())
        if not (MIN_WORDS <= nw <= MAX_WORDS):
            continue
        if NONPROC.search(ctx) and not (CRIM.search(ctx) or CIV.search(ctx)):
            continue
        # domain comes only from enacted structure (08_domain.py); never inferred from wording
        dom = r.get("domain")
        if dom not in ("criminal", "civil"):
            continue
        r["domain"] = dom
        r["nation"] = norm_nation(r["nation"])
        r["platform"] = urllib.parse.urlparse(r["source_url"]).netloc.lower().replace("www.", "")
        r["n_words"] = nw
        keep.append(r)
    print(f"after domain/length filter: {len(keep)}")

    # exact duplicates within a Nation
    seen, dedup = set(), []
    for r in keep:
        k = (r["nation"], hashlib.blake2b(re.sub(r"\W+", "", r["text"].lower()).encode(),
                                          digest_size=12).digest())
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    print(f"after within-Nation exact dedup: {len(dedup)}")

    # Nations with enough material
    cnt = collections.Counter(r["nation"] for r in dedup)
    ok = {n for n, c in cnt.items() if c >= MIN_PROV_PER_NATION}
    dedup = [r for r in dedup if r["nation"] in ok]
    print(f"after Nation threshold (>={MIN_PROV_PER_NATION}): {len(dedup)} from {len(ok)} Nations")

    # cross-Nation near-duplicate detection (Jaccard on 9-gram shingles, inverted index)
    sh = [shingle_hash(r["text"]) for r in dedup]
    inv = collections.defaultdict(list)
    for i, s in enumerate(sh):
        for g in list(s)[:60]:
            inv[g].append(i)
    dup_of = {}
    for g, idxs in inv.items():
        if len(idxs) < 2 or len(idxs) > 40:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if dedup[i]["nation"] == dedup[j]["nation"]:
                    continue
                if i in dup_of and j in dup_of:
                    continue
                u = len(sh[i] | sh[j])
                if u and len(sh[i] & sh[j]) / u >= 0.80:
                    dup_of.setdefault(max(i, j), min(i, j))
    for i, r in enumerate(dedup):
        r["xnation_dup"] = i in dup_of
    n_dup = sum(r["xnation_dup"] for r in dedup)
    print(f"cross-Nation near-duplicates flagged: {n_dup} ({n_dup/len(dedup)*100:.1f}%)")

    out = ROOT / "data/processed/corpus.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in dedup:
            f.write(json.dumps(r) + "\n")
    byn = collections.Counter(r["nation"] for r in dedup)
    byd = collections.Counter(r["domain"] for r in dedup)
    print(f"\nFINAL {len(dedup)} provisions | {len(byn)} Nations | {dict(byd)}")
    for n, c in byn.most_common():
        cr = sum(1 for r in dedup if r["nation"] == n and r["domain"] == "criminal")
        print(f"  {c:6d}  (crim {cr:5d} / civ {c-cr:5d})  {n[:52]}")


if __name__ == "__main__":
    main()
