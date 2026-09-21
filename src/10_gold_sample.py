"""Draw the human-verification sample.

Suggestions come from the rule cascade only, never from the zero-shot model, and two
thirds of the provisions it covers are shown with their suggested label for verification.
Everything else is drawn blind, giving an anchoring-free estimate of human-model agreement.
A subset of the blind items reappears much later in the queue, unmarked, so intra-annotator
reliability can be measured for the single coder.
"""
import collections, json, pathlib, random, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import SEED, corpus_path, stratified_sample

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
N, BLIND, REPEAT, LAG = 600, 0.33, 50, 220


def main():
    rows = [json.loads(l) for l in open(corpus_path())]
    rows = [r for r in rows if r.get("silver")]
    sample = stratified_sample(rows, N)
    rng = random.Random(SEED)
    rng.shuffle(sample)                       # interleave strata so fatigue is not confounded

    items = []
    for r in sample:
        items.append({"provision_id": r["provision_id"], "nation": r["nation"],
                      "domain": r["domain"], "heading": (r.get("heading") or "")[:120],
                      "text": r["text"][:1800],
                      "suggested": (r.get("rule_label")
                                    if r.get("rule_label") and rng.random() >= BLIND else None),
                      "rule": r.get("rule_label"), "nli": r.get("nli_label"),
                      "repeat_of": None})

    blind = [k for k, it in enumerate(items) if it["suggested"] is None and k < len(items) - LAG]
    for k in rng.sample(blind, min(REPEAT, len(blind))):
        dup = dict(items[k], repeat_of=items[k]["provision_id"])
        items.insert(rng.randint(k + LAG, len(items)), dup)

    for i, it in enumerate(items):
        it["i"] = i
    json.dump(items, open(ROOT / "data/interim/gold_sample.json", "w"), indent=1)

    nb = sum(1 for r in items if r["suggested"] is None and not r["repeat_of"])
    nr = sum(1 for r in items if r["repeat_of"])
    print(f"{len(items)} items | {nb} blind | {nr} repeats | "
          f"{len({r['nation'] for r in items})} Nations")
    for n, k in collections.Counter(r["nation"] for r in items).most_common():
        print(f"   {k:4d}  {n[:46]}")


if __name__ == "__main__":
    main()
