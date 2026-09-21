"""Assign NILL links to their document-type section and rank Nations by source accessibility."""
import json, pathlib, re, sys, urllib.parse, collections
from bs4 import BeautifulSoup
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch import get

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
PAYWALL = {"westlaw.com", "a.next.westlaw.com", "directory.westlaw.com", "lexisnexis.com",
           "casemakerlegal.com", "versuslaw.com", "wshein.com", "tinyurl.com",
           "nill.softlinkliberty.net", "fastcase.com"}
BLOCKED = {"codepublishing.com"}          # Cloudflare bot challenge; not bypassed
ARCHIVE = {"web.archive.org", "loc.gov"}


def host(u):
    return urllib.parse.urlparse(u).netloc.lower().replace("www.", "")


def sectioned(nill_url):
    html = get(nill_url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main") or soup
    out, cur = collections.defaultdict(list), None
    for el in main.find_all(["h3", "a"]):
        if el.name == "h3":
            t = el.get_text(" ", strip=True).rstrip(":")
            cur = t if t in ("Tribal Code", "Tribal Constitution", "Tribal Court Opinions") else None
        elif cur and el.get("href", "").startswith("http"):
            h = host(el["href"])
            if h in PAYWALL or "narf.org" in h:
                continue
            out[cur].append({"text": el.get_text(" ", strip=True), "url": el["href"], "host": h})
    return dict(out)


def main():
    idx = json.load(open(ROOT / "data/interim/nill_index.json"))
    rows = []
    for r in idx:
        sec = sectioned(r["nill_url"])
        code = sec.get("Tribal Code", [])
        live = [l for l in code if l["host"] not in BLOCKED and l["host"] not in ARCHIVE]
        if live:
            rows.append({"nation": r["nation"], "nill_url": r["nill_url"], "code_links": live})
    json.dump(rows, open(ROOT / "data/interim/candidates.json", "w"), indent=1)
    print(f"{len(rows)} Nations with a non-paywalled, non-blocked code link\n")
    hc = collections.Counter(l["host"] for r in rows for l in r["code_links"])
    for h, c in hc.most_common(30):
        print(f"  {c:4d}  {h}")


if __name__ == "__main__":
    main()
