"""Crawl the NILL Tribal Law Gateway and index each Nation's code/constitution links."""
import json, re, sys, pathlib
from bs4 import BeautifulSoup
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch import get

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
GATEWAY = "https://narf.org/nill/triballaw/index.html"
BASE = "https://narf.org/nill/"
SECTIONS = ["Tribal Code", "Tribal Constitution", "Tribal Court Opinions"]


def tribe_pages():
    soup = BeautifulSoup(get(GATEWAY), "lxml")
    out = {}
    for a in soup.find_all("a", href=True):
        if a["href"].startswith("../tribes/"):
            out[BASE + a["href"][3:]] = a.get_text(" ", strip=True)
    return out


def parse_tribe(url, name):
    html = get(url)
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("main") or soup.find("article") or soup
    text = body.get_text("\n", strip=True)
    rec = {"nation": name, "nill_url": url, "sections": {}}
    # Split page text on the bolded section labels to know which links sit where.
    for s in SECTIONS:
        m = re.search(re.escape(s) + r"\s*:", text)
        if not m:
            continue
        nxt = [re.search(re.escape(o) + r"\s*:", text[m.end():]) for o in SECTIONS if o != s]
        end = min([x.start() for x in nxt if x] or [len(text)])
        rec["sections"][s] = text[m.end():m.end() + end][:600]
    links = []
    for a in body.find_all("a", href=True):
        h, t = a["href"], a.get_text(" ", strip=True)
        if h.startswith("http") and "narf.org" not in h and "nill." not in h:
            links.append({"text": t, "url": h})
    rec["links"] = links
    return rec


def main():
    pages = tribe_pages()
    print(f"{len(pages)} Nation pages indexed", flush=True)
    out, seen = [], set()
    for i, (u, n) in enumerate(sorted(pages.items()), 1):
        if u in seen:
            continue
        seen.add(u)
        r = parse_tribe(u, n)
        if r:
            out.append(r)
        if i % 50 == 0:
            print(f"  {i}/{len(pages)}", flush=True)
    p = ROOT / "data/interim/nill_index.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} records -> {p}")


if __name__ == "__main__":
    main()
