"""Probe each candidate Nation's code URL: reachability, format, and procedure-domain signal."""
import json, pathlib, re, sys, concurrent.futures as cf
import requests
from bs4 import BeautifulSoup
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch import UA, allowed

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
CRIM = re.compile(r"criminal procedure|rules of criminal|law and order|criminal offense|arrest|arraign"
                  r"|sentenc|bail|bond|prosecut|misdemeanor", re.I)
CIV = re.compile(r"civil procedure|rules of civil|civil action|complaint|summons|pleading"
                 r"|service of process|small claims|judgment", re.I)


def probe(rec):
    best = None
    for l in rec["code_links"][:3]:
        u = l["url"]
        if not allowed(u):
            out = {"status": "robots", "url": u}
        else:
            try:
                r = requests.get(u, headers={"User-Agent": UA}, timeout=25, allow_redirects=True)
                ct = r.headers.get("content-type", "")
                txt = ""
                if "pdf" in ct.lower():
                    fmt = "pdf"
                elif r.status_code == 200:
                    fmt = "html"
                    txt = BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)
                else:
                    fmt = "err"
                chal = "Just a moment" in r.text[:2000] or "cf-browser-verification" in r.text[:4000]
                out = {"status": "challenge" if chal else str(r.status_code), "fmt": fmt, "url": r.url,
                       "bytes": len(r.content), "crim": len(CRIM.findall(txt)), "civ": len(CIV.findall(txt)),
                       "links": len(BeautifulSoup(r.text, "lxml").find_all("a")) if fmt == "html" else 0}
            except Exception as e:
                out = {"status": "exc:" + type(e).__name__, "url": u}
        score = (out.get("crim", 0) + out.get("civ", 0)) if out.get("status") == "200" else -1
        if best is None or score > best[0]:
            best = (score, out)
    return {"nation": rec["nation"], "probe": best[1], "score": best[0]}


def main():
    cands = json.load(open(ROOT / "data/interim/candidates.json"))
    res = []
    with cf.ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(probe, cands), 1):
            res.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(cands)}", flush=True)
    res.sort(key=lambda r: -r["score"])
    json.dump(res, open(ROOT / "data/interim/probe.json", "w"), indent=1)
    ok = [r for r in res if r["score"] > 0]
    print(f"\n{len(ok)} reachable with procedure signal\n")
    for r in ok[:45]:
        p = r["probe"]
        print(f"  {r['score']:5d} crim={p.get('crim',0):4d} civ={p.get('civ',0):4d} "
              f"{p.get('fmt','?'):5s} {r['nation'][:38]:40s} {p['url'][:60]}")


if __name__ == "__main__":
    main()
