"""Crawl a curated shortlist of Nation legal-code sites and discover procedure documents (HTML or PDF)."""
import json, pathlib, re, sys, urllib.parse, concurrent.futures as cf
import requests
from bs4 import BeautifulSoup
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch import UA, allowed

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
HDR = {"User-Agent": UA}

SEEDS = {
 # Pacific Northwest
 "Confederated Tribes of the Colville Reservation": "https://www.cct-cbc.com/current-code/",
 "Kalispel Tribe of Indians": "http://www.kalispeltribe.com/government/tribal-court/law-order-code",
 "Port Gamble S'Klallam Tribe": "https://www.pgst.nsn.us/government/law-and-order-code",
 "Quileute Tribe": "https://quileutenation.org/court/law-and-order-codes/",
 "Hoh Indian Tribe": "http://hohtribe-nsn.org/?page_id=1446",
 "Spokane Tribe of Indians": "http://spokanetribe.com/government/tribal-court/",
 # Southwest
 "Quechan Tribe": "http://quechantribalcourt.info/Quechan_Law_and_Order_Co.html",
 "Pueblo of Pojoaque": "http://pojoaque.org/community/legal-department/law-order-code/",
 "Fort McDowell Yavapai Nation": "http://www.fmyn.org/tribal-government/law-and-order-code/",
 "Hopi Tribe": "http://www.hopitribalcourts.com/hopi-laws/",
 "Pascua Yaqui Tribe": "https://www.pascuayaqui-nsn.gov/tribal-code-v1/",
 "Colorado River Indian Tribes": "http://www.crit-nsn.gov/crit_contents/ordinances/",
 "Gila River Indian Community": "http://www.gilariver.org/index.php/government/judical-branch",
 # Plains
 "Rosebud Sioux Tribe": "https://www.rstcourt.org/",
 "Sisseton-Wahpeton Oyate": "http://www.swo-nsn.gov/?page_id=851",
 "Mandan, Hidatsa and Arikara Nation": "https://www.mhanation.com/tribal-code/",
 "Ponca Tribe of Nebraska": "https://www.poncatribe-ne.org/tribal-documents/law-code-title/",
 # Great Lakes
 "Little Traverse Bay Bands of Odawa Indians": "https://ltbbodawa-nsn.gov/tribal-council-and-legislative-office/statutes/",
 "Oneida Nation": "https://oneida-nsn.gov/government/register/laws/",
 "Pokagon Band of Potawatomi Indians": "https://www.pokagonband-nsn.gov/government/tribal-court/tribal-code",
 # Southeast / Northeast
 "Alabama-Coushatta Tribe of Texas": "https://www.alabama-coushatta.com/",
 "Seminole Tribe of Florida": "http://www.semtribe.com/Government/TribalCourt/Codes.aspx",
 "Mashantucket Pequot Tribal Nation": "https://www.mptnlaw.com/laws.htm",
 # California
 "Karuk Tribe": "http://www.karuk.us/index.php/departments/judicial-system",
 "Los Coyotes Band of Cahuilla and Cupeno Indians": "http://www.sciljc.org/ordinances--codes---laws.html",
 # Oklahoma
 "Quapaw Nation": "https://www.quapawtribe.com/193/Tribal-Code",
 "Sac and Fox Nation of Missouri in Kansas and Nebraska": "https://www.sacandfoxks.com/",
 # Great Basin
 "Washoe Tribe of Nevada and California": "https://www.washoetribe.us/contents/organization/tribal-documents/law-and-order-code",
 # Alaska
 "Kenaitze Indian Tribe": "https://www.kenaitze.org/about/tribal-court/court-codes/",
 "Sitka Tribe of Alaska": "http://www.sitkatribe.org/government/council/Codes.htm",
}

WANT = re.compile(r"law\s*(and|&|\s)\s*order|criminal\s+procedure|civil\s+procedure|rules?\s+of\s+(civil|criminal)"
                  r"|judicial\s+code|court\s+code|code\s+of\s+(civil|criminal)|title\s+[IVXLC0-9]+", re.I)
PROC = re.compile(r"criminal|civil|procedure|law and order|court|judicial", re.I)


def fetch(u, timeout=30):
    if not allowed(u):
        return None
    try:
        return requests.get(u, headers=HDR, timeout=timeout, allow_redirects=True)
    except Exception:
        return None


def discover(item):
    nation, seed = item
    r = fetch(seed)
    if r is None or r.status_code != 200:
        return {"nation": nation, "seed": seed, "error": f"seed {getattr(r,'status_code','ERR')}", "docs": []}
    if "Just a moment" in r.text[:2000]:
        return {"nation": nation, "seed": seed, "error": "bot-challenge", "docs": []}
    soup = BeautifulSoup(r.text, "lxml")
    docs, seen = [], set()
    for a in soup.find_all("a", href=True):
        t = a.get_text(" ", strip=True)
        u = urllib.parse.urljoin(r.url, a["href"])
        if u in seen or u.startswith("mailto"):
            continue
        seen.add(u)
        if WANT.search(t) or (u.lower().endswith(".pdf") and PROC.search(t + " " + u)):
            docs.append({"title": t[:160], "url": u, "pdf": u.lower().endswith(".pdf")})
    return {"nation": nation, "seed": seed, "final": r.url, "error": None, "docs": docs[:120]}


def main():
    out = []
    with cf.ThreadPoolExecutor(10) as ex:
        for res in ex.map(discover, SEEDS.items()):
            out.append(res)
            n = len(res["docs"])
            print(f"  {('ERR:'+res['error']) if res['error'] else str(n)+' docs':22s} {res['nation'][:48]}", flush=True)
    json.dump(out, open(ROOT / "data/interim/discovered.json", "w"), indent=1)
    tot = sum(len(r["docs"]) for r in out)
    print(f"\n{sum(1 for r in out if not r['error'])}/{len(out)} seeds reachable; {tot} candidate documents")


if __name__ == "__main__":
    main()
