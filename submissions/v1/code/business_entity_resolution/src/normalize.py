"""Language-agnostic normalization of business names and addresses.

Produces per record: core name (legal suffixes removed), canonical legal-suffix
field, domain flag, normalized address, number set, and an address blocking key.
Rules are generic (accent stripping, punctuation, abbreviation tables that include
French forms) so unseen countries such as France are handled without training data.
"""
import json
import re
import unicodedata
from multiprocessing import Pool

import pandas as pd

from io_utils import CACHE, load
from translit import translit_text

# canonical legal forms (US, India, France, generic)
LEGAL = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "llp": "llp", "lp": "lp", "ltd": "ltd",
    "limited": "ltd", "pvt": "pvt", "private": "pvt", "pte": "pvt", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "plc": "plc", "pllc": "pllc", "pc": "pc", "sarl": "sarl", "sas": "sas",
    "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "opc": "opc",
    "public": "public", "lc": "llc", "ltda": "ltd", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl",
}
NAME_STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "a", "an"}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "sq": "square", "cir": "circle", "fl": "floor", "flr": "floor",
    "ste": "suite", "apt": "apartment", "nr": "near", "opp": "opposite", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west", "hno": "house", "no": "number", "imp": "impasse",
    "rte": "route", "ch": "chemin", "fbg": "faubourg", "r": "rue", "pte": "porte", "sect": "sector",
}
# street-type / filler words skipped when building the address key
ADDR_GENERIC = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "court", "place", "parkway", "highway",
    "square", "circle", "floor", "suite", "apartment", "unit", "near", "opposite", "number", "house",
    "rue", "impasse", "route", "chemin", "allee", "cours", "quai", "cite", "north", "south", "east",
    "west", "box", "po", "flat", "plot", "shop", "office", "block", "sector", "de", "du", "des", "la",
    "le", "d", "l", "the", "and", "of", "main", "cross", "null", "na", "building", "bldg", "room",
}
DOMAIN = re.compile(r"^(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|in|co|fr|us|biz|info|io|co\.in)$")
_DICT = {}


def strip_accents(s: str) -> str:
    """Remove combining marks after NFKD decomposition (e -> e for é)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def base_clean(s: str, country: str) -> str:
    """Shared cleanup: lowercase, transliterate Indic, strip accents, drop placeholders and punctuation."""
    s = translit_text(s, _DICT).lower()
    s = strip_accents(s)
    s = re.sub(r"<[^>]*>|\bn/a\b", " ", s)                 # <NULL>, <CITY_NAME>, n/a
    s = re.sub(r"\b[ldjnm]'|\bqu'", "", s)                 # French elision l', d'
    s = s.replace("'", "").replace("&", " and ")
    s = re.sub(r"\b(\w)\.(?=\w\b)", r"\1", s)              # l.l.c. -> llc.
    s = re.sub(r"\b(\w)\.(?=\w\b)", r"\1", s)
    s = re.sub(r"[^\w\s]|_", " ", s)
    c = strip_accents(country.lower())
    if c:
        s = re.sub(rf"\b{re.escape(c)}\b", " ", s)          # "(India)", "(France)" noise
    return s


def norm_name(name: str, country: str):
    """Return (core_name, legal_field, is_domain)."""
    low = strip_accents(name.lower().strip())
    m = DOMAIN.match(low)
    is_dom = int(bool(m))
    if m:
        name = m.group(1).replace("-", " ")
    toks = base_clean(name, country).split()
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    if not core:  # name made only of legal words: keep them as the name
        core = [t for t in toks if t not in NAME_STOP]
    return " ".join(core), " ".join(legal), is_dom


def norm_addr(addr: str, country: str):
    """Return (normalized_address, numbers, address_key)."""
    toks = [ADDR_ABBR.get(t, t) for t in base_clean(addr, country).split()]
    nums = []
    for t in toks:
        for n in re.findall(r"\d+", t):
            n = n.lstrip("0") or "0"
            if n not in nums:
                nums.append(n)
    key = ""
    for i, t in enumerate(toks):  # first number followed by first distinctive alpha token
        if t.isdigit():
            nxt = next((u for u in toks[i + 1:] if u.isalpha() and u not in ADDR_GENERIC and len(u) > 1), None)
            if nxt:
                key = f"{t.lstrip('0') or '0'} {nxt[:4]}"
                break
    return " ".join(toks), " ".join(nums), key


def _norm_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize one chunk (runs in a worker process)."""
    n = [norm_name(a, c) for a, c in zip(df.business_name, df.country)]
    a = [norm_addr(a, c) for a, c in zip(df.business_address, df.country)]
    return pd.DataFrame({
        "entity_id": df.entity_id.values, "country": df.country.values,
        "name": [x[0] for x in n], "legal": [x[1] for x in n], "is_dom": [x[2] for x in n],
        "addr": [x[0] for x in a], "nums": [x[1] for x in a], "akey": [x[2] for x in a],
    })


def _init(d):
    """Worker initializer: install the mined Indic dictionary."""
    _DICT.update(d)


def normalized(split: str, name: str) -> pd.DataFrame:
    """Return the normalized table for dataset/<split>/<split>_<name>.tsv, cached as parquet."""
    pq = CACHE / f"norm_{split}_{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = load(split, name)
    d = json.load(open(CACHE / "indic_dict.json"))
    chunks = [df.iloc[i:i + 100_000] for i in range(0, len(df), 100_000)]
    with Pool(10, initializer=_init, initargs=(d,)) as p:
        out = pd.concat(p.map(_norm_chunk, chunks), ignore_index=True)
    out.to_parquet(pq, index=False)
    return out


if __name__ == "__main__":
    assert norm_name("Schubert Adr L.L.C.", "US") == ("schubert adr", "llc", 0)
    assert norm_name("Pvt. Style Multi Limited", "India") == ("style multi", "ltd pvt", 0)
    assert norm_name("Apar Engineering (India) Private Limited", "India")[0] == "apar engineering"
    assert norm_name("hernandezs.com", "US") == ("hernandezs", "", 1)
    assert norm_name("[S.A.R.L.] TONTONS (FRANCE) MATERNELLE", "France") == ("tontons maternelle", "sarl", 0)
    assert norm_addr("6445- MOORS PL, DUBLIN, OH", "US")[2] == norm_addr("6445 Moors Place, Dublin, OH", "US")[2] == "6445 moor"
    assert norm_addr("51 RUE Clemenceau, Dunkerque", "France")[2] == "51 clem"
    assert norm_addr("005057 Macnamara Drive", "US")[1:] == ("5057", "5057 macn")
    print(norm_name("Rue d'Allonville Élégance SAS", "France"), norm_addr("N°8 IMPASSE CHARLIE PARKER, MÉRIGNAC", "France"))
    print("ok")
