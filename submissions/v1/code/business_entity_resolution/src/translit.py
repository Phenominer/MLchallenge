"""Rule-based transliteration of Indic scripts to Latin, plus a token dictionary
mined from training pairs (Indic token in S2/S3 -> Latin token in the matched S1).

All Indic Unicode blocks (Devanagari U+0900 ... Malayalam U+0D00) share the same
128-codepoint layout, so one table indexed by (codepoint % 0x80) covers them all.
Uses only the provided training data; no external resources.
"""
import json
import re
from collections import Counter, defaultdict

from rapidfuzz import fuzz

# offset within block -> (latin, kind) ; kind: C consonant, V independent vowel, M matra, X other
_TABLE = {
    0x01: ("n", "X"), 0x02: ("n", "X"), 0x03: ("h", "X"),
    0x05: ("a", "V"), 0x06: ("a", "V"), 0x07: ("i", "V"), 0x08: ("i", "V"), 0x09: ("u", "V"),
    0x0A: ("u", "V"), 0x0B: ("ri", "V"), 0x0E: ("e", "V"), 0x0F: ("e", "V"), 0x10: ("ai", "V"),
    0x12: ("o", "V"), 0x13: ("o", "V"), 0x14: ("au", "V"),
    0x3E: ("a", "M"), 0x3F: ("i", "M"), 0x40: ("i", "M"), 0x41: ("u", "M"), 0x42: ("u", "M"),
    0x43: ("ri", "M"), 0x46: ("e", "M"), 0x47: ("e", "M"), 0x48: ("ai", "M"), 0x4A: ("o", "M"),
    0x4B: ("o", "M"), 0x4C: ("au", "M"), 0x4D: ("", "M"),  # virama kills the inherent vowel
}
for _off, _lat in zip(range(0x15, 0x3A), "k kh g gh ng ch chh j jh ny t th d dh n t th d dh n n p ph b bh m y r r l l l v sh sh s h".split()):
    _TABLE[_off] = (_lat, "C")
for _off, _lat in zip(range(0x58, 0x60), "q kh g z r rh f y".split()):
    _TABLE[_off] = (_lat, "C")
for _off in range(0x66, 0x70):
    _TABLE[_off] = (str(_off - 0x66), "X")

INDIC = re.compile(r"[ऀ-ൿ]")


def translit_char(word: str) -> str:
    """Transliterate one word character by character (inherent 'a' after consonants, final schwa dropped)."""
    out, pending = [], False  # pending: last consonant still owes its inherent 'a'
    for ch in word:
        cp = ord(ch)
        if not 0x0900 <= cp <= 0x0D7F:
            if pending:
                out.append("a")
            pending = False
            if ch not in "‌‍":  # drop zero-width joiners
                out.append(ch)
            continue
        lat, kind = _TABLE.get(cp % 0x80, ("", "X"))
        if kind == "M":
            pending = False
        elif pending:
            out.append("a")
            pending = False
        out.append(lat)
        pending = kind == "C"
    return re.sub(r"aa+", "a", "".join(out))  # 'aa' vs 'a' is not distinctive in romanized data


def translit_text(text: str, table: dict) -> str:
    """Replace every Indic token by its mined translation, else by the char-level transliteration."""
    if not INDIC.search(text):
        return text
    return " ".join(table.get(t) or translit_char(t) if INDIC.search(t) else t for t in text.split())


def mine_dictionary(pairs, min_count: int = 2) -> dict:
    """Mine {indic_token: latin_token} from (indic_side_text, latin_side_text) pairs.

    Each Indic token is aligned to the Latin token most similar to its char
    transliteration; the most frequent aligned target wins if it covers >= half the votes.
    """
    votes = defaultdict(Counter)
    for src, tgt in pairs:
        cands = re.findall(r"[a-z]+", tgt.lower())
        if not cands:
            continue
        for tok in src.split():
            if INDIC.search(tok):
                tok = tok.strip(",.()")
                tr = translit_char(tok)
                best = max(cands, key=lambda c: fuzz.ratio(tr, c))
                if fuzz.ratio(tr, best) >= 50:
                    votes[tok][best] += 1
    out = {}
    for tok, c in votes.items():
        tgt, n = c.most_common(1)[0]
        if n >= min_count and n * 2 >= sum(c.values()):
            out[tok] = tgt
    return out


def build_dictionary(path):
    """Mine the Indic dictionary from train name and address pairs and save it as JSON."""
    from io_utils import load_sources, load_truth
    s1, s23 = load_sources("train")
    s23 = s23[s23.business_name.str.contains(INDIC.pattern) | s23.business_address.str.contains(INDIC.pattern)]
    owner = {m: k for k, v in load_truth().items() for m in v}
    s1i = s1.set_index("entity_id")
    s23 = s23[s23.entity_id.isin(owner)]
    ref = s1i.loc[[owner[e] for e in s23.entity_id]]
    pairs = list(zip(s23.business_name, ref.business_name)) + list(zip(s23.business_address, ref.business_address))
    d = mine_dictionary(pairs)
    json.dump(d, open(path, "w"), ensure_ascii=False)
    return d


if __name__ == "__main__":
    assert translit_char("लिमिटेड") == "limited", translit_char("लिमिटेड")
    assert translit_char("राम") == "ram", translit_char("राम")
    print(translit_char("प्राइवेट"), translit_char("ಕರ್ನಾಟಕ"), translit_char("महाराष्ट्र"))
    print(mine_dictionary([("राम मार्केटिंग प्राइवेट लिमिटेड", "Ram Marketing Private Limited")] * 2))
