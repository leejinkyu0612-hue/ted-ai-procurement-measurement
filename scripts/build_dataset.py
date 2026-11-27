# -*- coding: utf-8 -*-
"""Build the analysis table from raw TED API JSONL files.
Steps (pre-specified in S04 사전계획): deduplicate by publication-number; keep can-standard;
text = procedure title + procedure description (fallback: first lot title/description), in the
notice's first official language; tenders received = values whose code == 'tenders'.
Outputs data/notices.parquet and results/dataset_profile.json (counts and missingness only).
"""
import glob, json, os, re, math
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
rows = {}
for fn in sorted(glob.glob(os.path.join(BASE, "data", "raw", "*.jsonl"))):
    for line in open(fn, encoding="utf-8"):
        n = json.loads(line)
        rows[n["publication-number"]] = n
print("unique notices", len(rows))


def pick_text(d, lang):
    if not d:
        return ""
    if isinstance(d, dict):
        if lang and lang.lower() in d:
            v = d[lang.lower()]
        else:
            v = next(iter(d.values()))
        if isinstance(v, list):
            v = " ".join(v)
        return v or ""
    return str(d)


LANG3 = {"ENG": "eng"}
recs = []
for pid, n in rows.items():
    langs = n.get("official-language") or []
    lang = langs[0] if langs else None
    l3 = lang.lower() if lang else None
    title = pick_text(n.get("title-proc"), l3) or pick_text(n.get("title-lot"), l3)
    desc = pick_text(n.get("description-proc"), l3) or pick_text(n.get("description-lot"), l3)
    codes = n.get("received-submissions-type-code") or []
    vals = n.get("received-submissions-type-val") or []
    tenders = [int(v) for c, v in zip(codes, vals) if c == "tenders" and str(v).isdigit()]
    cpv = n.get("classification-cpv") or []
    main = cpv[0] if cpv else ""
    recs.append({
        "pub": pid, "lang": lang, "n_lang": len(langs), "country": (n.get("buyer-country") or [None])[0],
        "date": (n.get("publication-date") or "")[:10], "title": title.strip(), "desc": desc.strip(),
        "cpv_main": main, "cpv_div": main[:2], "cpv_all": " ".join(sorted(set(cpv))),
        "n_tender_fields": len(tenders), "tenders": tenders[0] if len(tenders) == 1 else None,
        "tenders_min": min(tenders) if tenders else None, "tenders_max": max(tenders) if tenders else None,
        "procedure": n.get("procedure-type"), "buyer_type": (n.get("buyer-legal-type") or [None])[0],
        "value": n.get("total-value"), "value_cur": (n.get("total-value-cur") or [None])[0],
        "winner_size": ";".join(sorted(set(n.get("winner-size") or []))),
    })
df = pd.DataFrame(recs)
df["text"] = (df["title"] + ". " + df["desc"]).str.slice(0, 2000)
df["year"] = df["date"].str.slice(0, 4)
df["quarter"] = df["date"].str.slice(0, 4) + "Q" + ((df["date"].str.slice(5, 7).astype(int) - 1) // 3 + 1).astype(str)
os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
df.to_parquet(os.path.join(BASE, "data", "notices.parquet"), index=False)
prof = {
    "unique_notices": len(df),
    "years": df["year"].value_counts().to_dict(),
    "languages_top": df["lang"].value_counts().head(25).to_dict(),
    "countries_n": int(df["country"].nunique()),
    "cpv_div": df["cpv_div"].value_counts().head(10).to_dict(),
    "empty_text": int((df["title"].str.len() + df["desc"].str.len() == 0).sum()),
    "single_tenders_field": int((df["n_tender_fields"] == 1).sum()),
    "no_tenders_field": int((df["n_tender_fields"] == 0).sum()),
    "multi_tenders_field": int((df["n_tender_fields"] > 1).sum()),
    "value_missing": int(df["value"].isna().sum()),
    "text_chars_median": float(df["text"].str.len().median()),
}
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
json.dump(prof, open(os.path.join(BASE, "results", "dataset_profile.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(prof, indent=1, ensure_ascii=False))
