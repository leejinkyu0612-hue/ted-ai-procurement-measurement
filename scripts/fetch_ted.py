# -*- coding: utf-8 -*-
"""Download TED eForms contract award notices (can-standard) for IT services (CPV 72) and
software packages (CPV 48), publication dates 2024-01-01..2025-12-31, via the anonymous
TED Search API v3. Raw API responses are stored as JSONL (one notice per line).
Data collection only; no outcome is computed here.
"""
import json, os, sys, time, datetime, urllib.request

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT, exist_ok=True)
API = "https://api.ted.europa.eu/v3/notices/search"
FIELDS = ["publication-number", "notice-type", "title-proc", "description-proc", "title-lot",
          "description-lot", "received-submissions-type-code", "received-submissions-type-val",
          "winner-size", "total-value", "total-value-cur", "buyer-legal-type", "procedure-type",
          "classification-cpv", "buyer-country", "publication-date", "official-language"]
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "ted_api_calls.jsonl")


def post(body, tries=6):
    data = json.dumps(body).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:
            time.sleep(5 * (t + 1))
            err = str(e)
    raise RuntimeError(err)


def month_ranges(y0=2024, y1=2025):
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            a = datetime.date(y, m, 1)
            b = (datetime.date(y + (m == 12), m % 12 + 1, 1) - datetime.timedelta(days=1))
            # split each month in halves to stay under page-depth limits
            mid = datetime.date(y, m, 15)
            yield a, mid
            yield mid + datetime.timedelta(days=1), b


def main():
    for cpv in ("72000000", "48000000"):
        for a, b in month_ranges():
            fn = os.path.join(OUT, f"can_{cpv[:2]}_{a:%Y%m%d}_{b:%Y%m%d}.jsonl")
            if os.path.exists(fn):
                continue
            q = f"notice-type=can-standard AND classification-cpv={cpv} AND PD>={a:%Y%m%d} AND PD<={b:%Y%m%d}"
            rows, token, page = [], None, 0
            while True:
                body = {"query": q, "fields": FIELDS, "limit": 250, "paginationMode": "ITERATION"}
                if token:
                    body["iterationNextToken"] = token
                r = post(body)
                page += 1
                ns = r.get("notices", [])
                for n in ns:
                    n.pop("links", None)
                    rows.append(n)
                with open(LOG, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps({"ts": datetime.datetime.now().isoformat(), "query": q, "page": page,
                                         "n": len(ns), "total": r.get("totalNoticeCount")}) + "\n")
                token = r.get("iterationNextToken")
                if not token or not ns:
                    break
                time.sleep(0.3)
            tmp = fn + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for n in rows:
                    f.write(json.dumps(n, ensure_ascii=False) + "\n")
            os.replace(tmp, fn)
            print(cpv, a, b, len(rows), flush=True)


if __name__ == "__main__":
    main()
