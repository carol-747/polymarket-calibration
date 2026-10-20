"""Fetch resolved binary Polymarket markets and their pre-resolution price paths.

Data sources (public, no API key):
  - Gamma API  https://gamma-api.polymarket.com/markets   (market metadata + final outcome)
  - CLOB API   https://clob.polymarket.com/prices-history (historical prices of the YES token)

Usage:
    python3 fetch_data.py                 # default: up to 3000 markets with volume >= $50k
    python3 fetch_data.py --max 1000      # smaller run

Outputs:
    data/markets.csv   one row per resolved binary market
    data/prices.csv    6-hourly YES-token price history over the final 30 days of each market
The script is resumable: re-running skips markets whose prices are already saved.
"""
import argparse, csv, json, os, sys, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com/prices-history"
HEADERS = {"User-Agent": "polymarket-calibration-research/1.0"}


def get_json(url, params, retries=4, quiet=False):
    full = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:200]
            if e.code != 429 and 400 <= e.code < 500:   # bad request: retrying will not help
                if not quiet:
                    print(f"  HTTP {e.code}: {body}", file=sys.stderr)
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  retry {attempt+1} after error: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except Exception:
        return None


def to_ts(s):
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00").replace(" ", "T", 1)
    if s[-3] in "+-" and s[-3:].lstrip("+-").isdigit():   # "+00" -> "+00:00"
        s = s + ":00"
    for cand in (s, s[:19]):
        try:
            d = datetime.fromisoformat(cand)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return int(d.timestamp())
        except Exception:
            continue
    return None


def parse_market(m):
    outcomes = parse_list(m.get("outcomes"))
    prices = parse_list(m.get("outcomePrices"))
    tokens = parse_list(m.get("clobTokenIds"))
    if not (outcomes and prices and tokens) or len(outcomes) != 2:
        return None
    if [o.lower() for o in outcomes] != ["yes", "no"]:
        return None
    try:
        p_yes = float(prices[0])
    except Exception:
        return None
    if p_yes not in (0.0, 1.0):   # skip unresolved / 50-50 resolutions
        return None
    end_ts = to_ts(m.get("closedTime")) or to_ts(m.get("endDate"))
    if not end_ts:
        return None
    events = m.get("events") or []
    return {
        "market_id": m.get("id"),
        "question": (m.get("question") or "").replace("\n", " "),
        "slug": m.get("slug"),
        "category": m.get("category") or "",
        "event_title": (events[0].get("title") if events else "") or "",
        "yes_token": tokens[0],
        "outcome_yes": int(p_yes == 1.0),
        "volume": m.get("volumeNum") or m.get("volume"),
        "start_date": m.get("startDate"),
        "end_ts": end_ts,
    }


def fetch_markets(max_markets, min_volume, since="2023-01-01"):
    """Walk backwards in 14-day windows of market end date; page within each window."""
    out, seen = [], set()
    win_end = datetime.now(timezone.utc)
    stop = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    step = 14 * 86400
    while len(out) < max_markets and win_end > stop:
        win_start = datetime.fromtimestamp(win_end.timestamp() - step, tz=timezone.utc)
        offset = 0
        while True:
            batch = get_json(GAMMA, {"closed": "true", "limit": 500, "offset": offset,
                                     "volume_num_min": min_volume,
                                     "end_date_min": win_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                     "end_date_max": win_end.strftime("%Y-%m-%dT%H:%M:%SZ")})
            if not batch:
                break
            for m in batch:
                if m.get("id") in seen:
                    continue
                seen.add(m.get("id"))
                row = parse_market(m)
                if row:
                    out.append(row)
            if len(batch) < 500 or offset >= 2000:
                break
            offset += 500
            time.sleep(0.2)
        print(f"window ending {win_end:%Y-%m-%d}: kept {len(out)} resolved binary markets so far")
        win_end = win_start
        time.sleep(0.2)
    return out[:max_markets]


def fetch_history(token, end, days):
    """Try the simple 'interval=max' query first; fall back to 6-day windows."""
    for fid in (360, 720, 1440):
        js = get_json(CLOB, {"market": token, "interval": "max", "fidelity": fid}, quiet=True)
        hist = (js or {}).get("history") or []
        if hist:
            return [h for h in hist if end - days * 86400 <= h.get("t", 0) <= end + 86400]
    hist = []
    for k in range(0, days, 6):
        js = get_json(CLOB, {"market": token, "startTs": end - (k + 6) * 86400,
                             "endTs": end - k * 86400, "fidelity": 60}, quiet=True)
        hist += (js or {}).get("history") or []
    return hist


def main():  # noqa
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--min-volume", type=float, default=50000)
    ap.add_argument("--days", type=int, default=30)
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    markets = fetch_markets(a.max, a.min_volume)
    with open("data/markets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(markets[0].keys()))
        w.writeheader(); w.writerows(markets)
    print(f"saved {len(markets)} markets to data/markets.csv")

    done = set()
    if os.path.exists("data/prices.csv"):
        with open("data/prices.csv") as f:
            done = {r["market_id"] for r in csv.DictReader(f)}
    new_file = not os.path.exists("data/prices.csv")
    with open("data/prices.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["market_id", "t", "p"])
        for i, m in enumerate(markets, 1):
            if m["market_id"] in done:
                continue
            end = m["end_ts"]
            hist = fetch_history(m["yes_token"], end, a.days)
            for h in hist:
                w.writerow([m["market_id"], h.get("t"), h.get("p")])
            if i % 50 == 0:
                f.flush(); print(f"prices: {i}/{len(markets)}")
            time.sleep(0.12)
    print("done. Files: data/markets.csv, data/prices.csv")


if __name__ == "__main__":
    main()
