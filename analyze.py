"""Calibration analysis of resolved Polymarket binary markets.

Question: when a Polymarket YES contract trades at price p, does the event happen
about p of the time? We check this 1, 7 and 30 days before each market closed.

Run after fetch_data.py:  python3 analyze.py
Outputs figures/ and results/.
"""
import json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HORIZONS = {"1d": 1, "7d": 7, "30d": 30}
TOL_DAYS = 1          # price must be observed within 1 day of the target time
BINS = np.linspace(0, 1, 11)
RNG = np.random.default_rng(7)

CATS = [
    ("Sports", r"\b(vs\.?|nba|nfl|nhl|mlb|ufc|epl|premier league|champions league|world cup|super bowl|grand prix|f1|tennis|open|match|game \d|win the \d{4}|playoffs?|finals?)\b"),
    ("Crypto", r"\b(bitcoin|btc|ethereum|eth|solana|sol|xrp|doge|crypto|token|airdrop|fdv)\b"),
    ("Politics", r"\b(election|elect|president|presidential|senate|house|governor|mayor|trump|biden|harris|democrat|republican|parliament|prime minister|nominee|primary|vote)\b"),
    ("Macro & Finance", r"\b(fed|interest rate|rates?|inflation|cpi|gdp|recession|s&p|nasdaq|stock|earnings|tariff|unemployment)\b"),
]


def categorize(text):
    t = text.lower()
    for name, pat in CATS:
        if re.search(pat, t):
            return name
    return "Other"


def price_at(g, target, tol):
    """Last price at or before the target time; if none, the nearest price within tol."""
    s = g[g["t"] <= target]
    if not s.empty and target - s["t"].iloc[-1] <= tol:
        return s["p"].iloc[-1]
    d = (g["t"] - target).abs()
    if d.min() <= tol:
        return g["p"].iloc[d.values.argmin()]
    return np.nan


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit_logistic(x, y, iters=50):
    """Recalibration fit: logit P(y=1) = a + b * logit(p). b<1 => prices too extreme."""
    X = np.column_stack([np.ones_like(x), x]); beta = np.zeros(2)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta)); W = mu * (1 - mu) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)   # tiny ridge for stability
        beta = beta + np.linalg.solve(H, X.T @ (y - mu))
    return beta


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n; d = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / d; h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def metrics(df):
    p, y = df["p"].values, df["y"].values
    clusters = df["event"].values
    brier = np.mean((p - y) ** 2)
    ll = -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
    idx = np.clip(np.digitize(p, BINS) - 1, 0, 9)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
              for b in range(10) if (idx == b).any())
    a, b = fit_logistic(logit(p), y)
    boots, uniq = [], np.unique(clusters)
    pos = {c: np.where(clusters == c)[0] for c in uniq}
    for _ in range(300):   # resample whole events, not single markets
        pick = RNG.choice(uniq, len(uniq))
        s = np.concatenate([pos[c] for c in pick])
        boots.append(fit_logistic(logit(p[s]), y[s])[1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    bss = 1 - brier / (y.mean() * (1 - y.mean()))
    return {"n": int(len(p)), "events": int(len(uniq)), "brier_skill": bss, "brier": brier, "log_loss": ll, "ece": ece,
            "base_rate": y.mean(), "slope": b, "slope_lo": lo, "slope_hi": hi, "intercept": a}


def main():
    mk = pd.read_csv("data/markets.csv", dtype={"market_id": str})
    pr = pd.read_csv("data/prices.csv", dtype={"market_id": str}).dropna()
    pr = pr.sort_values(["market_id", "t"])
    mk["category"] = (mk["question"].fillna("") + " " + mk["event_title"].fillna("")).map(categorize)

    rows = []
    groups = dict(tuple(pr.groupby("market_id")))
    for _, m in mk.iterrows():
        g = groups.get(m["market_id"])
        if g is None:
            continue
        for h, d in HORIZONS.items():
            p = price_at(g, m["end_ts"] - d * 86400, TOL_DAYS * 86400)
            if not np.isnan(p):
                rows.append({"market_id": m["market_id"], "horizon": h, "p": p, "y": m["outcome_yes"],
                             "event": m["event_title"] if isinstance(m["event_title"], str) and m["event_title"] else m["market_id"],
                             "category": m["category"], "volume": m["volume"]})
    obs = pd.DataFrame(rows)
    obs.to_csv("results/observations.csv", index=False)

    summary = {"markets_with_prices": int(obs["market_id"].nunique()), "by_horizon": {}, "by_category_7d": {}}
    for h in HORIZONS:
        summary["by_horizon"][h] = metrics(obs[obs.horizon == h])
    d7 = obs[obs.horizon == "7d"]
    for c, g in d7.groupby("category"):
        if len(g) >= 50:
            summary["by_category_7d"][c] = metrics(g)

    # calibration table (7d)
    tab = []
    idx = np.clip(np.digitize(d7["p"], BINS) - 1, 0, 9)
    for b in range(10):
        s = d7[idx == b]
        if len(s) == 0:
            continue
        lo, hi = wilson(s["y"].sum(), len(s))
        tab.append({"bin": f"{BINS[b]:.1f}-{BINS[b+1]:.1f}", "n": len(s), "mean_price": s["p"].mean(),
                    "observed_freq": s["y"].mean(), "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(tab).to_csv("results/calibration_7d.csv", index=False)

    # Figure 1: reliability diagram by horizon
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Perfect calibration")
    for h, col in zip(HORIZONS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        s = obs[obs.horizon == h]; i = np.clip(np.digitize(s["p"], BINS) - 1, 0, 9)
        xs, ys, es = [], [], []
        for b in range(10):
            t = s[i == b]
            if len(t) >= 10:
                lo, hi = wilson(t["y"].sum(), len(t))
                xs.append(t["p"].mean()); ys.append(t["y"].mean()); es.append([t["y"].mean() - lo, hi - t["y"].mean()])
        es = np.array(es).T
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=col, label=f"{h} before close (n={len(s)})")
    ax.set_xlabel("Market price of YES"); ax.set_ylabel("Fraction that resolved YES")
    ax.set_title("Polymarket calibration"); ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout(); fig.savefig("figures/reliability.png", dpi=150)

    # Figure 2: calibration error by bin (observed - price), 7d
    t = pd.DataFrame(tab)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t["bin"], t["observed_freq"] - t["mean_price"], color="#1f77b4")
    ax.errorbar(range(len(t)), t["observed_freq"] - t["mean_price"],
                yerr=[t["observed_freq"] - t["ci_lo"], t["ci_hi"] - t["observed_freq"]], fmt="none", c="k", capsize=3)
    ax.axhline(0, c="grey", lw=1); ax.set_ylabel("Observed YES rate minus price")
    ax.set_xlabel("Price bin (7 days before close)"); ax.set_title("Where prices are too high or too low")
    plt.xticks(rotation=45); fig.tight_layout(); fig.savefig("figures/calibration_error_7d.png", dpi=150)

    # Figure 3: Brier skill score by category (7d); skill adjusts for each category's base rate
    cat = pd.DataFrame(summary["by_category_7d"]).T.sort_values("brier_skill")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.barh(cat.index, cat["brier_skill"], color="#2ca02c")
    for i, (b, n) in enumerate(zip(cat["brier_skill"], cat["n"])):
        ax.text(b, i, f" {b:.2f} (n={int(n)})", va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("Brier skill score vs. base rate, 7 days before close (higher is better)")
    ax.set_title("Forecast skill by category"); fig.tight_layout(); fig.savefig("figures/skill_by_category.png", dpi=150)

    # Contested markets only (7d price between 0.1 and 0.9): near-certain markets inflate accuracy
    contested = d7[(d7.p >= 0.1) & (d7.p <= 0.9)]
    summary["contested_7d"] = metrics(contested)

    # Out-of-sample test of a recalibration "edge": fit on the earlier half, test on the later half
    d7e = d7.merge(mk[["market_id", "end_ts"]], on="market_id").sort_values("end_ts")
    cut = d7e["end_ts"].median()
    train, test = d7e[d7e.end_ts <= cut], d7e[d7e.end_ts > cut]
    a, b = fit_logistic(logit(train.p.values), train.y.values)
    q = 1 / (1 + np.exp(-(a + b * logit(test.p.values))))
    raw = (test.p.values - test.y.values) ** 2; rec = (q - test.y.values) ** 2
    ev = test["event"].values; uniq = np.unique(ev); pos = {c: np.where(ev == c)[0] for c in uniq}
    diffs = []
    for _ in range(1000):
        pick = RNG.choice(uniq, len(uniq)); s_ = np.concatenate([pos[c] for c in pick])
        diffs.append(raw[s_].mean() - rec[s_].mean())
    summary["oos_recalibration_7d"] = {
        "train_n": int(len(train)), "test_n": int(len(test)), "train_slope": b, "train_intercept": a,
        "test_brier_raw": raw.mean(), "test_brier_recalibrated": rec.mean(),
        "improvement": raw.mean() - rec.mean(),
        "improvement_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
