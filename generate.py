"""
ModelXI - headless prediction pipeline (curated world leagues).

Runs on a schedule (GitHub Actions). Pulls real data for a curated set of the
world's major leagues, fits a Dixon-Coles model per league, has Claude write
grounded analysis for the soonest fixtures, and writes predictions.json.

Keys come from environment variables (GitHub Secrets), never hard-coded:
    API_FOOTBALL_KEY, ANTHROPIC_API_KEY
"""

import os, json, math, time
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
from scipy.optimize import minimize
import anthropic

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
API_FOOTBALL_KEY  = os.environ["API_FOOTBALL_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Curated global set. Each has enough history for the model to be credible.
# NOTE: these are API-Football league ids to the best of our knowledge. A wrong
# id simply gets skipped (too little data) - so check the run log: each league
# prints its top teams, and if they don't belong to that league, fix the id.
LEAGUES = {
    # --- Europe ---
    "Premier League":     39,
    "La Liga":            140,
    "Serie A":            135,
    "Bundesliga":         78,
    "Ligue 1":            61,
    "Championship":       40,
    "Eredivisie":         88,
    "Primeira Liga":      94,
    "Belgian Pro League": 144,
    "Super Lig":          203,
    "Scottish Premiership": 179,
    "Liga I":             283,    # Romania (home league) - verify id if it looks off
    # --- Americas ---
    "Brazil Serie A":     71,
    "Argentina Primera":  128,
    "MLS":                253,
    "Liga MX":            262,
    # --- World ---
    "Saudi Pro League":   307,
    "J1 League":          98,
}

TRAIN_SEASONS      = [2024, 2025]
LIVE_SEASON        = 2026
DAYS_AHEAD         = 30          # wide enough to catch early-season fixtures weeks out
MATCHES_PER_LEAGUE = 10          # up to a full matchday per league
MAX_ANALYSES       = 100         # HARD cap on paid AI write-ups per run (cost control)
MODEL              = "claude-sonnet-5"   # or "claude-haiku-4-5" to spend less

# Cost note: only the soonest MAX_ANALYSES fixtures get a full Claude write-up;
# any overflow still appears with a free template summary. Raise/lower
# MAX_ANALYSES to trade quality vs spend.

MAX_GOALS = 10
API_BASE = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_FOOTBALL_KEY}
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# --------------------------------------------------------------------------- #
# Dixon-Coles model
# --------------------------------------------------------------------------- #
def _tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1.0 - lam*mu*rho
    if x == 0 and y == 1: return 1.0 + lam*rho
    if x == 1 and y == 0: return 1.0 + mu*rho
    if x == 1 and y == 1: return 1.0 - rho
    return 1.0

def _pois(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def fit_model(results, xi=0.0019):
    teams = sorted({r["home"] for r in results} | {r["away"] for r in results})
    if len(teams) < 2:
        raise ValueError("need >=2 teams")
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    def unpack(p):
        c, h, rho = p[0], p[1], p[2]
        atk = np.append(p[3:3+(n-1)], -np.sum(p[3:3+(n-1)]))
        dfc = np.append(p[3+(n-1):], -np.sum(p[3+(n-1):]))
        return c, h, rho, atk, dfc
    w = np.array([math.exp(-xi*r["age_days"]) for r in results])
    def nll(p):
        c, h, rho, atk, dfc = unpack(p)
        tot = 0.0
        for wi, r in zip(w, results):
            i, j = idx[r["home"]], idx[r["away"]]
            lam = math.exp(c+h+atk[i]+dfc[j]); mu = math.exp(c+atk[j]+dfc[i])
            x, y = r["hg"], r["ag"]
            tau = _tau(x, y, lam, mu, rho)
            if tau <= 0: return 1e9
            tot += wi*(math.log(tau) - lam + x*math.log(lam) - mu + y*math.log(mu))
        return -tot
    p0 = np.zeros(3+2*(n-1)); p0[0]=math.log(1.35); p0[1]=0.25; p0[2]=-0.05
    bounds = [(None,None),(None,None),(-0.2,0.2)] + [(-3,3)]*(2*(n-1))
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    c, h, rho, atk, dfc = unpack(res.x)
    return {"teams":teams, "attack":{t:atk[idx[t]] for t in teams},
            "defence":{t:dfc[idx[t]] for t in teams},
            "home_adv":h, "intercept":c, "rho":rho}

def _strength(p): return "strong" if p>=0.60 else "lean" if p>=0.53 else "toss-up"

def predict(m, home, away):
    lam = math.exp(m["intercept"]+m["home_adv"]+m["attack"][home]+m["defence"][away])
    mu  = math.exp(m["intercept"]+m["attack"][away]+m["defence"][home])
    g = np.zeros((MAX_GOALS+1, MAX_GOALS+1))
    for x in range(MAX_GOALS+1):
        for y in range(MAX_GOALS+1):
            g[x,y] = _tau(x,y,lam,mu,m["rho"])*_pois(x,lam)*_pois(y,mu)
    g /= g.sum()
    hw=float(np.tril(g,-1).sum()); dr=float(np.trace(g)); aw=float(np.triu(g,1).sum())
    over=float(sum(g[x,y] for x in range(MAX_GOALS+1) for y in range(MAX_GOALS+1) if x+y>=3))
    btts=float(sum(g[x,y] for x in range(1,MAX_GOALS+1) for y in range(1,MAX_GOALS+1)))
    sx, sy = np.unravel_index(int(g.argmax()), g.shape)
    result_side = max([(f"{home} win",hw),("Draw",dr),(f"{away} win",aw)], key=lambda t:t[1])

    # --- derive every market the goal model can honestly support ---
    def _c(cond):
        return float(sum(g[x,y] for x in range(MAX_GOALS+1) for y in range(MAX_GOALS+1) if cond(x,y)))
    o05=_c(lambda x,y:x+y>=1); o15=_c(lambda x,y:x+y>=2); o35=_c(lambda x,y:x+y>=4)
    dnb=hw+aw; dnb_h=hw/dnb if dnb>0 else 0.0; dnb_a=aw/dnb if dnb>0 else 0.0
    h_o15=_c(lambda x,y:x>=2); a_o15=_c(lambda x,y:y>=2)
    h_by2=_c(lambda x,y:x-y>=2); a_by2=_c(lambda x,y:y-x>=2)
    odd=_c(lambda x,y:(x+y)%2==1); cs_h=_c(lambda x,y:y==0); cs_a=_c(lambda x,y:x==0)
    flat=sorted(((x,y,float(g[x,y])) for x in range(MAX_GOALS+1) for y in range(MAX_GOALS+1)),
                key=lambda t:-t[2])[:3]
    def R(v): return round(float(v),4)
    markets=[
      {"group":"Result","rows":[["Home win",R(hw)],["Draw",R(dr)],["Away win",R(aw)]]},
      {"group":"Double chance","rows":[["Home or draw (1X)",R(hw+dr)],["Home or away (12)",R(hw+aw)],["Draw or away (X2)",R(dr+aw)]]},
      {"group":"Draw no bet","rows":[[home,R(dnb_h)],[away,R(dnb_a)]]},
      {"group":"Asian handicap","rows":[[f"{home} -1.5",R(h_by2)],[f"{away} -1.5",R(a_by2)]]},
      {"group":"Total goals","rows":[["Over 0.5",R(o05)],["Over 1.5",R(o15)],["Over 2.5",R(over)],["Over 3.5",R(o35)]]},
      {"group":"Both teams to score","rows":[["Yes",R(btts)],["No",R(1-btts)]]},
      {"group":"Team goals","rows":[[f"{home} over 1.5",R(h_o15)],[f"{away} over 1.5",R(a_o15)]]},
      {"group":"Clean sheet","rows":[[home,R(cs_h)],[away,R(cs_a)]]},
      {"group":"Goals odd / even","rows":[["Odd",R(odd)],["Even",R(1-odd)]]},
      {"group":"Correct score (top 3)","rows":[[f"{x}-{y}",R(pr)] for x,y,pr in flat]},
    ]

    ou = ("Over 2.5",over) if over>=0.5 else ("Under 2.5",1-over)
    bt = ("BTTS: Yes",btts) if btts>=0.5 else ("BTTS: No",1-btts)
    prim = max([result_side, ou, bt], key=lambda t:t[1])
    return {"home":home,"away":away,"xg_home":round(lam,2),"xg_away":round(mu,2),
            "prob_home":hw,"prob_draw":dr,"prob_away":aw,"over25":over,"btts":btts,
            "top_score":f"{int(sx)}-{int(sy)}","top_score_prob":float(g[sx,sy]),
            "primary_pick":prim[0],"primary_prob":prim[1],"primary_strength":_strength(prim[1]),
            "markets":markets}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _get(path, params):
    r = requests.get(API_BASE+path, headers=HEADERS, params=params, timeout=30)
    data = r.json()
    if data.get("errors"):
        print("  API-Football message:", data["errors"])
    return data

def fetch_all_fixtures(league_id, season):
    data = _get("/fixtures", {"league":league_id, "season":season})
    out = list(data.get("response", []))
    total = data.get("paging", {}).get("total", 1)
    for page in range(2, total+1):
        time.sleep(6)
        d = _get("/fixtures", {"league":league_id, "season":season, "page":page})
        out += d.get("response", [])
    return out

def _date(s):
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception: return None

def finished_results(league_id, seasons):
    now=datetime.now(timezone.utc); res=[]
    for s in seasons:
        for fx in fetch_all_fixtures(league_id, s):
            if fx["fixture"]["status"]["short"] not in ("FT","AET","PEN"): continue
            g=fx.get("goals", {})
            if g.get("home") is None or g.get("away") is None: continue
            d=_date(fx["fixture"]["date"])
            res.append({"home":fx["teams"]["home"]["name"], "away":fx["teams"]["away"]["name"],
                        "hg":int(g["home"]), "ag":int(g["away"]),
                        "age_days":max(0,(now-d).days) if d else 0})
    return res

def upcoming_fixtures(league_id, season, days_ahead, limit):
    now=datetime.now(timezone.utc); horizon=now+timedelta(days=days_ahead); ups=[]
    for fx in fetch_all_fixtures(league_id, season):
        if fx["fixture"]["status"]["short"]=="NS":
            d=_date(fx["fixture"]["date"])
            if d and now<=d<=horizon:
                ups.append((d, fx["teams"]["home"]["name"], fx["teams"]["away"]["name"]))
    ups.sort(key=lambda t:t[0])
    return [(d.isoformat(), h, a) for d,h,a in ups[:limit]]


# --------------------------------------------------------------------------- #
# AI analysis (grounded)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
 "You are the analyst for a football-analysis product. Write ONE short match "
 "preview explaining a statistical model's numbers in plain language.\n\n"
 "ABSOLUTE RULES:\n"
 "1. Use ONLY the facts in the context. If a fact is not there you do not know "
 "it. Never add injuries, transfers, history or motivation from your own memory.\n"
 "2. Never state a probability the model did not give you.\n"
 "3. Every number you write must appear in the context. When citing form, use "
 "the exact record and sequence provided (e.g. '3W 1D 1L, W W D L W'); NEVER "
 "invent your own win-draw-loss tallies or records.\n"
 "4. If a market is near 50/50, call it a coin flip. Do not fake conviction.\n"
 "5. You EXPLAIN the model's numbers; you never contradict, second-guess, or "
 "imply they are wrong or incomplete. Form and goals are supporting context.\n"
 "6. Be specific and quantitative, not generic. Tie every claim to a number.\n"
 "STRUCTURE: a one-line verdict, then 2-3 short sentences. Under 130 words."
)

def build_facts(all_results, team):
    ms=[r for r in all_results if r["home"]==team or r["away"]==team]
    ms=sorted(ms, key=lambda r:r["age_days"])[:5]
    seq=[]; w=d=l=0; gf=ga=0
    for r in ms:
        scored,conc=(r["hg"],r["ag"]) if r["home"]==team else (r["ag"],r["hg"])
        gf+=scored; ga+=conc
        if scored>conc: seq.append("W"); w+=1
        elif scored==conc: seq.append("D"); d+=1
        else: seq.append("L"); l+=1
    n=len(ms)
    return {"last5_sequence":" ".join(seq) if seq else "n/a",
            "last5_record":(f"{w}W {d}D {l}L" if n else "n/a"),
            "last5_goals":f"scored {gf}, conceded {ga} in last {n}"}

def _user_prompt(p, fh, fa):
    payload={"fixture":f"{p['home']} vs {p['away']}",
      "model_output":{
        "result":{p['home']:f"{p['prob_home']*100:.0f}%","draw":f"{p['prob_draw']*100:.0f}%",p['away']:f"{p['prob_away']*100:.0f}%"},
        "expected_goals":{p['home']:p['xg_home'],p['away']:p['xg_away']},
        "over_2_5":f"{p['over25']*100:.0f}%","btts":f"{p['btts']*100:.0f}%",
        "most_likely_score":{"score":p['top_score'],"prob":f"{p['top_score_prob']*100:.0f}%"},
        "primary_call":{"pick":p['primary_pick'],"prob":f"{p['primary_prob']*100:.0f}%","strength":p['primary_strength']}},
      "supporting_facts":{p['home']:fh, p['away']:fa}}
    return "Write the preview using ONLY these facts:\n\n"+json.dumps(payload, indent=2)

def _fallback(p, fh, fa):
    return (f"THE CALL - {p['primary_pick']} ({p['primary_prob']*100:.0f}%, {p['primary_strength']}). "
            f"Model: {p['home']} {p['prob_home']*100:.0f}% / draw {p['prob_draw']*100:.0f}% / "
            f"{p['away']} {p['prob_away']*100:.0f}%, expected goals {p['xg_home']} to {p['xg_away']}. "
            f"Form {p['home']} {fh['last5_record']} ({fh['last5_sequence']}), "
            f"{p['away']} {fa['last5_record']} ({fa['last5_sequence']}). "
            f"Most likely score {p['top_score']} (~{p['top_score_prob']*100:.0f}%), most probable of many.")

def write_analysis(p, all_results):
    fh=build_facts(all_results, p["home"]); fa=build_facts(all_results, p["away"])
    try:
        msg=client.messages.create(model=MODEL, max_tokens=350, system=SYSTEM_PROMPT,
             messages=[{"role":"user","content":_user_prompt(p, fh, fa)}])
        return "".join(b.text for b in msg.content if b.type=="text").strip()
    except Exception as e:
        print("  (LLM failed, using fallback):", e)
        return _fallback(p, fh, fa)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_league(name, league_id):
    print(f"\n=== {name} (id {league_id}) ===")
    res = finished_results(league_id, TRAIN_SEASONS)
    print(f"  {len(res)} training matches")
    if len(res) < 100:
        print("  !! too little data - skipping (check the league id or your plan)")
        return None
    m = fit_model(res)
    top = sorted(m["attack"], key=m["attack"].get, reverse=True)[:3]
    print("  top attacks:", ", ".join(top), " <- should belong to this league")
    return {"model":m, "results":res, "id":league_id}

def main():
    # 1) build each league and collect its upcoming fixtures (cheap, no AI yet)
    leagues_built = {}
    all_fixtures = []
    for name, lid in LEAGUES.items():
        lg = build_league(name, lid)
        if not lg:
            continue
        leagues_built[name] = lg
        known = set(lg["model"]["teams"])
        for kickoff, home, away in upcoming_fixtures(lid, LIVE_SEASON, DAYS_AHEAD, MATCHES_PER_LEAGUE):
            if home in known and away in known:
                all_fixtures.append((kickoff, name, home, away))

    all_fixtures.sort(key=lambda t: t[0])   # soonest first
    print(f"\n{len(all_fixtures)} fixtures across {len(leagues_built)} leagues; "
          f"full AI analysis on the soonest {MAX_ANALYSES}")

    # 2) predict every fixture; AI-write the soonest MAX_ANALYSES, template the rest
    preds = []
    for i, (kickoff, name, home, away) in enumerate(all_fixtures):
        lg = leagues_built[name]
        p = predict(lg["model"], home, away)
        p["league"] = name; p["league_id"] = lg["id"]; p["kickoff"] = kickoff
        if i < MAX_ANALYSES:
            p["analysis"] = write_analysis(p, lg["results"])
        else:
            fh = build_facts(lg["results"], home); fa = build_facts(lg["results"], away)
            p["analysis"] = _fallback(p, fh, fa)
        preds.append(p)

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "predictions": preds}
    with open("predictions.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(preds)} predictions across {len(leagues_built)} leagues to predictions.json")

if __name__ == "__main__":
    main()
