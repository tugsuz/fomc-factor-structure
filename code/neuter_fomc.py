"""Entity Neutering for FOMC statements: hide the meeting, then score raw and neutered text (19 Nov 2026).

The recipe follows Engelberg, Manela, Mullins and Vulicevic (Entity Neutering) and their code
(github.com/LukaVuli/Entity_Neutering): mask, paraphrase, test whether a model can still identify the text,
repeat for the texts that are still identified, then run the task on raw and on neutered text.
What is different here: there is one entity (the Fed), so the thing to hide is the DATE of the meeting.
The repo's neuter_data() cannot be used as it is: its check counts a text as identified when the firm name
is found, and its date rule is 7 days. This script has its own loop with a date-only rule. The repo folder
is not imported and not changed. The prompts below are written after the repo's prompts.py.

Stages
  baseline  the model guesses the date of each RAW statement (how much does it know?)
  neuter    mask -> paraphrase -> guess the date. If the guess is within --window-days of the true date,
            the text counts as identified and goes into the next round (at most --rounds).
  score     the model scores the raw and the neutered statement with the same prompt.
Every answer is appended to a cache file at once, so a run can stop and go on later without paying twice.

Keys: only from the environment (OPENAI_API_KEY or ANTHROPIC_API_KEY). No key is written, printed or logged.
Cost: --dry-run needs no key and no network. It counts tokens and prints the cost. A real run stops when the
running cost passes --max-usd.

Usage, from inside code/:
  python3 neuter_fomc.py --selftest
  python3 neuter_fomc.py --dry-run
  python3 neuter_fomc.py --provider openai --model gpt-5-nano-2025-08-07 --limit 10 --max-usd 0.50
  python3 neuter_fomc.py --provider openai --model gpt-5-nano-2025-08-07 --max-usd 5
  python3 neuter_fomc.py --provider anthropic --model claude-haiku-5-5 --max-usd 5
  python3 neuter_fomc.py --provider ollama --model llama3.1:8b
Output: data_public/fomc_statements/run_<provider>_<model>/ with cache.jsonl, neutered.csv, scores.csv, usage.json.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import paths

ROOT = os.path.join(paths.DATA, "fomc_statements")
# US dollars per million tokens (input, output). Checked on 19 Nov 2026. Haiku and Sonnet: Anthropic's price page.
# GPT-5 nano: pages that quote OpenAI's price page (I could not open that page itself). Check before a large run.
PRICES = {"gpt-5-nano": (0.05, 0.40), "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}

MASK = ("Your role is to ANONYMIZE a policy statement of a central bank committee, so that NOBODY, not even an expert on "
        "monetary policy, can tell on which date or in which year it was issued. Replace with neutral tags: every date, month, "
        "year and season (time_1, time_2); names of people (person_1); names of programs, facilities and operations (program_1); "
        "every LEVEL of an interest rate or target range (rate_level_1); every amount of money and every level given as a number, "
        "such as an unemployment or inflation rate (amount_1); every reference to a specific event such as a crisis, a war, a "
        "pandemic, a storm or a shutdown (event_1). KEEP the economic content: whether policy was tightened, eased or left "
        "unchanged; the direction and the size of every CHANGE (for example: raised rate_level_1 by 25 basis points); the view on "
        "growth, the labor market and inflation in words (stronger, weaker, elevated, moderate); the balance of risks; any "
        "guidance about future policy and its conditions; the reason of any dissent, without the name. Do not add information. "
        "Output the anonymized text and nothing else.")
MASK_ITER = ("You have already tried to anonymize this text and failed. An expert still found the date. These clues gave it "
             "away: {clues}. Be more aggressive. Remove or generalize these clues and anything like them. ")
PARA = ("You are to paraphrase a text so that its core information content is unchanged and someone who has memorized the "
        "original could not recognize it. Change the wording and the sentence structure completely. Use plain language. Keep "
        "every tag such as time_1 or rate_level_1 as it is. Keep whether policy was tightened, eased or unchanged, the direction "
        "and size of every change, the view on the economy, the balance of risks and the guidance about future policy. Replace "
        "stock phrases that are typical for one period. Keep about the same number of words. Output the paraphrased text and "
        "nothing else.")
PARA_ITER = "You have already failed once to make this text anonymous. Take no chances this time. "
IDENT = ("You will receive a statement of the U.S. Federal Open Market Committee. It may have been anonymized. You are "
         "omniscient. Use all of your knowledge and every clue in the text to identify the date of the meeting. Give your best "
         "guess even if you are unsure. Answer in exactly this format and nothing else: "
         "DATE: YYYY-MM-DD | CONFIDENCE: a number from 0 to 1 | CLUES: the words or facts that point to the date")
SCORE = ("You are an expert on U.S. monetary policy. Read the statement of the policy committee. Rate what it means against the "
         "stance before the meeting, on three dimensions. Each is a number from -2 (much easier) to 2 (much tighter); 0 means no "
         "change. TARGET: the decision on the policy rate today. PATH: the expected future path of the policy rate. BALANCE: "
         "asset purchases and the balance sheet. Then MOVE: your best guess of the change of the 2-year Treasury yield in the "
         "30 minutes after the release, in basis points (negative means it fell). Answer in exactly this format and nothing "
         "else: TARGET: x | PATH: x | BALANCE: x | MOVE: x")

LOCK = threading.Lock()
USAGE = {"in": 0, "out": 0, "calls": 0}


class Budget(Exception):
    pass


def n_tokens(text):
    """About four characters per token for English text. An estimate, on the high side for plain prose."""
    return int(len(text) / 4) + 1


def price(model, a):
    if a.price_in is not None and a.price_out is not None:
        return a.price_in, a.price_out
    if a.provider in ("ollama", "mock"):
        return 0.0, 0.0
    for name, p in PRICES.items():
        if model.startswith(name):
            return p
    sys.exit(f"No price for the model {model}. Give --price-in and --price-out (US dollars per million tokens).")


def post(url, headers, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:      # show the provider's message, never the request headers
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='ignore')[:500]}")


def mock(system, user):
    """A fake model for --selftest. It masks years by rule and guesses the right date only from an unmasked year."""
    if system.startswith("Your role is to ANONYMIZE") or system.startswith("You have already tried"):
        return re.sub(r"\b(19|20)\d\d\b", "time_1", user)
    if system.startswith("You are to paraphrase") or system.startswith("You have already failed"):
        return "In other words: " + user
    if system.startswith("You will receive"):
        m = re.search(r"\b((19|20)\d\d)\b", user)
        return f"DATE: {m.group(1) if m else '1990'}-12-16 | CONFIDENCE: 0.5 | CLUES: the year"
    return "TARGET: 1 | PATH: 0 | BALANCE: 0 | MOVE: 2"


def llm(a, system, user, max_out):
    with LOCK:
        pin, pout = price(a.model, a)
        if (USAGE["in"] * pin + USAGE["out"] * pout) / 1e6 >= a.max_usd:
            raise Budget()
    if a.provider == "mock":
        text, tin, tout = mock(system, user), n_tokens(system + user), 0
        tout = n_tokens(text)
    elif a.provider == "openai":
        body = {"model": a.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_completion_tokens": max_out * 4}        # room for hidden reasoning tokens of the GPT-5 family
        if a.reasoning_effort != "none":
            body["reasoning_effort"] = a.reasoning_effort
        r = post("https://api.openai.com/v1/chat/completions", {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}, body)
        text, tin, tout = r["choices"][0]["message"]["content"] or "", r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]
    elif a.provider == "anthropic":
        body = {"model": a.model, "max_tokens": max_out, "system": system, "messages": [{"role": "user", "content": user}]}
        r = post("https://api.anthropic.com/v1/messages",
                 {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}, body)
        text = "".join(c.get("text", "") for c in r["content"])
        tin, tout = r["usage"]["input_tokens"], r["usage"]["output_tokens"]
    else:                                                   # a local Ollama server, free
        body = {"model": a.model, "stream": False, "options": {"temperature": 0, "num_predict": max_out},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        r = post(a.ollama_url.rstrip("/") + "/api/chat", {}, body)
        text, tin, tout = r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0)
    with LOCK:
        USAGE["in"] += tin
        USAGE["out"] += tout
        USAGE["calls"] += 1
    return text.strip()


class Cache:
    """One JSON line for each answer. Key: date, stage, round."""

    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                self.d[(r["date"], r["stage"], r["round"])] = r["text"]

    def get(self, a, date, stage, rnd, system, user, max_out):
        key = (date, stage, rnd)
        if key not in self.d:
            text = llm(a, system, user, max_out)
            with LOCK:
                self.d[key] = text
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"date": date, "stage": stage, "round": rnd, "text": text}) + "\n")
        return self.d[key]


def parse_guess(text):
    m = re.search(r"DATE:\s*(\d{4})-(\d{1,2})-(\d{1,2})", text)
    c = re.search(r"CLUES:\s*(.*)", text, flags=re.S)
    try:
        d = dt.date(int(m.group(1)), int(m.group(2)), min(int(m.group(3)), 28)) if m else None
    except ValueError:
        d = None
    return d, (c.group(1).strip()[:600] if c else "")


def parse_scores(text):
    out = []
    for name in ["TARGET", "PATH", "BALANCE", "MOVE"]:
        m = re.search(name + r":\s*(-?\d+(?:\.\d+)?)", text)
        out.append(float(m.group(1)) if m else "")
    return out


def one_statement(a, cache, row):
    date = row["date"]
    true = dt.date.fromisoformat(date)
    raw = row["text"] + (" " + row["dissent"] if row.get("dissent") else "")       # the list of voters is left out
    hit = lambda d: d is not None and abs((d - true).days) <= a.window_days
    g0, _ = parse_guess(cache.get(a, date, "baseline", 0, IDENT, raw, 200))
    text, clues, rounds, g = raw, "", 0, None
    for rnd in range(1, a.rounds + 1):
        masked = cache.get(a, date, "mask", rnd, (MASK_ITER.format(clues=clues) if rnd > 1 else "") + MASK, text, n_tokens(text) + 300)
        text = cache.get(a, date, "para", rnd, (PARA_ITER if rnd > 1 else "") + PARA, masked, n_tokens(masked) + 300)
        g, clues = parse_guess(cache.get(a, date, "guess", rnd, IDENT, text, 200))
        rounds = rnd
        if not hit(g):
            break
    s_raw = parse_scores(cache.get(a, date, "score_raw", 0, SCORE, raw, 60))
    s_neu = parse_scores(cache.get(a, date, "score_neutered", rounds, SCORE, text, 60))
    yr = lambda d: "" if d is None else int(abs(d.year - true.year) <= 1)
    return {"date": date, "rounds": rounds, "guess_raw": g0, "raw_identified": int(hit(g0)), "raw_year_hit": yr(g0),
            "guess_neutered": g, "identified": int(hit(g)), "year_hit": yr(g), "words_raw": len(raw.split()),
            "words_neutered": len(text.split()), "text_neutered": text, "s_raw": s_raw, "s_neu": s_neu}


def load_statements(a):
    path = os.path.join(ROOT, "statements.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["status"] == "ok" and a.start <= r["date"] <= a.end]
    return rows[:a.limit] if a.limit else rows


def dry_run(a):
    rows = load_statements(a)
    if rows is None:
        n, words = 236, 450
        print("statements.csv is not there yet. The estimate assumes 236 statements of 450 words (run fomc_statements_pull.py).")
        toks = [int(words * 1.35)] * n
    else:
        toks = [n_tokens(r["text"] + " " + r.get("dissent", "")) for r in rows]
        print(f"{len(rows)} statements, {sum(len(r['text'].split()) for r in rows)} words, about {sum(toks)} tokens (4 characters = 1 token).")
    sys_tok = {k: n_tokens(v) for k, v in [("mask", MASK), ("para", PARA), ("ident", IDENT), ("score", SCORE)]}
    print(f"Prompt sizes in tokens: {sys_tok}")

    def total(rounds, overhead):
        tin = tout = calls = 0
        for t in toks:
            per_round_in = (sys_tok["mask"] + t) + (sys_tok["para"] + t) + (sys_tok["ident"] + t)
            per_round_out = (t + t + 80) * overhead
            tin += rounds * per_round_in + (sys_tok["ident"] + t) + 2 * (sys_tok["score"] + t)
            tout += rounds * per_round_out + (80 + 2 * 30) * overhead
            calls += 3 * rounds + 3
        return tin, tout, calls
    print(f"\nCalls for one statement: 3 in each round (mask, paraphrase, guess), 1 baseline guess, 2 scores.")
    print(f"Output tokens are multiplied by {a.reasoning_overhead} for hidden reasoning tokens (--reasoning-overhead).")
    print(f"{'model':22s} {'rounds':>7s} {'calls':>7s} {'input':>11s} {'output':>11s} {'US dollars':>11s}")
    for model, (pin, pout) in list(PRICES.items()) + [("ollama (local)", (0.0, 0.0))]:
        for label, rounds in [("expected", a.assume_rounds), ("worst", a.rounds)]:
            tin, tout, calls = total(rounds, a.reasoning_overhead)
            print(f"{model:22s} {rounds:3d} {label[:3]:>3s} {calls:7d} {tin:11,d} {int(tout):11,d} {(tin * pin + tout * pout) / 1e6:11.2f}")
    print("\nNothing was sent and nothing was spent. 'exp' assumes --assume-rounds rounds for every statement, 'wor' the maximum --rounds.")
    print("Prices: US dollars per million tokens (input, output):", PRICES, "as of 19 Nov 2026. Check the price page before a large run.")


def write_out(a, out, res):
    res = sorted(res, key=lambda r: r["date"])
    cols = ["date", "rounds", "guess_raw", "raw_identified", "raw_year_hit", "guess_neutered", "identified", "year_hit",
            "words_raw", "words_neutered", "text_neutered"]
    with open(os.path.join(out, "neutered.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows([[r[c] for c in cols] for r in res])
    with open(os.path.join(out, "scores.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "identified"] + [f"{n}_{v}" for v in ("raw", "neutered") for n in ("target", "path", "balance", "move")])
        w.writerows([[r["date"], r["identified"]] + r["s_raw"] + r["s_neu"] for r in res])
    pin, pout = price(a.model, a)
    cost = (USAGE["in"] * pin + USAGE["out"] * pout) / 1e6
    json.dump({**USAGE, "usd_this_run": round(cost, 4), "model": a.model, "provider": a.provider},
              open(os.path.join(out, "usage.json"), "w"))
    n = len(res)
    if n:
        print(f"\n{n} statements. Raw text: meeting identified in {sum(r['raw_identified'] for r in res)} of {n}.")
        print(f"After neutering: meeting identified in {sum(r['identified'] for r in res)} of {n}; year within one year in "
              f"{sum(r['year_hit'] == 1 for r in res)} of {n}; mean rounds {sum(r['rounds'] for r in res) / n:.1f}; "
              f"mean length {sum(r['words_neutered'] for r in res) / max(1, sum(r['words_raw'] for r in res)):.2f} of the raw text.")
    print(f"This run: {USAGE['calls']} calls, {USAGE['in']:,} input and {USAGE['out']:,} output tokens, {cost:.2f} US dollars. Files in {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic", "ollama", "mock"])
    p.add_argument("--model", default="gpt-5-nano-2025-08-07")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", default="1994-01-01")
    p.add_argument("--end", default="2099-12-31")
    p.add_argument("--rounds", type=int, default=8, help="maximum number of rounds, as in the repo")
    p.add_argument("--assume-rounds", type=int, default=4, help="rounds assumed by the dry run")
    p.add_argument("--window-days", type=int, default=21, help="a guess within this many days counts as identified")
    p.add_argument("--max-usd", type=float, default=2.0, help="the run stops when its cost passes this")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--reasoning-effort", default="low", help="for OpenAI GPT-5 models: minimal, low, medium, high, or none to leave it out")
    p.add_argument("--reasoning-overhead", type=float, default=2.0, help="dry run: factor on output tokens for hidden reasoning")
    p.add_argument("--price-in", type=float, default=None)
    p.add_argument("--price-out", type=float, default=None)
    p.add_argument("--ollama-url", default="http://localhost:11434")
    a = p.parse_args()
    if a.dry_run:
        return dry_run(a)
    if a.selftest:
        a.provider, a.model, a.rounds, a.workers = "mock", "mock", 3, 1
        rows = [{"date": "2015-12-16", "text": "The Committee decided in 2015 to raise the target range by 25 basis points.", "dissent": ""},
                {"date": "2008-12-16", "text": "The Committee cut the target. Conditions in 2008 were weak.", "dissent": "Voting against was X."}]
        out = os.path.join(paths.ROOT, "_scratch", "neuter_selftest")
        os.makedirs(out, exist_ok=True)
        cache = Cache(os.path.join(out, f"cache_{dt.datetime.now():%H%M%S}.jsonl"))
        res = [one_statement(a, cache, r) for r in rows]
        assert all(r["raw_identified"] == 1 and r["identified"] == 0 and r["rounds"] == 1 for r in res), res
        assert res[0]["s_raw"] == [1.0, 0.0, 0.0, 2.0] and "time_1" in res[0]["text_neutered"]
        assert parse_guess("DATE: 2019-07-31 | CONFIDENCE: 0.9 | CLUES: a, b")[0] == dt.date(2019, 7, 28)
        write_out(a, out, res)
        return print("selftest passed")
    key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(a.provider)
    if key and not os.environ.get(key):
        sys.exit(f"Set the environment variable {key} in your Terminal first. The key is never written to a file.")
    rows = load_statements(a)
    if rows is None:
        sys.exit("statements.csv is missing. Run: python3 fomc_statements_pull.py")
    out = os.path.join(ROOT, "run_" + re.sub(r"[^A-Za-z0-9.]+", "-", f"{a.provider}_{a.model}"))
    os.makedirs(out, exist_ok=True)
    cache = Cache(os.path.join(out, "cache.jsonl"))
    print(f"{len(rows)} statements, model {a.model}, at most {a.rounds} rounds, stop at {a.max_usd} US dollars. Answers already cached: {len(cache.d)}")
    res, stopped = [], False

    def work(r):
        try:
            return one_statement(a, cache, r)
        except Budget:
            return None
        except Exception as e:
            print(r["date"], "error:", str(e)[:300])
            return None
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, rows):
            if r is None:
                stopped = True
            else:
                res.append(r)
    write_out(a, out, res)
    if stopped:
        print("Some statements are not finished (cost limit or errors). Run the same command again to go on. Cached answers are not paid twice.")


if __name__ == "__main__":
    main()
