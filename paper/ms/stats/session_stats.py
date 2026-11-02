"""JB's input to the agent and the agent's token use, from the session transcript.

    python paper/ms/stats/session_stats.py TRANSCRIPT.jsonl [--until YYYY-MM-DD]

TRANSCRIPT is the Claude Code session file
(~/.claude/projects/<project>/<session id>.jsonl); its subagent transcripts are
read from <session id>/subagents/. Counts every message JB sent: typed at the
prompt, sent while the agent was working (folded into its current turn as a
queued command), and slash commands. Not counted: tool results, background-task
notifications, compaction summaries, and messages from other agents that
reached the session directly (<cross-session-message>, <agent-message>).
Material that JB relayed (reports from the H100 agent, Codex reviews, logs,
code) is split off heuristically (RELAY): from the line where it starts, the
rest of the message counts as relayed. Words are whitespace-separated; a page
is 500 words (single-spaced).

The API-equivalent cost prices every response's logged usage at the list
prices of the model that served it (PRICES, from
platform.claude.com/docs/en/about-claude/pricing on 2026-11-01; 31 Internet
Archive snapshots of that page, 2026-06-01 to 11-01, show that no model's price
or cache multiplier changed while it was in use; none of these models has a
long-context premium, and all requests ran at standard speed in the global
region): uncached input, cache writes (1.25x input for 5-minute
entries, 2x for 1-hour ones), cache reads, output, and web searches.
"""

import argparse
import collections
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("transcript")
ap.add_argument("--until", default=None, help="last day (UTC) to count")
args = ap.parse_args()
until = (args.until or "9999-12-31") + "T99"

AGENT = re.compile(r"^\s*<(cross-session-message|agent-message|teammate-message)")
REMINDER = re.compile(r"^\s*<system-reminder>.*?</system-reminder>\s*", re.S)
RELAY = re.compile(
    r"(here (is|are) (there|their|its|his|the) (message|notes|report)"
    r"|messages? (from|below)|from (the|its) (h100 )?agent|report from"
    r"|hand-off message|codex review|codex agent|came back with the following"
    r"|```|\n---\n|^\*\*|bovy@|traceback \(most recent|another version:"
    r"|from an agent|\[P[123]\]|^\s*def |^\s*\$ |^FAILED |^E {3}"
    r"|^\d{4}-\d\d-\d\dT\d\d:)",
    re.I | re.M,
)
WORDS_PER_PAGE = 500
# $/MTok: input, cache read, output
PRICES = {
    "claude-opus-4-8": (5, 0.50, 25),
    "claude-opus-5": (5, 0.50, 25),
    "claude-opus-5-5": (4, 0.20, 20),
    "claude-fable-5": (10, 1.00, 50),
    "claude-fable-5-1": (10, 0.25, 50),
    "claude-haiku-4-5-20251001": (1, 0.10, 5),
}
SEARCH_PRICE = 10 / 1000

msgs = {}  # (text, hour) -> (timestamp, text, how); a message is recorded twice
usage = {}  # API response id -> (timestamp, usage, model)


def add(ts, text, how):
    text = REMINDER.sub("", text).strip()
    if text and ts <= until and not AGENT.match(text):
        msgs.setdefault((text, ts[:13]), (ts, text, how))


with open(args.transcript) as fh:
    for line in fh:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = d.get("timestamp", "")
        if d.get("type") == "assistant":
            m = d.get("message", {})
            if m.get("usage") and m.get("id"):
                usage[m["id"]] = (ts, m["usage"], m.get("model"))
        elif d.get("type") == "user" and not (
            d.get("isMeta") or d.get("isCompactSummary") or d.get("toolUseResult")
        ):
            c = d.get("message", {}).get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if x.get("type") == "text")
            if not isinstance(c, str):
                continue
            s = c.lstrip()
            if s.startswith(("<task-notification", "<local-command-stdout", "[Request interrupted")):
                continue
            if s.startswith("<command-name>"):
                cmd = re.search(r"<command-name>([^<]*)", s).group(1)
                cargs = re.search(r"<command-args>([^<]*)", s)
                add(ts, cmd + " " + (cargs.group(1) if cargs else ""), "slash command")
            else:
                add(ts, c, "at the prompt")
        elif d.get("type") == "attachment":
            a = d.get("attachment", {})
            if a.get("type") == "queued_command" and isinstance(a.get("prompt"), str):
                if not a["prompt"].lstrip().startswith("<task-notification"):
                    add(ts, a["prompt"], "while the agent worked")

own = relayed = nrelay = ncodex = 0
lengths, days, how, months = [], set(), collections.Counter(), collections.Counter()
for ts, text, h in msgs.values():
    days.add(ts[:10])
    how[h] += 1
    months[ts[:7]] += 1
    n = len(text.split())
    m = RELAY.search(text)
    if m:
        nrelay += 1
        ncodex += bool(re.search("codex", text, re.I))
        end = text.find("\n", m.start())
        nown = 0 if m.start() < 3 else len(text[: end if end >= 0 else len(text)].split())
        own += nown
        relayed += n - nown
    else:
        own += n
        lengths.append(n)
lengths.sort()
print(f"JB's messages: {len(msgs):,} ({', '.join(f'{v:,} {k}' for k, v in how.items())})")
print(f"  on {len(days)} days, {min(days)} to {max(days)}; by month: {dict(sorted(months.items()))}")
print(f"  his own text: {own:,} words = {own / WORDS_PER_PAGE:,.0f} pages; "
      f"median message {lengths[len(lengths) // 2]} words (messages without relayed material)")
print(f"  relayed: {relayed:,} words = {relayed / WORDS_PER_PAGE:,.0f} pages, "
      f"in {nrelay} messages ({ncodex} with a Codex review)")


category = collections.Counter()  # cost by kind of token, over all rows


def cost(u, model, ttl):
    """List-price cost of one response; ttl is the cache-write duration when not itemized."""
    if model not in PRICES:  # "<synthetic>" placeholders for errors carry no usage
        assert not (u.get("output_tokens") or u.get("cache_read_input_tokens")), model
        return 0.0
    pin, pread, pout = PRICES[model]
    cc = u.get("cache_creation") or {}
    w5, w1 = cc.get("ephemeral_5m_input_tokens", 0), cc.get("ephemeral_1h_input_tokens", 0)
    rest = u.get("cache_creation_input_tokens", 0) - w5 - w1
    w5, w1 = (w5 + rest, w1) if ttl == "5m" else (w5, w1 + rest)
    searches = (u.get("server_tool_use") or {}).get("web_search_requests", 0)
    parts = {"input": u.get("input_tokens", 0) * pin / 1e6,
             "cache writes": (1.25 * w5 + 2 * w1) * pin / 1e6,
             "cache reads": u.get("cache_read_input_tokens", 0) * pread / 1e6,
             "output": u.get("output_tokens", 0) * pout / 1e6,
             "web searches": searches * SEARCH_PRICE}
    category.update(parts)
    return sum(parts.values())


def total(rows, label, ttl):
    rows = [r for r in rows if r[0] <= until]
    keys = ("output_tokens", "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    t = {k: sum(u.get(k, 0) for ts, u, model in rows) for k in keys}
    bymodel = collections.Counter()
    for ts, u, model in rows:
        bymodel[model] += cost(u, model, ttl)
    t["cost"] = sum(bymodel.values())
    print(f"{label}: {len(rows):,} responses; output {t['output_tokens'] / 1e6:,.1f}M tokens; input "
          f"{(t['input_tokens'] + t['cache_creation_input_tokens'] + t['cache_read_input_tokens']) / 1e9:,.1f}B "
          f"tokens ({t['cache_read_input_tokens'] / 1e9:,.1f}B read from the cache); "
          f"list-price cost ${t['cost']:,.0f} ({', '.join(f'{k} ${v:,.0f}' for k, v in bymodel.most_common() if v)})")
    return t


main = total(usage.values(), "main agent", "1h")
sub = {}
for f in glob.glob(os.path.join(os.path.splitext(args.transcript)[0], "subagents", "**", "*.jsonl"), recursive=True):
    with open(f) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = d.get("message", {}) if d.get("type") == "assistant" else {}
            if m.get("usage") and m.get("id"):
                sub[(f, m["id"])] = (d.get("timestamp", ""), m["usage"], m.get("model"))
subt = total(sub.values(), "subagents", "5m")
print(f"  output in all: {(main['output_tokens'] + subt['output_tokens']) / 1e6:,.0f}M tokens")
alltot = main["cost"] + subt["cost"]
print(f"  list-price cost in all: ${alltot:,.0f}: "
      + ", ".join(f"{k} {v / alltot:.0%}" for k, v in category.most_common()))
