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

Also counts JB's comments on GitHub on the agent's PRs since it got its own
account (jobovy-agent, 2026-07-26); before that it posted as jobovy.
"""

import argparse
import collections
import glob
import json
import os
import re
import subprocess

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

msgs = {}  # (text, hour) -> (timestamp, text, how); a message is recorded twice
usage = {}  # API response id -> (timestamp, usage)


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
                usage[m["id"]] = (ts, m["usage"])
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


def total(rows, label):
    keys = ("output_tokens", "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    t = {k: sum(u.get(k, 0) for ts, u in rows if ts <= until) for k in keys}
    n = sum(1 for ts, u in rows if ts <= until)
    print(f"{label}: {n:,} responses; output {t['output_tokens'] / 1e6:,.1f}M tokens; input "
          f"{(t['input_tokens'] + t['cache_creation_input_tokens'] + t['cache_read_input_tokens']) / 1e9:,.1f}B "
          f"tokens ({t['cache_read_input_tokens'] / 1e9:,.1f}B read from the cache)")
    return t


main = total(usage.values(), "main agent")
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
                sub[(f, m["id"])] = (d.get("timestamp", ""), m["usage"])
subt = total(sub.values(), "subagents")
print(f"  output in all: {(main['output_tokens'] + subt['output_tokens']) / 1e6:,.0f}M tokens")

# GitHub: JB's comments on the agent's PRs since it got its own account
since, last = "2026-07-26", (args.until or "2026-12-31")
gh = lambda *a: subprocess.run(["gh", *a], check=True, capture_output=True, text=True).stdout
q = f"repo:jobovy/galpy is:pr author:jobovy-agent created:{since}..{last}"
count = lambda extra: int(gh("api", "-X", "GET", "search/issues", "-f", f"q={q} {extra}", "--jq", ".total_count"))
nprs, nreviewed, ncommented = count(""), count("reviewed-by:jobovy"), count("commenter:jobovy")
agent_prs = set(gh("api", "-X", "GET", "search/issues", "-f", f"q={q}", "--paginate",
                   "--jq", ".items[].number").split())
comments = []
for endpoint, key in (("pulls/comments", "pull_request_url"), ("issues/comments", "issue_url")):
    out = gh("api", "--paginate", f"repos/jobovy/galpy/{endpoint}?since={since}T00:00:00Z&per_page=100",
             "--jq", f'.[] | select(.user.login=="jobovy") | {{pr: (.{key}|split("/")|last), body, created_at}}')
    comments += [json.loads(r) for r in out.splitlines()]
comments = [c for c in comments if c["pr"] in agent_prs and c["created_at"][:10] <= last]
codex = [c for c in comments if re.search(r"codex|\[P[123]\]", c["body"], re.I)]
words = sum(len(c["body"].split()) for c in comments if c not in codex)
print(f"GitHub, agent's PRs opened {since} to {last}: {nprs}; JB reviewed {nreviewed} and commented on "
      f"{ncommented}; {len(comments) - len(codex)} comments of his own ({words:,} words) on "
      f"{len({c['pr'] for c in comments if c not in codex})} PRs, plus {len(codex)} relayed Codex reviews")
