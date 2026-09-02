# ticket_agent

An LLM agent that turns unstructured support tickets into validated JSON.

```
"my fibre has been dead since Tuesday, call me on 06 1234567"
                          ↓
{"category": "network_outage", "severity": "high", "affected_service": "broadband",
 "sentiment": "angry", "summary": "...", "callback_number": "06 1234567",
 "needs_human": true}
```

A learning project: a minimal agent, and a sandbox for CI/CD and containers.

## How it works

```
  ticket text
      │
      ▼
 ① extract      ask the model for JSON
      │
      ▼
 ② validate     deterministic check, no LLM
      │
   valid? ──yes──► return
      │no
      ▼
 ③ decide       show the model the draft, the errors, and its tools
      │
      ▼
 ④ act          the model calls set_fields / regenerate / give_up
      │
      ▼
 ⑤ observe      execute it, re-validate, report back ──► ③
                (bounded by max_steps)
```

Step ③ is what makes this an agent rather than a pipeline: **the model chooses
the next action.** Our code executes what it picked and tells it what happened.

**One step is one model call**, the convention agent frameworks settle on
(LangChain counts iterations, the OpenAI Agents SDK counts turns). It is the
only definition under which `max_steps` bounds anything you pay for, so
`result.steps == client.usage.calls` is asserted on every path. Counting loop
iterations instead hid the extra call a `regenerate` makes: runs reported
`steps=3` while the meter recorded 4.

The validator is the source of truth throughout. The model is never trusted —
anything returned is guaranteed to satisfy the schema, because anything else
would not have been returned. A failed extraction returns `None` and the
reasons, never partial data.

| File | Role |
|---|---|
| `ticket_agent/schema.py` | The 7 fields, their types and allowed values. **One source of truth** — the prompt, the tool descriptions and the validator are all generated from it, so they cannot drift. |
| `ticket_agent/validator.py` | Deterministic checking. No LLM. Also *locates* the JSON in a reply (see below). |
| `ticket_agent/tools.py` | The actions the model may take, and their execution. |
| `ticket_agent/agent.py` | The loop above. |
| `ticket_agent/client.py` | The model boundary: a `Protocol` plus one class per provider. |
| `ticket_agent/config.py` | Config resolution: CLI flag → env var → `config.toml` → fallback. |
| `ticket_agent/transcript.py` | Opt-in recorder: wraps any client and keeps every prompt and reply. |

### Lenient about shape, strict about meaning

One rule decides every parsing question in this repo, and it is worth stating
before the tools:

> **Be lenient about the shape a model wraps its answer in. Be strict about
> what the answer means.**

Models bury the payload. They fence it in ```` ```json ````, they write "Here is
the JSON:", and a reasoning model emits `<think>...</think>` first. None of
those is a comprehension failure — the answer is right there — so
`extract_json_block()` finds the object and moves on:

1. the whole reply is a fence → unwrap it
2. the whole reply already parses → return it untouched
3. otherwise → scan for the first balanced `{...}` with `json.JSONDecoder.raw_decode`

Order matters. Step 2 runs before the scan so a reply that is *already* valid
JSON keeps its meaning: `[{...}]` is a list and is rejected as one, rather than
being silently rescued by digging the object out of it. The scan handles junk
*around* the JSON; it never rewrites JSON that parsed fine.

The same rule applies at the tool boundary. Small models routinely send
`{"category": "billing"}` where the schema says
`{"fields": {"category": "billing"}}`. `execute()` recognises the flattened
form **when any key is a real field name** — enough evidence that the intent is
not in doubt — and then checks every name exactly as it would through the
wrapper. Arguments with no field name in them at all are not read as flattened,
because guessing there would be inventing intent the model never expressed.

What none of this relaxes is *meaning*. An invented category inside a
perfectly-located object is still rejected; a flattened call still has its
values checked. Leniency buys back round-trips that a parser could have saved;
it never widens what counts as valid.

> Why it matters in money: a reasoning model scored **0/6** on the sample set
> while its output contained the correct answer every time. Spending an API call
> to repair something a parser can find locally for free is the wrong trade —
> fix deterministically what you can, and reserve the repair loop for what
> genuinely needs the model.

### The tools

A "tool" here is nothing mystical: it is a **JSON description of a function**,
sent to the model alongside the messages. The model cannot execute anything. It
replies with the *name* of the function it wants and the arguments it wants to
pass; our code decides whether to run it, runs it, and reports back.

`tools.py` holds both halves, deliberately in one file so they cannot drift:

| | |
|---|---|
| `TOOL_SCHEMAS` | What the model sees. Standard OpenAI `tools` format, accepted by Groq, Ollama and most others. |
| `execute()` | What actually runs. Takes `(name, args, draft)` and returns a `ToolResult`. |

#### `set_fields(fields: dict) -> ok / error`

Overwrite named keys of the current draft. **The workhorse** — most validation
failures are one or two wrong values, not a broken draft.

```
model asks:   set_fields({"category": "network_outage",
                          "callback_number": "+39 333 1234567"})
it gets back: "ok: set category='network_outage',
               callback_number='+39 333 1234567'. The extraction is now valid."
```

It takes a **dict of several fields, not one field at a time**. That is the
single most important design choice in this file: each tool call is a full
network round-trip, so fixing six fields in one call is six times cheaper — in
latency, tokens and steps — than six calls. The `description` string says so
explicitly, because the model has no other way to know it.

Rejected input becomes an observation, never an exception:

```
set_fields({})                    -> "error: 'fields' must be a non-empty object..."
set_fields({"catagory": "..."})   -> "error: not fields: catagory. Valid names: ..."
```

The list of valid names in both messages is generated from `schema.py`, so a
typo'd field name is reported against the real schema rather than a copy of it.

#### `regenerate() -> ok`

Throw the draft away and run the extraction prompt again from scratch. For when
the draft is **empty, unparseable, or wrong nearly everywhere** — patching an
empty draft key by key costs more round-trips than starting over.

Note what this is *not*: it is not our retry loop. Nothing in our code decides to
regenerate. The model looks at its own failed output and concludes that patching
is hopeless.

#### `give_up(reason: str) -> stop`

Declare the ticket genuinely un-extractable. Returns `ok=False`, `data=None`, and
the model's stated reason.

This exists because **the alternative to giving up is not success, it is
fabrication.** Without an honest exit, a model that cannot find a callback number
in a ticket that has none will eventually invent one to make the errors go away.
The description says "only when the text genuinely does not contain the
information, never merely because it is difficult" — that sentence is load-bearing.

#### How a tool call becomes the next turn

```
1. client.chat_with_tools(messages, TOOL_SCHEMAS)  → ToolTurn(tool_calls=[...])
2. toolkit.execute(call.name, call.arguments, draft) → ToolResult
3. validate(draft)          ← immediately, after every action
4. messages.append({"role": "tool", "tool_call_id": call.id,
                    "content": observation})
5. loop
```

Step 3 is why `ToolResult.observation` is a sentence and not a status code. What
comes back is the action's effect **and** the resulting state, in one string:

```
ok: set category='network_outage'. Remaining:
- "severity" must be one of ['critical', 'high', 'low', 'medium'], got 'urgent'
```

That feedback *is* the loop. A tool that returned nothing useful would leave the
model guessing whether its own fix landed, and it would guess wrong.

Step 4 is the agent's memory: `messages` accumulates, so on step 5 the model can
see what it already tried. Without it every turn would be a fresh guess and the
same wrong value would come back three times running.

#### Design notes

- **Tool descriptions are prompt engineering, not documentation.** They are the
  only thing telling the model when each action applies — and the *only* place
  that guidance lives. The system prompt does not repeat it: `description` is
  the field the API provides for this and is sent on the same request, so a
  second copy is one more place to edit when the rule changes. Same reasoning
  as `schema.py`. A vague description
  produces wrong tool choices exactly the way a vague error message produces
  wasted repairs.
- **`execute()` never raises on model input.** An unknown tool name or a
  malformed argument returns an error *observation*. Crashing would end the run;
  telling the model lets it try something else, which is the entire point of
  giving it a choice.
- **The tools do nothing dangerous** — they only ever touch a local dict. No network, no
  filesystem, no customer records. The input is untrusted customer text (see
  `samples/edge_02_injection.txt`), and a model with tools processing untrusted
  input is exactly where prompt injection stops being harmless. These are actions
  an attacker gains nothing from.
- **Tool calling needs a provider that supports it.** `groq`, `ollama` and `mock`
  do; `gemini` and `hf` go through bespoke SDKs and have no `chat_with_tools`, so
  the agent raises `TypeError` immediately rather than failing halfway through a
  ticket. Below roughly 1.5B parameters a model cannot call tools at all.

## Setup

```bash
conda create -n ai4i python=3.12 && conda activate ai4i
pip install -r requirements-dev.txt
cp .env.example .env          # then add your API keys
```

`.env` holds **secrets only** (gitignored). Everything that changes what the
agent produces — model, provider, retry limits — lives in `config.toml`, which
is committed so changes are reviewable.

## Run it

```bash
# No API key, no network, no cost — scripted responses that exercise a repair
python -m ticket_agent --file samples/ticket_02.txt --mock --verbose

# Against a real provider
python -m ticket_agent --file samples/ticket_02.txt --provider groq --verbose

# All sample tickets; writes logs/run_<stamp>.jsonl and .trace.jsonl
./scripts/run_batch.sh --provider groq
```

### Providers

| `--provider` | Needs | Notes |
|---|---|---|
| `mock` | nothing | Scripted. Free, deterministic, used by every test. |
| `gemini` | `GEMINI_API_KEY` | |
| `groq` | `GROQ_API_KEY` | Free tier. Not the same as Grok/xAI. |
| `hf` | `HF_TOKEN` | Routes to a third-party backend; pin it with `HF_PROVIDER`. |
| `ollama` | `brew install ollama` | Local, free, unlimited, no network. |

Adding an OpenAI-compatible provider (Together, Fireworks, a local vLLM) is a
block in `config.toml` — no code.

## Tests

```bash
pytest -q        # 155 tests, no API key, no network, no cost
ruff check .
```

Every test injects `MockLLMClient`, which replays scripted extractions **and
scripted tool choices**. That is what makes the agent examinable: you cannot ask
a real model to pick a particular tool on demand, so the choice is scripted and
what the loop does with it is asserted.

Integration tests against real providers are deliberately absent from CI — see
the comment in `.github/workflows/ci.yml`.

## Docker

```bash
docker build -t ticket-agent .
docker run --rm -v "$PWD/samples:/data:ro" ticket-agent --file /data/ticket_02.txt --mock
docker run --rm -e GROQ_API_KEY="$GROQ_API_KEY" -v "$PWD/samples:/data:ro" \
    ticket-agent --file /data/ticket_02.txt --provider groq
```

`config.toml` ships in the image; secrets never do — they arrive as environment
variables at run time.

## Inspecting a run

When a ticket fails, there are two different questions, and two records:

| | Answers | Where |
|---|---|---|
| **trace** | What the agent *did* — one line per step | `result.trace`, and `trace` in the log |
| **conversation** | What the model *saw and said*, verbatim | `conversation` in the log |

The trace tells you a step went wrong. The conversation tells you why.

```bash
# Narrated walkthrough of one ticket, ending with the full dialogue
python scripts/walkthrough.py samples/ticket_01.txt --provider ollama --conversation
python scripts/walkthrough.py samples/ticket_01.txt --conversation --full   # no eliding

# Machine-readable: --report carries both records by default
python -m ticket_agent --file samples/ticket_01.txt --provider ollama --report
python -m ticket_agent --file samples/ticket_01.txt --report --no-trace     # result only
```

Both are **on by default** inside `--report`, because the cost is asymmetric:
recording detail you never read wastes some disk, while failing to record it
means re-running the model — and a re-run is a different run, so the failure you
were chasing may not happen again.

Every batch therefore writes both files:

| File | Contents |
|---|---|
| `logs/run_<stamp>.jsonl` | Result, metadata and **per-call token usage**. Small enough to `jq`, grep and diff. |
| `logs/run_<stamp>.trace.jsonl` | The same records **plus** `trace` and `conversation`. |

### What is in a log record

One JSON object per ticket. Four groups:

**What produced this result** — without these, an output is not evidence of
anything, because you cannot say what made it.

| Key | Meaning |
|---|---|
| `timestamp` | UTC, when the run finished. |
| `source` | Ticket file, or `<inline>` for `--text`. |
| `provider` | `gemini` / `groq` / `hf` / `ollama` / `mock`. |
| `route` | Only meaningful for `hf`: which backend it was routed to. The same weights on a different backend can produce different output. `null` elsewhere. |
| `model` | The model id actually called. |
| `prompt_version` | Bumped whenever a prompt changes. Results are comparable only within one version. |
| `json_mode` | The **resolved** value, not the CLI flag — what was used, including when it came from `config.toml`. |

**What happened**

| Key | Meaning |
|---|---|
| `ok` | Did a validated extraction come back. |
| `data` | The extraction, or `null`. Never partial: `ok=false` always means `data=null`. |
| `errors` | The validation errors still outstanding at the end. Empty when `ok`. |
| `steps` | **Model calls.** Always equals `usage.calls` — enforced by a test on every path. |
| `stop_reason` | `validated` / `gave_up` / `model_stopped` / `budget_exhausted` / `empty_input`. |
| `gave_up` | The model's stated reason, if it called `give_up`. |
| `elapsed_s` | Wall clock. |

**What it cost** (`usage`) — tokens, not money: providers publish prices in docs,
not APIs, and a hardcoded rate would go stale silently.

| Key | Meaning |
|---|---|
| `calls` | Model round-trips. **The billable number.** |
| `prompt_tokens` / `completion_tokens` / `total_tokens` | Run totals. |
| `truncated_calls` | Call numbers cut off at `max_tokens`. Empty is the good case. |
| `per_call[]` | One entry per call: `n`, `kind` (`generate`/`chat`), its own token counts, `finish_reason`, `truncated`. |

**What the agent did** (`--trace` only)

`trace[]` — **one entry per decision.** The narrative.

| Key | Meaning |
|---|---|
| `step` | Its own sequence, `1..N`, no gaps. |
| `action` | `extract` (the first attempt) / `tool` (the model acted) / `stop` (it replied in prose instead). |
| `calls` | Which model calls this decision used. `[2, 3]` means it cost two. |
| `detail` | The draft, the observation, or the prose it stopped on. |
| `ok`, `errors` | Validation state after this decision. |

`conversation[]` — **one entry per model call.** The verbatim exchange.

| Key | Meaning |
|---|---|
| `n` | Call number. Joins to `usage.per_call[]` and to `trace[].calls`. |
| `call_type` | `generate` (plain completion, **no tools attached**) / `chat` (conversation + tool menu). |
| `tools_offered` | The tool names sent. Populated on `chat`, empty on `generate`. |
| `request` | Every message sent, verbatim. |
| `response` | `content` and `tool_calls`. |

**Two vocabularies, two key names.** `action` describes what the agent decided;
`call_type` describes what kind of API call it took. An earlier version called
both of them `kind`, so one key carried two disjoint value sets in one record
and the log could not be read without knowing which branch you were in.

Token counts are written **once**, in `usage.per_call`, joined by `n`.
`walkthrough.py --conversation` shows them beside each reply.

Reading a failed run:

```
steps=4  calls=4  stop_reason=model_stopped

trace step=1  action=extract  calls=[1]     draft, affected_service="internet" (invalid)
trace step=2  action=tool     calls=[2,3]   chose regenerate -> same invalid value
trace step=3  action=stop     calls=[4]     replied in prose instead of acting
```

### Did we hit the token cap?

`max_tokens` is a hard cap, and a reply cut off at the cap is the most
misleading failure this agent has: the model looks broken when the *budget* was
the problem. So every call records how it ended, and a truncated one is
announced on stderr without being asked:

```
warning: 2 model call(s) hit the max_tokens cap (30) and were cut off mid-answer: call(s) 1, 3.
         An empty or unparseable reply here is a BUDGET problem, not a model problem.
```

In the log, `usage` carries both the run total and every individual call:

```json
"usage": {
  "calls": 4, "total_tokens": 2628,
  "truncated_calls": [1, 3],
  "per_call": [
    {"n": 1, "kind": "generate", "prompt_tokens": 413, "completion_tokens": 30,
     "finish_reason": "length", "truncated": true},
    {"n": 2, "kind": "chat", "prompt_tokens": 822, "completion_tokens": 16,
     "finish_reason": "tool_calls", "truncated": false}
  ]
}
```

Truncation is detected from **`finish_reason`, not from the token count.**
`completion_tokens == max_tokens` is only a hint — a model may legitimately
finish on the last allowed token. `finish_reason == "length"` is the provider
saying it outright. Providers spell it differently (`length`, `MAX_TOKENS`), so
it is normalised to one value.

```bash
jq -r 'select(.usage.truncated_calls | length > 0) | .source' logs/run_<stamp>.jsonl
```

Recording is a wrapper (`transcript.py`) around whatever client you were going
to use, not a change to the agent — which is the `LLMClient` Protocol paying off
a third time. It is opt-in because keeping every prompt in memory is real memory
on a long batch.

Note that dumping the agent's `messages` list would **not** be the same thing:
the `regenerate` path calls `generate()` with a fresh prompt that never enters
the conversation, so the step you most want to inspect would be the one missing.
Recording at the client boundary catches every call by construction.

## Notes

- **The validator checks form, not truth.** It guarantees schema-conformant
  output. It cannot tell you the answer is correct — a model too weak to
  understand the ticket but strong enough to fill in the shape produces
  confident, valid, wrong data.
- **The loop fixes slips, not capability gaps.** Models below ~1B returned the
  same invalid value three times running after being told the allowed values
  verbatim, and cannot call tools at all.
- **A token budget must cover thinking AND answering.** A cap sized to the
  ~90-token answer left a reasoning model 2 tokens to answer with after it spent
  298 thinking, so it returned nothing and looked broken. `max_tokens` is set
  per provider for this reason.
- **`temperature=0` is not determinism.** The same ambiguous ticket was
  classified `billing` on one run and `other` on the next, same model, same
  prompt. Version the inputs and *store* the outputs; the log is the artifact.
- **Pin what you depend on, not just what you call.** `config.toml` argued
  for pinning model ids while `requirements.txt` left every library floating, so
  `docker build` produced a different program each month with nothing in git to
  explain it. Applications pin exactly; libraries declare ranges.
- **A setting that reaches some backends is worse than one that reaches none.**
  `[agent] max_tokens` was applied by two clients out of three — Gemini spells it
  `max_output_tokens` and never received it — so the CLI's truncation warning
  told you to raise a number that was never being sent. Silent partial
  application is how you stop trusting a config file.
- **A catalogue lists what exists, not what you may call.** Models appeared in
  provider listings that returned 404 or "not supported by any provider you have
  enabled".
