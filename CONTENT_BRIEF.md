# Content brief — fraud detection project

**Purpose:** hand this whole file to Claude (claude.ai) and ask it to write **one
complete post** about this project, for LinkedIn and X.

**Why it exists:** I'm job hunting. I want a single post that tells the whole story
— not a series of small ones. It has to work for two readers at once:

- a **recruiter or hiring manager** skimming for signal that I can actually build
  and reason about systems
- a **non-technical person** who should still finish it thinking "that sounds hard
  and he clearly knew what he was doing"

Every number here is real and measured. I need to defend all of it in an interview.

---

## 1. FILL THIS IN BEFORE USING (do not skip)

Claude cannot write in my voice from instructions alone. Replace the brackets.

```
Name:                    [your name]
Current role/status:     [e.g. final-year CS student / SDE-1 at X / job hunting]
Roles I'm targeting:     Data Engineer, ML Engineer, Data Scientist, full-stack
Location / market:       [e.g. India, remote-friendly]
How often I post:        [e.g. rarely — this would be my first technical post]
Link I'll include:       [GitHub repo URL and/or live demo URL]
How long I spent on it:  [e.g. about six weeks of evenings]
```

**Paste 2–3 things I've actually written** — old posts, a README, even a long
WhatsApp/Slack message explaining something technical. **This matters more than
every tone instruction in this file combined.** Claude should imitate these:

```
[paste sample 1]

[paste sample 2]
```

**My honest take, in my own words** (3–5 messy sentences — don't polish, Claude
will mine the phrasing):

```
[e.g. "I thought I was done when I hit 0.94. Then I realised I'd split a
time-series dataset randomly, which basically let the model peek at the future.
Fixing it dropped my score and that stung for a day. Then I kept auditing and
found five more things wrong, including a feature that was the same value for
every single row."]
```

---

## 2. What the project is

A real-time credit-card fraud detection system built on the **IEEE-CIS Fraud
Detection** dataset (Kaggle — 590,540 real transactions, 3.5% of them fraudulent).

Transactions stream through Kafka, a Python service scores each one with an XGBoost
model pulled from an MLflow model registry, results land in Redis, a FastAPI service
serves them, and Prometheus + Grafana monitor the whole thing. There's also a React
dashboard that streams live scoring and lets you drag the fraud threshold and watch
the tradeoff move in real time.

**Honest framing, and this is non-negotiable.** It's a personal project on a public
dataset. It has never run at a bank, never touched real customer money, and no post
should imply otherwise. The impressive part isn't "I built a pipeline" — plenty of
people have. It's that I audited my own finished project and found it was quietly
wrong in nine different ways, then fixed them and can prove each fix.

---

## 3. THE NARRATIVE — the single arc to follow

This is the structure. It's one story, told in order. Everything else in this
document is raw material that slots into these beats.

The arc works because it's a **reversal**: it looks like a success story, becomes a
story about being wrong, and ends as a story about judgment. That's what makes it
readable by anyone and credible to engineers.

**Beat 1 — The setup.**
I built a fraud detection system end to end. It ran. It scored 0.9469 AUC, which is
a strong number. I thought I was finished.

**Beat 2 — The turn.**
Before calling it done, I went back and audited my own work as if reviewing someone
else's. The first thing I found broke the main result.

`TransactionDT` in this dataset is a timestamp. I had split the data randomly into
train and test — which means the model trained on transactions from the future and
was tested on the past. The same card, the same device, on both sides of the split.

*Plain-language version: I'd let it study tomorrow's exam paper and then congratulated
it for passing today's test.*

**Beat 3 — The cost of honesty.**
I re-split the data chronologically: train on the earliest 70%, validate on the next
10%, test on the most recent 20% — the only setup that answers "how will this do on
tomorrow's transactions?"

The score dropped from **0.9469 to 0.8932**.

That 0.0537 was never skill. It was future knowledge. 0.8932 is the only number I'd
put in front of a risk team, and it's the number on the dashboard.

**Beat 4 — It wasn't the only thing.**
Once I started looking properly, I kept finding things. Pick the 4–5 most vivid from
section 4 and list them tightly. The strongest are:
- a feature that was the identical value for all 590,540 rows (the model ignored it entirely)
- the same feature calculated *differently* in training vs live scoring, disagreeing on 5.2% of rows
- monitoring that had been collecting exactly nothing, because two port numbers didn't match
- no real deployment gate — "production" silently served whatever I'd trained most recently

**Beat 5 — The realisation that changed how I think.**
The default decision threshold was 0.5, because that's the default. At 0.5, my system
blocked **8,185 legitimate customers to catch 2,687 frauds** — three-quarters of every
fraud alert was a false alarm.

That isn't a model problem. It's a business question wearing a model's clothes.

Same model, three different business assumptions:
- If a false alarm costs $5 and a missed fraud costs $100 → best threshold 0.38
- If customer friction is dearer, $25 vs $100 → best threshold 0.75. Half the fraud
  caught, but 85% fewer innocent customers blocked.
- If the requirement is "90% of our alerts must be right" → 0.98

Nothing about the model changed. Only what the business cared about. So I stopped
hardcoding the threshold and made it a versioned file the serving code reads on
startup — and put a slider in the dashboard so you can feel the tradeoff.

**Beat 6 — Then I made it fast.**
Scoring went from **48.69 ms to 0.34 ms per transaction** — about 140× — with no
change to the model. I'd been calling the model once per message and making four
separate round-trips to Redis for every transaction. Batching the scoring and
pipelining the writes did it.

**Beat 7 — The part I'm actually proud of.**
I wrote a test that asserts the training code and the live-scoring code produce
**identical inputs** for the same transaction. Two separate implementations existed
and had already drifted apart once.

The first time I ran it, it caught a bug **in my own fix for the earlier bug** —
because `None != None` is `False` in Python, while `NaN != NaN` is `True` in pandas.

Later I retrained with genuinely different preprocessing, and all six tests passed
with zero code changes — because the contract between training and serving is written
to disk and read at runtime, instead of assumed in two places.

**Beat 8 — The close.**
The honest ending, in my own words. Something like: the version of this project with
the better number was the worse project. Don't wrap it in a moral — just land it and
stop.

---

## 4. Verified facts (raw material — every number measured)

### Model
| | |
|---|---|
| Dataset | IEEE-CIS, 590,540 transactions, 3.5% fraud |
| Features | 439 (31 categorical, one with 1,787 distinct values) |
| Split | 413,378 train / 59,054 validation / 118,108 test — chronological |
| **AUC, random split** | **0.9469** ← the number that was lying |
| **AUC, chronological split** | **0.8932** ← the honest one |
| Gap from leakage | 0.0537 |
| Average precision | 0.4972 |
| Features the model actually uses | 417 of 438 — 21 contributed nothing |

### Thresholds — all one model, one 118,108-row holdout
| Threshold | Precision | Recall | Alerts | False alarms |
|---|---|---|---|---|
| **0.50 (the default)** | **0.247** | 0.661 | 10,872 | **8,185** |
| 0.70 | 0.416 | 0.504 | 4,927 | 2,878 |
| 0.83 (best F1) | 0.633 | 0.394 | 2,534 | 931 |
| 0.90 | 0.759 | 0.336 | 1,796 | 432 |
| 0.95 | 0.852 | 0.275 | 1,312 | 194 |

At 0.5: 8,185 innocent customers blocked to catch 2,687 frauds. Total fraud exposure
in that holdout: $609,934.

| Business assumption | Best threshold | Fraud $ caught | Innocent customers blocked |
|---|---|---|---|
| $5 false alarm / $100 missed fraud | 0.38 | $462,408 | 13,707 |
| $25 false alarm / $100 missed fraud | 0.75 | $253,058 | 2,006 |
| "90% of alerts must be correct" | 0.98 | $67,632 | 92 |

### The nine things I found
| What was wrong | Why it mattered |
|---|---|
| The transaction stream never joined the identity table | 41 features permanently missing when scoring live; **0.93% of transactions got a different verdict** |
| `addr_mismatch` feature was constant `1` for every row | It compared a billing *region* code to a billing *country* code — never equal, 0 matches in 50,000 rows. The model gave it **zero splits**. |
| That feature was computed *differently* in training vs serving | Disagreed on **5.2% of rows**: pandas says `NaN != NaN` is `True`, Python says `None != None` is `False` |
| Prometheus scraped port 8001, the service published on 8002 | **Zero metrics collected, ever.** Prometheus wasn't even in the compose file. |
| Latency histogram used the library's default buckets | Those are in *seconds*; I recorded *milliseconds*. Every sample fell in the overflow bucket — all percentiles meaningless. |
| A missing `import json` | One API endpoint returned a 500 on every successful lookup |
| Model stage `"Production"` didn't exist | MLflow 3 removed stages, so the call always failed — into a bare `except` that fell back to "latest". **No deployment gate at all.** |
| MLflow server started without `--serve-artifacts` | Every model download failed with a 500; the service couldn't load *any* model |
| No SIGTERM handling | Docker stops containers with SIGTERM. The service died without saving its place in the stream, losing in-flight work. |

### Performance
| | Before | After |
|---|---|---|
| Scoring latency | 48.69 ms/txn | **0.34 ms/txn** (~140×) |
| Kafka partitions | 1 — couldn't scale past one worker | 6 |
| Delivery guarantee | At-most-once (could silently lose transactions) | At-least-once |

### One more detail worth including if there's room
Once the split was chronological, the rate of never-before-seen categories — new
devices, new email domains — on the test period was **~19× higher** than on
validation. That's real-world drift becoming visible the moment the evaluation
stopped flattering itself.

---

## 5. Plain-language translations (for the non-technical reader)

Use these instead of jargon, or immediately after it. This is what makes the post
land with someone who doesn't code.

| Technical | Say this instead |
|---|---|
| Data leakage from a random split | "I'd accidentally let it study tomorrow's exam paper, then congratulated it for passing today's test" |
| AUC 0.8932 | "it ranks a random fraud above a random legitimate charge about 89% of the time" |
| Precision 0.247 at threshold 0.5 | "three out of every four fraud alerts were false alarms" |
| Decision threshold | "how suspicious is suspicious enough to block a card" |
| Train/serve skew | "the model was fed slightly different information in testing than in real use — so it quietly gave different answers" |
| A constant feature | "one of the clues I gave it was the same for every single transaction, so it was no clue at all" |
| Micro-batching | "instead of processing transactions one at a time, handle them in small groups" |
| The parity test | "an automatic check that the practice environment and the real one are actually identical" |
| Model registry / promotion gate | "a deliberate step to put a new model live, instead of whatever I happened to train last" |

**The single best sentence for a non-technical reader** (adapt, don't quote
verbatim): *"Setting that one number too low meant blocking 8,185 innocent customers
to catch 2,687 criminals — and nobody would have noticed, because the model's
accuracy score looked fine either way."*

---

## 6. What to write

**One story, three containers.** Same narrative, three lengths:

1. **LinkedIn — the main post.** 400–600 words. Longer than typical, which is fine
   for a narrative with real numbers, *provided the first two lines earn the click*.
   Everything after line 2 is hidden behind "see more" — so the hook has to carry it.
   Short paragraphs, 1–2 sentences. Line breaks are load-bearing.

2. **X — a thread.** 8–12 tweets, one beat per tweet. The first tweet must work
   completely alone. Link in the last one.

3. **A short version.** ~120 words, for an X single post or a portfolio blurb.
   Keeps beats 1, 2, 3 and 5 only.

**Rules for all three:**
- Numbers, not adjectives. "0.9469 to 0.8932", never "a significant drop"
- One link, at the end
- 0–3 hashtags on LinkedIn, none on X
- The mistakes are the content. Don't sand them into "learnings"
- Don't end with a moral. Land the last line and stop.

---

## 7. Voice rules — sounding like a person

**Do:**
- Write like explaining to one competent colleague over coffee
- Vary sentence length. Some short. Then a longer one carrying the real idea.
- Say what I *believed* before I found each bug — the wrongness is the story
- Contractions: "I'd", "didn't", "wasn't"
- Let a sentence end without a lesson attached

**Don't:**
- "Not just X, but Y" / "It's not about X. It's about Y."
- Open with a rhetorical question
- Em-dash-heavy rhythm; three-item lists in every sentence
- These words: leverage, delve, robust, seamless, game-changer, journey, unlock,
  elevate, testament, landscape, realm, meticulous, showcase, dive deep
- "🚀 Excited to share", "I'm humbled/thrilled", "Here's the thing:", "Let that sink
  in", bold-unicode text, one-word lines for drama, "Agree?"
- Emoji as bullet points
- Any claim of production use, scale, or business impact

**The test:** would I say this sentence out loud to a friend? If it sounds odd
spoken, rewrite it.

---

## 8. Honesty guardrails (non-negotiable)

Every sentence must survive "tell me more about that" from an interviewer.

- **Personal project, public Kaggle dataset.** Never imply production, real
  customers, real money, or a team.
- Don't say "saved $X". The dollar figures are the dataset's own transaction amounts
  on a held-out split — not business impact. Phrase as "of the fraud in the test
  set, this threshold catches $X".
- **Never lead with 0.9469** except as the thing being corrected.
- The bugs were mine. Not "common pitfalls I observed in the industry."
- Latency figures are single-machine and local.
- If a claim needs a caveat, include the caveat or cut the claim.

---

## 9. Prompt to paste into Claude

> I'm attaching a brief about a fraud detection project I built. I'm job hunting and
> want to post about it.
>
> Read the whole thing. Section 1 has my writing samples — matching that voice is
> the top priority. Section 3 is the narrative arc I want followed, in order.
> Section 5 has plain-language translations. Section 8 is non-negotiable.
>
> I do NOT want several posts on different themes. I want **one complete story**
> that covers everything, written three ways:
>
> 1. **LinkedIn post**, 400–600 words, following the eight beats in section 3.
> 2. **X thread**, 8–12 tweets, same story.
> 3. **Short version**, ~120 words, beats 1/2/3/5 only.
>
> It has to work for two readers at once: a recruiter skimming for evidence I can
> build and reason about systems, and a non-technical person who should finish it
> thinking "that sounds genuinely hard and he knew what he was doing." Use section
> 5 to keep the second reader with me — technical term first, plain explanation
> immediately after, no condescension.
>
> Constraints:
> - Only numbers from section 4. Invent nothing.
> - Personal project on a public dataset — never imply production deployment.
> - None of the banned words or openings in section 7.
> - Don't end with a tidy moral.
>
> Then give me three alternative opening lines for the LinkedIn version, and tell me
> which you'd pick and why.

---

## 10. Follow-up prompts

- "The LinkedIn version reads too polished. Rewrite it rougher — how I'd explain it
  to a friend who also codes."
- "Cut it by 30% without losing a single number."
- "The middle sags. Tighten beats 4 and 6 and give beat 5 more room."
- "Rewrite the opening so it works for someone who has no idea what AUC is."
- "An interviewer read this and wants to go deeper. What are the 5 hardest follow-up
  questions, and how should I answer them?"
- "Give me a 2-sentence version for my LinkedIn headline / CV summary."

---

## 11. Before posting

1. **Read it aloud.** Anything you stumble on gets rewritten.
2. **Check every number** against section 4.
3. **Have the repo link live and the README readable** — anyone actually hiring will
   click before they comment.
4. **Reply to every comment in the first 6 hours.** Does more for reach than the
   post itself.
5. Post the LinkedIn version and the X thread the same week — different audiences,
   minimal overlap.
