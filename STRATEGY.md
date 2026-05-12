# NVIDIA Nemotron Reasoning Challenge — Winning Strategy

> Goal: top-3 final leaderboard *and* qualify for the Open Progress Prize at the
> 2026-04-09 midpoint. The path to both is the same plan, executed quickly.

---

## 1. The unfair advantage

The benchmark is **six narrow, programmatically solvable puzzle types**, all
framed identically ("In Alice's Wonderland..."). Each prompt contains *the
examples that define the rule*, plus the query. That means a small Python
function can score ~100% on every category given enough effort:

| Category     | Train share | Solvability |
|--------------|-------------|-------------|
| `bit_manip`  | 1602        | enumerate over a small operator family until a candidate matches all examples |
| `cipher`     | 1576        | align word-pairs by length to recover the letter substitution table |
| `numeral`    | 1576        | detect base/system from examples (Roman + maybe others), convert |
| `unit_conv`  | 1594        | linear regression `y = a·x + b` over the (input, output) pairs |
| `gravity`    | 1597        | average `g = 2·d / t²` across examples, plug in |
| `eq_symbols` | 1555        | symbol→symbol map or char-arithmetic (ASCII delta); hardest, but tractable |

**Therefore: don't try to teach the model raw reasoning from 9,500 examples.
Generate millions of perfect, fully-worked solutions with your own solvers and
SFT the model on those.** This converts the contest from "can the model reason?"
to "can the model imitate a known-correct reasoning trace?", which a 30B model
trained on rank-32 LoRA can do near-perfectly.

This same advantage is available to every competitor — the differentiator is
who builds the cleanest solvers, the highest-quality CoT traces, and trains the
adapter with the fewest distribution-shift bugs.

---

## 2. Submission constraints (do not violate these)

- LoRA adapter for `nemotron-3-nano-30b-a3b-bf16`, **rank ≤ 32**.
- `submission.zip` must contain `adapter_config.json` (+ `adapter_model.safetensors`).
- Eval runs on vLLM with `temperature=0.0`, `max_tokens=7680`, `max_model_len=8192`.
- Final answer must appear inside `\boxed{...}`. Metric falls back to "last
  numeric value" — that fallback is a *trap* for non-numeric answers (cipher,
  bit_manip, numeral, eq_symbols). **Always emit `\boxed{}`** at the end of
  every training trace.
- For prize eligibility: publish a Kaggle notebook *and* a write-up.

Headroom: with `temperature=0` you don't need self-consistency. With 7680 output
tokens you have ample room for reasoning, but stay disciplined — long traces
hurt latency and add error surface.

---

## 3. Phased plan

### Phase 0 — Local sanity (½ day)

- ✅ Data is unzipped at `data/train.csv` / `data/test.csv`.
- Build a `categorize(prompt) -> str` function (already trivial — first-line
  keyword match).
- Stratify the 9,500 rows: 9k train / 500 holdout (preserve category balance).
  Holdout is for *ground-truth eval of every solver and adapter version*.

### Phase 1 — Build perfect Python solvers (2–3 days, the most leveraged work)

For each of the 6 categories, write `solve_<cat>(prompt) -> answer` and verify
≥99% on the 9k slice. **Do not start LoRA training until every solver is at or
near 100% on its category.**

**`unit_conv`** — easiest. Parse the float pairs with regex, fit `y = a*x + b`
via least-squares (numpy `polyfit`). Round to two decimals to match training
labels.

**`gravity`** — closed-form. From each example, compute `g_i = 2*d_i / t_i²`,
take mean (or median for robustness), then `d = 0.5 * g_mean * t_query²`. Round
to two decimals.

**`numeral`** — start with a Roman numeral encoder/decoder; verify all numeral
puzzles match. If any don't, inspect — could be Greek, base-N, or another
system. Branch on detection.

**`cipher`** — for each (src_word, tgt_word) where lengths match, record
position-aligned letter mappings. Build the union table; resolve any conflicts
by majority vote over all aligned positions across all word pairs. Then map
each letter of the query. (Edge case: missing letters — fall back to the most
likely Caesar shift inferred from known mappings, or leave gap.)

**`bit_manip`** — enumerate a candidate function family:
- Rotations: `rotl(x, k)`, `rotr(x, k)` for `k ∈ {1..7}`
- XOR with constant `c ∈ {0..255}`
- NOT
- Compositions: `f(x) = rotl(x XOR c1, k) XOR c2`
- Possibly `(x AND m1) | (rotr(x, k) AND m2)`

For each candidate, check it matches *all* 8 example pairs. If multiple match,
take any. With ~256³ × 8 candidates max, this is milliseconds. (If you find a
prompt with no match, broaden the family.)

**`eq_symbols`** — hardest. Hypotheses to test in order:
1. Per-character substitution map `c -> c'` (table lookup), output = map applied
   to input.
2. Per-character ASCII arithmetic: `out_i = in_i + Δ` (mod printable range),
   possibly position-dependent.
3. Pairwise/expression rewrite: input is parsed as an expression in some
   symbol→digit encoding, evaluated, and re-encoded.
4. Output length differs from input length → not a pure char map; possibly a
   filter (drop chars matching a predicate) or a duplication/run-length rule.

Build a small library of structural detectors and try each on the example set.
If all six categories sit ≥99%, you are positioned to win.

### Phase 2 — Synthetic CoT data generation (1–2 days)

For every solver, also write a `explain_<cat>(prompt) -> str` that emits a
worked-example reasoning trace ending in `\boxed{<answer>}`. Style guidelines:

- Open with category recognition: *"This is a unit-conversion puzzle. I'll fit
  a linear relation y = a·x + b…"*
- Show the actual computation (numbers visible). Don't fake the chain — produce
  the same numbers your solver computed.
- Keep traces 200–600 tokens. Long enough to teach the structure, short enough
  to fit hundreds of examples per training batch.
- End every trace with `\n\nFinal answer: \boxed{<exact_answer>}`.

Volume target: **50k–200k examples**, balanced across categories. Generation
parameters should *vary* (number of examples in the prompt, value ranges, edge
cases like negative shifts, conversion factors near 1, all-same inputs) so the
adapter generalizes.

Hold out a true-distribution validation slice from the *original* train.csv —
never train on it.

### Phase 3 — LoRA fine-tuning

**Adapter config** (mirroring the demo, which is correct):

```python
LoraConfig(
    r=32, lora_alpha=64,           # alpha 2× rank for stronger update
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
```

`in_proj`/`out_proj` cover the Mamba SSM blocks; `up_proj`/`down_proj` cover
the MLPs. With ~880M trainable params at rank 32, the adapter has plenty of
capacity for this task.

**Training recipe**:
- Single SFT epoch, possibly two. Loss-mask the prompt (only score the
  reasoning + answer tokens).
- Constant or cosine LR ~`1e-4` to `2e-4`. Warmup 3% steps.
- Effective batch ~64 sequences (use grad-accum). Sequence length 2k is fine —
  prompts are <500 chars, traces <600 tokens.
- bf16 weights, **paged_adamw_8bit** optimizer.
- Save checkpoints every ~500 steps; eval each on holdout.

**Where to train**:

*Kaggle (recommended for the actual submission notebook)* — same env as eval,
no surprises. Notebooks have time limits, but you can train offline elsewhere
and only run packaging on Kaggle.

*Hendrix* — single L40s 48GB cannot host the 30B model in bf16 (~60GB). For
training there, you need **QLoRA**: load base in 4-bit (NF4), train LoRA on
top. This drops base memory to ~15GB and trains on a single L40s. The LoRA
weights themselves are bf16-compatible — at eval they are merged onto the
bf16 base. (Quantize base only for *training*; submission ships only the LoRA
adapter, so the deploy-time quantization mismatch is not a correctness issue,
but it can hurt SFT quality slightly. Validate this empirically before
committing to the QLoRA path.)

If you can grab 2× L40s on Hendrix or use multi-GPU, a non-quantized bf16 +
LoRA setup is preferable.

### Phase 4 — Iterate aggressively

After each training run:
1. Run greedy decode on the 500-row holdout. Compute per-category accuracy.
2. Look at every error. They cluster: solver bug? trace style? distribution
   shift in the synthetic generator? Fix the *root cause* and regenerate that
   slice.
3. Retrain (or warm-start the adapter) and resubmit.

Set a daily submission budget so you keep moving up the public LB while still
having room for the final week.

### Phase 5 — Push for the last 2–3 points

Most teams will plateau here. To break above:

- **Test-time program execution disguised as reasoning**: Train traces that
  walk through the actual algorithm step-by-step (e.g., for cipher: build the
  table letter by letter, then apply). The model effectively becomes a "soft
  interpreter" of your solver — far more robust than abstract reasoning.
- **RLVR (RL with verifiable rewards)** as a finishing pass: rollout from the
  SFT'd model, score each rollout with your gold solver, do GRPO/PPO on the
  binary reward. Lifts long-tail correctness 1–3 points typically.
- **Rejection sampling SFT** (cheaper alternative to RL): sample N traces per
  prompt, keep only ones that produce the gold answer, retrain. Requires no
  RL machinery.
- **Diverse trace augmentation**: for each training example, generate 2–3
  *different* valid reasoning traces (e.g., for `unit_conv`, regress in two
  ways) — improves robustness on unseen prompt phrasings.
- **eq_symbols deep dive**: this category is likely where the field separates.
  Spend disproportionate effort here.

---

## 4. Risk register (real ones, not handwaving)

- **Hidden test categories**: the held-back test set might include one or two
  *new* puzzle types not in the public train. Detect this with a "category
  classifier" in your trace prefix — if confidence is low, fall back to a
  general reasoning trace (the base model's natural style). Mitigate by
  *not* hard-stopping unknown prompts during training.
- **Numeric-tolerance trap on non-numeric answers**: the metric prefers
  `\boxed{}` but falls back to "last numeric value". If your trace mentions
  numbers after the `\boxed{}`, you're fine; if it accidentally emits a number
  *instead of* a `\boxed{}` for a string-answer puzzle, you lose. **Always
  end with `\boxed{}` on the last line, period.**
- **LoRA-rank ceiling**: rank 32 is binding. Don't waste capacity on style.
  Loss-masking the prompt helps; compact, structured traces help more.
- **Distribution shift**: synthetic prompts must match real prompt phrasing
  *exactly* (same wording, same example counts, same number formats). Build
  your synthetic generators by sampling templates from train.csv and
  substituting only the rule.
- **eq_symbols undefined**: if the rule family is broader than your detectors,
  you'll silently produce nonsense. Always run synthetic coverage against
  *real* train prompts and confirm match-rate ≥ 99%.

---

## 5. Forty-day sprint to the final deadline (today 2026-05-06 → 2026-06-15)

The midpoint prize is gone; only the final leaderboard and the Open Contribution
Awards (top-10% LB + write-up in Data / RL / Fine-tuning) are still live. Plan
backwards from 2026-06-15.

**Week 1 (May 6–12) — Solvers + first submission.**
- Day 1–2: solvers for `unit_conv`, `gravity`, `numeral` (closed-form, easy
  ≥99% wins).
- Day 3–4: solver for `cipher` (letter-table alignment) and `bit_manip`
  (operator enumeration).
- Day 5: rough `eq_symbols` solver — at least the dominant rule families.
- Day 6: synthetic CoT generator skeletons for all six. Generate first 50k
  traces.
- Day 7: **first SFT submission** to the LB. Establishes a baseline you can
  iterate against.

**Week 2 (May 13–19) — Close the gap on `eq_symbols` + scale data.**
- Reverse-engineer remaining `eq_symbols` rule families until solver is ≥99%.
- Scale synthetic dataset to 200k examples; rebalance by error rate.
- Train LoRA from scratch (not warm-start) on the larger dataset.
- Submit each retrained checkpoint.

**Week 3 (May 20–26) — Rejection-sampling SFT and RLVR.**
- Sample N=8 rollouts per prompt from your best SFT model on a held-out
  synthetic slice. Keep only correct ones; retrain on the survivors. This is
  cheap and almost always lifts 1–2 pts.
- If LB still has headroom: GRPO-style RL with verifiable rewards using your
  gold solver as the verifier. Single epoch, small KL coefficient, freeze base
  + train only the LoRA adapter.

**Week 4 (May 27 – Jun 2) — Hardening and ablations.**
- Audit every holdout error. Patch generators or solvers; never paper over a
  failure with more compute.
- Run an ablation: drop each category's training data and measure the cliff —
  any category whose accuracy is robust to dropping is over-represented; any
  category that collapses is the one to grow.
- Decide if you're chasing top-3 or top-10% — if top-3 looks closed, focus on
  the cleanest possible methodology write-up for Open Contribution.

**Week 5 (Jun 3–8) — Lock-in candidates + team merger deadline 2026-06-08.**
- Freeze 2–3 candidate adapters. Submit each.
- If you're going to merge with anyone, **do it before 2026-06-08 EOD UTC.**
- Begin the public Kaggle notebook + write-up (required for any prize).

**Final week (Jun 9–15) — Polish, not panic.**
- Last LB pushes only if you have a clearly-better candidate from week 4.
- Finalize and publish the Kaggle notebook + write-up.
- Submit the best adapter as the official final entry well before 2026-06-15
  23:59 UTC. **Do not wait for the last hour** — Kaggle queue depth spikes at
  deadlines.

---

## 6. File layout suggestion

```
nemotron/
├── data/                       # train.csv, test.csv
├── solvers/                    # per-category solvers + CoT generators
│   ├── bit_manip.py
│   ├── cipher.py
│   ├── numeral.py
│   ├── unit_conv.py
│   ├── gravity.py
│   └── eq_symbols.py
├── synth/                      # synthetic dataset builder
│   └── generate.py
├── train/                      # SFT scripts (Hendrix + Kaggle variants)
│   ├── sft_qlora.py
│   └── slurm_sft.sh
├── eval/                       # offline eval against holdout
│   └── score.py
├── notebooks/                  # the public Kaggle notebook
└── STRATEGY.md
```

---

## 7. Stop and rethink if…

- A solver plateaus below 95% on its category — the rule family is wrong, not
  the regression. Don't pile on more candidates; reverse-engineer 20 prompts
  by hand.
- After SFT, holdout accuracy is dramatically below solver accuracy — the
  model isn't imitating the trace. Check loss masking, learning rate, and
  whether the prompt format in training matches eval exactly.
- The adapter overfits one category and regresses others — increase data
  balance, lower LR, or add dropout.

---

The shortest path is: **solvers → synthetic CoT → SFT → iterate**. Do not skip
the solvers. Everything compounds from there.
