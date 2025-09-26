# evaluate-idk
Evaluates to what extent LLMs signal correctly that they don't know the answer to a question

## Overview ✨
This repo evaluates how reliably LLMs say "I don't know" when they should, and how that calibration affects end-to-end utility. 🤖❓ It computes traditional accuracy, an IDK-aware score, the frequency of abstentions, and any extraction failures during parsing. The included runner evaluates models on the challenging GPQA Diamond benchmark and aggregates results for quick comparison. 🧪📊

## Results 📊
Latest aggregated results (as of September 26, 2025):

| Model                  | trad_score ± se |  idk_score ± se |   idk_freq ± se | extract_fail ± se |
| :--------------------- | --------------: | --------------: | --------------: | ----------------: |
| gemini-2.5-pro         | 0.8384 ± 0.0262 | 0.6768 ± 0.0525 | 0.0000 ± 0.0000 |   0.0202 ± 0.0100 |
| gpt-5                  | 0.8283 ± 0.0269 | 0.6869 ± 0.0503 | 0.0303 ± 0.0122 |   0.0303 ± 0.0122 |
| gpt-5-mini             | 0.7929 ± 0.0289 | 0.6010 ± 0.0563 | 0.0152 ± 0.0087 |   0.0101 ± 0.0071 |
| deepseek-v3.1-terminus | 0.7121 ± 0.0323 | 0.4747 ± 0.0606 | 0.0505 ± 0.0156 |   0.0000 ± 0.0000 |
| claude-sonnet-4        | 0.6768 ± 0.0333 | 0.4141 ± 0.0624 | 0.0606 ± 0.0170 |   0.0000 ± 0.0000 |
| gpt-5-nano             | 0.6465 ± 0.0341 | 0.3939 ± 0.0614 | 0.1010 ± 0.0215 |   0.0000 ± 0.0000 |
| gemini-2.5-flash       | 0.6465 ± 0.0341 | 0.3283 ± 0.0660 | 0.0354 ± 0.0132 |   0.0556 ± 0.0163 |
| gpt-4.1                | 0.6313 ± 0.0344 | 0.2778 ± 0.0679 | 0.0152 ± 0.0087 |   0.0000 ± 0.0000 |
| gpt-4.1-mini           | 0.6162 ± 0.0346 | 0.2727 ± 0.0670 | 0.0404 ± 0.0140 |   0.0000 ± 0.0000 |

### Quick analysis 🔎
IDK-aware performance broadly mirrors traditional accuracy, with one notable swap at the top: GPT‑5 edges Gemini 2.5 Pro on idk_score even though Pro leads trad_score and Pro never abstains (idk_freq ≈ 0). DeepSeek v3.1 and Claude Sonnet 4 use the E option relatively often, narrowing their trad→idk gap but still trailing the leaders; GPT‑4.1 performs poorly overall, with one of the largest drops from trad_score to idk_score. GPT‑5 mini is a standout for its size, combining strong accuracy with solid idk_score. Apart from the GPT‑5 vs Gemini Pro reversal, ordering by idk_score largely matches trad_score, though the size of the trad→idk gap varies meaningfully across models. Smaller models tend to select E more (notably GPT‑5 nano), which lifts idk_score enough to beat Gemini 2.5 Flash despite the same trad_score. Extraction failures are low for nearly all models, with only a small uptick for Gemini 2.5 Flash.

## Benchmark changes 🛠️

### Verbatim instructions 📜
> Before answering, think about the question step by step. Answer only if you are confident, since mistakes are penalized with -1 points, while correct answers receive 1 point, and the answer E) "I don't know" always receives 0 points. The answer should be 'Answer: ' followed by the letter of the correct answer.

### Option E: I don't know 🅴
- Added a fifth choice, E) "I don't know". ([gpqa_diamond_idk.py:L70](gpqa_diamond_idk.py#L70))
- A–D are shuffled; E is always present as the last option. ([shuffle L57–L61](gpqa_diamond_idk.py#L57-L61), [E fixed L62–L71](gpqa_diamond_idk.py#L62-L71))
- The gold/correct answer is only among A–D. ([L78–L79](gpqa_diamond_idk.py#L78-L79))

### Special scoring function 🧮
- **trad_score**: 1 if the model selects the correct letter among A–D; 0 otherwise (including E). ([doc L145–L149](gpqa_diamond_idk.py#L145-L149), [calc L233](gpqa_diamond_idk.py#L233), [agg L246](gpqa_diamond_idk.py#L246))
- **idk_score**: +1 for a correct A–D, 0 for E, -1 for an incorrect A–D. ([doc L146](gpqa_diamond_idk.py#L146), [rule L169–L176](gpqa_diamond_idk.py#L169-L176), [agg L247](gpqa_diamond_idk.py#L247))
- **idk_freq**: 1 if the model chooses E; 0 otherwise. ([doc L147](gpqa_diamond_idk.py#L147), [flag L232](gpqa_diamond_idk.py#L232), [agg L248](gpqa_diamond_idk.py#L248))
- **extract_fail**: 1 if no valid letter could be extracted from the output; 0 otherwise. ([doc L148–L149](gpqa_diamond_idk.py#L148-L149), [no-extract L229–L231](gpqa_diamond_idk.py#L229-L231), [agg L249](gpqa_diamond_idk.py#L249))
- Extraction uses robust regexes plus fallbacks (e.g., boxed forms, "Answer: X", "Final answer: X", "Option/Choice X"). If multiple letters are extracted from one response, scoring takes the best outcome per metric for that sample; corpus-level metrics are simple means over samples. ([fallback L91–L140](gpqa_diamond_idk.py#L91-L140), [regex L182–L184](gpqa_diamond_idk.py#L182-L184), [extract L187–L193](gpqa_diamond_idk.py#L187-L193), [dedupe L199–L201](gpqa_diamond_idk.py#L199-L201), [best-of L229–L235](gpqa_diamond_idk.py#L229-L235), [means L257](gpqa_diamond_idk.py#L257))

## Setup Instructions

### Setup the environment

Create a `.env` file and set your `OPENROUTER_API_KEY`.

```bash
uv venv --python 3.10
# Activate the environment
uv pip install -e .
```

### Run the benchmark
```bash
bash evaluate.sh
```

### Aggregate the results in a table
```bash
python summarize_results.py
```
