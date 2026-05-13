# Zero2Text

This is my attempt at implementing **Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings** (Kim et al., 2026) based on `Zero2Text.pdf` in this folder.

The goal is simple: given a **victim embedding** $e_v$, try to recover the original text with:

- a generator LLM (Qwen3-0.6B)
- a local attacker embedder (all-mpnet-base-v2)
- online queries to the victim embedder
- an online ridge-regression alignment matrix $W^t$
- the paper's scoring $S(\cdot)$

> This is still a prototype and the reconstructions are not good yet (see [Current Results](#current-results)).

---

## How It Works

The code follows Fig. 2 / Sec. 3.3–3.4 of the paper:

| Step | Description |
|------|-------------|
| **❶ Generate candidates** | For each beam prefix, the LLM proposes next tokens. We keep $K_S$ diverse continuations (diversity measured by cosine similarity between *local* embeddings; threshold $T_{hw}$). |
| **❷ Local embeddings** | Embed every candidate sentence with the attacker embedder. |
| **❸/❹ Select who to query** | First iteration: query $3 \times K_A$ sentences (paper special case). Later iterations: query $K_A \cdot \gamma^{t-1}$. |
| **❺ Update alignment** | Update $W^t$ with ridge regression (Eq. 3) using all queried pairs accumulated so far. |
| **❻ Re-project non-queried** | For candidates not queried this iteration, approximate victim embedding with $eW^t$. |
| **❼ Score + sort** | Paper Eq. 4 (see below). |
| **❽ Beam select** | Keep the top $K_B$ candidates. |

### Scoring (Eq. 4)

$$S(e_i, t) = Z(y_i) + \text{conf}_t \cdot Z(\cos(e_i, e_v))$$

where $y_i$ is the LLM score carried along the beam, $Z$ is z-score over the candidate pool, and $\text{conf}_t$ is Eq. 5 (with the paper's special-case scaling for $t=1$).

---

## Hyperparameters

Defaults from paper §4.1, set in `main.py`:

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Beam size | $K_B$ | 10 |
| Survivor candidates | $K_S$ | 5 |
| Query budget | $K_A$ | 50 |
| Query decay | $\gamma$ | 0.8 |
| Diversity threshold | $T_{hw}$ | 0.9 |
| Max length | $T$ | 32 |
| Ridge regularization | $\lambda$ | 0.1 |
| First-iter logit penalty | — | −5 on non-alphabetic tokens |

---

## Usage

```bash
./venv/bin/python main.py
```

---

## 12-05-2026 Results

The pipeline runs end-to-end, but beams drift into "prompty / Q&A / template" text rather than converging on the actual `target_text`.

**Target text:**

```
Isaac Newton discovered the law of gravity.
```

**Example output at step 25:**

```
Step 25: [
  'GivenQuestion = "What is the correct name of the person who discovered the first successful experiment in magnetic fields?"\nA. Einstein\n',
  'GivenQuestion = "What is the correct name of the person who discovered the first successful experiment in magnetic fields?"\nA. Nelson Mandela',
  'GivenQuestion = "What is the correct name of the person who discovered the theory of relativity?";\n\nAnswer = "Albert Einstein',
  'GivenQuestion = "What is the correct name of the person who discovered the theory of evolution?"\n\nThe answer is "Von Ne',
  'GivenQuestion = "What is the correct name of the person who discovered the theory of evolution?"\n\nThe answer is "Darwin"\n\n',
  'GivenQuestion = "What is the correct name of the person who discovered the theory of relativity?";\n\nconst { search } =',
  'GivenQuestion = "What is the correct name of the person who discovered the first successful experiment in magnetic fields?"\nOptions:\n- A',
  'GivenQuestion = "What is the correct name of the person who discovered the first successful experiment in magnetic fields?"\nA. Nelson and',
  'GivenQuestion = "What is the correct name of the person who discovered the first successful experiment in magnetic fields?"\nContext: "In',
  'GivenQuestion = "What is the correct name of the person who discovered the first successful experiment in magnetic fields?"\nContext: "Ant'
]
```

The beam is converging near the target's embedding region, but the LLM's prior post-training dominates and steers every beam toward Q&A / template-style completions.


## 13-05-2026 Results

Target Text : The cat sat on the wall

13th Generations : (beam_size=10, K_S=1000, K_A=50, gamma=0.8, T_hw=0.9, max_len=32, different attacter and victim embedding models)

Step 13: ['QuestionDescription: Wally the white cat is sitting on a platform that', 'QuestionDescription: Wally the white cat is sitting on a floor.', 'QuestionDescription: Wally the white cat is sitting on a mat.', 'QuestionDescription: Wally the white cat is sitting quietly in a room', 'QuestionDescription: Wally the white cat is sitting on a bed.', 'QuestionDescription: Wally the white cat is sitting on a rectangular table', 'QuestionDescription: Wally the white cat is sitting on a rectangular floor', 'QuestionDescription: Wally the white cat is sitting on a window sill', 'QuestionDescription: Wally the white cat is sitting randomly on a chair', 'QuestionDescription: Wally the white cat is sitting on stairs. The']