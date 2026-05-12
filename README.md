# Zero2Text (this repo)

This is my attempt at implementing **Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings** (Kim et al., 2026) based on `Zero2Text.pdf` in this folder.

The goal is simple: given a **victim embedding** \(e_v\), try to recover the original text with:

- a generator LLM (Qwen3-0.6B),
- a local attacker embedder (all-mpnet-base-v2),
- online queries to the victim embedder,
- an online ridge-regression alignment matrix \(W^t\),
- and the paper’s scoring \(S(\cdot)\).

This is still a prototype and the reconstructions are not good yet (see “Current results”).

## How the method works (paper mapping)

The code follows Fig. 2 / Sec. 3.3–3.4:

- **Generate candidates (❶)**: for each beam prefix, the LLM proposes next tokens. We keep **\(K_S\)** diverse continuations (diversity measured by cosine similarity between *local* embeddings; threshold \(T_{hw}\)).
- **Local embeddings (❷)**: embed every candidate sentence with the attacker embedder.
- **Select who to query (❸/❹)**:
  - First iteration: query **\(3 \times K_A\)** sentences (paper special case).
  - Later iterations: query **\(K_A \cdot \gamma^{t-1}\)**.
- **Update alignment (❺)**: update \(W^t\) with ridge regression (Eq. (3)) using all queried pairs accumulated so far.
- **Re-project non-queried (❻)**: for candidates not queried this iteration, approximate victim embedding with \(eW^t\).
- **Score + sort (❼)**: paper Eq. (4)

\[
S(e_i, t) = Z(y_i) + conf_t \cdot Z(\cos(e_i, e_v))
\]

where `y_i` is the LLM score we carry along the beam, `Z` is z-score over the candidate pool, and `conf_t` is Eq. (5) (with the paper’s special-case scaling for \(t=1\)).

- **Beam select (❽)**: keep the top \(K_B\) candidates.

## Files

- `main.py`: runnable demo script.
- `beam_search.py`: the Zero2Text loop (Fig. 2) + Eq. (4)/(5) scoring.
- `alingment.py`: online ridge regression for \(W^t\) (Eq. (3)).
- `generator.py`: LLM next-token distribution + ASCII restriction + first-iteration logit penalty; also handles empty `init_text`.
- `embedder.py`: attacker + victim embedders.

## Hyperparameters (paper §4.1 defaults)

Current defaults in `main.py`:

- `beam_size` (\(K_B\)) = 10
- `K_S` = 5
- `K_A` = 50
- `gamma` = 0.8
- `T_hw` = 0.9
- `max_len` (\(T\)) = 32
- `lambda_reg` (\(\lambda\)) = 0.1
- first-iteration logit penalty = -5 on non-alphabetic tokens

## Run

```bash
./venv/bin/python main.py
```

## Current results (snapshot)

Right now it runs end-to-end but the beams drift into “prompty / Q&A / template” text instead of landing near the actual `target_text`.

Example at step 25:

```text
Step 25: [
  'GivenQuestion = \"What is the correct name of the person who discovered the first successful experiment in magnetic fields?\"\\nA. Einstein\\n',
  'GivenQuestion = \"What is the correct name of the person who discovered the first successful experiment in magnetic fields?\"\\nA. Nelson Mandela',
  'GivenQuestion = \"What is the correct name of the person who discovered the theory of relativity?\";\\n\\nAnswer = \"Albert Einstein',
  'GivenQuestion = \"What is the correct name of the person who discovered the theory of evolution?\"\\n\\nThe answer is \"Von Ne',
  'GivenQuestion = \"What is the correct name of the person who discovered the theory of evolution?\"\\n\\nThe answer is \"Darwin\"\\n\\n',
  'GivenQuestion = \"What is the correct name of the person who discovered the theory of relativity?\";\\n\\nconst { search } =',
  'GivenQuestion = \"What is the correct name of the person who discovered the first successful experiment in magnetic fields?\"\\nOptions:\\n- A',
  'GivenQuestion = \"What is the correct name of the person who discovered the first successful experiment in magnetic fields?\"\\nA. Nelson and',
  'GivenQuestion = \"What is the correct name of the person who discovered the first successful experiment in magnetic fields?\"\\nContext: \"In',
  'GivenQuestion = \"What is the correct name of the person who discovered the first successful experiment in magnetic fields?\"\\nContext: \"Ant'
]
```

So the next work item is: figure out why the scoring/alignment isn’t pulling the generations toward the target sentence and why the LLM prior dominates.

