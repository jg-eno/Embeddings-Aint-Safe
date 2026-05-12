import torch
import numpy as np


class Beam:
  def __init__(self, text, score):
    self.text = text
    self.score = score


class BeamSearch:
  """
  Zero2Text (Kim et al., 2026): Sec. 3.3–3.4 and Fig. 2.
  """

  def __init__(self,generator,attacker,victim,aligner,
                target_embedding,beam_size=10,K_S=5,K_A=50,gamma=0.8,T_hw=0.9):
    
    self.generator = generator
    self.attacker = attacker
    self.victim = victim
    self.aligner = aligner
    self.target_embedding = np.asarray(target_embedding, dtype=np.float64).reshape(-1)
    self.beam_size = beam_size
    self.K_S = K_S
    self.K_A = K_A
    self.gamma = gamma
    self.T_hw = T_hw
    self.conf_prev = None

  def reset_state(self):
    self.conf_prev = None

  def zscore(self, x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    s = float(np.std(x))
    if s < eps:
      return np.zeros_like(x)
    return (x - np.mean(x)) / s

  def cos_to_target(self, rows, eps=1e-8):
    rows = np.asarray(rows, dtype=np.float64)
    t = np.asarray(self.target_embedding, dtype=np.float64).reshape(-1)
    t = t / (np.linalg.norm(t) + eps)
    nr = np.linalg.norm(rows, axis=1) + eps
    return (rows @ t) / nr

  def row_cos(self, a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / (na * nb + eps)

  def is_complete(self, beam, max_tokens=32):
    tok = self.generator.tokenizer
    eos_id = tok.eos_token_id
    ids = tok.encode(beam.text, add_special_tokens=False)
    if eos_id is not None and ids and ids[-1] == eos_id:
      return True
    return len(ids) >= max_tokens

  def query_budget(self, step, n):
    if step == 0:
      q = 3 * self.K_A
    else:
      q = int(round(self.K_A * (self.gamma**step)))
    return max(1, min(n, q))

  def score(self, y, embs, conf):
    z_y = self.zscore(y)
    z_cos = self.zscore(self.cos_to_target(embs))
    return z_y + conf * z_cos

  def confidence(self, step, W_prev, W_new, local_sub, victim_sub):
    if local_sub.shape[0] == 0:
      return 0.0
    if step == 0:
      proj = local_sub @ W_new
      return 0.7 * float(np.mean(self.row_cos(proj, victim_sub)))
    proj = local_sub @ W_prev
    return float(np.mean(self.row_cos(proj, victim_sub)))

  def expand(self, beam, step):
    logprobs = self.generator.next_token_logprob(beam.text)
    logprobs = self.generator.apply_logit_penalty(logprobs, step)
    logprobs = self.generator.mask_non_ascii_tokens(logprobs)

    order = torch.argsort(logprobs, descending=True)
    picked = []
    out = []

    for token_id in order:
      tid = int(token_id.item())
      lp = float(logprobs[tid].item())
      if not np.isfinite(lp) or lp == -float("inf"):
        continue

      token = self.generator.tokenizer.decode([tid])
      text = beam.text + token

      emb = self.attacker.embed([text])[0].astype(np.float64)
      emb = emb / (np.linalg.norm(emb) + 1e-8)

      if not picked:
        ok = True
      else:
        cosims = [float(np.dot(emb, e)) for e in picked]
        ok = max(cosims) < self.T_hw

      if ok:
        out.append(Beam(text=text, score=beam.score + lp))
        picked.append(emb)
        if len(out) >= self.K_S:
          break

    if not out:
      tid = int(order[0].item())
      token = self.generator.tokenizer.decode([tid])
      lp = float(logprobs[tid].item())
      out.append(Beam(text=beam.text + token, score=beam.score + lp))

    return out

  def step(self, beams, step, max_tokens=32):
    cands = []
    for beam in beams:
      if self.is_complete(beam, max_tokens=max_tokens):
        cands.append(beam)
      else:
        cands.extend(self.expand(beam, step))

    if not cands:
      return beams

    texts = [b.text for b in cands]
    n = len(texts)

    local_embs = self.attacker.embed(texts)
    y = np.array([b.score for b in cands], dtype=np.float64)
    W_prev = self.aligner.W

    victim_all = None
    if W_prev is None:
      victim_all = self.victim.embed(texts)
      cos_v = self.cos_to_target(victim_all)
      S_sel = self.zscore(cos_v) + 0.05 * self.zscore(y)
    else:
      conf = self.conf_prev if self.conf_prev is not None else 0.0
      hat = local_embs @ W_prev
      S_sel = self.score(y, hat, conf)

    q = self.query_budget(step, n)
    q_idx = np.argsort(-S_sel)[:q].tolist()
    q_set = set(q_idx)

    if victim_all is not None:
      idx = np.asarray(q_idx, dtype=int)
      victim_sub = victim_all[idx]
      local_sub = local_embs[idx]
    else:
      q_texts = [texts[i] for i in q_idx]
      victim_sub = self.victim.embed(q_texts)
      local_sub = local_embs[q_idx]

    self.aligner.update(local_sub, victim_sub)
    W_new = self.aligner.W

    conf_t = self.confidence(step, W_prev, W_new, local_sub, victim_sub)

    d2 = W_new.shape[1]
    unified = np.empty((n, d2), dtype=np.float64)
    for j, i in enumerate(q_idx):
      unified[i] = victim_sub[j]
    for i in range(n):
      if i not in q_set:
        unified[i] = local_embs[i] @ W_new

    S_final = self.score(y, unified, conf_t)

    keep = np.argsort(-S_final)[:self.beam_size]
    out = []
    for i in keep:
      b = cands[i]
      b.score = float(S_final[i])
      out.append(b)

    self.conf_prev = conf_t
    return out
