import torch
import numpy as np


class Beam:
    def __init__(self, text, score):
        self.text = text
        self.score = score


def _zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    s = float(np.std(x))
    if s < eps:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def _rowwise_cosine(a, b, eps=1e-8):
    """Cosine similarity between rows of a and rows of b (same shape)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / (na * nb + eps)


def _cosine_to_target(rows, target, eps=1e-8):
    """Cosine similarity of each row to unit-norm target vector e_v."""
    rows = np.asarray(rows, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    nt = np.linalg.norm(t) + eps
    t = t / nt
    nr = np.linalg.norm(rows, axis=1) + eps
    return (rows @ t) / nr


class BeamSearch:
    """
    Zero2Text (Kim et al., 2026): Sec. 3.3–3.4 and Fig. 2.
    Diversity-aware K_S expansion, victim query budget, ridge W (Eq. 3),
    hybrid score S (Eq. 4) and confidence conf_t (Eq. 5).
    """

    def __init__(
        self,
        generator,
        attacker,
        victim,
        aligner,
        target_embedding,
        beam_size=10,
        K_S=5,
        K_A=50,
        gamma=0.8,
        T_hw=0.9,
    ):
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
        self._conf_prev = None

    def reset_state(self):
        self._conf_prev = None

    def _beam_finished(self, beam, max_tokens=32):
        tok = self.generator.tokenizer
        eos_id = tok.eos_token_id
        ids = tok.encode(beam.text, add_special_tokens=False)
        if eos_id is not None and ids and ids[-1] == eos_id:
            return True
        return len(ids) >= max_tokens

    def expand_beam_diverse(self, beam, step):
        """
        Sec. 3.3: pick up to K_S continuations so pairwise cosine similarity
        of their *local* sentence embeddings stays below T_hw.
        """
        log_probs = self.generator.next_token_logprob(beam.text)
        log_probs = self.generator.apply_logit_penalty(log_probs, step)
        log_probs = self.generator.mask_non_ascii_tokens(log_probs)

        order = torch.argsort(log_probs, descending=True)
        selected_embs = []
        new_beams = []

        for token_id in order:
            tid = int(token_id.item())
            lp = float(log_probs[tid].item())
            if not np.isfinite(lp) or lp == -float("inf"):
                continue
            token = self.generator.tokenizer.decode([tid])
            text = beam.text + token
            emb = self.attacker.embed([text])[0].astype(np.float64)
            en = np.linalg.norm(emb) + 1e-8
            emb = emb / en

            if not selected_embs:
                ok = True
            else:
                cosims = [float(np.dot(emb, e)) for e in selected_embs]
                ok = max(cosims) < self.T_hw

            if ok:
                log_p = lp
                new_beams.append(
                    Beam(text=text, score=beam.score + log_p),
                )
                selected_embs.append(emb)
                if len(new_beams) >= self.K_S:
                    break

        if not new_beams:
            for token_id in order:
                tid = int(token_id.item())
                lp = float(log_probs[tid].item())
                if np.isfinite(lp) and lp > -float("inf"):
                    token = self.generator.tokenizer.decode([tid])
                    new_beams.append(
                        Beam(text=beam.text + token, score=beam.score + lp)
                    )
                    break

        return new_beams

    def _query_budget(self, step, num_candidates):
        """Sec. 3.4: first iteration (t=1) uses 3 * K_A; later K_A * gamma^{t-1} with step = t-1."""
        if step == 0:
            q = 3 * self.K_A
        else:
            q = int(round(self.K_A * (self.gamma**step)))
        return max(1, min(num_candidates, q))

    def _confidence_after_query(self, step, W_prev, W_new, local_subset, victim_subset):
        """Eq. (5): conf_1 uses W^1 and factor 0.7; conf_t for t>=2 uses W^{t-1}."""
        if local_subset.shape[0] == 0:
            return 0.0
        if step == 0:
            proj = local_subset @ W_new
            c = float(np.mean(_rowwise_cosine(proj, victim_subset)))
            return 0.7 * c
        proj = local_subset @ W_prev
        return float(np.mean(_rowwise_cosine(proj, victim_subset)))

    def _S_batch(self, y, emb_for_cos, conf_weight):
        """Eq. (4): S = Z(y) + conf * Z(cos(emb, e_v))."""
        z_y = _zscore(y)
        cos_e = _cosine_to_target(emb_for_cos, self.target_embedding)
        z_c = _zscore(cos_e)
        return z_y + conf_weight * z_c

    def step(self, beams, step, max_tokens=32):
        candidates = []
        for beam in beams:
            if self._beam_finished(beam, max_tokens=max_tokens):
                candidates.append(beam)
            else:
                candidates.extend(self.expand_beam_diverse(beam, step))

        if not candidates:
            return beams

        texts = [b.text for b in candidates]
        N = len(texts)
        local_embs = self.attacker.embed(texts)

        W_prev = self.aligner.W
        y = np.array([b.score for b in candidates], dtype=np.float64)

        # Cold start (no W^0): Sec. 3.4 bypasses grouping ❸; ranking on Z(y) alone
        # ignores e_v and picks generic high-LM prefixes. Use victim cos(e_v, φ(s)) on
        # all candidates (one batch) to choose which pairs seed W^1, plus tiny LM tie-break.
        victim_all = None
        if W_prev is None:
            conf_sel = 0.0
            victim_all = self.victim.embed(texts)
            cos_v = _cosine_to_target(victim_all, self.target_embedding)
            lm_w = 0.05
            S_select = _zscore(cos_v) + lm_w * _zscore(y)
        else:
            conf_sel = self._conf_prev if self._conf_prev is not None else 0.0
            hat_e = local_embs @ W_prev
            S_select = self._S_batch(y, hat_e, conf_sel)

        query_count = self._query_budget(step, N)
        query_order = np.argsort(-S_select)[:query_count]
        query_list = query_order.tolist()
        query_set = set(query_list)

        if victim_all is not None:
            idx = np.asarray(query_list, dtype=int)
            victim_subset = victim_all[idx]
            local_subset = local_embs[idx]
        else:
            query_texts = [texts[i] for i in query_list]
            victim_subset = self.victim.embed(query_texts)
            local_subset = local_embs[query_list]

        self.aligner.update(local_subset, victim_subset)
        W_new = self.aligner.W

        conf_t = self._confidence_after_query(step, W_prev, W_new, local_subset, victim_subset)

        d2 = W_new.shape[1]
        unified = np.empty((N, d2), dtype=np.float64)
        for j, i in enumerate(query_list):
            unified[i] = victim_subset[j]
        for i in range(N):
            if i not in query_set:
                unified[i] = local_embs[i] @ W_new

        S_final = self._S_batch(y, unified, conf_t)

        prune_idx = np.argsort(-S_final)[: self.beam_size]
        pruned = []
        for i in prune_idx:
            b = candidates[i]
            b.score = float(S_final[i])
            pruned.append(b)

        self._conf_prev = conf_t
        return pruned
