import torch
import torch.nn.functional as F
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


class Generator:
  def __init__(self, model="Qwen/Qwen3-0.6B"):
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.model = AutoModelForCausalLM.from_pretrained(model)
    self.model.eval()

  def _lm_inputs(self, content):
    device = next(self.model.parameters()).device
    enc = self.tokenizer(content, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    if input_ids.shape[1] == 0:
      bos = getattr(self.model.config, "bos_token_id", None)
      if bos is None:
        bos = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
      if bos is None:
        raise ValueError("Empty prompt but no BOS/PAD/EOS token id.")
      input_ids = torch.tensor([[bos]], dtype=torch.long, device=device)

    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

  def next_token_logprob(self, content):
    inputs = self._lm_inputs(content)
    with torch.no_grad():
      outputs = self.model(**inputs)
    logits = outputs.logits
    next_token_logits = logits[:, -1, :]
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    return log_probs.squeeze(0)

  def apply_logit_penalty(self, log_probs, step, penalty=-5.0):
    if step != 0:
      return log_probs

    penalized = log_probs.clone()
    for token_id in range(len(log_probs)):
      token = self.tokenizer.decode([token_id])
      if not re.match(r"^[a-zA-Z]+$", token.strip()):
        penalized[token_id] += penalty
    return penalized

  def mask_non_ascii_tokens(self, log_probs):
    out = log_probs.clone()
    for token_id in range(len(log_probs)):
      token = self.tokenizer.decode([token_id])
      if not token.isascii():
        out[token_id] = -float("inf")
    return out
