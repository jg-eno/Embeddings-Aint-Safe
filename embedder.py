from sentence_transformers import SentenceTransformer
import torch

class BaseEmbedder:
  def __init__(self,model_name):
    self.model = SentenceTransformer(model_name)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {self.device}")
    print(f"Model : {model_name}")
    self.model.to(self.device)

  def embed(self, sentences, batch_size=32, show_progress_bar=False):
    return self.model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )

  def cos_sim(self, embd1, embd2):
    return embd1 @ embd2.T

class VictimEmbedder(BaseEmbedder):
  def __init__(self):
    print("Victim Embedder Loading...")
    super().__init__("Qwen/Qwen3-Embedding-0.6B")

class AttackerEmbedder(BaseEmbedder):
  def __init__(self):
    print("Attacker Embedder Loading...")
    super().__init__("sentence-transformers/all-mpnet-base-v2")