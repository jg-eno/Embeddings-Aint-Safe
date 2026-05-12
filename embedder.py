from sentence_transformers import SentenceTransformer

class BaseEmbedder:
  def __init__(self,model_name):
    self.model = SentenceTransformer(model_name)

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
    super().__init__("Qwen/Qwen3-Embedding-0.6B")

class AttackerEmbedder(BaseEmbedder):
  def __init__(self):
    super().__init__("sentence-transformers/all-mpnet-base-v2")