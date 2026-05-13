from embedder import VictimEmbedder, AttackerEmbedder
from alingment import AlignmentModel
from generator import Generator
from beam_search import BeamSearch, Beam


class Zero2Text:
  def __init__(self,generator,victim,attacker,aligner,beam_search,max_len=32):
    self.generator = generator
    self.victim = victim
    self.attacker = attacker
    self.aligner = aligner
    self.beam_search = beam_search
    self.max_len = max_len

  def run(self, target_embedding, init_text=""):
    self.beam_search.reset_state()
    self.beam_search.target_embedding = target_embedding

    beams = [Beam(init_text, 0.0)]

    for step in range(self.max_len):
      beams = self.beam_search.step(beams, step, max_tokens=self.max_len)
      print(f"Step {step}: {[b.text for b in beams]}")

    return beams


victim = VictimEmbedder()
attacker = AttackerEmbedder()
aligner = AlignmentModel(lambda_reg=0.1)

generator = Generator()

target_text = "Paris is the capital of France"
target_embedding = victim.embed([target_text])[0]

beam_search = BeamSearch(
    generator=generator,
    attacker=attacker,
    victim=victim,
    aligner=aligner,
    target_embedding=target_embedding,
    beam_size=10,
    K_S=1000,
    K_A=50,
    gamma=0.8,
    T_hw=0.9,
)

z2t = Zero2Text(
    generator=generator,
    victim=victim,
    attacker=attacker,
    aligner=aligner,
    beam_search=beam_search,
    max_len=32,
)

final_beams = z2t.run(
    target_embedding=target_embedding,
    init_text="",
)

for beam in final_beams:
    print(beam.text, beam.score)
