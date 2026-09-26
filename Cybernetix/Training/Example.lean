import Cybernetix.Training.SGD

namespace Cybernetix.Training.Example

open Exact

/-- Four corners labeled by the affine function `2*x₀ - 3*x₁ + 1`. -/
def batchA : Batch Rat := ⟨⟨-1, -1, 2⟩, [⟨1, 1, 0⟩]⟩
def batchB : Batch Rat := ⟨⟨-1, 1, -4⟩, [⟨1, -1, 6⟩]⟩
def evaluationBatch : Batch Rat :=
  ⟨batchA.first, batchA.rest ++ batchB.samples⟩

def initial : State := ⟨⟨0, 0, 0⟩, 0⟩
def rate : Rat := 1 / 8
def epoch : List Update := [⟨batchA, rate⟩, ⟨batchB, rate⟩]
def schedule : List Update := epoch ++ epoch ++ epoch ++ epoch

/-- Expected result is literal data, separately checked against the trainer. -/
def expected : State := ⟨⟨15 / 8, -45 / 16, 58975 / 65536⟩, 8⟩

end Cybernetix.Training.Example
