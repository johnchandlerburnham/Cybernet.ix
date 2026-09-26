/-!
A two-feature affine regressor and nonempty minibatches. Definitions are shared
by the exact executable (`Rat`) and the mathematical interpretation (`Real`).
No field laws or floating-point behavior are assumed by this module.
-/

namespace Cybernetix.Training

structure Parameters (α : Type) where
  w₀ : α
  w₁ : α
  bias : α
  deriving DecidableEq, Repr

namespace Parameters

def map (f : α → β) (p : Parameters α) : Parameters β :=
  ⟨f p.w₀, f p.w₁, f p.bias⟩

def add [Add α] (p q : Parameters α) : Parameters α :=
  ⟨p.w₀ + q.w₀, p.w₁ + q.w₁, p.bias + q.bias⟩

def scale [Mul α] (t : α) (p : Parameters α) : Parameters α :=
  ⟨t * p.w₀, t * p.w₁, t * p.bias⟩

def dot [Add α] [Mul α] (p q : Parameters α) : α :=
  p.w₀ * q.w₀ + p.w₁ * q.w₁ + p.bias * q.bias

def shift [Add α] [Mul α] (p v : Parameters α) (t : α) : Parameters α :=
  p.add (v.scale t)

def subtractScaled [Sub α] [Mul α] (p : Parameters α) (rate : α)
    (g : Parameters α) : Parameters α :=
  ⟨p.w₀ - rate * g.w₀, p.w₁ - rate * g.w₁, p.bias - rate * g.bias⟩

end Parameters

structure Sample (α : Type) where
  x₀ : α
  x₁ : α
  target : α
  deriving DecidableEq, Repr

def Sample.map (f : α → β) (s : Sample α) : Sample β :=
  ⟨f s.x₀, f s.x₁, f s.target⟩

/-- A batch cannot be empty, so its mean has a positive integer divisor. -/
structure Batch (α : Type) where
  first : Sample α
  rest : List (Sample α) := []
  deriving DecidableEq, Repr

def Batch.samples (b : Batch α) : List (Sample α) := b.first :: b.rest

def Batch.size (b : Batch α) : Nat := b.rest.length + 1

def Batch.map (f : α → β) (b : Batch α) : Batch β :=
  ⟨b.first.map f, b.rest.map (Sample.map f)⟩

theorem Batch.size_pos (b : Batch α) : 0 < b.size := Nat.zero_lt_succ _

def predict [Add α] [Mul α] (p : Parameters α) (s : Sample α) : α :=
  p.w₀ * s.x₀ + p.w₁ * s.x₁ + p.bias

def residual [Add α] [Sub α] [Mul α] (p : Parameters α) (s : Sample α) : α :=
  predict p s - s.target

def sampleLoss [Add α] [Sub α] [Mul α] (p : Parameters α) (s : Sample α) : α :=
  let r := residual p s
  r * r

def sampleGradient [Add α] [Sub α] [Mul α] [OfNat α 2]
    (p : Parameters α) (s : Sample α) : Parameters α :=
  let r := 2 * residual p s
  ⟨r * s.x₀, r * s.x₁, r⟩

/-- Right-associated sums are part of this exact reference's evaluation order. -/
def lossSum [Add α] [Sub α] [Mul α] [OfNat α 0]
    (p : Parameters α) : List (Sample α) → α
  | [] => 0
  | s :: ss => sampleLoss p s + lossSum p ss

def gradientSum [Add α] [Sub α] [Mul α] [OfNat α 0] [OfNat α 2]
    (p : Parameters α) : List (Sample α) → Parameters α
  | [] => ⟨0, 0, 0⟩
  | s :: ss => (sampleGradient p s).add (gradientSum p ss)

def meanLoss [Add α] [Sub α] [Mul α] [Div α] [OfNat α 0] [NatCast α]
    (p : Parameters α) (b : Batch α) : α :=
  lossSum p b.samples / (b.size : α)

def meanGradient [Add α] [Sub α] [Mul α] [Div α] [OfNat α 0] [OfNat α 2]
    [NatCast α] (p : Parameters α) (b : Batch α) : Parameters α :=
  let g := gradientSum p b.samples
  let n : α := b.size
  ⟨g.w₀ / n, g.w₁ / n, g.bias / n⟩

def sgdStep [Add α] [Sub α] [Mul α] [Div α] [OfNat α 0] [OfNat α 2]
    [NatCast α] (p : Parameters α) (b : Batch α) (rate : α) : Parameters α :=
  p.subtractScaled rate (meanGradient p b)

end Cybernetix.Training
