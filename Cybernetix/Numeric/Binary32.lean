import Init.Data.Float.Model.Float32
import Cybernetix.Training.Example

/-! A small binary32 experiment using Lean's pure logical model, never native floats.
This pins primitive behavior to the Lean toolchain. It is not the complete P32
profile: the toy's sums are right-associated and special values propagate.
-/

namespace Cybernetix.Numeric.Binary32

open Cybernetix.Training

local instance : OfNat Float32.Model n := ⟨Float32.Model.ofNat n⟩
local instance : NatCast Float32.Model := ⟨Float32.Model.ofNat⟩

def profile : String := "cybernetix.binary32-sgd-toy.v1.lean-4.33.1"

/-- Explicit conversion program: round numerator and denominator, then divide.
Only the small dyadic inputs in the toy experiment have exact-conversion claims. -/
def fromRat (q : Rat) : Float32.Model :=
  Float32.Model.ofInt q.num / Float32.Model.ofNat q.den

/-- Exact value of finite bits. Signed zeros both denote rational zero. -/
def toRat? (f : Float32.Model) : Option Rat :=
  match f.unpack with
  | .notANumber | .infinity _ => none
  | .zero _ => some 0
  | .finite sign mantissa exponent _ =>
    let magnitude : Rat := (mantissa : Rat) * (2 : Rat) ^ exponent
    some (match sign with | .positive => magnitude | .negative => -magnitude)

def step (p : Parameters Float32.Model) (u : Exact.Update) : Parameters Float32.Model :=
  sgdStep p (u.batch.map fromRat) (fromRat u.rate)

def run (initial : Parameters Rat) (us : List Exact.Update) : Parameters Float32.Model :=
  us.foldl step (initial.map fromRat)

def words (p : Parameters Float32.Model) : Parameters UInt32 := p.map (·.toBits)

/-- Three little-endian binary32 words, ordered `w₀`, `w₁`, `bias`. -/
def encodeParameters (p : Parameters Float32.Model) : ByteArray := Id.run do
  let mut out := ByteArray.empty
  for w in #[p.w₀.toBits, p.w₁.toBits, p.bias.toBits] do
    for shift in #[0, 8, 16, 24] do
      out := out.push ((w >>> shift).toUInt8)
  return out

inductive Op where
  | add | sub | mul | div
  deriving Repr

def Op.eval : Op → Float32.Model → Float32.Model → Float32.Model
  | .add => (· + ·)
  | .sub => (· - ·)
  | .mul => (· * ·)
  | .div => (· / ·)

structure Case where
  name : String
  op : Op
  left : UInt32
  right : UInt32
  expected : UInt32

/-- Adversarial primitive cases for comparing implementations against the model. -/
def cases : List Case := [
  ⟨"halfway to even, low", .add, 0x3f800000, 0x33800000, 0x3f800000⟩,
  ⟨"halfway to even, high", .add, 0x3f800001, 0x33800000, 0x3f800002⟩,
  ⟨"subnormal addition", .add, 0x00000001, 0x00000001, 0x00000002⟩,
  ⟨"subnormal to normal", .add, 0x007fffff, 0x00000001, 0x00800000⟩,
  ⟨"underflow to positive zero", .mul, 0x00000001, 0x3f000000, 0x00000000⟩,
  ⟨"underflow to negative zero", .mul, 0x80000001, 0x3f000000, 0x80000000⟩,
  ⟨"overflow", .mul, 0x7f7fffff, 0x40000000, 0x7f800000⟩,
  ⟨"division rounding", .div, 0x3f800000, 0x40400000, 0x3eaaaaab⟩,
  ⟨"cancellation", .sub, 0x3f800001, 0x3f800000, 0x34000000⟩]

def Case.check (c : Case) : Bool :=
  (c.op.eval (.ofBits c.left) (.ofBits c.right)).toBits == c.expected

end Cybernetix.Numeric.Binary32
