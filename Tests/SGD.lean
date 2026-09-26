import Cybernetix.Numeric.Binary32
import Init.Data.Float.Float32

namespace Cybernetix.Tests.SGD

open Cybernetix.Training
open Cybernetix.Numeric

local instance : OfNat Float32 n := ⟨Float32.ofModel (Float32.Model.ofNat n)⟩
local instance : NatCast Float32 := ⟨fun n => Float32.ofModel (Float32.Model.ofNat n)⟩

private def check (ok : Bool) (message : String) : IO Unit :=
  unless ok do throw <| IO.userError message

private def nativeInput (q : Rat) : Float32 :=
  Float32.ofBits (Binary32.fromRat q).toBits

private def nativeOp : Binary32.Op → Float32 → Float32 → Float32
  | .add => (· + ·)
  | .sub => (· - ·)
  | .mul => (· * ·)
  | .div => (· / ·)

def run : IO Unit := do
  let actual := Exact.run Example.initial Example.schedule
  check (Exact.checkRun Example.initial Example.schedule Example.expected) "exact final state"
  check (decide (Exact.run Example.initial [] = Example.initial)) "empty schedule"
  for split in [:Example.schedule.length + 1] do
    let saved := Exact.run Example.initial (Example.schedule.take split)
    check (decide (Exact.run saved (Example.schedule.drop split) = actual))
      s!"checkpoint/resume at {split}"
  check (!(Exact.checkRun Example.initial Example.schedule
    { Example.expected with steps := 9 })) "accepted wrong step counter"
  check (!(Exact.checkRun Example.initial Example.schedule
    { Example.expected with parameters := { Example.expected.parameters with w₀ := 2 } }))
    "accepted wrong weights"
  check (!(Exact.checkRun Example.initial (Example.schedule.take 7) Example.expected))
    "accepted shortened run"
  check (!(Exact.checkRun { Example.initial with parameters := ⟨1, 0, 0⟩ }
    Example.schedule Example.expected)) "accepted changed initialization"
  check (!(Exact.checkRun Example.initial
    (Example.schedule.map fun u => { u with rate := 1 / 4 }) Example.expected))
    "accepted changed learning rate"
  check (!(Exact.checkRun Example.initial
    (Example.schedule.map fun u =>
      { u with batch := { u.batch with first := { u.batch.first with target := 7 } } })
    Example.expected)) "accepted changed labels"
  let u : Exact.Update := ⟨⟨⟨1, 0, 1⟩, []⟩, 1 / 8⟩
  let v : Exact.Update := ⟨⟨⟨1, 0, 0⟩, []⟩, 1 / 8⟩
  check (decide (Exact.run Example.initial [u, v] ≠ Exact.run Example.initial [v, u]))
    "schedule order was ignored"
  check (Exact.encodeState actual ==
    "cybernetix.exact-rational-sgd.v1\nsteps=8\nw0=15/8\nw1=-45/16\nbias=58975/65536\n")
    "exact state encoding"

  let mut native := Example.initial.parameters.map nativeInput
  let mut reference := Example.initial.parameters.map Binary32.fromRat
  for u in Example.schedule do
    reference := Binary32.step reference u
    native := sgdStep native (u.batch.map nativeInput) (nativeInput u.rate)
    check (decide (native.map (·.toBits) = Binary32.words reference))
      "native binary32 update differs from the bit model"
  check (decide (Binary32.words reference = ⟨0x3ff00000, 0xc0340000, 0x3f665f00⟩))
    "binary32 final words"
  check (decide ((Binary32.encodeParameters reference).data =
    #[0, 0, 240, 63, 0, 0, 52, 192, 0, 95, 102, 63])) "binary32 byte order"
  for c in Binary32.cases do
    check c.check s!"logical binary32 fixture: {c.name}"
    let result := nativeOp c.op (Float32.ofBits c.left) (Float32.ofBits c.right)
    check (result.toBits == c.expected) s!"native binary32 fixture: {c.name}"
  IO.println "SGD replay, claim rejection, and native binary32 comparison tests passed."

end Cybernetix.Tests.SGD
