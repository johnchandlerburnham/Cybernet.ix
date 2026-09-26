import Cybernetix.Numeric.Binary32

namespace Cybernetix.Numeric.Binary32.Proofs

open Cybernetix.Training

set_option maxRecDepth 16384
set_option maxHeartbeats 8000000

/-- All primitive fixtures are established by kernel reduction of the bit model. -/
theorem cases_correct : cases.all Case.check = true := by decide +kernel

/-- Every checkpoint of this particular rounded run equals the exact rational run.
This is deliberately a concrete claim, not a general floating-gradient theorem. -/
theorem toy_prefixes_exact : ∀ n : Fin 9,
    (run Example.initial.parameters (Example.schedule.take n.val)).map toRat? =
      (Exact.run Example.initial (Example.schedule.take n.val)).parameters.map some := by
  decide +kernel

theorem toy_final_words :
    words (run Example.initial.parameters Example.schedule) =
      (⟨0x3ff00000, 0xc0340000, 0x3f665f00⟩ : Parameters UInt32) := by
  decide +kernel

theorem toy_final_bytes :
    (encodeParameters (run Example.initial.parameters Example.schedule)).data =
      #[0, 0, 240, 63, 0, 0, 52, 192, 0, 95, 102, 63] := by
  decide +kernel

/-- The same terms with a different reduction tree can give different answers. -/
theorem reassociation_changes_answer :
    let a := Float32.Model.ofBits 0x4b800000
    let b := Float32.Model.ofBits 0x3f800000
    let c := Float32.Model.ofBits 0xcb800000
    ((a + b) + c).toBits = 0 ∧ (a + (b + c)).toBits = 0x3f800000 := by
  decide +kernel

end Cybernetix.Numeric.Binary32.Proofs
