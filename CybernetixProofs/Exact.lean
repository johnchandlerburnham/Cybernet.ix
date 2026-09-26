import Cybernetix.Training.Example
import CybernetixProofs.Gradient

/-! Connect the executable rational updates to the same model over the reals.
Concrete evaluation proofs below use kernel reduction, never `native_decide`.
-/

namespace Cybernetix.Training.Proofs

noncomputable section

def realParameters (p : Parameters Rat) : Parameters ℝ := p.map (fun q : Rat => (q : ℝ))
def realSample (s : Sample Rat) : Sample ℝ := s.map (fun q : Rat => (q : ℝ))
def realBatch (b : Batch Rat) : Batch ℝ := b.map (fun q : Rat => (q : ℝ))

theorem realParameters_add (p q : Parameters Rat) :
    realParameters (p.add q) = (realParameters p).add (realParameters q) := by
  simp [realParameters, Parameters.map, Parameters.add]

theorem sampleGradient_cast (p : Parameters Rat) (s : Sample Rat) :
    realParameters (sampleGradient p s) = sampleGradient (realParameters p) (realSample s) := by
  simp [realParameters, realSample, Parameters.map, Sample.map, sampleGradient, residual, predict]

theorem gradientSum_cast (p : Parameters Rat) (ss : List (Sample Rat)) :
    realParameters (gradientSum p ss) = gradientSum (realParameters p) (ss.map realSample) := by
  induction ss with
  | nil => simp [gradientSum, realParameters, Parameters.map]
  | cons s ss ih =>
    simp only [List.map_cons, gradientSum, realParameters_add, sampleGradient_cast, ih]

theorem meanGradient_cast (p : Parameters Rat) (b : Batch Rat) :
    realParameters (meanGradient p b) = meanGradient (realParameters p) (realBatch b) := by
  have h := gradientSum_cast p b.samples
  simp only [realParameters, Parameters.map] at h
  simp only [realParameters, Parameters.map, meanGradient, realBatch, Batch.map,
    Batch.samples, Batch.size, List.length_map]
  have h₀ := congrArg Parameters.w₀ h
  have h₁ := congrArg Parameters.w₁ h
  have hb := congrArg Parameters.bias h
  simp only [Batch.samples, List.map_cons] at h₀ h₁ hb
  simp only [Rat.cast_div, Rat.cast_natCast, h₀, h₁, hb, realSample]
  rfl

/-- Exact executable SGD commutes with the rational-to-real interpretation. -/
theorem sgdStep_cast (p : Parameters Rat) (b : Batch Rat) (rate : Rat) :
    realParameters (sgdStep p b rate) =
      sgdStep (realParameters p) (realBatch b) (rate : ℝ) := by
  have h := meanGradient_cast p b
  simp only [realParameters, Parameters.map] at h
  have h₀ := congrArg Parameters.w₀ h
  have h₁ := congrArg Parameters.w₁ h
  have hb := congrArg Parameters.bias h
  dsimp only at h₀ h₁ hb
  simp only [realParameters, Parameters.map, sgdStep, Parameters.subtractScaled,
    Rat.cast_sub, Rat.cast_mul, h₀, h₁, hb]

/-- The whole rational run is a sequence of the mathematically verified real updates. -/
theorem run_cast (s : Exact.State) (us : List Exact.Update) :
    realParameters (Exact.run s us).parameters =
      us.foldl (fun p u => sgdStep p (realBatch u.batch) (u.rate : ℝ))
        (realParameters s.parameters) := by
  induction us generalizing s with
  | nil => rfl
  | cons u us ih =>
    change realParameters (Exact.run (Exact.step s u) us).parameters = _
    rw [ih]
    simp only [List.foldl_cons, Exact.step, sgdStep_cast]

open Exact Example

set_option maxRecDepth 8192 in
set_option maxHeartbeats 4000000 in
theorem example_run : run initial schedule = expected := by decide +kernel

theorem example_valid : Runs initial schedule expected :=
  example_run ▸ run_sound initial schedule

set_option maxRecDepth 8192 in
theorem example_initial_loss : meanLoss initial.parameters evaluationBatch = (14 : Rat) := by
  decide +kernel

set_option maxRecDepth 8192 in
theorem example_final_loss :
    meanLoss expected.parameters evaluationBatch = (261150529 / 4294967296 : Rat) := by
  decide +kernel

set_option maxRecDepth 8192 in
theorem example_loss_bound : meanLoss expected.parameters evaluationBatch < (1 / 16 : Rat) := by
  decide +kernel

set_option maxRecDepth 8192 in
theorem example_prediction :
    predict (run initial schedule).parameters (⟨3, -2, 13⟩ : Sample Rat) =
      (796255 / 65536 : Rat) := by
  rw [example_run]
  decide +kernel

end

end Cybernetix.Training.Proofs
