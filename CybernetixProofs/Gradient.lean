import Cybernetix.Training.Linear
import CybernetixProofDependencies

/-! The actual minibatch MSE gradient, its directional derivative, and a
conditional descent theorem. These theorems concern real arithmetic.
-/

namespace Cybernetix.Training.Proofs

noncomputable section

open Parameters

/-- Curvature along a direction, before dividing by the minibatch size. -/
def energySum (v : Parameters ℝ) : List (Sample ℝ) → ℝ
  | [] => 0
  | s :: ss => predict v s ^ 2 + energySum v ss

def meanEnergy (v : Parameters ℝ) (b : Batch ℝ) : ℝ :=
  energySum v b.samples / b.size

theorem sampleLoss_shift (p v : Parameters ℝ) (s : Sample ℝ) (t : ℝ) :
    sampleLoss (p.shift v t) s =
      sampleLoss p s + t * (sampleGradient p s).dot v + t ^ 2 * predict v s ^ 2 := by
  simp only [sampleLoss, residual, predict, shift, add, scale, sampleGradient, dot]
  ring

theorem lossSum_shift (p v : Parameters ℝ) (ss : List (Sample ℝ)) (t : ℝ) :
    lossSum (p.shift v t) ss =
      lossSum p ss + t * (gradientSum p ss).dot v + t ^ 2 * energySum v ss := by
  induction ss with
  | nil => simp [lossSum, gradientSum, dot, energySum]
  | cons s ss ih =>
    simp only [lossSum, gradientSum, energySum, sampleLoss_shift, ih, add, dot]
    ring

/-- An exact quadratic expansion identifies the gradient without finite differences. -/
theorem meanLoss_shift (p v : Parameters ℝ) (b : Batch ℝ) (t : ℝ) :
    meanLoss (p.shift v t) b =
      meanLoss p b + t * (meanGradient p b).dot v + t ^ 2 * meanEnergy v b := by
  simp only [meanLoss, meanGradient, meanEnergy, lossSum_shift, dot]
  ring

/-- The implemented gradient dotted with *any* direction is the true derivative
of the actual mean batch loss along that direction. -/
theorem meanGradient_correct (p v : Parameters ℝ) (b : Batch ℝ) :
    HasDerivAt (fun t => meanLoss (p.shift v t) b) ((meanGradient p b).dot v) 0 := by
  simp_rw [meanLoss_shift]
  have hLinear : HasDerivAt (fun t : ℝ => t * (meanGradient p b).dot v)
      ((meanGradient p b).dot v) 0 := by
    simpa using (hasDerivAt_id (0 : ℝ)).mul_const ((meanGradient p b).dot v)
  have hQuadratic : HasDerivAt (fun t : ℝ => t ^ 2 * meanEnergy v b) 0 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).pow 2).mul_const (meanEnergy v b)
  convert! ((hasDerivAt_const (0 : ℝ) (meanLoss p b)).add hLinear).add hQuadratic
    using 1
  simp

/-- Exact loss change for the implementation's SGD update, including curvature. -/
theorem sgdStep_loss (p : Parameters ℝ) (b : Batch ℝ) (rate : ℝ) :
    meanLoss (sgdStep p b rate) b = meanLoss p b -
      rate * (meanGradient p b).dot (meanGradient p b) +
      rate ^ 2 * meanEnergy (meanGradient p b) b := by
  have h : sgdStep p b rate = p.shift (meanGradient p b) (-rate) := by
    simp [sgdStep, subtractScaled, shift, add, scale, sub_eq_add_neg]
  rw [h, meanLoss_shift]
  ring

/-- A sufficient step-size condition for descent on this batch. It does not assert
monotonic improvement on other batches, a whole dataset, or arbitrary rates. -/
theorem sgdStep_descent (p : Parameters ℝ) (b : Batch ℝ) (rate : ℝ)
    (hRate : 0 ≤ rate)
    (hSmall : rate * meanEnergy (meanGradient p b) b ≤
      (meanGradient p b).dot (meanGradient p b)) :
    meanLoss (sgdStep p b rate) b ≤ meanLoss p b := by
  rw [sgdStep_loss]
  nlinarith [mul_le_mul_of_nonneg_left hSmall hRate]

end

end Cybernetix.Training.Proofs
