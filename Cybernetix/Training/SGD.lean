import Cybernetix.Training.Linear
import Init.Data.Rat

/-! Exact, deterministic SGD conditioned on an explicit ordered minibatch schedule.
Sampling a schedule is outside this program. No stochastic-estimator claim is made.
-/

namespace Cybernetix.Training.Exact

/-- This arithmetic profile does not round or use host floating-point operations. -/
def profile : String := "cybernetix.exact-rational-sgd.v1"

structure State where
  parameters : Parameters Rat
  steps : Nat
  deriving DecidableEq, Repr

structure Update where
  batch : Batch Rat
  rate : Rat
  deriving DecidableEq, Repr

def step (s : State) (u : Update) : State :=
  ⟨sgdStep s.parameters u.batch u.rate, s.steps + 1⟩

/-- An explicit schedule includes the batch and rate at every update. -/
def run (initial : State) (schedule : List Update) : State :=
  schedule.foldl step initial

/-- A concrete run relation, independently usable as the subject of a claim. -/
inductive Runs : State → List Update → State → Prop where
  | nil (s) : Runs s [] s
  | cons (s u us finish) : Runs (step s u) us finish → Runs s (u :: us) finish

theorem run_sound (s : State) (us : List Update) : Runs s us (run s us) := by
  induction us generalizing s with
  | nil => exact .nil s
  | cons u us ih => exact .cons s u us _ (ih (step s u))

theorem Runs.result {s finish : State} {us : List Update}
    (h : Runs s us finish) : finish = run s us := by
  induction h with
  | nil => rfl
  | cons _ _ _ _ _ ih => exact ih

theorem Runs.deterministic {s a b : State} {us : List Update}
    (ha : Runs s us a) (hb : Runs s us b) : a = b :=
  ha.result.trans hb.result.symm

theorem run_append (s : State) (firstPart secondPart : List Update) :
    run s (firstPart ++ secondPart) = run (run s firstPart) secondPart := by
  simp [run, List.foldl_append]

theorem run_steps (s : State) (us : List Update) :
    (run s us).steps = s.steps + us.length := by
  induction us generalizing s with
  | nil => rfl
  | cons u us ih =>
    change (run (step s u) us).steps = s.steps + (u :: us).length
    rw [ih]
    simp [step, Nat.add_comm, Nat.add_left_comm]

/-- Check against consumer-supplied inputs, rather than a producer-chosen plan. -/
def checkRun (initial : State) (schedule : List Update) (claimed : State) : Bool :=
  decide (run initial schedule = claimed)

theorem checkRun_iff (initial : State) (schedule : List Update) (claimed : State) :
    checkRun initial schedule claimed = true ↔ Runs initial schedule claimed := by
  simp only [checkRun, decide_eq_true_eq]
  constructor
  · intro h
    rw [← h]
    exact run_sound initial schedule
  · intro h
    exact h.result.symm

/-- Canonical decimal numerator/positive denominator, including `0/1`. -/
def encodeRat (q : Rat) : String := s!"{q.num}/{q.den}"

/-- Versioned ASCII output; state equality determines identical encoded bytes.
This encodes a state, not a complete experiment manifest or an Ix certificate. -/
def encodeState (s : State) : String :=
  profile ++ "\nsteps=" ++ toString s.steps ++
    "\nw0=" ++ encodeRat s.parameters.w₀ ++
    "\nw1=" ++ encodeRat s.parameters.w₁ ++
    "\nbias=" ++ encodeRat s.parameters.bias ++ "\n"

theorem Runs.encoded_result_eq {s a b : State} {us : List Update}
    (ha : Runs s us a) (hb : Runs s us b) :
    (encodeState a).toUTF8 = (encodeState b).toUTF8 := by
  rw [ha.deterministic hb]

end Cybernetix.Training.Exact
