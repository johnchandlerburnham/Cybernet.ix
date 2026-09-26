import Cybernetix.Numeric.Binary32

open Cybernetix.Training
open Cybernetix.Numeric

def main : IO Unit := do
  IO.println "Two-feature regression, alternating minibatches, exact learning rate 1/8"
  let mut state := Example.initial
  IO.println s!"step {state.steps}: MSE = {Exact.encodeRat (meanLoss state.parameters Example.evaluationBatch)}"
  for update in Example.schedule do
    state := Exact.step state update
    IO.println s!"step {state.steps}: MSE = {Exact.encodeRat (meanLoss state.parameters Example.evaluationBatch)}"
  unless Exact.checkRun Example.initial Example.schedule state do
    throw <| IO.userError "run validation failed"
  IO.println "\nExact checkpoint:"
  IO.print (Exact.encodeState state)
  let rounded := Binary32.run Example.initial.parameters Example.schedule
  IO.println s!"\n{Binary32.profile} parameter bytes: {(Binary32.encodeParameters rounded).data}"
  IO.println "Proofs: lake build CybernetixProofs (including axiom audit)."
