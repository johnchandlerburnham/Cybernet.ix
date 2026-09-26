import CybernetixProofs.Exact
import CybernetixProofs.Binary32
import CybernetixProofs.Tensor
import CybernetixProofs.LinearClassifier
import CybernetixProofs.LinearArtifact
import CybernetixProofs.MNIST
import Lean.Util.CollectAxioms

/-! Fail the build if a release theorem depends on an unapproved axiom.
In particular, `sorryAx` and native-evaluation axioms are not allowed.
-/

open Lean Elab Command in
run_cmd do
  let allowed := #[``propext, ``Classical.choice, ``Quot.sound]
  let roots := #[
    ``Cybernetix.Training.Proofs.meanGradient_correct,
    ``Cybernetix.Training.Proofs.sgdStep_loss,
    ``Cybernetix.Training.Proofs.sgdStep_descent,
    ``Cybernetix.Training.Proofs.sgdStep_cast,
    ``Cybernetix.Training.Proofs.run_cast,
    ``Cybernetix.Training.Exact.checkRun_iff,
    ``Cybernetix.Training.Exact.run_append,
    ``Cybernetix.Training.Exact.run_steps,
    ``Cybernetix.Training.Exact.Runs.encoded_result_eq,
    ``Cybernetix.Training.Proofs.example_valid,
    ``Cybernetix.Training.Proofs.example_initial_loss,
    ``Cybernetix.Training.Proofs.example_final_loss,
    ``Cybernetix.Training.Proofs.example_loss_bound,
    ``Cybernetix.Training.Proofs.example_prediction,
    ``Cybernetix.Numeric.Binary32.Proofs.cases_correct,
    ``Cybernetix.Numeric.Binary32.Proofs.toy_prefixes_exact,
    ``Cybernetix.Numeric.Binary32.Proofs.toy_final_words,
    ``Cybernetix.Numeric.Binary32.Proofs.toy_final_bytes,
    ``Cybernetix.Numeric.Binary32.Proofs.reassociation_changes_answer,
    ``Cybernetix.Tensor.assembleWord_wordByte,
    ``Cybernetix.Tensor.Buffer.getWord_ofFn,
    ``Cybernetix.Tensor.Buffer.view_ofFn,
    ``Cybernetix.Tensor.Buffer.ofBytes_roundtrip,
    ``Cybernetix.Tensor.Buffer.ofBytes_reject,
    ``Cybernetix.Tensor.Buffer.getWordFast_eq,
    ``Cybernetix.Tensor.Buffer.getWordU_eq,
    ``Cybernetix.Numeric.Reduction.depth_covers,
    ``Cybernetix.Numeric.Reduction.depth_minimal,
    ``Cybernetix.Numeric.Reduction.native_model,
    ``Cybernetix.Numeric.Reduction.nativeTreeU_eq,
    ``Cybernetix.Model.LinearClassifier.validate_eq,
    ``Cybernetix.Model.LinearClassifier.forward_eq_reference,
    ``Cybernetix.Model.LinearClassifier.argmax_correct,
    ``Cybernetix.Model.LinearClassifier.decodeParameters_encode,
    ``Cybernetix.Corpus.MNIST.pixelWord_eq,
    ``Cybernetix.Corpus.MNIST.input_word,
    ``Cybernetix.Corpus.MNIST.pixels_finite,
    ``Cybernetix.Corpus.MNIST.train_validation_disjoint]
  for name in roots do
    let _ ← getConstInfo name
    let axioms ← collectAxioms name
    for ax in axioms do
      unless allowed.contains ax do
        throwError "{name} depends on unapproved axiom {ax}"
  logInfo m!"Cybernet.ix axiom audit passed for {roots.size} theorem roots."
