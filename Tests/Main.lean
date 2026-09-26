import Tests

def main : IO Unit := do
  Cybernetix.Tests.FFI.run
  Cybernetix.Tests.SGD.run
  Cybernetix.Tests.LinearClassifier.run
  Cybernetix.Tests.MNIST.run
