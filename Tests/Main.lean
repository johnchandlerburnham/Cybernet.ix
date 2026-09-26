import Tests

def main : IO Unit := do
  Cybernetix.Tests.FFI.run
  Cybernetix.Tests.SGD.run
