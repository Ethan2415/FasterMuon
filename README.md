# FasterMuon

FasterMuon is an optimized implementation of the Muon optimizer that significantly reduces optimizer update time by batching Newton–Schulz iterations across parameters with identical shapes.

Muon is well known for its strong token efficiency, but its update step can become a noticeable bottleneck in certain training workloads. This is especially true when training relatively small models with fast forward passes and a large number of parameters optimized by Muon. In such cases, the optimizer update may take a comparable amount of time to the forward and backward passes combined.

For example, in one of my training workloads, the forward pass took roughly 200 ms per step, while the Muon update implemented in existing repositories consumed another ~200 ms. In contrast, a standard AdamW update required only a few milliseconds. Under these conditions, the optimizer overhead substantially reduces the practical training efficiency of Muon.

FasterMuon addresses this issue with a simple idea: parameters sharing the same shape are grouped together and processed using batched Newton–Schulz iterations. By replacing many small matrix operations with larger batched operations, the optimizer can make much better use of GPU parallelism and significantly reduce update latency.

The speedup is most noticeable when:

* The model has a relatively fast forward/backward pass.
* A large number of parameters are optimized using Muon.
* Many of those parameters share identical shapes.

In workloads matching these characteristics, FasterMuon can substantially reduce optimizer overhead and improve overall training throughput.

More implementation details can be found directly in the source code.

**Note:** The current implementation has only been tested under Distributed Data Parallel (DDP) training. Other parallel training strategies may require additional modifications.
