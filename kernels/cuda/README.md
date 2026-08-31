# CUDA kernels

Empty on purpose.

A kernel gets written here only after `src/forge/training/profiling.py` produces a
trace showing that a specific stage is a real bottleneck. Writing a fused kernel for a
stage that is 3 percent of step time is decoration, not optimization, and it does not
survive an interviewer asking "how much did it help?".

Candidates, in the order profiling is likely to rank them:
1. fused window extraction + tokenization on the preprocessing path
2. a fused pooling/head operation if the head shows up in the trace
3. attention path work only if a measurement says the stock kernel is the limit

Each kernel that lands here ships with: the profile trace that motivated it, a
correctness test against the PyTorch reference, and a before/after benchmark.
