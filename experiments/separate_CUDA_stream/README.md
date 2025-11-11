In this experiment, I test the effect that a separate inference workload (not batched, rather a separate CUDA stream) has
on the original sequence's activations.

This folder is incomplete. There are only concurrent prefill workloads tested here (exp1), but the results should generalize to concurrent decode as well. Concurrent streams did not affect numerics here, only execution time due to resource contentions.
