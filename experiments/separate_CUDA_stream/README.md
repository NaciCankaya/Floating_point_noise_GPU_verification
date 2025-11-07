In this experiment, I test the effect that a separate inference workload (not batched, rather a separate CUDA stream) has
on the original sequence's activations.

Key findings:
1. Detectable when GPU utilization is high and race conditions need to schedule between original inference and the other stream
2. Not detectable at low utilization: CUDA streams operate on separate SMs in parallel, as if they were on different devices.
3. NOT REPRODUCIBLE. The only kind of workload (so far) that has produced real statistical noise in my experiments. This is
   due to atomics and race conditions in scheduling, which depends on runtime-fluctuating factors such as temperature, clock
   speed etc.
