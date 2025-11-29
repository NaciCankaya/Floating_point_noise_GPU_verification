In this experiment, I test the effect that a separate inference workload (not batched, rather a separate CUDA stream) has
on the original sequence's activations.



We see two experiments here: Two concurrent prefill streams and two concurrent decode streams. In both cases, activations were unaffected by the presence of other streams. 

The profiler confirms that execution was indeed overlapping, not interleaved or serialized. 



Hiding parallel ML work in a separate stream is probably not a perfect evasion method however, if the reported streams do not under-utilize hardware. The reason is resource contention: The experiments confirm that one can not simply add an additional stream of workloads wothout competing with the default stream on compute or memory bandwidth.And if you try to evade resource contention by secretly using more efficient kernels for the default stream, you are back to leaving a numerical noise fingerprint.

