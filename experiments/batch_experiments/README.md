Two categories of experiments here:

1. Batch size:
   Same hardware, add parallel batches next to sequence0 (the one whose activations we
   are measuring and comparing across different batch sizes). I did this by naively adding dummy sequences that
   are automatically repeated text. They are much shorter than sequence0. Another experiment, "unique_sequences" batches
   more "realistic" samples together.
   Result: Measurable deviations, but repeatable for fixed batch size.
2. Batch composition:
   Fixed batch size, but different content (different, equal length text). No padding tokens. No effect on sequence0 
   activations.
