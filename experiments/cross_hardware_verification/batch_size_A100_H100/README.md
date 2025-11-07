Probably the most interesting experiment in this repo:
Do we need to match hardware to catch secret changes to batch size?

The results seem to strongly suggest that we do not.

Two experiments here:
1. One where we measure the cross-hardware distance baseline for different batch sizes
2. One where we compare across
   A) same hardware, different batch size
   B) same batch size, different hardware (basically like the earlier control experiment)
   C) different hardware, different batch size
   
Turns out that C) is distinguishable from B) with high confidence. 1. produced comparable deviations to 2.B) for bs1, 
which is also reassuring.
