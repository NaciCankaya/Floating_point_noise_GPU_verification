This CUDA update came with some substantial changes to the CUDA backend, with new cuDNN and stuff. I figured this would be
a useful "model organism" for a change in kernels that boost inference efficiency at a low level. 

Tests showed that this is update is detectable in the sense that it affects activations. As expected from different numerics.
