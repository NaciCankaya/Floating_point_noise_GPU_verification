Activation comparison across a suite of different GPUs.
Different activations for different GPU types, same for the same type. Same for A100 PCIe and A100 SXM. 
Same for H100 and H200. Both as expected, as logic is the same. 

Peculiar behaviour of H100, H200 and B200: No difference across batch size at first. Reason identified: The first experiment
naively batched the same sequence with itself. When using unique sequences, batch size affected activations again.
