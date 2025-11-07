This is an experiment that -in contrast to most experiments in this repo- I did in decode rather than prefill. 
Why? On the one hand, because not all experiments should be in prefill. After all, both matter for inference. For the 
purposesof the experiments here, results of prefill experiments should extrapolate to decode inference though - 
computationally, both turned out to be determinstic within our conditions (default CUDA stream, fixed algorithms 
and batch size).

Implementation wise, decode actually did reveal a little pain with collecting activations. In this experiment, we are
comparing an eager inference run (uncompiled mode) vs. a compiled one (torch.compile). This is another model organism for what "secret backend optimization" might be like, as in the CUDA version test. A problem arised when trying to extract hidden states from a compiled model in decode inference, which was no issue in prefill: CUDA graphs. These immediately over-write hidden states after use. I could not do my experiments on hidden states for that reason, which is why I did these only by comparing key vectors from attn layers. 

This worked so well, that I decided to use keys in every experiment from now on, since they are just less invasive to extract in realistic inference monitoring and verification scenarios.
Why? Because they are cached by default (needed for decode), and are smaller (especially for GQA, MLA etc. architectures).
Activations (hidden states), in contrast, would need to be cached and transmitted/hashed, which is additional memory and I/O 
overhead.

There is also a speed benchmark where I test if a compiled model really does decode faster.
