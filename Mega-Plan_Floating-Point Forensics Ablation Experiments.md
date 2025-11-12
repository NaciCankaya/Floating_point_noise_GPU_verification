# **Mega-Plan: Floating-Point Forensics Ablation Experiments**

**Purpose:** Systematic testing of detection capabilities across hardware platforms (A100, H100) for various inference setup modifications.

**Research Question:** For each variable that affects inference execution, can we detect it via floating-point forensics (activation/key/logprob differences)? How does detectability compare to cross-hardware baseline deviation?

---

## **Executive Summary**

**Total Experiments:** 8  
 **Hardware Platforms:** A100-80GB, H100  
 **Model:** Qwen3-30B-A3B-AWQ-Int4 (\~15GB, MoE architecture) QuixiAI/Qwen3-30B-A3B-AWQ  
 **Sequence Configuration:** \~6k token prompt (prefill)l \+ 30 tokens decode  
 **Repetitions per Config:** 3 (establishes within-setup reproducibility)  
 **Total Data Volume:** (8 JSON files, one for each experiment)  
 **Pod Requirements:** 4 instances: 

(\[(1 instance for experiments 1 to 6 with single GPU) \+ (1 instance for experiments 7 and 8 using four SXM GPUs)\]  2, since we are doing this for both A100 SXM and H100 SXM, respectively

For most of the experiments seen here, I already have implementations ready in my repo. But they do not fit the standardized format used here, with identical model, prompt, and json format. This is why we are repeating them in one go. But still, if you are not sure what to do, look into the repo.

### **Experiment 0: Reference Baseline**

**Configuration (all experiments reuse this as baseline):**

* `batch_size`: 1  
* `compile`: False  
* `quant_method`: AWQ  
* `attention_impl`: flash\_attention\_2  
* `concurrent_work`: False (default CUDA stream)  
* `cuda_version`: Pod default (\~12.8)  
* `tp_size`: 1  
* `ep_size`: 1

---

| Experiment | Variable Tested | num configs |
| ----- | ----- | :---: |
| 0\. Reference | None.  | 1 |
| 1\. Batch Size | batch\_size: 2, 4 | 2 |
| 2\. Compilation | compile: True | 1 |
| 3\. Quantization | quant\_method: gptq, bnbQuixiAI/Qwen3-30B-A3B-AWQunsloth/Qwen3-30B-A3B-bnb-4bit | 2 |
| 4\. Attention | attention\_impl: eager | 1 |
| 5\. Concurrent Streams | concurrent\_work: True | 1 |
| 6\. CUDA Version | cuda\_version: 11.8, 12.1 (or cu118, cu 121\) | 2 |
| 7\. Tensor Parallelism | tp\_size: 2, 4 | 2 |
| 8\. Expert Parallelism | ep\_size: 2, 4 | 2 |

Regarding implementation, you can take heavy inspiration from the experiment files in my repo. BUT: STICK TO THE TEMPLATE. Experiments 1-5 may be a single python script, if you prefer. Choose whatever architecture makes the most sense to you, though.

## **Experimental Variables**

---

## **Signal Extraction Specification**

### **Layers to Extract**

* **Layers:** 1, 2, 4, 12, last (beware indexing)  
* **Rationale:** Sample across depth for propagation analysis

### **Token Positions**

* **Positions:** \-3, \-2, \-1 (final three tokens at each decode step)  
* **Total measurements per sequence:** 30 decode steps × 3 positions \= 90 timepoints. Only store the measurement for the last repetition in json IF all three repetitions were bit-identical.

### **Signal Types**

1. **Hidden States:** 3584-dimensional vectors (5 layers × 3 positions)  
2. **Key Vectors:** 512-dimensional GQA keys (concatenated) (5 layers × 3 positions)  
3. **Logprobs:** Top-10 token probabilities (3 positions)

---

## **Uniform JSON Schema**

All 8 experiment files follow this identical structure:

{  
  "experiment\_metadata": {  
    "experiment\_type": "batch\_size",  
    "variable\_tested": "batch\_size",  
    "model": "Qwen3-30B-A3B-GPTQ-Int4",  
    "model\_size": "30B",  
    "architecture": "MoE",  
    "sequence\_length": 8192,  
    "decode\_steps": 30,  
    "extraction\_config": {  
      "layers": \[1, 2, 4, 12, 39\],  
      "positions": \[-3, \-2, \-1\],  
      "hidden\_dim": 3584,  
      "key\_dim": 512,  
      "top\_k\_logprobs": 10  
    },  
    "date\_created": "YYYY-MM-DD HH:MM:SS"  
  },  
    
  "configurations": \[  
    {  
      "config\_id": "A100\_bs1",  
      "hardware": "A100-80GB",  
      "provider": "RunPod",  
      "variable\_value": 1,  
      "cuda\_version": "12.8",  
      "torch\_version": "2.x.x",  
      "transformers\_version": "4.x.x",  
      "flash\_attn\_version": "2.x.x",  
      "python\_version": "3.x",  
      "fixed\_params": {  
        "compile": false,  
        "attention\_impl": "flash\_attention\_2",  
        "quantization": "gptq-int4",  
        "tp\_size": 1,  
        "ep\_size": 1,  
        "concurrent\_streams": false  
      }  
    }  
  \],  
    
  "runs": \[  
    {  
      "config\_id": "A100\_bs1",  
      "rep\_id": 0,  
      "timestamp": "YYYY-MM-DD HH:MM:SS",  
      "runtime\_seconds": 123.45,  
      "prompt\_text": "...",  
        
      "decode\_steps": \[  
        {  
          "step": 0,  
          "token\_id": 12345,  
          "token\_text": "Hello",  
            
          "hidden\_states": {  
            "layer\_1": {  
              "pos\_-3": \[3584 floats\],  
              "pos\_-2": \[3584 floats\],  
              "pos\_-1": \[3584 floats\]  
            },  
            "layer\_2": {},  
            "layer\_4": {},  
            "layer\_12": {},  
            "layer\_39": {}  
          },  
            
          "key\_vectors": {  
            "layer\_1": {  
              "pos\_-3": \[512 floats\],  
              "pos\_-2": \[512 floats\],  
              "pos\_-1": \[512 floats\]  
            },  
            "layer\_2": {},  
            "layer\_4": {},  
            "layer\_12": {},  
            "layer\_39": {}  
          },  
            
          "logprobs": {  
            "pos\_-3": {  
              "token\_ids": \[10 ints\],  
              "log\_probs": \[10 floats\]  
            },  
            "pos\_-2": {},  
            "pos\_-1": {}  
          }  
        }  
      \]  
    }  
  \]  
}

**Key Schema Features:**

* Self-documenting: All versions and configurations embedded  
* Uniform across experiments: Single comparison script will work for all experiment jsons  
* Linkable: config\_id, variable\_tested connect runs to configurations  
* Complete: Includes metadata for reproducibility verification

## **Pod Execution Plan**

### **Pod 1: Single GPU (A100-SXM or H100-SXM)**

**Experiments:** 0, 1, 2, 3, 4, 5, 6

**Execution Sequence:**

1. **Setup** (\~20 min)  
   * **Setup:** Install dependencies (just test them by running the experiment code and install whatever the error print wants you to pip install)  
   * **Experiment 1 \- Batch Size:** Download base model: `QuixiAI/Qwen3-30B-A3B-AWQ`  
   * Download quantization variants: GPTQ, BNB versions  
   * Verify installations and model loading  
2. **Experiment 0: Reference Baseline**  
   * Run 3 reps with baseline config  
   * Verify bit-exact reproducibility  
   * Save to `reference_baseline.json` (for documentation)  
   * This data will be reused as baseline for Experiments 1-6  
3. **Experiment 1: Batch Size**  
   * Load AWQ model  
   * Run bs1, bs2, bs4 configs. Make sure to use distinct token sequences, all \~6k tokens long, same as experiment 0\. Beware padding tokens to not measure the wrong thing.   
   * Run bs=2 (3 reps)  
   * Run bs=4 (3 reps)  
   * Fetch json from Experiment 0 bs=1 data as baseline  
   * Save all three’s results to `batch_size_experiment.json`  
4. **Experiment 2: Compilation**  
   * Load AWQ model with `torch.compile(model)`  
   * Run compile=True (3 reps)  
   * Reuse Experiment 0 compile=False data as baseline  
   * Save to `compile_experiment.json`  
5. **Experiment 3: Quantization**  
   * Load GPTQ model variant (3 reps)  
   * Load BNB model variant (3 reps)  
   * Reuse Experiment 0 AWQ data as baseline  
   * Save to `quantization_experiment.json`  
   * Delete the other two quants, keep awq  
6. **Experiment 4: Attention**  
   * Load AWQ model with `attn_implementation="eager"`  
   * Run eager attention (3 reps)  
   * Reuse Experiment 0 FA2 data as baseline  
   * Save to `attention_experiment.json`  
7. **Experiment 5: Concurrent Streams**  
   * Load AWQ model  
   * Launch concurrent workload on stream 1 (\~50% GPU util)  
   * Run inference on stream 0 (3 reps)  
   * make sure both overlap (check out the profiler script in my repo)  
   * Reuse Experiment 0 single-stream data as baseline  
   * Save to `concurrent_streams_experiment.json`  
8. **CUDA Version Switch to cu118** (\~30 min)

bash

  *\# Install CUDA 11.8*

   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local\_installers/cuda\_11.8.0\_520.61.05\_linux.run

   sudo sh cuda\_11.8.0\_520.61.05\_linux.run \--silent \--toolkit

   export PATH\=/usr/local/cuda-11.8/bin:$PATH

   export LD\_LIBRARY\_PATH=/usr/local/cuda-11.8/lib64:$LD\_LIBRARY\_PATH

   

   *\# Reinstall PyTorch for cu118*

   pip uninstall torch \-y

   pip install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu118

   

   *\# Verify CUDA version*

   python \-c "import torch; print(torch.version.cuda)"

9. **Experiment 6a: CUDA 11.8**  
   * Load AWQ model  
   * Run cu118 config (3 reps)  
10. **CUDA Version Switch to cu121**

bash

   *\# Install CUDA 12.1*

    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local\_installers/cuda\_12.1.0\_530.30.02\_linux.run

    sudo sh cuda\_12.1.0\_530.30.02\_linux.run \--silent \--toolkit

    export PATH\=/usr/local/cuda-12.1/bin:$PATH

    export LD\_LIBRARY\_PATH=/usr/local/cuda-12.1/lib64:$LD\_LIBRARY\_PATH

    

    *\# Reinstall PyTorch for cu121*

    pip uninstall torch \-y

    pip install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu121

11. **Experiment 6b: CUDA 12.1**  
    * Load AWQ model  
    * Run cu121 config (3 reps)  
    * Combine with cu118 data and Experiment 0 (cu128) as baseline  
    * Save to `cuda_version_experiment.json`

**Output Files:**

* `reference_baseline.json` (optional documentation)  
* `batch_size_experiment.json`  
* `compile_experiment.json`  
* `quantization_experiment.json`  
* `attention_experiment.json`  
* `concurrent_streams_experiment.json`  
* `cuda_version_experiment.json`

---

### **Pod 2: Multi-GPU (4×A100-SXM or 4×H100-SXM)**  **Experiments:** 7, 8

**Execution Sequence:**

1. **Setup** (\~30 min)  
   * Download model to shared storage (NVMe recommended)  
   * Verify multi-GPU communication (NCCL)  
2. **Experiment 7: Tensor Parallelism**  
   * **TP=2 Config:**  
     * Load model with `tensor_parallel_size=2`  
     * Run 3 reps  
   * **TP=4 Config:**  
     * Load model with `tensor_parallel_size=4`  
     * Run 3 reps  
   * **Include Experiment 0 TP=1 data** from Pod 1 as baseline  
   * Save to `tensor_parallel_experiment.json`  
3. **Experiment 8: Expert Parallelism**  
   * **EP=2 Config:**  
     * Load model with `expert_parallel_size=2`  
     * Run 3 reps  
   * **EP=4 Config:**  
     * Load model with `expert_parallel_size=4`  
     * Run 3 reps  
   * **Include Experiment 0 EP=1 data** from Pod 1 as baseline  
   * Save to `expert_parallel_experiment.json`

**Output Files:**

* `tensor_parallel_experiment.json`  
* `expert_parallel_experiment.json`

---

## **Experiment-Specific Details**

### **Experiment 0: Reference Baseline**

**Variable:** None (establishes baseline)  
 **Purpose:**

1. Verify bit-exact reproducibility within identical setups  
2. Like all experiments, run it on both A100 and H100.  
3. Provide reference configuration for all subsequent experiments  
4. Validate extraction pipeline and data collection

**Configuration:**

* All parameters at "default" values  
* Will be reused as baseline comparison for Experiments 1-6

**Runs:**

* 3 repetitions per hardware  
* Verify L2=0 across repetitions (bit-exact reproducibility)  
* signals only from 3rd repetition for storage

## **Python Code Structure**

experiments/  
├── common/  
│   ├── \_\_init\_\_.py  
│   ├── model\_loader.py      \# Load Qwen3 with various configs  
│   ├── extraction.py         \# Extract hidden states, keys, logprobs  
│   ├── runner.py             \# Run inference with extraction  
│   └── json\_writer.py        \# Write to experiment JSON format  
│   └── json\_reader.py        \# Used to fetch reference measurement and store all the variant’s data in \#fused json. Yes, this is redundant across experiments, but makes later comparisons simpler.  
├── ablation_cross_hardware/  
│   ├── exp0\_reference.py 
│   ├── exp1\_batch\_size.py  
│   ├── exp2\_compile.py  
│   ├── exp3\_quantization.py  
│   ├── exp4\_attention.py  
│   ├── exp5\_concurrent\_streams.py  
│   ├── exp6\_tensor\_parallel.py  
│   ├── exp7\_expert\_parallel.py  
│   └── exp8\_cuda\_version.py  
└── analysis/  
    └── compare\_experiments.py

## **Analysis Framework**

### **Single Script for All Experiments**

\# analysis/compare\_experiments.py
```python
import json  
import numpy as np  
from scipy.spatial.distance import euclidean  
from pathlib import Path

class ExperimentAnalyzer:  
    def \_\_init\_\_(self, json\_path: str):  
        """Load experiment JSON."""  
        with open(json\_path) as f:  
            self.data \= json.load(f)  
          
        self.experiment\_type \= self.data\["experiment\_metadata"\]\["experiment\_type"\]  
        self.variable \= self.data\["experiment\_metadata"\]\["variable\_tested"\]  
      
    def compute\_l2\_distance(  
        self,  
        run1: dict,  
        run2: dict,  
        signal\_type: str \= "hidden\_states",  \# or "key\_vectors" or "logprobs"  
        layer: int \= 39  
    ) \-\> float:  
        """  
        Compute L2 distance between two runs for specified signal type.  
          
        Returns:  
            Average L2 distance across all decode steps and positions  
        """  
        distances \= \[\]  
          
        for step\_idx in range(len(run1\["decode\_steps"\])):  
            step1 \= run1\["decode\_steps"\]\[step\_idx\]  
            step2 \= run2\["decode\_steps"\]\[step\_idx\]  
              
            for pos in \["pos\_-3", "pos\_-2", "pos\_-1"\]:  
                vec1 \= np.array(step1\[signal\_type\]\[f"layer\_{layer}"\]\[pos\])  
                vec2 \= np.array(step2\[signal\_type\]\[f"layer\_{layer}"\]\[pos\])  
                  
                l2\_dist \= euclidean(vec1, vec2)  
                distances.append(l2\_dist)  
          
        return np.mean(distances)  
      
    def within\_config\_reproducibility(self, config\_id: str) \-\> dict:  
        """  
        Measure reproducibility within a single config (3 reps).  
        Should be bit-exact (L2 \= 0\) for deterministic setups.  
        """  
        runs \= \[r for r in self.data\["runs"\] if r\["config\_id"\] \== config\_id\]  
          
        if len(runs) \!= 3:  
            raise ValueError(f"Expected 3 reps for {config\_id}, found {len(runs)}")  
          
        \# Compare rep0 vs rep1, rep0 vs rep2, rep1 vs rep2  
        distances \= {  
            "rep0\_vs\_rep1": self.compute\_l2\_distance(runs\[0\], runs\[1\]),  
            "rep0\_vs\_rep2": self.compute\_l2\_distance(runs\[0\], runs\[2\]),  
            "rep1\_vs\_rep2": self.compute\_l2\_distance(runs\[1\], runs\[2\])  
        }  
          
        return {  
            "config\_id": config\_id,  
            "within\_config\_distances": distances,  
            "mean": np.mean(list(distances.values())),  
            "max": np.max(list(distances.values()))  
        }  
      
    def cross\_hardware\_baseline(self) \-\> dict:  
        """  
        Measure L2 distance between A100 and H100 for same variable value.  
        This is the "baseline noise" from hardware differences alone.  
        """  
        a100\_configs \= \[c for c in self.data\["configurations"\] if "A100" in c\["hardware"\]\]  
        h100\_configs \= \[c for c in self.data\["configurations"\] if "H100" in c\["hardware"\]\]  
          
        results \= {}  
          
        for a100\_cfg in a100\_configs:  
            \# Find matching H100 config (same variable value)  
            matching\_h100 \= \[  
                c for c in h100\_configs   
                if c\["variable\_value"\] \== a100\_cfg\["variable\_value"\]  
            \]  
              
            if not matching\_h100:  
                continue  
              
            h100\_cfg \= matching\_h100\[0\]  
              
            \# Get first rep from each (assuming reproducibility)  
            a100\_run \= \[r for r in self.data\["runs"\]   
                       if r\["config\_id"\] \== a100\_cfg\["config\_id"\]   
                       and r\["rep\_id"\] \== 0\]\[0\]  
            h100\_run \= \[r for r in self.data\["runs"\]   
                       if r\["config\_id"\] \== h100\_cfg\["config\_id"\]   
                       and r\["rep\_id"\] \== 0\]\[0\]  
              
            distance \= self.compute\_l2\_distance(a100\_run, h100\_run)  
              
            results\[f"{a100\_cfg\['variable\_value'\]}"\] \= {  
                "a100\_config": a100\_cfg\["config\_id"\],  
                "h100\_config": h100\_cfg\["config\_id"\],  
                "l2\_distance": distance  
            }  
          
        return results  
      
    def variable\_detectability(self, hardware: str \= "A100-80GB") \-\> dict:  
        """  
        Measure L2 distance when changing the variable within same hardware.  
        This is the "signal" we're trying to detect.  
        """  
        configs \= \[c for c in self.data\["configurations"\]   
                  if c\["hardware"\] \== hardware\]  
          
        \# Get baseline config (typically first one or smallest value)  
        baseline\_config \= configs\[0\]  
        baseline\_run \= \[r for r in self.data\["runs"\]  
                       if r\["config\_id"\] \== baseline\_config\["config\_id"\]  
                       and r\["rep\_id"\] \== 0\]\[0\]  
          
        results \= {}  
          
        for config in configs\[1:\]:  \# Skip baseline  
            test\_run \= \[r for r in self.data\["runs"\]  
                       if r\["config\_id"\] \== config\["config\_id"\]  
                       and r\["rep\_id"\] \== 0\]\[0\]  
              
            distance \= self.compute\_l2\_distance(baseline\_run, test\_run)  
              
            results\[config\["variable\_value"\]\] \= {  
                "config\_id": config\["config\_id"\],  
                "baseline\_value": baseline\_config\["variable\_value"\],  
                "test\_value": config\["variable\_value"\],  
                "l2\_distance": distance  
            }  
          
        return results  
      
    def cross\_hardware\_detectability(self) \-\> dict:  
        """  
        Can we detect variable changes even across different hardware?  
        Compare A100\_variant vs H100\_baseline.  
        """  
        \# Implementation similar to above  
        pass  
      
    def generate\_report(self) \-\> dict:  
        """  
        Generate complete analysis report for this experiment.  
        """  
        report \= {  
            "experiment": self.experiment\_type,  
            "variable": self.variable,  
              
            \# Question 1: Is the setup reproducible?  
            "reproducibility": {  
                cfg\["config\_id"\]: self.within\_config\_reproducibility(cfg\["config\_id"\])  
                for cfg in self.data\["configurations"\]  
            },  
              
            \# Question 2: What's the cross-hardware baseline?  
            "cross\_hardware\_baseline": self.cross\_hardware\_baseline(),  
              
            \# Question 3: Can we detect variable changes within hardware?  
            "within\_hardware\_detectability": {  
                "A100": self.variable\_detectability("A100-80GB"),  
                "H100": self.variable\_detectability("H100")  
            },  
              
            \# Question 4: Can we detect across hardware?  
            "cross\_hardware\_detectability": self.cross\_hardware\_detectability()  
        }  
          
        \# Compute signal-to-noise ratios  
        baseline\_noise \= np.mean(\[  
            v\["l2\_distance"\]   
            for v in report\["cross\_hardware\_baseline"\].values()  
        \])  
          
        for hw in \["A100", "H100"\]:  
            detect \= report\["within\_hardware\_detectability"\]\[hw\]  
            for var\_value, data in detect.items():  
                data\["snr"\] \= data\["l2\_distance"\] / baseline\_noise if baseline\_noise \> 0 else float('inf')  
          
        return report

def compare\_all\_experiments():  
    """  
    Run analysis on all 8 experiment files.  
    """  
    experiment\_files \= \[  
        "batch\_size\_experiment.json",  
        "compile\_experiment.json",  
        "quantization\_experiment.json",  
        "attention\_experiment.json",  
        "concurrent\_streams\_experiment.json",  
        "tensor\_parallel\_experiment.json",  
        "expert\_parallel\_experiment.json",  
        "cuda\_version\_experiment.json"  
    \]  
      
    all\_reports \= {}  
      
    for exp\_file in experiment\_files:  
        if not Path(exp\_file).exists():  
            print(f"Skipping {exp\_file} (not found)")  
            continue  
          
        print(f"Analyzing {exp\_file}...")  
        analyzer \= ExperimentAnalyzer(exp\_file)  
        report \= analyzer.generate\_report()  
        all\_reports\[analyzer.experiment\_type\] \= report  
      
    \# Create summary table  
    summary \= create\_summary\_table(all\_reports)  
      
    return all\_reports, summary

def create\_summary\_table(reports: dict) \-\> dict:  
    """  
    Create comparative summary across all experiments.  
      
    Returns table showing:  
    \- Variable tested  
    \- Cross-hardware baseline L2  
    \- Within-hardware detection L2 (min, max)  
    \- Signal-to-noise ratio  
    \- Detectability classification (Strong/Moderate/Weak)  
    """  
    summary \= \[\]  
      
    for exp\_type, report in reports.items():  
        baseline\_l2 \= np.mean(\[  
            v\["l2\_distance"\]   
            for v in report\["cross\_hardware\_baseline"\].values()  
        \])  
          
        \# Get max detection signal across both hardware and all variable values  
        max\_signal \= 0  
        for hw in \["A100", "H100"\]:  
            if hw in report\["within\_hardware\_detectability"\]:  
                hw\_signals \= \[  
                    data\["l2\_distance"\]  
                    for data in report\["within\_hardware\_detectability"\]\[hw\].values()  
                \]  
                max\_signal \= max(max\_signal, max(hw\_signals) if hw\_signals else 0\)  
          
        snr \= max\_signal / baseline\_l2 if baseline\_l2 \> 0 else float('inf')  
          
        \# Classify detectability  
        if snr \> 10:  
            classification \= "Strong"  
        elif snr \> 3:  
            classification \= "Moderate"  
        elif snr \> 1:  
            classification \= "Weak"  
        else:  
            classification \= "Not Detectable"  
          
        summary.append({  
            "experiment": exp\_type,  
            "variable": report\["variable"\],  
            "baseline\_l2": baseline\_l2,  
            "max\_signal\_l2": max\_signal,  
            "snr": snr,  
            "detectability": classification  
        })  
      
    return summary

if \_\_name\_\_ \== "\_\_main\_\_":  
    reports, summary \= compare\_all\_experiments()  
      
    \# Print summary table  
    print("\\n" \+ "="\*80)  
    print("DETECTABILITY SUMMARY")  
    print("="\*80)  
    print(f"{'Experiment':\<20} {'Variable':\<20} {'Baseline L2':\<15} {'Max Signal L2':\<15} {'SNR':\<10} {'Detection':\<15}")  
    print("-"\*80)  
      
    for row in summary:  
        print(f"{row\['experiment'\]:\<20} {row\['variable'\]:\<20} {row\['baseline\_l2'\]:\<15.4f} "  
              f"{row\['max\_signal\_l2'\]:\<15.4f} {row\['snr'\]:\<10.2f} {row\['detectability'\]:\<15}")  
      
    \# Save detailed reports  
    with open("analysis\_reports.json", "w") as f:  
        json.dump(reports, f, indent=2)  
      
    with open("summary.json", "w") as f:  
        json.dump(summary, f, indent=2)  
      
    print("\\nDetailed reports saved to analysis\_reports.json")  
    print("Summary saved to summary.json")

```

## **Critical Implementation Notes**

### **1\. Reproducibility Verification**

Every experiment must verify bit-exact reproducibility within identical setups:

* Compare 3 reps within each config  
* If L2 \> 0, investigate non-determinism source  
* Document any non-deterministic behaviors, immediately stop and report back to me.

### **2\. CUDA Stream Isolation (Experiment 5\)**

\# Concurrent stream implementation  
stream0 \= torch.cuda.Stream()  \# Main inference  
stream1 \= torch.cuda.Stream()  \# Concurrent work

with torch.cuda.stream(stream0):  
    \# Run model inference here  
    outputs \= model.generate(...)

with torch.cuda.stream(stream1):  
    \# Run synthetic workload  
    \# Must be substantial: target \~50% GPU utilization  
    for \_ in range(large\_number):  
        dummy \= torch.randn(8192, 8192, device='cuda') @ torch.randn(8192, 8192, device='cuda')

torch.cuda.synchronize()

### **3\. Prompt Construction**

All experiments use identical prompt to ensure comparability, except batch\_size, where we add batch neighbours to the reference sequence0 (same as in exp0):

I recommend using Qwen’s standard chat template. Prompt should be a text pulled from a long pdf, cut to 8k token length. Same with batch neighbours, but use different pdfs. This text is simply combined with the prompt: “Provide a summary” or whatever.

### **4\. Memory Management**

Between runs:

torch.cuda.empty\_cache()  
import gc  
gc.collect()

### **5\. Timing Verification**

While not primary metric, track runtime:

import time  
start \= time.time()  
\# ... run inference ...  
runtime \= time.time() \- start

Include in JSON for timing forensics analysis. Average of three repetitions, and variance from average.

---

## **Validation Checklist**

Before running production experiments:

* \[ \] Model loads correctly on target hardware  
* \[ \] Extraction pipeline captures all required signals  
* \[ \] JSON schema validates against specification  
* \[ \] Bit-exact reproducibility confirmed (3 reps, L2=0)  
* \[ \] Prompt generates exactly 6k tokens from the pdf, plus the summarization request  
* \[ \] CUDA version switching works (Experiment 5\)  
* \[ \] Multi-GPU setup functional (Experiments 7-8)  
* \[ \] Storage capacity sufficient (when downloading three models, the pod’s disk space needs to suffice)
