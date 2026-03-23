<div align="center">

<h2>BiMoE: Brain-Inspired Experts for EEG-Dominant Affective State Recognition</a></h2>
This work has been submitted anonymously to ICME 2026.<sup></sup>

<div align="left">

##  Introduction 

EEG-based Multimodal Sentiment Analysis plays a key role in building robust brain–computer interface (BCI) systems.
However, current methods still face three main limitations: they often treat EEG signals as uniform, overlooking the region-specific mechanisms of emotion processing; they lack effective ways to capture both local and global spatiotemporal features of EEG; and they struggle to fully integrate EEG with complementary peripheral physiological signals.
To address these issues, we propose BiMoE—a Brain-Inspired Mixture of Experts framework. BiMoE incorporates brain-topology-aware partitioning to model region-specific EEG dynamics, where each expert uses a dual-stream encoder to extract local and global contextual features. A separate expert processes PPS with multi-scale large-kernel convolutions. All experts are dynamically integrated through adaptive routing and optimized with a joint loss that encourages balanced, diverse, and accurate collaboration.

<p align="center">
<img src="https://anonymous.4open.science/r/BiMoE-78D7/BiMoE_framework.png" width=75% height=75% 
class="center">
</p>

## News  
**[2025/12]** Submit and [**Open source**](https://anonymous.4open.science/r/BiMoE-78D7)
**[2026/03]** Accept by _ICME 2026_.

## Datasets
We conducted extensive experiments on the [**DEAP**]([https://ieeexplore.ieee.org/abstract/document/5871728]) and [**DREAMER**]([https://ieeexplore.ieee.org/abstract/document/7887697]).

Due to the privacy policy, you need to apply for the dataset through the link, which is very simple.

DEAP: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/

DREAMER: https://zenodo.org/records/546113

After download datastes, replace the **data_path** and **save_path**, run _"save_deap.py"_ and _"save_dreamer.py"_.

Run _"load_deap.py"_ and _"load_dreamer.py"_ to perform data set preprocessing.

## Run
Run _"BiMoE_deap.py"_ and _"BiMoE_dreamer.py"_.

