<div align="center">

<h2>BiMoE: Brain-Inspired Experts for EEG-Dominant Affective State Recognition</a></h2>
This work has been submitted anonymously to ICME 2026.<sup></sup>

<p align="center">
<img src="https://anonymous.4open.science/r/BiMoE-78D7/BiMoE_framework.png" width=75% height=75% 
class="center">
</p>

##  Introduction 

<div align="left">
Electroencephalogram (EEG)-based Multimodal Sentiment Analysis (MSA) plays a key role in building robust brain–computer interface (BCI) systems.
However, current methods still face three main limitations: they often treat EEG signals as uniform, overlooking the region-specific mechanisms of emotion processing; they lack effective ways to capture both local and global spatiotemporal features of EEG; and they struggle to fully integrate EEG with complementary peripheral physiological signals (PPS).
To address these issues, we propose BiMoE—a Brain-Inspired Mixture of Experts framework. BiMoE incorporates brain-topology-aware partitioning to model region-specific EEG dynamics, where each expert uses a dual-stream encoder to extract local and global contextual features. A separate expert processes PPS with multi-scale large-kernel convolutions. All experts are dynamically integrated through adaptive routing and optimized with a joint loss that encourages balanced, diverse, and accurate collaboration.

## News  
**[2025/12]** Submit and [**Open source**](https://anonymous.4open.science/r/BiMoE-78D7)

## Datasets
