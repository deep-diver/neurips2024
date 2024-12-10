---
title: Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern
  Generators
summary: Bio-inspired CPG-PE enhances spiking neural networks' sequential modeling
  by efficiently encoding position information, outperforming conventional methods
  across various tasks.
categories: []
tags:
- "\U0001F3E2 Microsoft Research"
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} kQMyiDWbOG {{< /keyword >}}
{{< keyword icon="writer" >}} Changze Lv et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=kQMyiDWbOG" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93894" target="_blank" >}}
↗ NeurIPS Proc.
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=kQMyiDWbOG&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/kQMyiDWbOG/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Spiking Neural Networks (SNNs), while energy-efficient and biologically plausible, struggle with sequential tasks due to ineffective positional encoding (PE).  Existing PE methods are either not compatible with SNNs' spike-based communication or lack hardware-friendliness.  This paper tackles this challenge.

The authors propose a novel PE technique, CPG-PE, drawing inspiration from the Central Pattern Generators (CPGs) in the human brain.  They demonstrate that CPG-PE outperforms existing methods across various sequential tasks, such as time-series forecasting and natural language processing.  Furthermore, the analysis of CPG-PE elucidates the underlying mechanisms of neural computation and suggests a new direction for designing more effective and biologically realistic SNNs. **CPG-PE is shown to be highly effective and addresses a critical limitation in SNN research.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} CPG-PE, a novel positional encoding technique inspired by Central Pattern Generators, significantly improves SNN performance. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} SNNs with CPG-PE outperform conventional methods in time-series forecasting, natural language processing, and image classification. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} CPG-PE provides a biologically plausible and hardware-friendly solution for positional encoding in SNNs. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working with spiking neural networks (SNNs). It directly addresses the challenge of creating effective positional encoding for SNNs, a critical limitation hindering their application in various sequential tasks. The proposed CPG-PE method offers a biologically plausible and computationally efficient solution, potentially opening new avenues for SNN development and applications.  Its consistent performance gains across diverse domains showcase its broad applicability and potential to advance SNN research significantly.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/kQMyiDWbOG/figures_2_1.jpg)

> This figure compares three different positional encoding (PE) methods for neural networks: (a) Traditional PE in ANN Transformers using sine and cosine functions; (b) Existing PE methods in spike-based Transformers, which have limitations in terms of uniqueness and spike-form representation; (c) The proposed CPG-PE method, which leverages central pattern generators (CPGs) for a biologically-inspired and more effective PE tailored for spiking neural networks (SNNs). Finally, (d) shows the improved performance of CPG-PE compared to existing methods across different tasks, including time-series forecasting, text classification, and image classification.





![](https://ai-paper-reviewer.com/kQMyiDWbOG/tables_6_1.jpg)

> This table presents the results of time series forecasting experiments using various models (ANNs and SNNs with and without positional encoding).  It shows the R-squared (R2) and Root Relative Squared Error (RSE) values for different prediction lengths (6, 24, 48, 96 time steps) across four benchmark datasets (Metr-la, Pems-bay, Solar, Electricity).  The best performing SNN models for each dataset and prediction length are highlighted in bold. The table helps compare the performance of SNNs using CPG-PE (the proposed method) against other SNNs and their corresponding ANN counterparts.





### In-depth insights


#### CPG-PE: A Novel PE
The proposed CPG-PE (Central Pattern Generator-based Positional Encoding) offers a **biologically-inspired** approach to positional encoding in Spiking Neural Networks (SNNs).  Unlike traditional methods, CPG-PE leverages the rhythmic, patterned output of CPGs—neural circuits producing rhythmic patterns without rhythmic input—to generate spike-based positional information. This approach is **inherently compatible with SNNs** and neuromorphic hardware, addressing the challenge of creating effective and hardware-friendly positional encoding strategies in SNNs.  By mathematically demonstrating the relationship between sinusoidal positional encoding and CPG membrane potential dynamics, CPG-PE provides a **strong theoretical foundation**. The method's effectiveness is further validated through extensive experiments across diverse domains, showcasing superior performance compared to conventional counterparts. The **uniqueness of CPG-PE lies in its biologically plausible mechanism and hardware-friendliness**, making it a significant advancement in enabling SNNs to effectively handle sequential data.

#### SNN Sequential Tasks
Spiking Neural Networks (SNNs), while energy-efficient and biologically plausible, face challenges in handling sequential data.  **Effective positional encoding (PE)** is crucial for SNNs to understand the order of information in sequences, but methods adapted from Artificial Neural Networks (ANNs) often don't translate well to the spiking domain. This is because ANN PE techniques rely on continuous representations, which are not naturally compatible with the discrete nature of spikes. The paper explores the use of Central Pattern Generators (CPGs) as a biologically-inspired approach to PE for SNNs.  **CPGs generate rhythmic patterns without rhythmic input**, offering a potential solution to the challenges of creating hardware-friendly, spike-based positional information.  The proposed CPG-PE method shows promising results across various sequential tasks, outperforming conventional SNN approaches.  The connection between CPGs and sinusoidal PE commonly used in ANNs is highlighted, strengthening the biological plausibility and mathematical foundation of the proposed technique.  **However, limitations exist in applying CPG-PE directly to image data**, which requires adaptations like patch-based processing. Future research may focus on hybrid models combining CPG-PE with convolutional layers to enhance performance on spatial data.

#### CPG & PE Analogy
The core idea of the CPG & PE analogy lies in drawing a parallel between Central Pattern Generators (CPGs) in neuroscience and Positional Encoding (PE) in deep learning.  **CPGs, neural circuits generating rhythmic patterns without rhythmic input, are analogous to PE, which adds temporal or spatial information to sequential data**. The authors suggest that the commonly used sinusoidal PE in transformers is mathematically a specific instance of the membrane potential dynamics within a particular CPG model. This analogy is not just a superficial similarity but proposes a biologically plausible mechanism for positional encoding within spiking neural networks (SNNs).  It bridges the gap between biologically inspired models and modern deep learning techniques, suggesting that **CPGs, which are a cornerstone of biological rhythm generation, could provide a foundation for building more efficient and biologically realistic positional encoders for SNNs.** By leveraging the inherent rhythmic properties of CPGs, the authors propose a novel positional encoding scheme (CPG-PE) specifically tailored for SNNs, demonstrating its effectiveness across various sequential tasks. This connection offers a deeper understanding of neural computation and suggests new avenues for developing more energy-efficient and biologically plausible artificial neural networks.  The strength of the analogy rests on its predictive power: it proposes a novel approach to PE and offers valuable insights into fundamental principles of neural computation.

#### Hardware-Friendly SNN
A hardware-friendly spiking neural network (SNN) design is crucial for efficient and scalable neuromorphic computing.  **Energy efficiency** is a primary advantage of SNNs over traditional artificial neural networks (ANNs), but realizing this advantage requires specialized hardware. This necessitates that the SNN architecture and training algorithms be adapted for efficient implementation in hardware.  **Event-driven computation**, inherent in SNNs, allows for significant power savings as neurons only compute and communicate when they receive a spike.  However, this event-driven nature can pose challenges for mapping SNNs onto existing hardware architectures.  **Memory access** and communication are critical bottlenecks that must be minimized.  Consequently,  **specialized memory structures** that enable efficient storage and retrieval of spiking activity are essential.  Moreover, the choice of neuron model directly impacts hardware efficiency, with simpler models like the leaky integrate-and-fire (LIF) neuron being preferable. Furthermore, training algorithms must consider the hardware constraints.  **Approaches like surrogate gradient methods** make it possible to train SNNs effectively using standard backpropagation while also being compatible with hardware limitations.   Ultimately, a successful hardware-friendly SNN design hinges on the synergistic integration of these aspects: an efficient architecture, appropriate neuron models, optimized training strategies, and supporting hardware infrastructure.

#### Future Research
The paper's 'Future Research' section hints at several promising avenues.  **Extending CPG-PE's applicability to image data** is crucial, possibly through hybrid models combining CPG-PE with convolutional layers to handle both spatial and sequential information effectively.  This would involve exploring adaptive patch sizes or dynamically adjusting the patching mechanism.  **Developing learnable relative positional encodings for SNNs** is another key area.  These encodings must maintain the spike-form nature of SNNs while ensuring positional uniqueness. This could improve the model's ability to handle temporal dynamics in various data types.  Furthermore, **investigating the adaptability of CPGs in the brain** is suggested, exploring whether CPGs can learn and adapt to data like learnable PEs in ANNs. This neuroscience-inspired direction could significantly enhance the theoretical understanding of SNNs and potentially unlock new algorithmic improvements.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/kQMyiDWbOG/figures_4_1.jpg)

> This figure illustrates the concept of Central Pattern Generators (CPGs). Panel (a) shows a schematic diagram of two CPG neurons mutually inhibiting each other through spiking activity. Panel (b) presents spike trains of four CPG neurons, depicting the rhythmic and coordinated spiking pattern produced by a CPG network.  The curves represent the neurons' membrane potential over time, with vertical lines indicating spike events.


![](https://ai-paper-reviewer.com/kQMyiDWbOG/figures_5_1.jpg)

> This figure illustrates the implementation of the CPG-PE method in SNNs.  It shows how the positional encoding (CPG-PE) is integrated into the network, maintaining binary spike signals.  A linear layer adjusts the dimensionality after concatenation of the input and encoded signals, preparing the data for a spiking neuron layer.


![](https://ai-paper-reviewer.com/kQMyiDWbOG/figures_8_1.jpg)

> This figure compares different positional encoding methods for neural networks, including traditional methods used in ANN transformers and spike transformers, and the novel CPG-PE method proposed in the paper.  Panel (a) shows the sinusoidal positional encoding in ANN Transformers. Panel (b) illustrates the relative positional encoding previously used in spike-based transformers. Panel (c) presents the proposed CPG-PE (Central Pattern Generator Positional Encoding) method. Finally, Panel (d) demonstrates the superior performance of CPG-PE across various tasks like time-series forecasting, text classification, and image classification, highlighting its effectiveness as a positional encoding technique for spiking neural networks (SNNs).


![](https://ai-paper-reviewer.com/kQMyiDWbOG/figures_17_1.jpg)

> This figure illustrates how the proposed CPG-PE method is integrated into spiking neural networks (SNNs). It shows the process of positional encoding using CPG-PE, followed by concatenation with the input spike matrix and a linear layer to map the feature dimension back to the original size.  The figure highlights that the entire process maintains the spike format for hardware-friendly compatibility.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/kQMyiDWbOG/tables_6_2.jpg)
> This table presents the accuracy results achieved by various Spiking Neural Network (SNN) models and a fine-tuned BERT model on six different text classification benchmark datasets.  The table compares the performance of Spikformer models with different positional encoding methods (no positional encoding, random positional encoding, float positional encoding, and CPG-PE). The results show that the Spikformer model with CPG-PE achieves the best accuracy, surpassing other SNN models and approaching the performance of the BERT model. The results are averaged across five random seeds, showcasing the consistency of CPG-PE's effectiveness.

![](https://ai-paper-reviewer.com/kQMyiDWbOG/tables_7_1.jpg)
> This table presents the results of image classification experiments conducted on three benchmark datasets: CIFAR10, CIFAR10-DVS, and CIFAR100.  It compares the performance of various Spikformer models with different positional encoding methods (no positional encoding, random positional encoding, float PE, RPE, and the proposed CPG-PE).  The table shows the number of parameters for each model and its accuracy on each dataset. The best performing models are highlighted.

![](https://ai-paper-reviewer.com/kQMyiDWbOG/tables_13_1.jpg)
> This table presents the experimental results for time-series forecasting on four benchmark datasets (Metr-la, Pems-bay, Solar, and Electricity) using various prediction lengths (6, 24, 48, 96).  It compares the performance of different Spiking Neural Networks (SNNs) with and without positional encoding (PE), including both conventional ANN-based models for comparison.  The table shows R-squared (R2) and Root Relative Squared Error (RSE) values for each model configuration, indicating the impact of different PE methods on the prediction accuracy. The best results for SNNs are highlighted in bold.

![](https://ai-paper-reviewer.com/kQMyiDWbOG/tables_17_1.jpg)
> This table presents the results of image classification experiments conducted on the ImageNet dataset using three different Spikformer models: one without positional encoding, one with relative positional encoding (RPE), and one with the proposed CPG-PE. The table shows the number of parameters (in millions) and the accuracy achieved by each model.  The results demonstrate that the CPG-PE model outperforms the other two models, achieving higher accuracy with a comparable number of parameters. This highlights the effectiveness of the CPG-PE method for image classification tasks.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kQMyiDWbOG/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}