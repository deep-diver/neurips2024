---
title: "Neuronal Competition Groups with Supervised STDP for Spike-Based Classification"
summary: "Neuronal Competition Groups (NCGs) enhance supervised STDP training in spiking neural networks by promoting balanced competition and improved class separation, resulting in significantly higher classi..."
categories: ["AI Generated", ]
tags: ["Machine Learning", "Deep Learning", "🏢 Univ. Lille",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} GeE5qF6ICg {{< /keyword >}}
{{< keyword icon="writer" >}} Gaspard Goupy et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=GeE5qF6ICg" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/GeE5qF6ICg" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/GeE5qF6ICg/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Spiking Neural Networks (SNNs) are energy-efficient alternatives to traditional Artificial Neural Networks. However, training SNNs using local learning rules like Spike Timing-Dependent Plasticity (STDP) remains a challenge. While unsupervised STDP with Winner-Takes-All (WTA) competition is effective for feature extraction, applying it to supervised classification poses difficulties due to unbalanced competition. This limits the ability of SNNs to accurately learn various class-specific patterns.

This research introduces Neuronal Competition Groups (NCGs), which improve the classification capabilities of SNNs trained with supervised STDP. NCGs use a two-compartment threshold mechanism for balanced competition regulation within each class. Experimental results on image datasets demonstrate that NCGs significantly improve the accuracy of state-of-the-art supervised STDP rules by learning multiple patterns per class. This makes it possible to achieve comparable accuracy to fully-supervised methods with improved efficiency.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Neuronal Competition Groups (NCGs) improve spiking neural network classification accuracy by promoting the learning of various patterns per class. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The proposed two-compartment threshold mechanism in NCGs ensures balanced competition and fair decision-making during training. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} NCGs significantly improve the accuracy of state-of-the-art supervised STDP rules on image recognition datasets, achieving performance comparable to fully-supervised methods with greater efficiency. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **spike-based neural networks** and **neuromorphic computing**. It addresses the challenge of efficient and balanced learning in supervised scenarios, offering a novel architecture and competition regulation mechanism.  This improves accuracy and opens avenues for building more efficient and powerful spiking neural networks for various applications. The proposed methods are relevant to the current trend of using local learning rules in SNNs and minimize the dependence on labeled data.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/GeE5qF6ICg/figures_4_1.jpg)

> 🔼 This figure illustrates the architecture of a spiking classification layer using Neuronal Competition Groups (NCGs). Each class has its own NCG, a group of neurons that compete for activation using intra-class Winner-Takes-All (WTA) competition.  Lateral inhibition ensures that only the first neuron to spike in the group wins and receives a weight update, based on a temporal error signal. A competition regulation mechanism, based on adaptive thresholds, ensures fair competition among neurons within each NCG. This method allows for learning of multiple class-specific patterns per class.
> <details>
> <summary>read the caption</summary>
> Figure 1: Spiking classification layer with Neuronal Competition Groups (NCGs). In this layer, each class is mapped to an NCG and the prediction is based on the first spike. An NCG is a group of M neurons connected with lateral inhibition to enable intra-class WTA competition: the first neuron to fire inhibits the other ones and undergoes a weight update based on a temporal error (which depends on the learning rule considered). The sign and amplitude of the error pushes neurons to fire earlier (positive sign) or later (negative sign). Competition regulation occurs only within the NCG mapped to the class of the input sample to ensure balanced competition among neurons on samples of their class. NCGs improve the classification capabilities of a layer by promoting the learning of various patterns per class.
> </details>





![](https://ai-paper-reviewer.com/GeE5qF6ICg/tables_7_1.jpg)

> 🔼 This table presents the accuracy results of different supervised spike-timing-dependent plasticity (STDP) methods applied to spiking classification layers.  The classification layers are placed on top of two different unsupervised feature extraction networks (STDP-CSNN and SoftHebb-CNN).  The table shows the mean and standard deviation of accuracy, achieved across 10-fold cross-validation, for various methods (R-STDP, SSTDP, S2-STDP) and variations using the proposed Neuronal Competition Group (NCG) architecture. The number of neurons per class is varied to evaluate the impact on performance.  The results illustrate the improvement in accuracy achieved through the use of the NCG architecture, especially when combined with the S2-STDP training rule.
> <details>
> <summary>read the caption</summary>
> Table 1: Accuracy of spiking classification layers trained with STDP-based methods, on top of Hebbian-based unsupervised feature extractors.
> </details>





### In-depth insights


#### Supervised STDP
Supervised Spike-Timing-Dependent Plasticity (STDP) is a crucial advancement in spiking neural network (SNN) training, addressing the limitations of traditional backpropagation.  **Instead of relying on global error signals, supervised STDP leverages local learning rules**, modifying synaptic weights based on the precise timing of pre- and post-synaptic spikes, along with an error signal. This approach offers significant advantages for neuromorphic hardware implementation due to its inherent locality and biological plausibility.  Several variations of supervised STDP have been developed, each with its own strengths and weaknesses concerning accuracy, efficiency, and compatibility with different coding schemes.  A key challenge in supervised STDP, particularly when combined with winner-takes-all (WTA) competition, lies in achieving balanced learning across neurons within a class, preventing the dominance of certain neurons and ensuring the learning of diverse patterns per class. This necessitates sophisticated competition regulation mechanisms to promote robust and effective learning. The exploration and development of robust and efficient supervised STDP algorithms are central to realizing the potential of SNNs for energy-efficient and biologically-inspired computing.

#### NCG Architecture
The Neuronal Competition Group (NCG) architecture is a novel approach to enhance spiking neural network (SNN) classification.  **NCGs improve classification by promoting the learning of diverse patterns within each class.**  Instead of a single neuron per class, an NCG comprises multiple neurons that compete internally (intra-class WTA), learning distinct features. A key innovation is the **two-compartment threshold mechanism**, regulating this competition.  One threshold governs decision-making during inference, ensuring fairness; the other adapts during training, balancing competition among neurons. This dual-threshold system prevents unfair dominance by a single neuron, leading to **improved accuracy and better generalization**. The NCG architecture is combined with state-of-the-art supervised STDP rules, demonstrating significant accuracy gains across diverse image recognition benchmarks.

#### Competition Regulation
The sub-heading 'Competition Regulation' highlights a critical aspect of the proposed Neuronal Competition Group (NCG) architecture.  Standard Winner-Takes-All (WTA) mechanisms in supervised learning scenarios often lead to **imbalanced competition**, where a single neuron dominates, hindering the learning of diverse patterns within a class. The authors ingeniously address this by introducing a **two-compartment threshold system**.  Each neuron possesses a fixed threshold for decision-making and a dynamic threshold for weight updates, activated only for samples of its class. This clever design promotes balanced intra-class competition by ensuring that neurons aren't unfairly disadvantaged due to varying thresholds.  The **adaptive threshold adjustment** further refines this balance, preventing a single neuron from monopolizing updates, resulting in a more robust and effective learning process. This mechanism is crucial for improved class separation and overall classification accuracy, as demonstrated by experimental results, making it a significant contribution to supervised spike-based learning.

#### Experimental Results
The Experimental Results section of a research paper is crucial for demonstrating the validity and effectiveness of the proposed methods.  A strong presentation would begin with a clear description of the experimental setup, including datasets used, evaluation metrics, and the parameters of the methods. It's vital to show a rigorous comparison with existing state-of-the-art techniques, demonstrating a clear improvement.  **Statistical significance** should be rigorously established using appropriate methods, and the results should be clearly visualized via graphs and tables.  **Ablation studies**, systematically removing components of the model, help isolate the contribution of each part.  **Error analysis** can further enhance understanding of strengths and weaknesses.  Finally, a discussion of limitations and potential future work based on the observations would strengthen the results section.

#### Future Research
Future research directions stemming from this work could explore several promising avenues.  **Extending the NCG architecture to multi-layer networks** is crucial, requiring investigation into effective inter-layer competition mechanisms.  The current single-layer implementation provides a strong foundation, but a multi-layer approach would unlock the potential for processing more complex data and achieving significantly improved performance.  Further exploration of the **interaction between NCGs and diverse STDP learning rules** beyond the ones tested here (R-STDP, SSTDP, S2-STDP) should yield valuable insights into optimized learning strategies. Investigating **alternative competition regulation mechanisms** that offer enhanced balance and efficiency while minimizing computational overhead presents another exciting challenge.  Finally, a comprehensive examination of the **generalizability of NCGs across various datasets and tasks**, moving beyond image classification to other domains such as time series analysis or natural language processing, would solidify its practical applicability and broaden its impact.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/GeE5qF6ICg/figures_8_1.jpg)

> 🔼 This figure compares the number of weight updates received by neurons in class 0 trained with S2-STDP+NCG, both with and without competition regulation, over 20 epochs.  The data is from the CIFAR-10 dataset, and the features used were extracted using STDP-CSNN.  Neurons n1 through n4 are classified as 'target' neurons, while neuron n5 is a 'non-target' neuron.  The graph visually demonstrates the impact of competition regulation on balanced weight updates across neurons within a class.
> <details>
> <summary>read the caption</summary>
> Figure 2: Number of weight updates per epoch received by the neurons of class 0 trained with S2-STDP+NCG, with and without competition regulation, on CIFAR-10. n₁ to n₄ are labeled as target neurons and n₅ is labeled as non-target. The features are extracted with STDP-CSNN.
> </details>



![](https://ai-paper-reviewer.com/GeE5qF6ICg/figures_8_2.jpg)

> 🔼 This figure displays the impact of competition regulation on weight updates per epoch for neurons in a class (class 0 here).  The left panel shows the updates without competition regulation, demonstrating unbalanced updates, with one neuron receiving the vast majority. The right panel shows updates with competition regulation, resulting in more balanced weight updates across neurons.
> <details>
> <summary>read the caption</summary>
> Figure 2: Number of weight updates per epoch received by the neurons of class 0 trained with S2-STDP+NCG, with and without competition regulation, on CIFAR-10. n₁ to n₄ are labeled as target neurons and n₅ is labeled as non-target. The features are extracted with STDP-CSNN.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/GeE5qF6ICg/tables_8_1.jpg)
> 🔼 This table presents the classification accuracy results achieved using different supervised spike-timing-dependent plasticity (STDP) training methods on a spiking classification layer. The methods are evaluated using two different unsupervised feature extractors: STDP-CSNN and SoftHebb-CNN.  The table compares the performance variations across different datasets (MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100) with varying numbers of neurons per class in the classification layer.  The impact of the proposed Neuronal Competition Group (NCG) architecture is assessed by comparing results with and without NCGs, alongside other modifications like competition regulation mechanisms and neuron labeling.
> <details>
> <summary>read the caption</summary>
> Table 1: Accuracy of spiking classification layers trained with STDP-based methods, on top of Hebbian-based unsupervised feature extractors.
> </details>

![](https://ai-paper-reviewer.com/GeE5qF6ICg/tables_9_1.jpg)
> 🔼 This table presents the accuracy results of different supervised Spike-Timing-Dependent Plasticity (STDP) methods applied to spiking classification layers.  The methods are compared using two different unsupervised feature extractors (STDP-CSNN and SoftHebb-CNN) and four image datasets of varying complexity (MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100).  The table shows the impact of using the proposed Neuronal Competition Group (NCG) architecture on the accuracy of two state-of-the-art STDP rules (SSTDP and S2-STDP) and the performance of a commonly used rule (R-STDP).  The number of neurons per class is varied to demonstrate the effect of the NCG architecture on different numbers of neurons within the classification layer.
> <details>
> <summary>read the caption</summary>
> Table 1: Accuracy of spiking classification layers trained with STDP-based methods, on top of Hebbian-based unsupervised feature extractors.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GeE5qF6ICg/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}