---
title: "FEEL-SNN: Robust Spiking Neural Networks with Frequency Encoding and Evolutionary Leak Factor"
summary: "FEEL-SNN enhances spiking neural network robustness by mimicking biological visual attention and adaptive leak factors, resulting in improved resilience against noise and attacks."
categories: ["AI Generated", ]
tags: ["AI Theory", "Robustness", "🏢 College of Computer Science and Technology, Zhejiang University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} TuCQdBo4NC {{< /keyword >}}
{{< keyword icon="writer" >}} Mengting Xu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=TuCQdBo4NC" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/TuCQdBo4NC" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=TuCQdBo4NC&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/TuCQdBo4NC/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Spiking Neural Networks (SNNs), inspired by the biological nervous system, are promising but suffer from vulnerability to noise and adversarial attacks.  Current methods often lack theoretical support and use simplified simulations of biological neurons, limiting their generalizability.  This paper addresses these issues by proposing a unified framework for SNN robustness.  

The paper introduces FEEL-SNN, a novel approach that leverages **Frequency Encoding (FE)** to mimic selective visual attention and **Evolutionary Leak factor (EL)** to simulate adaptive membrane potential leaks in biological neurons.  FE filters out unwanted noise based on frequencies, while EL allows each neuron to dynamically adjust its leak factor to ensure optimal robustness at different times. This combined approach significantly improves SNN robustness against various noise types and attacks, showing higher accuracy and resilience in rigorous experimental evaluations.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A unified theoretical framework for SNN robustness was proposed, suggesting that improved encoding and adaptive leak factors can enhance SNN robustness. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The FEEL-SNN model, employing frequency encoding (FE) and an evolutionary leak factor (EL), significantly improved SNN robustness against various types of noise and attacks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Experimental results confirmed the efficacy of both FE and EL methods, either independently or in conjunction with existing robustness enhancement techniques. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it offers a novel approach to enhancing the robustness of spiking neural networks (SNNs) by directly addressing the limitations of existing methods.  **The unified theoretical framework presented provides valuable insights into SNN robustness constraints**, paving the way for the development of more resilient and reliable SNNs across various applications.  **The proposed FEEL-SNN architecture, incorporating frequency encoding and an evolutionary leak factor, is a significant advancement in the field, offering a more bio-inspired and effective solution to improve SNN resilience against noise and attacks.** This work is highly relevant to the current research trend focusing on improving the robustness of AI systems, contributing to broader efforts to develop more secure and reliable AI applications.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/TuCQdBo4NC/figures_1_1.jpg)

> 🔼 This figure shows two key biological mechanisms that inspire the FEEL-SNN model.  Panel (a) illustrates selective visual attention, where the brain focuses on specific frequencies of visual information, effectively filtering out irrelevant details. Panel (b) depicts the non-fixed membrane potential leak in biological neurons.  The membrane potential leak in biological neurons isn't constant; it changes based on internal and external factors like ion concentration differences across the cell membrane. This variability enables neurons to better adapt to and handle noise.
> <details>
> <summary>read the caption</summary>
> Figure 1: Illustration of the (a) selective visual attention and (b) non-fixed membrane potential leak in biological nervous system.
> </details>





![](https://ai-paper-reviewer.com/TuCQdBo4NC/tables_7_1.jpg)

> 🔼 This table presents the performance of the Frequency Encoding (FE) and Evolutionary Leak Factor (FEEL) methods in improving the robustness of Spiking Neural Networks (SNNs) against various attacks.  Different training strategies (vanilla, adversarial training (AT), regularized adversarial training (RAT), and stochastic gating (StoG)) are used to evaluate the effectiveness of the proposed methods.  The table shows the accuracy on the CIFAR-100 dataset using the VGG11 network with various attack methods (Gaussian noise (GN), Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Basic Iterative Method (BIM), and Carlini & Wagner (CW) attacks). The improvements achieved by adding FE and FEEL to the various training strategies are shown in parentheses.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of the proposed FE and FEEL with different training strategies. The perturbation  = 8/255 for all attacks, and iterative step k = 7, step size  = 0.01 for PGD, BIM. The dataset is CIFAR100 with T = 8, the network is VGG11. The improvement brought by our method is shown in parentheses.
> </details>





### In-depth insights


#### SNN Robustness
The robustness of Spiking Neural Networks (SNNs) is a critical area of research, given their potential for real-world applications.  **Current understanding suggests inherent robustness stems from SNNs' bio-inspired nature**, but this understanding lacks theoretical grounding and often relies on empirical observations that lack generalizability. The paper highlights this gap, **emphasizing the need for a unified theoretical framework to understand and improve SNN robustness.**  This framework could guide the development of new SNN architectures, training methodologies, and encoding techniques.  **The study proposes a novel model, FEEL-SNN, incorporating frequency encoding and an evolutionary leak factor to mimic biological mechanisms of selective attention and adaptive membrane potential leaks.** This approach aims to enhance SNNs' ability to handle diverse noise types, thereby increasing their reliability and trustworthiness, particularly for safety-critical applications.

#### FEEL-SNN Design
The FEEL-SNN design integrates **frequency encoding (FE)** and an **evolutionary leak factor (EL)** to enhance the robustness of spiking neural networks (SNNs).  FE mimics selective visual attention by processing varying frequencies at different timesteps, effectively filtering noise while preserving information. This is a significant departure from traditional direct encoding methods which simply repeat noisy inputs.  EL, inspired by non-fixed membrane potential leaks in biological neurons, allows each neuron to learn its optimal leak factor across time, further enhancing robustness.  The combination of FE and EL, as validated by experimental results, demonstrably improves SNN resilience against various noise types and attacks. This is **biologically inspired**, leading to a model that is both accurate and robust, a major advance over simpler SNN designs that rely on fixed parameters and less sophisticated input encoding.  The unified theoretical framework presented supports the design, showing how improvements in both encoding and the evolutionary leak factor directly affect robustness.

#### Frequency Encoding
The concept of "Frequency Encoding" in the context of spiking neural networks (SNNs) offers a compelling approach to enhance robustness.  **It leverages the inherent ability of biological systems to selectively process different frequency components of visual information.**  Instead of simply repeating input images multiple times, which amplifies noise, frequency encoding transforms the input into its frequency representation. This allows the network to **selectively focus on low-frequency components of the input image, which typically contain the most relevant information, while suppressing high-frequency noise.** This is analogous to how the biological visual system filters out irrelevant details, focusing on salient features. The efficacy of this method hinges on **carefully designing the frequency filtering process over time**.  The implementation of frequency encoding involves transforming images into the frequency domain (using Fourier transforms), then strategically suppressing or amplifying specific frequencies using a frequency mask. This mask’s properties are carefully controlled, for example, by varying its width over time, mimicking biological attention mechanisms. The overall impact is a more robust SNN capable of handling noisy inputs while maintaining accuracy and efficiency. **This biologically-inspired technique is not only theoretically sound but has also demonstrated significant improvements in SNN robustness through experimental evaluations.**

#### Leak Factor Evol.
The concept of 'Leak Factor Evol.' in a spiking neural network (SNN) context suggests a mechanism for adapting the neuron's membrane leak factor over time.  This dynamic adjustment is **biologically inspired**, mimicking how biological neurons adjust their leak based on various factors, such as ion concentrations and environmental stimuli.  A static leak factor is a simplification of the complex biological reality. A more advanced method using 'Leak Factor Evol.' can enhance the SNN's robustness to noise and adversarial attacks by allowing it to **adapt to varying noise characteristics**. For example, a neuron could dynamically increase its leak factor during periods of high noise to reduce the impact of spurious signals. This **adaptation ability is a key strength** because it enables the SNN to maintain performance even in unpredictable or noisy environments.  However, the computational cost and complexity of implementing such a dynamic system is a considerable challenge.  Successfully implementing a leak factor evolution algorithm would require careful consideration of the trade-off between biological accuracy and computational efficiency.  Furthermore, it's crucial to investigate the impact of different leak factor evolution strategies on the network's learning process and overall performance.

#### Future of SNNs
The future of spiking neural networks (SNNs) is bright, driven by their biological plausibility and potential for energy-efficient, high-performance computing.  **Hardware advancements** in neuromorphic computing are crucial, enabling SNNs to move beyond simulations and into real-world applications.  **Algorithmic innovations** are also vital; more efficient training methods and improved robustness against noise and adversarial attacks are needed to unlock their full potential.  **Bridging the gap** between SNNs and traditional artificial neural networks (ANNs) is a key challenge.  Developing hybrid models that combine the strengths of both approaches may be a fruitful direction.  **Exploration of novel architectures** and learning paradigms, inspired by biological systems, such as incorporating more sophisticated neuron models and synaptic plasticity rules, could significantly enhance SNN capabilities.  Research into SNN applications in areas such as brain-computer interfaces, event-driven vision, and robotics will be critical in demonstrating the real-world value of SNNs. **Addressing the challenges** of training, scalability, and interpretability will remain key research themes shaping the future of this exciting field.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/TuCQdBo4NC/figures_4_1.jpg)

> 🔼 This figure visualizes the frequency spectrums of original CIFAR10 images and their corresponding noise-added versions.  It demonstrates that original images have most information concentrated at low frequencies (center), while added noise (FGSM, PGD, BIM, CW) spreads across various frequencies, with higher-frequency components present at the edges.
> <details>
> <summary>read the caption</summary>
> Figure 2: Visualization frequency spectrums for data observation. The first column shows three cases of original CIFAR10 images. The second column shows the corresponding frequency spectrums of the images in the first column. The third column to the seventh column shows the frequency spectrums of corresponding added noises to the images in the first column, where added noise maps the difference between the noise image and the original one. The center of each frequency spectrum represents the low-frequency information, and the edge area is the high-frequency information.
> </details>



![](https://ai-paper-reviewer.com/TuCQdBo4NC/figures_4_2.jpg)

> 🔼 This figure illustrates the FEEL-SNN architecture, highlighting two key components: Frequency Encoding (FE) and Evolutionary Leak Factor (EL).  FE mimics selective visual attention by processing input images in the frequency domain and suppressing noise at different frequencies across time steps.  EL simulates the non-fixed membrane potential leak in biological neurons by allowing each neuron to learn its own optimal leak factor over time, enhancing robustness. The figure shows how FE transforms an original image into a sequence of frequency-encoded images which are then processed by the SNN. The EL component is illustrated as a set of trainable leak factors associated with different neurons at different time steps, allowing the network to adaptively adjust its robustness to various noise levels.
> <details>
> <summary>read the caption</summary>
> Figure 3: Illustration of the proposed FEEL-SNN. (a) Frequency encoding to simulate the selective visual attention in biological brain and (b) Evolutionary leak factor to simulate the non-fixed membrane potential leak in biological nervous system.
> </details>



![](https://ai-paper-reviewer.com/TuCQdBo4NC/figures_6_1.jpg)

> 🔼 This figure shows the performance of Frequency Encoding (FE) and Frequency Encoding and Evolutionary Leak Factor (FEEL) methods under different white-box adversarial attacks (FGSM, PGD, BIM, CW).  The results are presented for four different settings: CIFAR10 with VGG11 and T=4, CIFAR100 with VGG11 and T=4, CIFAR100 with WideResNet16 and T=8, and Tiny-ImageNet with ResNet19 and T=4.  The attack perturbation is consistent across all experiments. The graphs show that FEEL consistently outperforms the vanilla (baseline) method and FE in terms of accuracy when subjected to adversarial attacks.
> <details>
> <summary>read the caption</summary>
> Figure 4: Performance of the proposed FE and FEEL under different white-box attacks. The attack perturbation  = 4/255 for all attacks, iterative step k = 4, and step size a = 0.01 for PGD, BIM.
> </details>



![](https://ai-paper-reviewer.com/TuCQdBo4NC/figures_6_2.jpg)

> 🔼 This figure presents the performance comparison of vanilla SNNs, SNNs with Frequency Encoding (FE), and SNNs with FEEL (Frequency Encoding and Evolutionary Leak factor) under different white-box adversarial attacks.  The results show the accuracy achieved by each method on CIFAR-10 and CIFAR-100 datasets using VGG11 and WideResNet16 architectures with different time steps (T).  The bar chart shows a significant improvement in accuracy when using FE and FEEL, demonstrating their effectiveness in improving the robustness of SNNs against white-box attacks.
> <details>
> <summary>read the caption</summary>
> Figure 4: Performance of the proposed FE and FEEL under different white-box attacks. The attack perturbation = 4/255 for all attacks, iterative step k = 4, and step size = 0.01 for PGD, BIM.
> </details>



![](https://ai-paper-reviewer.com/TuCQdBo4NC/figures_8_1.jpg)

> 🔼 This figure shows the performance comparison of the vanilla model and the FEEL model under PGD white-box and black-box attacks with different perturbation levels (epsilon) on CIFAR-10 and CIFAR-100 datasets using VGG11 network.  The iterative step k is fixed at 4. It demonstrates the robustness of the FEEL model against adversarial attacks.
> <details>
> <summary>read the caption</summary>
> Figure 6: Performance of the white-box (WB) and black-box (BB) scenarios under PGD attack with different perturbation ∈, the iterative step k = 4, the network is VGG11.
> </details>



![](https://ai-paper-reviewer.com/TuCQdBo4NC/figures_8_2.jpg)

> 🔼 This figure visualizes the performance comparison between the vanilla method and FEEL-SNN under different perturbation levels (epsilon) and iterative steps (k) using the Projected Gradient Descent (PGD) attack.  The results are shown separately for white-box (WB) and black-box (BB) attack scenarios on the CIFAR-10 and CIFAR-100 datasets using the VGG11 network. It demonstrates the improved robustness of FEEL-SNN against PGD attacks across various settings, exhibiting a slower decrease in accuracy as both the perturbation level and iterative steps increase compared to the vanilla approach.
> <details>
> <summary>read the caption</summary>
> Figure 6: Performance of the white-box (WB) and black-box (BB) scenarios under PGD attack with different perturbation e, the iterative step k = 4, the network is VGG11.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/TuCQdBo4NC/tables_8_1.jpg)
> 🔼 This table compares the performance of the proposed Frequency Encoding (FE) method against a baseline method (Inverse FE) and the vanilla method across various attack types. The results demonstrate FE's effectiveness in improving robustness while preserving accuracy, unlike Inverse-FE.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of the proposed Frequency Encoding (FE) and the alternative strategy Inverse-FE (IFE). The perturbation  = 4/255 for all attacks, and iterative step k = 4, step size a = 0.01 for PGD. The dataset is CIFAR10 with time step T = 4, the network is VGG11.
> </details>

![](https://ai-paper-reviewer.com/TuCQdBo4NC/tables_9_1.jpg)
> 🔼 This table shows the impact of different frequency masking radius r values on the robustness of the FEEL-SNN model against PGD attacks.  It compares three different encoding strategies: a direct encoding approach using a fixed radius, a frequency encoding approach with a uniform radius across all time steps, and the proposed frequency encoding approach with a variable radius that changes across time steps. The results show that the proposed approach consistently outperforms the other two methods across different radius values.
> <details>
> <summary>read the caption</summary>
> Table 3: Effect of frequency masking radius r on robustness. The attack is PGD with perturbation \(\epsilon = 4/255\), iterative step \(\alpha = 0.01\), and iterative step \(k = 4\). The dataset is CIFAR10 with \(T = 4\), the network is VGG11.
> </details>

![](https://ai-paper-reviewer.com/TuCQdBo4NC/tables_13_1.jpg)
> 🔼 This table presents a quantitative comparison of the proposed Frequency Encoding (FE) and FEEL methods against various state-of-the-art training strategies. It assesses the model's performance under different attacks (clean, GN, FGSM, PGD, BIM, CW) on the CIFAR100 dataset. The results showcase the improvements achieved by integrating FE and FEEL, especially in enhancing robustness against adversarial attacks.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of the proposed FE and FEEL with different training strategies. The perturbation  = 8/255 for all attacks, and iterative step k = 7, step size a = 0.01 for PGD, BIM. The dataset is CIFAR100 with T = 8, the network is VGG11. The improvement brought by our method is shown in parentheses.
> </details>

![](https://ai-paper-reviewer.com/TuCQdBo4NC/tables_14_1.jpg)
> 🔼 This table presents the performance of the FEEL-SNN model against various attacks under different training strategies.  It shows the clean accuracy and accuracy under different attacks (Gaussian Noise, FGSM, PGD, BIM, CW) for various SNN training methods (Vanilla, Vanilla+FE, Vanilla+FEEL, AT, AT+FE, AT+FEEL, RAT, RAT+FE, RAT+FEEL, StoG, StoG+FE, StoG+FEEL, AT+StoG, AT+StoG+FE, AT+StoG+FEEL, RAT+StoG, RAT+StoG+FE, RAT+StoG+FEEL). The results highlight the improvements achieved by incorporating the Frequency Encoding (FE) and Evolutionary Leak Factor (EL) methods, both individually and in combination with other training strategies, demonstrating enhanced robustness against adversarial attacks.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of the proposed FE and FEEL with different training strategies. The perturbation € = 8/255 for all attacks, and iterative step k = 7, step size a = 0.01 for PGD, BIM. The dataset is CIFAR100 with T = 8, the network is VGG11. The improvement brought by our method is shown in parentheses.
> </details>

![](https://ai-paper-reviewer.com/TuCQdBo4NC/tables_14_2.jpg)
> 🔼 This table compares the performance of different methods for enhancing the robustness of Spiking Neural Networks (SNNs) against adversarial attacks.  It contrasts the performance of the proposed Evolutionary Leak Factor (EL) with various other techniques (Vanilla, FEEL with different fixed λ, FEEL with L2 regularization, and FEEL with learnable λ) and shows that EL consistently improves both clean accuracy and robustness under various attacks.
> <details>
> <summary>read the caption</summary>
> Table 6: Performance (%) of the proposed evolutionary leak factor λ (EL) with other strategies, where 'FEEL, (||||2)' represents EL with L2 norm regularization, 'GP' represents gradient penalty, which adds L2 norm constraint to the model gradient. The perturbation ε = 4/255 for all attacks, and iterative step k = 4, step size α = 0.01 for PGD. The dataset is CIFAR10 with T = 4, the network is VGG11.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TuCQdBo4NC/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}