---
title: "SpGesture: Source-Free Domain-adaptive sEMG-based Gesture Recognition with Jaccard Attentive Spiking Neural Network"
summary: "SpGesture: A source-free domain-adaptive SEMG gesture recognition system using a novel Spiking Jaccard Attentive Neural Network achieves real-time performance with high accuracy."
categories: ["AI Generated", ]
tags: ["AI Applications", "Human-AI Interaction", "🏢 Hong Kong University of Science and Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} GYqs5Z4joA {{< /keyword >}}
{{< keyword icon="writer" >}} Weiyu Guo et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=GYqs5Z4joA" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/GYqs5Z4joA" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/GYqs5Z4joA/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Existing sEMG-based gesture recognition systems suffer from high latency, energy consumption, and sensitivity to variations in real-world conditions.  The inherent instability of sEMG signals and distribution shifts across different forearm postures make creating robust models challenging.  This limits the applicability of sEMG in real-world applications.



SpGesture, a novel framework based on Spiking Neural Networks (SNNs), addresses these challenges. It introduces a Spiking Jaccard Attention mechanism to enhance feature representation and a Source-Free Domain Adaptation technique to improve model robustness.  Experimental results on a new sEMG dataset show that SpGesture achieves high accuracy and real-time performance (latency below 100ms), outperforming existing methods.  The code for SpGesture is publicly available.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} SpGesture, a new framework for sEMG-based gesture recognition using Spiking Neural Networks (SNNs), achieves real-time performance and high accuracy. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The Spiking Jaccard Attention mechanism significantly improves the efficiency and accuracy of SNNs for processing sparse sEMG data. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} SpGesture introduces Source-Free Domain Adaptation to SNNs, enabling robust performance across varying forearm postures without requiring source data. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in sEMG-based gesture recognition and neuromorphic computing.  It **addresses the critical issues of high latency and energy consumption in existing sEMG systems**, offering a novel and efficient solution.  The introduction of source-free domain adaptation to spiking neural networks is a significant advance, opening up new avenues for research in robust and adaptable machine learning models. Its **real-time performance and high accuracy on a new sEMG dataset** make it highly relevant to current trends and future developments in human-computer interaction.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_3_1.jpg)

> 🔼 This figure illustrates the architecture of the proposed Spiking Jaccard Attention Neural Network.  Raw surface electromyography (sEMG) data is initially converted into spike trains using a convolutional leaky integrate-and-fire (ConvLIF) layer. These spike trains then pass through multiple ConvLIF layers with varying numbers of channels (N and 2N) to extract features.  The extracted features then undergo a Spiking Jaccard Attention mechanism before classification by a multi-layer perceptron (MLP) and finally a LIF classifier layer. A membrane potential memory component is also shown, highlighting a key element of the Source-Free Domain Adaptation strategy.
> <details>
> <summary>read the caption</summary>
> Figure 1: The pipeline of Jaccard Attention Spike Neural Network: Raw sEMG Data is first encoded into Spike Signals using ConvLIF. These signals pass through ConvLIF layers with N and 2N channels. The processed data then goes through the Spiking Jaccard Attention mechanism.
> </details>





![](https://ai-paper-reviewer.com/GYqs5Z4joA/tables_7_1.jpg)

> 🔼 This table compares the performance of the proposed SpGesture method against various state-of-the-art methods for sEMG-based gesture recognition.  It shows the top-1 accuracy and standard deviation achieved by different models, including Convolutional Neural Networks (CNNs), Temporal Convolutional Networks (TCNs), Transformers, GRUs, and Informers, as well as various Spiking Neural Networks (SNNs). The table highlights SpGesture's superior performance compared to other DNN and SNN baselines.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison with previous works on sEMG-based gesture estimation.
> </details>





### In-depth insights


#### sEMG Gesture recog
Surface electromyography (sEMG) based gesture recognition is a field of study focused on using electrical signals from muscles to control devices.  **sEMG signals offer a natural and intuitive interaction modality**, but challenges remain.  Existing methods often suffer from **high computational latency and energy consumption**.  Additionally, **sEMG signals are inherently unstable and sensitive to distribution shifts**, affecting model robustness.  New approaches are exploring Spiking Neural Networks (SNNs) due to their energy efficiency and potential for real-time processing.  However, **SNNs face challenges in achieving comparable accuracy to traditional Artificial Neural Networks (ANNs)**.  Therefore, current research focuses on improving SNN performance through techniques such as attention mechanisms and domain adaptation, to create robust and efficient sEMG-based gesture recognition systems suitable for real-world applications.  **The field continues to seek efficient algorithms and hardware implementations** to improve accuracy, latency, and energy efficiency of the technology.

#### Spiking Jaccard Attn
The proposed Spiking Jaccard Attention mechanism offers a novel approach to attention in Spiking Neural Networks (SNNs).  Traditional attention mechanisms struggle with the sparsity inherent in SNNs, often requiring conversion to continuous representations which negates the energy efficiency benefits of SNNs. **Spiking Jaccard Attention directly computes attention using the Jaccard similarity measure on spike trains**, avoiding this costly conversion. This not only preserves the energy efficiency of SNNs but also aligns well with the binary operations found in neuromorphic hardware. **The Jaccard similarity is particularly well-suited for binary data like spike trains.** Furthermore, it offers improved computational efficiency compared to traditional attention methods because it leverages the sparsity of spike trains, directly calculating similarity between pairs of spike sequences.  The use of a querying mechanism further refines the focus on relevant temporal aspects of the data.  **Overall, the Spiking Jaccard Attention represents a significant advancement in applying attention mechanisms effectively to SNNs, enhancing accuracy and maintaining the energy efficiency that is a primary advantage of this type of neural network.**

#### SFDA in SNNs
The application of Source-Free Domain Adaptation (SFDA) to Spiking Neural Networks (SNNs) presents a unique challenge and opportunity.  **SFDA's core strength is adapting a model to a new domain without access to the original source data**, crucial for privacy-sensitive applications.  However, SNNs, with their binary and sparse nature, pose difficulties for traditional SFDA methods which often rely on continuous representations and complex similarity calculations.  This paper innovatively tackles this by using **membrane potential as a memory list** for pseudo-labeling, enabling knowledge transfer without source data sharing, and introducing a **novel Spiking Jaccard Attention mechanism**. This attention mechanism is designed to work effectively with the sparse nature of SNNs, enhancing feature representation and significantly improving accuracy. The combined approach of Spiking Jaccard Attention and the memory-based SFDA is particularly effective in scenarios with distribution shifts, such as those caused by varying forearm postures in sEMG gesture recognition. **The effectiveness is demonstrated by achieving state-of-the-art accuracy**, showcasing the potential of this approach to improve the practicality and robustness of SNNs for real-world applications.

#### Forearm Posture Var
The heading 'Forearm Posture Var' likely refers to a section exploring the impact of different forearm positions on surface electromyography (sEMG) signals used for gesture recognition.  This is a crucial consideration because **forearm posture significantly affects muscle activation patterns**, leading to variations in the measured sEMG data.  The research likely investigates how these variations impact the accuracy and robustness of gesture recognition models.  The analysis might involve comparing model performance across various postures, potentially revealing **the need for robust models that generalize well across different positions**.  Furthermore, the study might explore techniques to mitigate the negative effects of forearm posture changes, such as data augmentation or domain adaptation strategies, to improve model generalizability and real-world applicability. **Dataset composition** and the inclusion of diverse forearm postures during data collection would be a critical aspect of this analysis.  The results might demonstrate the limitations of models trained on limited postures and highlight the importance of considering forearm posture variability during model development and testing for improved sEMG-based gesture recognition.

#### Future Research
Future research directions stemming from this work could explore extending the Spiking Source-Free Domain Adaptation (SSFDA) method to handle diverse distribution shifts beyond forearm postures.  **Investigating the robustness of the SSFDA and Spiking Jaccard Attention (SJA) across a broader range of SNN architectures** would further validate their generalizability and highlight potential limitations.  **Evaluating performance on neuromorphic hardware** is crucial for realizing the energy efficiency benefits of SNNs.  Furthermore, exploring alternative attention mechanisms and incorporating other advanced techniques from deep learning could potentially enhance accuracy and robustness.  Finally, **expanding the dataset to include a wider variety of gestures and subjects, alongside a more comprehensive analysis of the impact of noise and electrode placement**, would make the model more practical and reliable for real-world applications.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_4_1.jpg)

> 🔼 This figure compares the proposed Spiking Jaccard Attention (SJA) module with the existing MA-SNN attention module.  MA-SNN uses fully connected layers and pooling, resulting in continuous intermediate values and reduced efficiency. In contrast, SJA utilizes spike values, which improves efficiency and accuracy by enabling direct computation of Jaccard similarity on the sparse spike trains.
> <details>
> <summary>read the caption</summary>
> Figure 2: Comparison of MA-SNN and Spiking Jaccard Attention Modules. MA-SNN [67] uses fully connected layers with pooling but lacks a querying mechanism, leading to continuous intermediate values and lower efficiency. Our Spiking Jaccard Attention uses spike values for intermediate representations, enhancing efficiency and accuracy.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_5_1.jpg)

> 🔼 This figure illustrates the Spiking Source-Free Domain Adaptation (SSFDA) process.  Raw sEMG data is fed into the JASNN model. The model's membrane potential is recorded and stored in a 'Membrane Potential Memory List'.  For each new unlabeled sample, the k-nearest neighbors from this memory are found using Pearson correlation.  A probabilistic approach generates pseudo-labels based on these neighbors (using the mode or random selection).  The model is then trained using a loss function combining Smooth Negative Log-Likelihood (SNLL) and Kullback-Leibler (KL) divergence. The memory list is updated after each training epoch.
> <details>
> <summary>read the caption</summary>
> Figure 3: Computation flow of Spiking Source-Free Domain Adaptation. The process starts with selecting the k-nearest samples from the membrane potential memory using the Pearson correlation coefficient. Probabilistic Label Generation then produces pseudo-labels based on these k samples. Gradients are computed with Smooth NLL and KL divergence loss. The membrane potential memory list is updated at each epoch's end.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_8_1.jpg)

> 🔼 Violin plots showing the accuracy of three different domain adaptation methods (Ours, Duan et al., Huang et al.) before and after applying Spiking Source-Free Domain Adaptation (SSFDA) on two different scenarios.  The plots compare performance when the model is trained on forearm posture P1 and tested on forearm postures P2 (4a) and P3 (4b). The results illustrate the impact of SSFDA in improving robustness and generalizability across different forearm postures.
> <details>
> <summary>read the caption</summary>
> Figure 4: Comparison of performance before and after applying SSFDA for various methodologies: Figures 4a and 4b are Violin Plots demonstrating this disparity.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_9_1.jpg)

> 🔼 This figure compares the inference speed and RAM usage of three different attention mechanisms (Raw Attention, Efficient Attention, and Spiking Jaccard Attention) using both spike and float data types.  The results demonstrate that the Spiking Jaccard Attention method is significantly more efficient than the other two methods in terms of both speed and memory usage, especially as the amount of data increases.
> <details>
> <summary>read the caption</summary>
> Figure 5: Inference speed and RAM usage comparison between spike and float data for Raw Attention [60], Efficient Attention [54], and our Spiking Jaccard Attention: The first column shows inference time for float data, and the second for spike data. The third and fourth columns show RAM usage for these data types. The x-axis represents different data row counts, and the y-axis is logarithmic to highlight performance differences. Each experiment was conducted 100 times, with averaged results.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_16_1.jpg)

> 🔼 This figure shows an overview of the dataset used in the SpGesture research.  It displays ten different hand gestures performed across three different forearm postures (horizontal, diagonal, and elevated). Each gesture is represented by a number (0-9), and each forearm posture has a unique color-coded background. The 'REST' column shows images of the hand in a relaxed state before and after performing each gesture.
> <details>
> <summary>read the caption</summary>
> Figure 6: Overview of our dataset: the compilation contains sEMG data for ten distinct actions, each across three postures. Varied background colors represent distinct forearm postures, while the digits ranging from 0 to 9 correspond to specific gesture actions. The 'Rest' label at the top denotes a static hand gesture when no action is being performed.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_17_1.jpg)

> 🔼 The figure shows a summary of the collected sEMG data. The left side shows three different forearm postures. The middle section shows the raw sEMG data from one subject for the three postures and the right section shows the RMS features extracted from the sEMG data.
> <details>
> <summary>read the caption</summary>
> Figure 7: Summary of our collected data: The three postures on the left illustrate distinct preparatory arm actions. The central data graph represents the sEMG data captured from a subject under the three postures, with unique colors assigned to different channels. The data graph on the right showcases the acquired data after Root Mean Square (RMS) processing.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_19_1.jpg)

> 🔼 This figure illustrates the architecture of the proposed Jaccard Attentive Spiking Neural Network (JASNN).  Raw surface electromyography (sEMG) data is initially converted into spike signals using a convolutional leaky integrate-and-fire (ConvLIF) encoder. These spike signals then pass through multiple ConvLIF layers with varying numbers of channels (N and 2N), extracting features and increasing dimensionality. Finally, a Spiking Jaccard Attention mechanism is applied to focus on the most relevant features before classification.
> <details>
> <summary>read the caption</summary>
> Figure 1: The pipeline of Jaccard Attention Spike Neural Network: Raw sEMG Data is first encoded into Spike Signals using ConvLIF. These signals pass through ConvLIF layers with N and 2N channels. The processed data then goes through the Spiking Jaccard Attention mechanism.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_21_1.jpg)

> 🔼 This figure presents a comparison of model performance before and after applying Spiking Source-Free Domain Adaptation (SSFDA) using three different methodologies. The violin plots show the distribution of accuracy across different subjects for two different scenarios: (a) comparing model trained on posture 1's performance on posture 2, and (b) comparing model trained on posture 1's performance on posture 3. The results demonstrate the effectiveness of SSFDA in improving model robustness by reducing the performance gap between different forearm postures.  The proposed method shows significantly improved accuracy and reduced variance compared to other methods.
> <details>
> <summary>read the caption</summary>
> Figure 4: Comparison of performance before and after applying SSFDA for various methodologies: Figures 4a and 4b are Violin Plots demonstrating this disparity.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_22_1.jpg)

> 🔼 This figure compares the performance of three different pseudo-label generation methods (Ours, Duan et al., and Huang et al.) for Spiking Source-Free Domain Adaptation (SSFDA) before and after applying the adaptation technique.  Violin plots are used to show the distribution of accuracy across 14 subjects for two scenarios: when transferring models trained on forearm posture P1 to postures P2 (Figure 4a) and P3 (Figure 4b).  The plots demonstrate that the proposed method (Ours) shows a significant improvement in accuracy and reduced variability across subjects compared to the other methods, after applying SSFDA.
> <details>
> <summary>read the caption</summary>
> Figure 4: Comparison of performance before and after applying SSFDA for various methodologies: Figures 4a and 4b are Violin Plots demonstrating this disparity.
> </details>



![](https://ai-paper-reviewer.com/GYqs5Z4joA/figures_22_2.jpg)

> 🔼 This figure illustrates the architecture of the proposed Jaccard Attentive Spiking Neural Network (JASNN).  Raw surface electromyography (sEMG) data is initially converted into spike signals using a convolutional leaky integrate-and-fire (ConvLIF) encoder. These spike signals then pass through multiple ConvLIF layers, with varying numbers of channels (N and 2N). The processed data is subsequently fed into a Spiking Jaccard Attention module, which enhances the network's ability to focus on relevant features and improve its performance.
> <details>
> <summary>read the caption</summary>
> Figure 1: The pipeline of Jaccard Attention Spike Neural Network: Raw sEMG Data is first encoded into Spike Signals using ConvLIF. These signals pass through ConvLIF layers with N and 2N channels. The processed data then goes through the Spiking Jaccard Attention mechanism.
> </details>



</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GYqs5Z4joA/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}