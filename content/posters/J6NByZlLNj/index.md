---
title: "WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks"
summary: "WaveAttack: A new backdoor attack method leveraging asymmetric frequency obfuscation for high stealthiness and effectiveness in Deep Neural Networks."
categories: []
tags: ["Machine Learning", "Deep Learning", "🏢 East China Normal University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} J6NByZlLNj {{< /keyword >}}
{{< keyword icon="writer" >}} Jun Xia et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=J6NByZlLNj" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95737" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=J6NByZlLNj&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/J6NByZlLNj/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Deep Neural Networks (DNNs) are increasingly vulnerable to backdoor attacks, which maliciously manipulate training data to trigger incorrect predictions.  Existing methods suffer from low fidelity poisoned samples and latent space transfer issues, making them easily detectable.  This poses a significant challenge to the security and trustworthiness of DNNs in various applications. 

This paper introduces WaveAttack, a new backdoor attack method addressing these limitations.  **WaveAttack leverages Discrete Wavelet Transform (DWT) to generate stealthy backdoor triggers in high-frequency image components.** It also employs asymmetric frequency obfuscation for improved trigger impact, enhancing both stealthiness and effectiveness.  **Extensive experiments demonstrate that WaveAttack significantly outperforms state-of-the-art attacks, achieving higher success rates and better image quality.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} WaveAttack uses Discrete Wavelet Transform (DWT) to embed backdoor triggers in high-frequency image components, enhancing stealth. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} An asymmetric frequency obfuscation method improves trigger impact during both training and inference, increasing attack effectiveness. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} WaveAttack outperforms existing backdoor attacks in terms of attack success rate, image fidelity (PSNR, SSIM), and imperceptibility (IS). {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it introduces a novel backdoor attack method, WaveAttack, that is highly effective and stealthy.  **WaveAttack surpasses existing methods in terms of both attack success rate and image fidelity,** making it a significant threat to deep learning systems.  This research is particularly relevant given the growing concerns about the security and trustworthiness of DNNs, highlighting the need for robust defense mechanisms. The approach also opens up new avenues for studying frequency-based obfuscation techniques in adversarial machine learning.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_3_1.jpg)

> The figure shows an example of adding noise to different frequency components of an image.  It demonstrates that adding noise to the high-frequency component (HH) results in a less perceptible change than adding the same noise to the low-frequency components (LL, LH, HL). This motivates the authors' choice to inject backdoor triggers into the high-frequency component, as it improves the stealthiness of the attack.





![](https://ai-paper-reviewer.com/J6NByZlLNj/tables_4_1.jpg)

> This table compares the performance of WaveAttack against seven other state-of-the-art (SOTA) backdoor attack methods.  The comparison is done across four benchmark datasets (CIFAR-10, CIFAR-100, GTSRB, and ImageNet) using two key metrics: Attack Success Rate (ASR) and Benign Accuracy (BA).  Higher ASR indicates more effective backdoor attacks, while higher BA means the model maintains its accuracy on clean data.  The table highlights the best and second-best performance for each metric on each dataset.





### In-depth insights


#### WaveAttack Intro
The hypothetical "WaveAttack Intro" section would likely introduce the core concept of WaveAttack, a novel backdoor attack method targeting deep neural networks (DNNs).  It would highlight the **increasing prevalence of backdoor attacks** and the limitations of existing methods.  The introduction would emphasize WaveAttack's key innovation: employing **asymmetric frequency obfuscation** using Discrete Wavelet Transform (DWT) to embed highly stealthy backdoor triggers.  This approach aims to **improve the effectiveness** of backdoor attacks while maintaining **high fidelity** in poisoned samples, thus evading detection algorithms. The introduction would likely conclude by outlining the paper's contributions, such as the detailed explanation of WaveAttack's methodology and comprehensive experimental evaluation demonstrating its superiority over state-of-the-art methods.

#### Asymmetric Obs.
The concept of "Asymmetric Obs." in the context of a research paper likely refers to a technique that introduces **unequal or unbalanced obscuring effects** during different phases of a process, such as training and inference in a machine learning model. This asymmetry is likely deliberate, serving a specific purpose.  In the case of a backdoor attack, for example, the goal might be to maintain the model's performance on legitimate data while significantly altering its behavior when presented with specific "trigger" inputs. The **asymmetry would help disguise the backdoor**, as the changes made to the model during training might be imperceptible in normal operation, only manifesting when the trigger is used.  **Stealthiness** is a key advantage; this method aims to minimize the impact of the inserted backdoor on the standard performance metrics, thereby making detection harder.  One crucial aspect to consider is the nature of this obfuscation.  Is it designed to work in the image space directly, or does it manipulate internal model representations in a higher-dimensional latent space?  Further investigation into the specific implementation will reveal more details.

#### DWT Trigger Gen.
The heading 'DWT Trigger Gen.' suggests a method for generating backdoor triggers using Discrete Wavelet Transform (DWT).  **DWT excels at separating image components into different frequency sub-bands**, allowing for the creation of triggers embedded within high-frequency details, making them **imperceptible to human observation**. This approach aims to enhance the stealthiness of backdoor attacks.  The effectiveness of this generation method likely depends on the specific DWT used, the choice of sub-bands for trigger insertion, and the method of embedding the trigger within the selected sub-band.  A crucial aspect would be evaluating the fidelity of the resulting poisoned images; **high fidelity ensures that poisoned samples closely resemble benign ones**, evading detection by visual inspection or simple statistical analysis.  Overall, 'DWT Trigger Gen.' proposes a sophisticated technique for generating highly stealthy backdoor triggers, potentially improving the effectiveness and survivability of backdoor attacks against deep neural networks.

#### Defense Robustness
A robust defense against backdoor attacks needs to consider multiple aspects.  **WaveAttack's stealthiness, achieved through high-frequency trigger generation and asymmetric obfuscation, makes it particularly challenging for traditional detection methods** which often focus on identifying visible or easily detectable patterns in the input data. Therefore, defenses would need to incorporate techniques that go beyond simple trigger detection.  **Robust defenses should focus on methods that analyze the model's internal representations**, perhaps examining the latent space or intermediate activations, rather than solely relying on input features.  Furthermore, **a strong defense requires a multi-pronged approach, combining multiple defense mechanisms** to address the various attack strategies.  Methods focusing on model retraining, feature extraction modifications, or latent-space analysis might prove more effective than single-faceted approaches.  **Addressing the asymmetric nature of WaveAttack—the difference between training and inference phases—would require advanced defense techniques** capable of dynamically adapting to changing model behaviors.

#### Future Works
Future research directions stemming from this work could involve exploring **more sophisticated trigger generation methods** that are even more robust to detection.  Investigating **different frequency obfuscation techniques** beyond the asymmetric approach used here would also be valuable, potentially incorporating adaptive or learned methods.  Further exploration of the **generalizability of WaveAttack** across diverse DNN architectures and datasets is crucial.  Finally, a particularly important avenue is to explore **defensive mechanisms** against WaveAttack, thus furthering the ongoing arms race between attackers and defenders in the field of backdoor attacks.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_3_2.jpg)

> This figure illustrates the WaveAttack method.  It starts with input images, which undergo a Discrete Wavelet Transform (DWT) to separate them into four frequency components (LL, LH, HL, HH). The high-frequency component (HH) is then processed by an encoder-decoder network (E and D) to generate residuals.  These residuals are multiplied by a coefficient (α) to create a modified HH component (HH'). This modified component, along with the other frequency components from the DWT, is then fed into an Inverse Discrete Wavelet Transform (IDWT) to reconstruct the image. The benign samples, samples with added residuals (payload samples), and regularization samples are randomly split and used to train a classifier (C) to generate the backdoored model.


![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_6_1.jpg)

> This figure compares the poisoned samples and their residuals generated by different backdoor attack methods, including BadNets, Blend, IAD, WaNet, BppAttack, Adapt-Blend, FTrojan, and WaveAttack. The top row shows the poisoned samples, while the bottom row shows their corresponding residuals magnified five times.  It visually demonstrates the stealthiness of each method by showing how much the poisoned image differs from the original. WaveAttack shows the least visible difference.


![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_6_2.jpg)

> This figure compares the poisoned samples generated by seven different backdoor attacks, including WaveAttack.  The top row shows the poisoned samples, while the bottom row displays the residuals (magnified 5x) showing the difference between the poisoned and original images. The goal is to visually assess the stealthiness of each attack, with smaller and less noticeable residuals indicating a more stealthy approach. WaveAttack aims to generate samples that are visually indistinguishable from clean samples.


![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_7_1.jpg)

> This figure shows the results of applying the STRIP defense method to the WaveAttack backdoor.  The histograms display the normalized entropy of both clean and poisoned samples for three datasets: CIFAR-10, CIFAR-100, and GTSRB. The overlapping distributions of clean and poisoned samples demonstrate that WaveAttack successfully evades detection by STRIP, indicating its strong stealthiness against this particular defense technique.  The x-axis represents the normalized entropy, while the y-axis shows the number of inputs (samples).


![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_8_1.jpg)

> This figure presents a comparison of GradCAM heatmaps for clean and WaveAttack backdoored models. GradCAM is a technique used to visualize which parts of an image a neural network focuses on to make a prediction. The left side shows the heatmaps from a clean model, which correctly classifies the images. The right side shows heatmaps from models that have been attacked using WaveAttack. The similarity in heatmaps between the clean and attacked models demonstrates the stealthiness of the WaveAttack method, making it difficult to distinguish between clean and poisoned images using this technique.


![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_8_2.jpg)

> This figure shows the performance comparison between WaveAttack and seven SOTA attack methods on CIFAR-10 by resisting Fine-Pruning.  The x-axis represents the number of filters pruned, while the y-axis shows both the Attack Success Rate (ASR) and Benign Accuracy (BA).  The plots illustrate how the ASR and BA of each attack method change as more and more filters are pruned.  This demonstrates the robustness of WaveAttack against the Fine-Pruning defense method.


![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_8_3.jpg)

> This figure shows a bar chart comparing the anomaly index of clean and backdoored models for four different datasets: CIFAR-10, CIFAR-100, GTSRB, and ImageNet.  The anomaly index is a measure used by the Neural Cleanse defense method to detect backdoors.  Lower anomaly indices suggest higher stealthiness.  The chart indicates that WaveAttack's backdoored models have anomaly indices below the threshold of 2, making them difficult for Neural Cleanse to detect.


![](https://ai-paper-reviewer.com/J6NByZlLNj/figures_9_1.jpg)

> This figure shows an example to motivate the design of the backdoor trigger on high-frequency components. It compares the effects of adding the same noise to different frequency components (LL, LH, HL, HH) of an image obtained through the Haar wavelet transform. The result shows that it is more difficult to identify the difference between the original image and the poisoned counterpart when noise is added to the HH component (high-frequency component) compared to other components. Therefore, the authors argue that it is more suitable to inject triggers into the HH component for backdoor attacks.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/J6NByZlLNj/tables_5_1.jpg)
> This table presents a comparison of the effectiveness of WaveAttack against seven other state-of-the-art (SOTA) backdoor attack methods.  The comparison is made across four benchmark datasets (CIFAR-10, CIFAR-100, GTSRB, and ImageNet) using two key metrics: Attack Success Rate (ASR) and Benign Accuracy (BA).  Higher ASR indicates better attack effectiveness, while higher BA shows less negative impact on the model's performance on benign data.  The table highlights the best and second-best performing methods for each dataset, demonstrating WaveAttack's superior performance.

![](https://ai-paper-reviewer.com/J6NByZlLNj/tables_5_2.jpg)
> This table presents the performance of WaveAttack on different deep neural networks (DNNs) including VGG16, SENet18, ResNeXt29, and DenseNet121.  The results show the benign accuracy (BA) and attack success rate (ASR) for each network under WaveAttack and a baseline of no attack. The data indicates the effectiveness of WaveAttack across various network architectures.

![](https://ai-paper-reviewer.com/J6NByZlLNj/tables_6_1.jpg)
> This table presents the results of the WaveAttack method using different Discrete Wavelet Transforms (DWTs).  It compares the performance of WaveAttack using Haar and Daubechies (DB) wavelets across three datasets (CIFAR-10, CIFAR-100, and GTSRB) using metrics such as Inception Score (IS), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Benign Accuracy (BA), and Attack Success Rate (ASR).  The lower the IS, the better; the higher the PSNR, SSIM, BA, and ASR, the better.

![](https://ai-paper-reviewer.com/J6NByZlLNj/tables_7_1.jpg)
> This table compares the performance of WaveAttack against seven other state-of-the-art (SOTA) backdoor attack methods.  The comparison is done using two metrics: Attack Success Rate (ASR) and Benign Accuracy (BA).  Higher ASR indicates a more effective attack, while higher BA means the attack has less impact on the model's performance on benign data. The table shows the results for four different datasets: CIFAR-10, CIFAR-100, GTSRB, and ImageNet.

![](https://ai-paper-reviewer.com/J6NByZlLNj/tables_9_1.jpg)
> This table compares the backdoor detection rate (BDR) of several backdoor attack methods against a frequency detection method.  It shows that WaveAttack has a significantly lower BDR than other state-of-the-art methods, indicating its higher stealthiness against this specific defense mechanism.  The lower BDR for WaveAttack suggests that its generated poisoned samples are less easily detected as malicious by the frequency-based method compared to other methods. The table helps to demonstrate the effectiveness of WaveAttack's strategy in evading detection.

![](https://ai-paper-reviewer.com/J6NByZlLNj/tables_9_2.jpg)
> This table compares the performance of different backdoor attack methods against a frequency detection method.  The Backdoor Detection Rate (BDR) is shown for each method, indicating their susceptibility to detection. Lower BDR values suggest better stealthiness.

![](https://ai-paper-reviewer.com/J6NByZlLNj/tables_14_1.jpg)
> This table shows the details of the datasets used in the experiments of the paper.  It lists the dataset name, input image size (width x height x color channels), number of classes, the number of training images, and the number of test images.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/J6NByZlLNj/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}