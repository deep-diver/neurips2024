---
title: "SampDetox: Black-box Backdoor Defense via Perturbation-based Sample Detoxification"
summary: "SampDetox uses diffusion models to purify poisoned machine learning samples by strategically adding noise to eliminate backdoors without compromising data integrity."
categories: []
tags: ["Machine Learning", "Deep Learning", "🏢 Singapore Management University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Y6RV6z98Pk {{< /keyword >}}
{{< keyword icon="writer" >}} Yanxin Yang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Y6RV6z98Pk" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94718" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=Y6RV6z98Pk&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/Y6RV6z98Pk/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Backdoor attacks on machine learning models pose a serious threat, particularly in cloud-based services where model internals are hidden.  Existing defenses often require access to training data or model parameters which is not always feasible. This necessitates robust black-box defense mechanisms. The lack of effective black-box defense methods that can handle various types of backdoors highlights a critical need in the field.

SampDetox addresses this challenge by using diffusion models. It employs a two-stage process: first, applying global noise to neutralize low-visibility backdoors; second, identifying and targeting high-visibility backdoors with localized noise.  The diffusion model then reconstructs the purified samples.  Extensive experiments demonstrate SampDetox's superior performance compared to existing techniques, showcasing its ability to effectively defend against a wide range of sophisticated backdoor attacks without requiring model access. This innovative approach represents a significant step forward in securing machine learning systems.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} SampDetox effectively defends against various state-of-the-art backdoor attacks in a black-box setting. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method leverages diffusion models for sample purification, providing a model-agnostic approach. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} SampDetox's two-stage process strategically combines lightweight and intensive noise to eliminate low- and high-visibility backdoors. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in machine learning security as it introduces a novel black-box defense against backdoor attacks, a significant challenge in deploying trustworthy ML systems.  The **model-agnostic nature** of SampDetox makes it broadly applicable, while its effectiveness against diverse state-of-the-art attacks showcases its potential impact.  The **theoretical foundations** and empirical results provide a strong basis for further research into perturbation-based defenses.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/Y6RV6z98Pk/figures_3_1.jpg)

> This figure shows examples of poisoned samples with their visibility (v) and robustness (nr) scores.  The visibility score (v) quantifies how noticeable the backdoor trigger is, while the robustness score (ηr) measures how resistant the poisoned sample is to modifications from random noise. The figure visually demonstrates that backdoors with low visibility are less robust to noise (easily destroyed by lightweight noise) while those with high visibility are more robust but easier to detect.





![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_7_1.jpg)

> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and a no-defense baseline.  The comparison is done across ten different backdoor attacks (BadNets, SIG, LC, TrojanNN, Dynamic, Blended, LF, WaNet, ISSBA, BPP), evaluating each defense method using three metrics: Clean sample Accuracy (CA), Poisoned sample Accuracy (PA), and Attack Success Rate (ASR). Bold values indicate the best performance, while underlined values indicate second-best performance for each attack and metric.





### In-depth insights


#### Backdoor Defense
Backdoor defenses in machine learning aim to mitigate the malicious insertion of hidden triggers that cause models to misbehave.  **Existing methods often rely on access to training data or model internals (white-box approaches), limiting their applicability to real-world scenarios, especially in the context of Machine Learning as a Service (MLaaS).**  Black-box defenses, conversely, operate without such access.  They frequently employ techniques like sample purification to cleanse data of malicious triggers before feeding it to a potentially compromised model.  This often involves adding noise to perturb the trigger patterns, which requires careful consideration to avoid destroying crucial sample information. **Advanced strategies combine various noise levels or utilize generative models to selectively remove triggers, highlighting the importance of understanding the relationship between trigger visibility and robustness.**  Effective methods also weigh the balance between maintaining accuracy on clean data and successfully neutralizing backdoors.  The goal is not just detection but ensuring that the overall system remains reliable and functional even in the face of attacks.

#### Diffusion Models
Diffusion models are a powerful class of generative models that have shown remarkable success in various applications.  They function by gradually adding noise to data until it becomes pure noise, then learning to reverse this process to generate new samples. **This forward diffusion process is relatively straightforward**, while the **reverse process is the core of the model's generative power**, often implemented using a neural network trained to predict and remove noise at each step.  **The key strength lies in their ability to generate high-quality, diverse samples**, often surpassing other methods. However, **training diffusion models requires significant computational resources and data**, presenting a barrier to entry.  Furthermore, **the inherent stochasticity of diffusion models can sometimes lead to less controllable outputs** compared to deterministic methods.  Despite these limitations, ongoing research continues to refine their efficiency, controllability, and applicability across diverse fields such as image generation, speech synthesis, and drug discovery.

#### Noise Perturbation
Noise perturbation, in the context of defending against backdoor attacks in machine learning models, involves strategically adding noise to input samples to disrupt or eliminate malicious trigger patterns.  **The core idea is to leverage the inherent trade-off between trigger visibility and robustness.**  Highly visible triggers are often easily detectable and thus easily disrupted by even slight noise, while less visible, more robust triggers require stronger noise perturbations.  **Effective noise perturbation strategies therefore need to carefully balance the intensity and type of noise** to maximize backdoor removal while preserving the integrity of the original data's semantics for accurate classification.  This often requires sophisticated techniques, such as using diffusion models to add and subsequently remove noise, or employing adaptive methods that adjust noise levels based on trigger characteristics.  **Successful implementation hinges on a deep understanding of the backdoor attack mechanisms and the characteristics of the noise that can effectively counteract them.** Ultimately, the goal is to create a robust pre-processing step that neutralizes threats without significantly degrading the model's overall performance on clean data.

#### SampDetox Method
The SampDetox method is a novel, **black-box backdoor defense** approach that leverages perturbation-based sample detoxification.  It strategically combines lightweight and high-intensity noise to eliminate different types of backdoor triggers.  **Low-visibility triggers** are removed using low-level noise and a diffusion model for restoration.  **High-visibility triggers**, identified through structural similarity comparison between the original and denoised samples, are targeted with intensive noise in specific regions. This two-stage process effectively destroys various backdoor triggers while preserving overall sample semantics. The approach is **model-agnostic**, meaning it doesn't need access to model parameters.  **Theoretically grounded**, SampDetox's effectiveness is supported by its reliance on the properties of diffusion models and a preliminary study highlighting the correlation between trigger visibility and robustness.  This makes it a potentially highly effective approach to safeguard against untrustworthy third-party ML services.

#### Future Directions
Future research could explore several promising avenues. **Improving SampDetox's robustness against adaptive attacks** is crucial; attackers might strategically modify triggers to evade detection.  Investigating the impact of different diffusion models and noise strategies warrants attention.  **Extending SampDetox to other data modalities** (audio, text) would enhance its general applicability, while evaluating the method's performance on larger-scale datasets is necessary to demonstrate scalability.  **A formal theoretical analysis** could provide deeper insights into SampDetox's capabilities and limitations. Finally, research into the interplay between trigger visibility, robustness, and purification effectiveness requires further exploration to create a more effective defense against this increasingly sophisticated class of attacks.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/Y6RV6z98Pk/figures_4_1.jpg)

> The figure illustrates the SampDetox approach, a two-stage process for purifying poisoned samples. Stage 1 (Global Detoxification) involves adding noise globally to the poisoned samples to eliminate low-visibility backdoors and then using a diffusion model to restore the samples. Stage 2 (Local Detoxification) involves comparing the restored samples to the original clean samples to identify the locations of high-visibility backdoors.  Targeted noise is then applied to these areas, followed by denoising with the diffusion model. The output is a detoxified sample.


![](https://ai-paper-reviewer.com/Y6RV6z98Pk/figures_7_1.jpg)

> This figure shows the results of applying SampDetox to ten different backdoor attacks.  The top row displays a clean sample alongside its poisoned versions created by each of the ten attacks. The middle row visualizes the triggers identified by the SampDetox algorithm during its second stage (local detoxification).  The bottom row presents the corresponding detoxified samples produced by SampDetox after both stages of processing.


![](https://ai-paper-reviewer.com/Y6RV6z98Pk/figures_9_1.jpg)

> This figure shows the framework and workflow of the SampDetox approach.  It illustrates a two-stage process: Stage 1 (Global Detoxification) applies noise to the entire sample to remove low-visibility backdoors. Stage 2 (Local Detoxification) uses pixel-wise comparison with the original sample to identify high-visibility triggers and then selectively applies noise to those areas. Diffusion models are used in both stages to reconstruct clean samples after noise application.


![](https://ai-paper-reviewer.com/Y6RV6z98Pk/figures_22_1.jpg)

> This figure shows 30 examples of poisoned images from the CIFAR-10 dataset, each with its corresponding visibility score (v) and robustness score (nr). These scores are calculated using the metrics defined in Section 3.1 of the paper.  The visibility score (v) represents how easily the backdoor trigger is detectable in the image, while the robustness score (nr) indicates how resistant the poisoned sample is to noise or modifications. This figure visually demonstrates the correlation observed between trigger visibility and sample robustness, which is a key finding of the preliminary study presented in Section 3 of the paper.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_8_1.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and without any defense.  The comparison is done across ten different backdoor attacks (BadNets, SIG, LC, TrojanNN, Dynamic, Blended, LF, WaNet, ISSBA, and BPP). For each attack, the table shows the Clean Accuracy (CA), Poisoned Accuracy (PA), and Attack Success Rate (ASR) for each defense method.  Bold and underlined values highlight the best and second-best performance for each metric and each attack.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_8_2.jpg)
> This table compares the performance of SampDetox against three other state-of-the-art sample purification-based backdoor defense methods (Sancdifi, BDMAE, and ZIP) across ten different backdoor attacks.  The table shows the Clean Accuracy (CA), Poisoned Accuracy (PA), and Attack Success Rate (ASR) for each defense method under each attack.  Bold values indicate the best performance, while underlined values indicate the second-best performance for each metric under each attack.  This allows for a direct comparison of the effectiveness of SampDetox against existing methods across a variety of attack scenarios.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_16_1.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses: Sancdifi, BDMAE, and ZIP.  The comparison is done across ten different backdoor attacks, evaluating three key metrics for each: Clean sample Accuracy (CA), Poisoned sample Accuracy (PA), and Attack Success Rate (ASR).  Bold and underlined values highlight the best and second-best results for each metric and attack.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_17_1.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and without any defense.  The comparison is performed across ten different backdoor attacks (BadNets, SIG, LC, TrojanNN, Dynamic, Blended, LF, WaNet, ISSBA, and BPP). For each attack and defense method, the table shows the Clean Accuracy (CA), Poisoned Accuracy (PA), and Attack Success Rate (ASR).  Bold and underlined values highlight the best and second-best performance for each metric within each attack type.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_17_2.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and a no-defense baseline.  The comparison is made across ten different backdoor attacks (BadNets, SIG, LC, TrojanNN, Dynamic, Blended, LF, WaNet, ISSBA, and BPP). For each attack and defense method, the Clean Accuracy (CA), Poisoned Accuracy (PA), and Attack Success Rate (ASR) are reported. Bold values indicate the best performance for a given metric, and underlined values show the second-best performance.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_17_3.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and a no-defense baseline.  The results are shown for ten different backdoor attacks (BadNets, SIG, LC, TrojanNN, Dynamic, Blended, LF, WaNet, ISSBA, BPP) across three metrics: Clean Accuracy (CA), Poisoned Accuracy (PA), and Attack Success Rate (ASR).  Bold and underlined values indicate the best and second-best performance for each metric and attack, respectively. The table highlights SampDetox's superior performance, particularly in achieving high Clean Accuracy while maintaining high Poisoned Accuracy and significantly lowering the Attack Success Rate.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_18_1.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and a no-defense baseline.  The comparison is done across ten different backdoor attacks, evaluating each method's Clean Accuracy (CA), Poisoned sample Accuracy (PA), and Attack Success Rate (ASR).  Bold and underlined values indicate the best and second-best performance for each metric, respectively, across the ten attacks.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_18_2.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and a no-defense baseline.  The comparison is performed across ten different backdoor attacks (BadNets, SIG, LC, TrojanNN, Dynamic, Blended, LF, WaNet, ISSBA, and BPP), each evaluated using Clean Accuracy (CA), Poisoned sample Accuracy (PA), and Attack Success Rate (ASR). Bold values indicate the best performance for each metric across all defense methods, while underlined values indicate the second-best. The table allows for a comprehensive comparison of the effectiveness of different defense strategies.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_19_1.jpg)
> This table presents the results of the SampDetox approach tested with four different diffusion models on a subset of the MS-Celeb-1M dataset.  The models tested include one trained on a subset of MS-Celeb-1M, one trained on a different subset of the same dataset, one trained on ImageNet, and a fourth model that is fine-tuned from the ImageNet model using a subset of MS-Celeb-1M.  The performance metrics (CA, PA, ASR) are shown for two attack types: BadNets (visible) and WaNet (invisible), demonstrating how SampDetox performs with diffusion models trained on different distributions.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_19_2.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and without any defense for ten different backdoor attacks.  The table shows Clean sample Accuracy (CA), Poisoned sample Accuracy (PA), and Attack Success Rate (ASR) for each method and attack.  Bold and underlined values indicate the best and second-best results, respectively, offering a clear performance comparison across various defense strategies and attack types.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_19_3.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses: Sancdifi, BDMAE, and ZIP.  For each of ten different backdoor attacks (including BadNets, SIG, LC, TrojanNN, Dynamic, Blended, LF, WaNet, ISSBA, and BPP), the table shows the Clean Accuracy (CA), Poisoned Accuracy (PA), and Attack Success Rate (ASR) for each defense method, including the proposed SampDetox.  Bold values indicate the best performance for each metric, while underlined values indicate the second-best.  This allows for a direct comparison of the effectiveness of the different methods in mitigating backdoor attacks.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_20_1.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) and without any defense.  The comparison is made across ten different backdoor attacks (BadNets, SIG, LC, TrojanNN, Dynamic, Blended, LF, WaNet, ISSBA, and BPP), evaluating the Clean sample Accuracy (CA), Poisoned sample Accuracy (PA), and Attack Success Rate (ASR) for each method and attack.  Bold and underlined values indicate the best and second-best results, respectively, highlighting SampDetox's superior performance in many cases.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_21_1.jpg)
> This table compares the performance of SampDetox against three state-of-the-art sample purification-based backdoor defenses (Sancdifi, BDMAE, and ZIP) across ten different backdoor attacks.  For each attack and defense method, it shows the clean sample accuracy (CA), poisoned sample accuracy (PA), and attack success rate (ASR).  Bold values represent the best performance for each metric, while underlined values indicate the second-best performance. The table highlights SampDetox's superior performance in terms of CA, frequently achieving the best or second-best PA and ASR across various attacks.

![](https://ai-paper-reviewer.com/Y6RV6z98Pk/tables_21_2.jpg)
> This table presents the performance comparison of SampDetox against three semantic backdoor attacks (Refool, DSFT, and CBA) across three datasets (CIFAR-10, GTSRB, and Tiny-ImageNet).  The metrics used for comparison are Clean Accuracy (CA), Poisoned Accuracy (PA), and Attack Success Rate (ASR).  The results show how well SampDetox defends against these types of attacks which focus on manipulating the semantics of the data rather than directly embedding explicit triggers.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Y6RV6z98Pk/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}