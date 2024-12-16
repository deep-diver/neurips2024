---
title: "DataStealing: Steal Data from Diffusion Models in Federated Learning with Multiple Trojans"
summary: "Attackers can steal massive private data from federated learning diffusion models using multiple Trojans and an advanced attack, AdaSCP, which circumvents existing defenses."
categories: ["AI Generated", ]
tags: ["Machine Learning", "Federated Learning", "🏢 Zhejiang University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 792txRlKit {{< /keyword >}}
{{< keyword icon="writer" >}} Yuan Gan et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=792txRlKit" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/792txRlKit" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/792txRlKit/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Federated learning (FL), while designed for privacy, faces new threats from generative diffusion models.  **Existing FL defenses against single backdoors are inadequate against attacks exploiting multiple malicious triggers planted in the model by multiple compromised clients.**  This paper identifies this serious vulnerability and details how attackers can leverage it to steal a substantial amount of training data.



The researchers introduce **DataStealing**, a novel framework and attack methodology,  demonstrating its effectiveness. They also propose **AdaSCP**, a sophisticated attack that dynamically scales malicious updates to evade detection by distance-based defenses. AdaSCP targets critical parameters of the model and cleverly adjusts the magnitude of updates to blend in with benign updates.  **Extensive experiments confirm the risk of large-scale data leakage and the effectiveness of AdaSCP in defeating existing defenses.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Diffusion models in federated learning are vulnerable to data theft via multiple strategically placed Trojans. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The proposed AdaSCP attack effectively bypasses advanced distance-based defenses in federated learning. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} ComboTs allow for the efficient extraction of thousands of private images by generating specific triggers. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in federated learning and diffusion models. It highlights a critical privacy vulnerability in using diffusion models within federated learning, demonstrating how attackers can steal sensitive data using multiple Trojans.  The proposed AdaSCP attack and ComboTs significantly advance our understanding of the risks and provide avenues for developing more robust defenses. This work is timely given the increasing adoption of diffusion models in sensitive applications.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/792txRlKit/figures_1_1.jpg)

> 🔼 The figure illustrates the DataStealing attack.  Multiple Trojan models are embedded within a federated learning system training a diffusion model.  After the global model is released, the attacker uses Combinatorial Triggers (ComboTs) to extract high-quality private data from users who utilize the poisoned model.  The process occurs despite strict data management measures, highlighting the vulnerability of diffusion models to data theft in a federated learning setting.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of DataStealing. To steal data under strict management, the attacker can poison the global diffusion model in FL with multiple Trojans. After releasing the global model for users, the attacker extracts massive local data in high quality from outside with multiple triggers (ComboTs).
> </details>





![](https://ai-paper-reviewer.com/792txRlKit/tables_6_1.jpg)

> 🔼 This table presents the results of DataStealing attacks on three datasets (CIFAR-10, CelebA, and LSUN Bedroom) with non-IID data distributions.  It compares the performance of the proposed AdaSCP attack against several state-of-the-art (SOTA) attack methods, namely Data Poison, Model Poison, PGD Poison, and BC Layer Sub. The performance is evaluated under several advanced defenses, namely FedAvg, Krum, Multi-Krum, Foolsgold, RFA, and Multi-metrics, using FID and MSE as metrics.  Lower values are better for FID and MSE. Red highlights the best score and blue highlights the second-best score in each setting. The '*' indicates that the average score ignores the results where model training failed (collapsed).
> <details>
> <summary>read the caption</summary>
> Table 1: DataStealing in Non-IID Datasets. Performance of AdaSCP compared to the SOTA attack methods with various advanced defenses in non-IID distribution. '↓': lower is better. Red: the 1st score. Blue: the 2nd score. (*: averaging by ignoring the collapsed result.)
> </details>





### In-depth insights


#### ComboTs Trojan
The concept of "ComboTs Trojan" suggests a sophisticated attack method targeting diffusion models within a federated learning framework.  It likely involves combining multiple independent Trojan triggers (ComboTs) to enhance the effectiveness and stealthiness of data extraction.  **The combinatorial nature of the triggers allows for a greater number of target images to be mapped**, increasing the potential damage.  This approach is especially relevant in situations where the diffusion model is used to generate data, thus allowing the extraction of sensitive information without direct access to the underlying training datasets.  **The effectiveness of ComboTs would heavily depend on the ability to successfully inject these triggers into the model's parameters** during the federated learning process, circumventing the defensive mechanisms implemented to protect the privacy of the contributing clients' data.  The success would also hinge on the ability to extract information from the model's outputs after training, requiring a careful balance between the impact on the model's functionality and the amount of data that can be successfully exfiltrated.  **It is a critical issue to consider the possibility of multiple triggers being used in combination** to significantly enhance the attack's efficacy and the challenge this presents to the security of diffusion models within a federated learning setting.

#### AdaSCP Attack
The AdaSCP attack, as described in the provided research paper excerpt, is a sophisticated method designed to circumvent existing defenses in federated learning (FL) systems, specifically focusing on the vulnerabilities of diffusion models.  **The core of AdaSCP lies in its adaptive approach**, adjusting the scale of malicious updates to blend seamlessly with benign updates. This is achieved by evaluating the importance of model parameters, focusing on those most influential in the model's generative process and applying a carefully calculated scale factor to magnify their impact.  **This adaptive scaling technique makes it significantly harder for distance-based defenses to identify and filter the malicious update**.  By only modifying crucial parameters and scaling them precisely, the malicious contributions appear much more similar to genuine model updates, thereby evading detection mechanisms.  **The strategy also incorporates an indicator mechanism**, monitoring the model's acceptance or rejection by the central server to further refine the scaling strategy, making it adaptive and resilient to changing system responses. In essence, AdaSCP presents a significant advancement in backdoor attacks for diffusion models in FL, emphasizing the crucial need for more robust security measures to protect against such sophisticated attacks.

#### FL Privacy Risks
Federated Learning (FL), while designed to enhance data privacy, presents unique challenges.  **The decentralized nature of FL, involving multiple clients training a shared model without directly sharing data, introduces vulnerabilities.**  Malicious actors could exploit these vulnerabilities to infer sensitive information.  **Data poisoning attacks**, where malicious clients inject corrupted data, can significantly impact model accuracy and potentially reveal private information.  **Model inversion attacks** aim to reconstruct training data from the model updates shared by clients, posing a substantial threat to privacy.  Additionally, **the heterogeneity and non-IID nature of data** across clients may lead to unexpected information leakage.  Therefore, robust defense mechanisms are crucial to mitigate these risks and ensure privacy preservation in FL.  **Advanced defenses**, employing techniques such as secure aggregation, differential privacy, and robust optimization, are needed to protect against increasingly sophisticated attacks.  **The effectiveness of these defenses and the extent of potential privacy breaches remain active areas of research.**

#### Defense Evasion
In the realm of adversarial machine learning, **defense evasion** is a critical aspect focusing on the ability of an attacker to circumvent implemented security measures.  Successfully evading defenses requires a deep understanding of the underlying mechanisms and potential weaknesses.  **Adaptive strategies** are crucial, as static attacks are often easily detected and neutralized.  Therefore, attackers must develop methods that dynamically adjust to the specific defense in place, making detection and mitigation significantly more challenging.  **Analyzing the limitations of existing defenses** is a key element of the evasion process, as this reveals exploitable vulnerabilities.  **Adversarial training**, a common defensive technique, can be evaded through novel attack methods that exploit the inherent limitations of this approach.  The effectiveness of evasion techniques often depends on the specific **threat model** being utilized, including the resources and knowledge available to the attacker.  Ultimately, **robust defenses** require a multi-faceted approach that considers various potential attack vectors and dynamically adapts to emerging evasion techniques.

#### Future Defenses
Future defenses against DataStealing attacks on federated learning (FL) with diffusion models must be multifaceted.  **Strengthening distance-based defenses** is crucial, perhaps by incorporating more robust anomaly detection methods that are less susceptible to adversarial manipulation.  **Improving gradient-based defenses** to better handle diffusion models' unique properties is also needed. This might involve designing novel techniques to filter malicious updates more effectively and to make the identification of critical parameters more robust.  Exploring **algorithmic defenses** focused on identifying and mitigating the impact of multiple trojans, rather than single backdoors, is essential.  Finally, enhancing data sanitization and privacy-preserving mechanisms within the FL framework, such as differential privacy, is necessary.  **Research should focus on developing techniques to actively detect and counter the manipulation of critical model parameters**, as this is the core vulnerability exploited by AdaSCP and similar attacks.  A layered defense strategy combining multiple approaches will likely be needed to effectively protect the privacy of sensitive data within FL settings using diffusion models.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/792txRlKit/figures_2_1.jpg)

> 🔼 This figure illustrates the ComboTs and AdaSCP methods.  (a) shows how ComboTs select multiple trigger points to map to target images. (b) details the Trojan diffusion process, showing how the poisoned model can reconstruct target images from the added noise. (c) depicts the AdaSCP attack, which adapts the scale of updates to bypass defenses.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overview of ComboTs and AdaSCP. (a) ComboTs choose two points from candidate positions to form multiple triggers for mapping target images. (b) The forward and backward Trojan diffusion process. After training with ComboTs, the poisoned model can restore the target images in high quality from Trojan noise, thereby enabling DataStealing. (c) AdaSCP achieves the purpose of DataStealing and defeats the advanced defenses by training critical parameters and adaptively scaling the updates before uploading.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_5_1.jpg)

> 🔼 This figure illustrates the proposed ComboTs and AdaSCP attack methods.  (a) shows how ComboTs selects multiple trigger points to map to target images for data exfiltration. (b) demonstrates the Trojan diffusion process, where the poisoned model uses ComboTs to reconstruct target images from Trojan noise.  Finally, (c) explains AdaSCP, which circumvents defenses by training on critical parameters and adaptively scaling updates before uploading.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overview of ComboTs and AdaSCP. (a) ComboTs choose two points from candidate positions to form multiple triggers for mapping target images. (b) The forward and backward Trojan diffusion process. After training with ComboTs, the poisoned model can restore the target images in high quality from Trojan noise, thereby enabling DataStealing. (c) AdaSCP achieves the purpose of DataStealing and defeats the advanced defenses by training critical parameters and adaptively scaling the updates before uploading.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_6_1.jpg)

> 🔼 This figure shows a qualitative comparison of the generated images produced by different attack methods under various defense mechanisms.  The image with the lowest mean squared error (MSE) for each attack and defense combination is displayed. This visually demonstrates the effectiveness (or lack thereof) of each attack in generating realistic images while evading the defenses. More detailed visual results are available in Appendix A.10.
> <details>
> <summary>read the caption</summary>
> Figure 3: Qualitative Comparison. The generated image with the lowest MSE in one trigger-target pair is presented under every attack and defense method. More visual results are in Appendix A.10.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_7_1.jpg)

> 🔼 The figure shows the impact of the number of triggers used in the AdaSCP attack on the FID and MSE scores.  The results indicate a trade-off: using more triggers can lead to more successful data extraction (lower MSE) but comes at the cost of potentially harming the generative quality of the model (higher FID).  The experiment used the FedAvg and Multi-Krum defense mechanisms in a non-IID setting on the CIFAR10 dataset.
> <details>
> <summary>read the caption</summary>
> Figure 4: Ablation Study on Trigger Number.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_14_1.jpg)

> 🔼 This figure shows the results of an experiment designed to demonstrate that existing defense mechanisms struggle to prevent successful attacks by adversaries over extended training periods.  The experiment used the AdaSCP attack against the Multi-Krum defense on the CIFAR10 dataset with a non-IID data distribution.  The x-axis represents the training round, while the y-axis shows both FID (Frechet Inception Distance) and MSE (Mean Squared Error). The plot shows that while AdaSCP initially fails to succeed within 300 rounds, it becomes successful after extending the training duration to 1500 rounds.  This indicates that longer training times can make the AdaSCP attack more effective against Multi-Krum. The images embedded within the graph illustrate sample generated images at different points in the training process.
> <details>
> <summary>read the caption</summary>
> Figure 5: Attacking in an Extended Training. Given more time, AdaSCP can attack successfully under the defense of Multi-Krum in the Non-IID distribution of CIFAR10.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_14_2.jpg)

> 🔼 This figure shows the results of an experiment designed to demonstrate that existing defense mechanisms struggle to prevent successful attacks by adversaries over extended training periods.  The experiment used the AdaSCP attack against the Multi-Krum defense mechanism on the CIFAR10 dataset with a non-IID data distribution.  The x-axis represents the training round, and the y-axis shows both FID (Frechet Inception Distance) and MSE (Mean Squared Error) values. The plot shows that AdaSCP attack initially fails within 300 rounds, but after extending the training to 1500 rounds, AdaSCP successfully defeats the Multi-Krum defense, indicating that longer training times can make the attack more stealthy and effective.
> <details>
> <summary>read the caption</summary>
> Figure 5: Attacking in an Extended Training. Given more time, AdaSCP can attack successfully under the defense of Multi-Krum in the Non-IID distribution of CIFAR10.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_15_1.jpg)

> 🔼 This figure shows the relative distance between malicious updates and the mean distance of benign updates during the first 100 training rounds when using AdaSCP attack against Krum and Multi-Krum defense mechanisms on CIFAR10 dataset.  The graph illustrates how AdaSCP successfully reduces the distance of malicious updates to the mean benign updates over time, allowing it to evade the defenses.
> <details>
> <summary>read the caption</summary>
> Figure 7: Relative Distance. Ratio of malicious updates distance to mean benign updates distance when attacking with AdaSCP on CIFAR10 in the first 100 rounds.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_18_1.jpg)

> 🔼 This figure illustrates the ComboTs and AdaSCP methods.  ComboTs (a) selects multiple points to create triggers for embedding target images into a diffusion model. The Trojan diffusion process (b) adds noise to the original images and uses the triggers to embed the target images. The AdaSCP attack (c) adaptively scales the malicious updates based on gradient importance and evades distance-based defenses by making the malicious updates appear similar to benign updates.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overview of ComboTs and AdaSCP. (a) ComboTs choose two points from candidate positions to form multiple triggers for mapping target images. (b) The forward and backward Trojan diffusion process. After training with ComboTs, the poisoned model can restore the target images in high quality from Trojan noise, thereby enabling DataStealing. (c) AdaSCP achieves the purpose of DataStealing and defeats the advanced defenses by training critical parameters and adaptively scaling the updates before uploading.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_19_1.jpg)

> 🔼 This figure illustrates the ComboTs and AdaSCP attack methods.  (a) shows how ComboTs selects multiple trigger points within an image to enable the mapping of multiple target images. (b) depicts the forward and reverse diffusion processes involved in embedding and extracting these images, using Trojan noise. Finally, (c) illustrates how AdaSCP adaptively scales critical parameters in the poisoned model updates to evade distance-based defenses, thereby successfully performing DataStealing.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overview of ComboTs and AdaSCP. (a) ComboTs choose two points from candidate positions to form multiple triggers for mapping target images. (b) The forward and backward Trojan diffusion process. After training with ComboTs, the poisoned model can restore the target images in high quality from Trojan noise, thereby enabling DataStealing. (c) AdaSCP achieves the purpose of DataStealing and defeats the advanced defenses by training critical parameters and adaptively scaling the updates before uploading.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_20_1.jpg)

> 🔼 This figure shows the visual results of applying various attacks (Data Poison, Model Poison, PGD Poison, BC Layer Sub, AdaSCP) against LSUN bedroom dataset under different defense mechanisms (FedAvg, Krum, Multi-Krum, Foolsgold, RFA, Multi-metrics).  Each cell in the figure represents the generated image for a specific attack and defense combination.  The goal is to show the effectiveness of different attacks against various defenses in stealing private data from the diffusion model. The image quality and the success of the attack vary depending on the combination of attack and defense.
> <details>
> <summary>read the caption</summary>
> Figure 10: Visual Results of LSUN Bedroom in Non-IID Distribution.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_20_2.jpg)

> 🔼 This figure shows the visual results of the LSUN Bedroom dataset under various attack methods (Data Poison, Model Poison, PGD Poison, BC Layer Sub, AdaSCP) and defense mechanisms (FedAvg, Krum, Multi-Krum, Foolsgold, RFA, Multi-metrics) in a non-IID data distribution. Each cell in the figure represents a generated image under a specific attack and defense combination. The target image is displayed in the first column. The figure demonstrates the effectiveness of AdaSCP in generating high-quality images, even when defenses are in place, and how different attacks and defenses affect the quality and fidelity of the generated images.
> <details>
> <summary>read the caption</summary>
> Figure 10: Visual Results of LSUN Bedroom in Non-IID Distribution.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_21_1.jpg)

> 🔼 This figure visualizes images sampled from a diffusion model that has been attacked using the AdaSCP method.  The images are generated using the same trigger but with different Gaussian noise added during the sampling process. This shows the effect of noise on the generation process and the ability of the attack method to produce slightly varied outputs from the same trigger. The figure contains two parts, one for the CIFAR10 dataset and another for the CelebA dataset.
> <details>
> <summary>read the caption</summary>
> Figure 11: Sampled Images with Different Noise. We visualize the sampled images in CIFAR10 and CelebA attacked by our AdaSCP under FedAvg. These images are generated with the same trigger and different Gaussian noise.
> </details>



![](https://ai-paper-reviewer.com/792txRlKit/figures_21_2.jpg)

> 🔼 This figure illustrates the DataStealing attack.  Multiple poisoned clients inject Trojan backdoors into a global diffusion model trained via federated learning (FL).  The attacker then uses these backdoors to extract private data by using multiple triggers, called ComboTs, to manipulate the generated images from the released model.  The figure shows the process from the poisoned clients, the central server with defenses, and the final generation of data from triggers. 
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of DataStealing. To steal data under strict management, the attacker can poison the global diffusion model in FL with multiple Trojans. After releasing the global model for users, the attacker extracts massive local data in high quality from outside with multiple triggers (ComboTs).
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/792txRlKit/tables_7_1.jpg)
> 🔼 This table presents the results of DataStealing experiments conducted on three datasets (CIFAR10, CelebA, and LSUN Bedroom) using non-IID data distribution.  It compares the performance of the proposed AdaSCP attack against several state-of-the-art (SOTA) attack methods.  The comparison includes various advanced defenses like FedAvg, Krum, Multi-Krum, Foolsgold, RFA, and Multi-metrics. The metrics used for evaluation are FID (Frechet Inception Distance) and MSE (Mean Squared Error). Lower values for FID and MSE indicate better performance.  The table highlights the effectiveness of AdaSCP in surpassing other methods across different defense mechanisms.
> <details>
> <summary>read the caption</summary>
> Table 1: DataStealing in Non-IID Datasets. Performance of AdaSCP compared to the SOTA attack methods with various advanced defenses in non-IID distribution. '': lower is better. Red: the 1st score. Blue: the 2nd score. (*: averaging by ignoring the collapsed result.)
> </details>

![](https://ai-paper-reviewer.com/792txRlKit/tables_8_1.jpg)
> 🔼 This table presents the results of the DataStealing attack using AdaSCP and several other state-of-the-art (SOTA) attack methods on three different datasets (LSUN Bedroom, CelebA, and CIFAR10) under various defense mechanisms in a non-IID data distribution setting.  The table shows the FID (Frechet Inception Distance) and MSE (Mean Squared Error) for each attack method and defense mechanism. Lower FID and MSE values generally indicate better performance. The table highlights AdaSCP's performance compared to other methods and its ability to bypass various advanced defenses.
> <details>
> <summary>read the caption</summary>
> Table 1: DataStealing in Non-IID Datasets. Performance of AdaSCP compared to the SOTA attack methods with various advanced defenses in non-IID distribution. '↓': lower is better. Red: the 1st score. Blue: the 2nd score. (*: averaging by ignoring the collapsed result.)
> </details>

![](https://ai-paper-reviewer.com/792txRlKit/tables_15_1.jpg)
> 🔼 This table presents a comparison of the DataStealing performance of AdaSCP against state-of-the-art (SOTA) attack methods across different defense mechanisms.  It specifically focuses on non-identically and independently distributed (non-IID) datasets. The table shows FID (Fréchet Inception Distance) and MSE (Mean Squared Error) scores, where lower values are better, indicating successful data exfiltration. The table highlights AdaSCP's effectiveness in surpassing SOTA approaches, particularly when various advanced defenses are employed.
> <details>
> <summary>read the caption</summary>
> Table 1: DataStealing in Non-IID Datasets. Performance of AdaSCP compared to the SOTA attack methods with various advanced defenses in non-IID distribution. '': lower is better. Red: the 1st score. Blue: the 2nd score. (*: averaging by ignoring the collapsed result.)
> </details>

![](https://ai-paper-reviewer.com/792txRlKit/tables_16_1.jpg)
> 🔼 This table presents a comparison of the DataStealing performance using AdaSCP against other state-of-the-art (SOTA) attack methods. The comparison is done across various advanced defenses in a non-IID (non-independent and identically distributed) data setting.  The table shows the FID (Fréchet Inception Distance) and MSE (Mean Squared Error) scores for different attacks and defenses. Lower FID and MSE values are better, indicating better performance. The table highlights AdaSCP's superior performance in most scenarios.
> <details>
> <summary>read the caption</summary>
> Table 1: DataStealing in Non-IID Datasets. Performance of AdaSCP compared to the SOTA attack methods with various advanced defenses in non-IID distribution. '↓': lower is better. Red: the 1st score. Blue: the 2nd score. (*: averaging by ignoring the collapsed result.)
> </details>

![](https://ai-paper-reviewer.com/792txRlKit/tables_16_2.jpg)
> 🔼 This table presents the results of DataStealing experiments conducted on non-IID datasets using various state-of-the-art (SOTA) attack methods and advanced defense mechanisms.  The table compares the performance of the proposed AdaSCP attack against other methods in terms of FID and MSE scores under different defense strategies (FedAvg, Krum, Multi-Krum, Foolsgold, RFA, Multi-metrics).  Lower FID and MSE scores indicate better performance. The table highlights AdaSCP's superior performance in most scenarios, particularly when facing advanced defenses.
> <details>
> <summary>read the caption</summary>
> Table 1: DataStealing in Non-IID Datasets. Performance of AdaSCP compared to the SOTA attack methods with various advanced defenses in non-IID distribution. '': lower is better. Red: the 1st score. Blue: the 2nd score. (*: averaging by ignoring the collapsed result.)
> </details>

![](https://ai-paper-reviewer.com/792txRlKit/tables_16_3.jpg)
> 🔼 This table presents the results of an ablation study conducted to determine the optimal patch size for ComboTs in the DataStealing attack.  The study varied the patch size from 1x1 to 15x15 pixels and measured the performance using the FID and MSE metrics under the FedAvg defense mechanism. The results show that a 3x3 patch size achieves a good balance between attack effectiveness and the impact on the generative quality of the model. Smaller patches were insufficient to distinguish the triggers from the noise, while larger patches degraded the image quality and hence model performance. This study guided the selection of the patch sizes used in subsequent experiments.
> <details>
> <summary>read the caption</summary>
> Table 7: Ablation Study on the Patch Size of ComboTs.
> </details>

![](https://ai-paper-reviewer.com/792txRlKit/tables_17_1.jpg)
> 🔼 This table presents the results of the DataStealing attacks on three datasets (LSUN Bedroom, CelebA, and CIFAR10) under different non-IID data distributions.  The table compares the performance of AdaSCP against other state-of-the-art (SOTA) attack methods. Each method is evaluated with various advanced defenses (FedAvg, Krum, Multi-Krum, Foolsgold, RFA, Multi-metrics).  The metrics used to evaluate performance are FID (Frechet Inception Distance) and MSE (Mean Squared Error).  Lower values for both FID and MSE indicate better performance. The table highlights AdaSCP's superior performance in most scenarios, even when compared to other advanced attack methods.
> <details>
> <summary>read the caption</summary>
> Table 1: DataStealing in Non-IID Datasets. Performance of AdaSCP compared to the SOTA attack methods with various advanced defenses in non-IID distribution. '': lower is better. Red: the 1st score. Blue: the 2nd score. (*: averaging by ignoring the collapsed result.)
> </details>

![](https://ai-paper-reviewer.com/792txRlKit/tables_25_1.jpg)
> 🔼 This table presents the results of DataStealing attacks on non-IID datasets using various advanced defenses.  It compares the performance of the proposed AdaSCP method against state-of-the-art (SOTA) attack methods.  The metrics used are FID and MSE, with lower values indicating better performance. The table highlights AdaSCP's effectiveness in overcoming various defenses, even in non-IID data scenarios.
> <details>
> <summary>read the caption</summary>
> Table 1: DataStealing in Non-IID Datasets. Performance of AdaSCP compared to the SOTA attack methods with various advanced defenses in non-IID distribution. '↓': lower is better. Red: the 1st score. Blue: the 2nd score. (*: averaging by ignoring the collapsed result.)
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/792txRlKit/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/792txRlKit/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}