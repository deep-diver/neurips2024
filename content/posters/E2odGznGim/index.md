---
title: "Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack"
summary: "Researchers discover that existing backdoor defenses leave vulnerabilities, allowing for easy re-activation of backdoors through subtle trigger manipulation. "
categories: ["AI Generated", ]
tags: ["Computer Vision", "Image Classification", "🏢 School of Data Science,The Chinese University of Hong Kong",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} E2odGznGim {{< /keyword >}}
{{< keyword icon="writer" >}} Mingli Zhu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=E2odGznGim" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/E2odGznGim" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/E2odGznGim/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Deep neural networks are susceptible to backdoor attacks, where malicious triggers cause misclassification. While defense mechanisms exist, their effectiveness is questionable.  This paper investigates whether backdoor threats are truly eliminated after defense. 

The researchers introduce a novel metric to measure backdoor existence, finding that existing defense strategies fail to fully remove backdoors; they become dormant.  The paper then proposes re-activation attacks to reactivate these dormant backdoors, exploring white-box, black-box, and transfer attack scenarios.  **The findings highlight a critical vulnerability in existing defense strategies** and underscore the need for more robust and advanced defense mechanisms.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Existing backdoor defenses may not completely eliminate backdoors; they often remain dormant. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Dormant backdoors can be easily re-activated by manipulating the original trigger using a universal adversarial attack. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The proposed backdoor re-activation attacks work in both white-box and black-box scenarios, highlighting the robustness of the vulnerability. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it reveals a critical vulnerability in existing backdoor defenses**, demonstrating that backdoors often lie dormant rather than being eliminated. This finding emphasizes the urgent need for more robust defense mechanisms and opens avenues for research into more effective and advanced strategies.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/E2odGznGim/figures_1_1.jpg)

> 🔼 This figure compares the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  The x-axis represents the ASR, showing the success rate of backdoor attacks. The y-axis represents the BEC, a novel metric introduced in the paper to quantify the persistence of backdoors in a model, even after defense mechanisms are applied.  The plot reveals that even when ASR is reduced to near zero by defense strategies (meaning the backdoor is seemingly deactivated), the BEC remains high, indicating that the backdoor still exists in a dormant state.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>





![](https://ai-paper-reviewer.com/E2odGznGim/tables_3_1.jpg)

> 🔼 This table summarizes the four stages of a backdoor attack and defense pipeline.  The stages are: (I) Pre-training and Training (backdoor injection); (II) Post-training (backdoor defense); (III) Inference (backdoor activation attempts); and (IV) Backdoor Re-activation (the goal of this paper's attack). For each stage, the table specifies the task description, inputs/outputs, and the goal of that stage, showing how the model changes through the process and how attacks and defenses affect the results.  The clean model, backdoored model, defense model and final attack result are shown in terms of input and output values.
> <details>
> <summary>read the caption</summary>
> Table 1: Illustration of the pipeline of backdoor attack and defense.
> </details>





### In-depth insights


#### Backdoor Re-activation
The concept of "Backdoor Re-activation" in the context of deep neural network security exposes a critical vulnerability.  Existing defenses, while seemingly effective at reducing immediate attack success rates, often fail to truly eliminate backdoors.  **These dormant backdoors can be re-activated by subtly modifying the original trigger**, exploiting residual vulnerabilities within the model's architecture. This re-activation can occur even in black-box scenarios where the adversary only has inference access. **The research highlights the need for defense strategies that go beyond simply reducing immediate attack success**, emphasizing the importance of  detecting and fundamentally removing backdoor mechanisms instead of just masking their immediate effects. **Novel evaluation metrics, such as the proposed "backdoor existence coefficient," are necessary** to accurately assess the persistence of backdoors after defense, promoting the development of more robust and resilient security measures against sophisticated backdoor attacks.

#### BEC: A Novel Metric
The proposed metric, Backdoor Existence Coefficient (BEC), offers a novel approach to evaluating the persistence of backdoors in deep neural networks, even after defense mechanisms are applied.  **Unlike traditional metrics that focus solely on attack success rate (ASR), BEC directly assesses the similarity of activation patterns between backdoored and defense models.** This is crucial because ASR alone can be misleading, as a low ASR doesn't necessarily mean the backdoor is eliminated; it might simply be dormant. By comparing activation patterns, BEC provides a more nuanced measure of the backdoor's presence, revealing its persistence even if it remains inactive under the original trigger. This highlights **the importance of BEC as a supplementary evaluation tool for assessing the effectiveness of backdoor defenses.**  A high BEC after defense, even with a low ASR, is a strong indicator of a vulnerability, suggesting the need for more robust defense strategies. The paper emphasizes that **BEC effectively measures the residual backdoor effect in defense models**, contributing to a more comprehensive understanding of the backdoor threat.

#### Black-box Attacks
In the realm of deep learning security, **black-box attacks** represent a significant challenge.  Unlike white-box attacks which assume full model knowledge, black-box attacks operate under the constraint of limited or no access to model internals.  This limitation necessitates the development of innovative attack strategies that rely on external model interactions, typically via query-based methods.  These attacks focus on exploiting the model's output behavior to infer vulnerabilities and craft adversarial examples that induce misclassifications, even without knowing model parameters.  The impact of successful black-box attacks is profound, as they demonstrate the fragility of machine learning models in real-world scenarios where complete model transparency is unrealistic.  **Effective black-box attacks underscore the critical need for robust model defenses that are not easily circumvented by adversaries with limited information.**  A key area of focus within black-box attacks is the development of methods that minimize the number of queries required for successful attacks, balancing effectiveness with practicality. This emphasizes the ongoing arms race between attackers and defenders in the field of adversarial machine learning. The development of more sophisticated black-box attacks constantly pushes the boundaries of defensive strategies.

#### Defense Vulnerabilities
The concept of 'Defense Vulnerabilities' in the context of backdoor attacks on deep neural networks is crucial.  Existing defenses, while seemingly effective in reducing attack success rates, often fail to truly eliminate backdoors.  **These backdoors can remain dormant, lying hidden within the model until re-activated by cleverly manipulated triggers.** This highlights a critical vulnerability: the false sense of security provided by current defense mechanisms.  **The research emphasizes the need for novel defense strategies that address this fundamental weakness,**  actively preventing the persistence of backdoors rather than merely mitigating their immediate effects. The current methods focus primarily on reducing the attack success rate (ASR), and overlook the deeper problem of complete backdoor elimination. A key contribution is the identification of this gap and the introduction of a new metric for quantifying residual backdoor presence, providing valuable insights into the true robustness of existing defense methods.

#### Future Defenses
Future defenses against backdoor attacks must move beyond reactive patching.  **Robust defenses need to be proactive**, potentially integrated during model training, focusing on inherent model properties rather than solely on trigger identification. This requires exploring techniques that improve model robustness to adversarial examples generally, not just those with specific triggers.  **Methods focusing on feature disentanglement or robust feature extraction** would be beneficial, aiming to decouple the trigger from genuine features. **Multimodal defenses that consider the interaction between different modalities** (e.g., text and image) are also critical for applications involving multiple data types.  Further research into the **development of novel metrics** that accurately measure backdoor presence is needed, moving beyond simple attack success rates.  Ultimately, a layered defense strategy, combining multiple techniques across various stages of model development and deployment, is the most promising approach to mitigate the persistent threat of backdoor attacks.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/E2odGznGim/figures_7_1.jpg)

> 🔼 This figure compares the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  The BEC measures the similarity of activation among backdoor-related neurons in poisoned samples between a backdoored model and its corresponding defense model. The ASR is the attack success rate.  Surprisingly, the figure shows that even when defense mechanisms reduce the ASR to near zero, the BEC remains high, indicating that backdoors are dormant rather than eliminated.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_8_1.jpg)

> 🔼 This figure displays a comparative analysis of the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various models.  It shows the relationship between BEC and ASR across different attack and defense methods.  The key observation is that even when defense methods significantly reduce the ASR (close to zero, indicating successful defense), the BEC remains high, implying that the backdoors are merely dormant and not eliminated.  This finding motivates the core research question of the paper regarding the possibility of re-activating these dormant backdoors.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_19_1.jpg)

> 🔼 This figure presents a comparative analysis of the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  The x-axis represents the Backdoor Activation Rate (ASR), and the y-axis represents the Backdoor Existence Coefficient (BEC). Different shapes and colors represent different attack and defense methods. The key finding is that even when defense methods reduce the ASR to near zero, implying successful defense, the BEC remains significantly high. This indicates that the original backdoors are not eliminated but rather lie dormant within the defense models.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_19_2.jpg)

> 🔼 This figure shows a comparison of clean images with images that have been poisoned using five different backdoor attack methods: Badnet, Blended, SIG, and TrojanVQA.  Each column displays the same base image modified according to the respective attack strategy, illustrating how each method subtly alters the image to embed a backdoor trigger. The differences are often barely perceptible, highlighting the difficulty of detecting these attacks.
> <details>
> <summary>read the caption</summary>
> Figure 5: Visualization of poisoned samples for different backdoor attacks on ImageNet-1K dataset.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_23_1.jpg)

> 🔼 This figure shows a comparative analysis of the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  The x-axis represents the backdoor activation rate, and the y-axis represents the backdoor existence coefficient.  Each point represents a different model, with different shapes and colors used to distinguish various attack and defense methods. The figure demonstrates that even though defense methods can reduce the ASR, the BEC often remains high, indicating that the backdoors are merely dormant and not completely removed.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_24_1.jpg)

> 🔼 This figure displays a comparative analysis of the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various models.  The backdoor existence coefficient is a novel metric introduced in the paper to quantify the persistence of backdoors in models even after defense mechanisms are applied. The x-axis represents the backdoor activation rate (ASR), which measures the attack success rate. The y-axis represents the backdoor existence coefficient (BEC).  Different shapes and colors represent different attack and defense methods. The figure shows that even when defense models achieve near-zero ASR (meaning the backdoors appear to be eliminated), the BEC values remain significantly high, indicating that the backdoors are merely dormant rather than truly removed. This observation highlights the vulnerability of existing defense strategies.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_25_1.jpg)

> 🔼 This figure displays a comparative analysis of the backdoor existence coefficient (BEC) and the backdoor activation rate (ASR) across various attack and defense methods.  The x-axis represents the backdoor activation rate (ASR), indicating the success rate of the backdoor attack. The y-axis represents the backdoor existence coefficient (BEC), a novel metric introduced in the paper to quantify the extent of backdoor presence in a model.  Different shapes and colors represent different attack and defense methods. The figure highlights a key finding of the paper: that while defense strategies significantly reduce ASR (bringing it close to zero),  the BEC remains high, indicating that the backdoors are merely dormant rather than eliminated.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_26_1.jpg)

> 🔼 The figure displays a comparative analysis of the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  It demonstrates that while defense strategies significantly reduce ASR (almost to zero, indicating successful defense), the BEC values remain high.  This suggests that the original backdoors are not eliminated but rather lie dormant in the defense models, implying a vulnerability that existing defense mechanisms fail to address.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_27_1.jpg)

> 🔼 This figure presents a comparative analysis of the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  The x-axis represents the backdoor activation rate (ASR), and the y-axis represents the backdoor existence coefficient (BEC). Different shapes and colors represent different combinations of attack and defense methods. The figure demonstrates that even when defense models achieve near-zero ASR (implying successful defense), the BEC remains high, suggesting that backdoors still exist in the model but are dormant.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_28_1.jpg)

> 🔼 This figure presents a comparative analysis of the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  The x-axis represents the backdoor activation rate (ASR), and the y-axis represents the backdoor existence coefficient (BEC).  Different shapes and colors represent different combinations of attack and defense methods. The figure demonstrates that even when defense strategies reduce the ASR to near zero, implying successful mitigation, the BEC remains significantly high.  This indicates that backdoors are not eliminated but rather lie dormant, implying a vulnerability in existing defense techniques.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_28_2.jpg)

> 🔼 This figure compares the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  The BEC measures the similarity of activation among backdoor-related neurons in poisoned samples between the original backdoored model and its corresponding defense model.  The ASR represents the attack success rate.  Surprisingly, the figure shows that even when defense methods reduce the ASR to near zero, indicating successful defense, the BEC remains high. This suggests that backdoors are not eliminated but merely dormant in the defense models.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_29_1.jpg)

> 🔼 This figure compares the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various attack and defense methods.  The x-axis represents the backdoor activation rate (ASR), indicating the model's misclassification rate when presented with the backdoor trigger. The y-axis represents the backdoor existence coefficient (BEC), a novel metric introduced in the paper to quantify the extent of backdoor presence within models after defense. The plot shows that even when defense strategies significantly reduce the ASR (nearly to zero),  the BEC remains substantially high, implying that the backdoors are merely dormant and not completely eliminated by existing defenses. Different shapes and colors represent different combinations of attacks and defenses.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_29_2.jpg)

> 🔼 This figure displays a comparative analysis of the backdoor existence coefficient (BEC) and the backdoor activation rate (ASR) across various models.  The x-axis represents the backdoor activation rate (ASR), showing how effectively the backdoor was activated in the model.  The y-axis represents the backdoor existence coefficient (BEC), a novel metric introduced in the paper to quantify the extent to which the backdoor remains in the model even after defense mechanisms are applied. Different shapes and colors represent different attack and defense methods.  The key observation is that even when the ASR is very low (near zero, meaning the defense was successful in preventing backdoor activation), the BEC remains relatively high, indicating the presence of dormant backdoors that could potentially be reactivated.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



![](https://ai-paper-reviewer.com/E2odGznGim/figures_29_3.jpg)

> 🔼 This figure displays a comparative analysis of the backdoor existence coefficient (BEC) and backdoor activation rate (ASR) across various models.  Different shapes and colors represent different attack and defense methods. The key observation is that even when ASR is reduced to near zero by defense methods (implying successful defense), BEC remains high, indicating that the backdoors are dormant but not eliminated.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparative analysis of backdoor existence coefficient and backdoor activation rate across different models.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/E2odGznGim/tables_6_1.jpg)
> 🔼 This table presents the performance of white-box and black-box backdoor re-activation attacks against various defenses on the CIFAR-10 dataset using the PreAct-ResNet18 model.  The attacks use an l∞-norm bound of 0.05.  The table shows the attack success rate (ASR) for each attack method and defense strategy. The best results for each setting are highlighted in bold.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_6_2.jpg)
> 🔼 This table presents the performance of white-box and black-box backdoor re-activation attacks against various defense mechanisms on the CIFAR-10 dataset using the PreAct-ResNet18 model.  The attacks used seven different backdoor attack methods. The effectiveness of each attack is measured by its attack success rate (ASR), presented as a percentage. The table shows the ASR for each attack against different defenses, with the best result in each row highlighted in boldface.  It demonstrates the success rate of the re-activation attacks against existing defense models.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_6_3.jpg)
> 🔼 This table presents the performance of white-box and black-box backdoor re-activation attacks against various defense methods on the CIFAR-10 dataset using the PreAct-ResNet18 model.  The attacks utilize a perturbation with an l∞-norm bound of 0.05. The table shows the attack success rates (ASR) for several different attack methods, both before and after the application of the defense methods.  The best results (highest ASR) for each attack and defense combination are highlighted in bold.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_7_1.jpg)
> 🔼 This table presents the performance of white-box and black-box backdoor re-activation attacks against various post-training defenses on the CIFAR-10 dataset using the PreAct-ResNet18 model.  The attacks use a perturbation bound (l∞-norm) of 0.05.  The table shows the attack success rate (ASR) for each attack method (WBA and BBA) against different defenses, allowing comparison of the effectiveness of re-activation attacks against existing defense strategies.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_8_1.jpg)
> 🔼 This table presents the results of a backdoor re-activation attack on the CIFAR-10 dataset using the PreAct-ResNet18 model.  It compares the attack success rates (ASR) of white-box (WBA) and black-box (BBA) re-activation attacks against several different defense mechanisms.  The best performing attack for each defense is highlighted in bold. The l∞-norm bound for the perturbation was set to 0.05.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_8_2.jpg)
> 🔼 This table presents the Centered Kernel Alignment (CKA) scores, which measures the similarity between different feature representations.  It compares the CKA scores between the original backdoor attack (OBA), the re-activation attack (RBA), and a general universal adversarial perturbation attack (gUAA). The comparison is done for two different defense methods (i-BAU and FT-SAM) and two different attack methods (BadNets and Blended).  The table helps to demonstrate that RBA's mechanism is highly similar to OBA, and both differ significantly from gUAA.
> <details>
> <summary>read the caption</summary>
> Table 9: CKA scores between OBA, RBA, and gUAA.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_9_1.jpg)
> 🔼 This table presents the Attack Success Rate (ASR) achieved by the Re-activation Backdoor attack (RBA) and the general Universal Adversarial Attack (gUAA) against two different defense models (i-BAU and FT-SAM).  The ASR is measured for various query numbers (1000, 3000, 5000, and 7000). The results demonstrate that RBA significantly outperforms gUAA across different query numbers and defense mechanisms, highlighting its effectiveness in reactivating backdoors.
> <details>
> <summary>read the caption</summary>
> Table 10: ASR (%) of RBA and gUAA with different query numbers.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_9_2.jpg)
> 🔼 This table presents the Attack Success Rate (ASR) for three different attack methods: Original Backdoor Attack (OBA), Re-activation Backdoor Attack (RBA), and General Universal Adversarial Attack (gUAA).  The ASR is measured under different levels of random noise applied (l∞-norm of 0, 0.03, 0.06, and 0.09).  The results show the robustness of the RBA and OBA attacks compared to the gUAA attack in the presence of noise.
> <details>
> <summary>read the caption</summary>
> Table 11: ASR (%) of OBA, RBA, and gUAA under different l∞-norm of random noise.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_21_1.jpg)
> 🔼 This table presents the performance comparison of white-box (WBA) and black-box (BBA) backdoor re-activation attacks against various defense methods on the CIFAR-10 dataset using the PreAct-ResNet18 model.  The attacks aim to reactivate dormant backdoors in models that have already undergone defense mechanisms. The results are shown as percentages and the best performing attack method is highlighted in bold for each defense.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_22_1.jpg)
> 🔼 This table presents the performance of white-box and black-box backdoor re-activation attacks against various defense methods on the CIFAR-10 dataset using the PreAct-ResNet18 model.  The attacks aim to re-activate dormant backdoors in defense models by slightly perturbing the original trigger. The table shows the attack success rate (ASR) for each attack and defense combination, with the best ASR for each defense method bolded.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_22_2.jpg)
> 🔼 This table presents the performance of white-box and black-box backdoor re-activation attacks against various defense methods on the CIFAR-10 dataset using the PreAct-ResNet18 model.  The attacks use an l∞-norm bound of 0.05. The best performing attack for each defense method is shown in bold.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_23_1.jpg)
> 🔼 This table presents the results of the backdoor re-activation attack on the Tiny ImageNet dataset using the PreAct-ResNet18 model.  It compares the attack success rates (ASR) of white-box (WBA) and query-based black-box (BBA) attacks against several different defense mechanisms. The best results for each combination of attack and defense are highlighted in bold.
> <details>
> <summary>read the caption</summary>
> Table 12: Performance (%) of backdoor re-activation attack on both white-box (WBA) and query-based black-box (BBA) attacks with l∞-norm bound p = 0.05 against different defenses with Tiny ImageNet on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_25_1.jpg)
> 🔼 This table presents the Attack Success Rate (ASR) of the proposed backdoor re-activation attack against two recent defenses, SEAM and CT. The ASR is shown for four different original backdoor attacks (BadNets, Blended, Input-Aware, and LF) with and without the re-activation attack (WBA and BBA).  The results demonstrate the effectiveness of the re-activation attack against these defenses, even for those defenses designed to mitigate backdoor attacks.
> <details>
> <summary>read the caption</summary>
> Table 16: ASR (%) of our attack against SEAM and CT.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_27_1.jpg)
> 🔼 This table presents the results of transfer-based re-activation attacks.  It shows the attack success rate (ASR) achieved when using different source models (WideResNet28-2, ResNet18, VGG19-BN) to attack the target model (PreAct-ResNet18) that has undergone post-training defense. The results demonstrate the transferability of the proposed backdoor re-activation attack across various model architectures.  The table highlights the ASR for each defense method (i-BAU, FT-SAM, SAU) and attack type (BadNets, Blended).
> <details>
> <summary>read the caption</summary>
> Table 17: Transfer re-activation attack preformance (ASR %) against the target model PreAct-ResNet18, using different architectures of source models.
> </details>

![](https://ai-paper-reviewer.com/E2odGznGim/tables_27_2.jpg)
> 🔼 This table presents the results of the backdoor re-activation attack on the CIFAR-10 dataset using the PreAct-ResNet18 model.  It compares the attack success rates (ASR) under white-box (WBA) and black-box (BBA) scenarios against several defense methods. The best ASR values are highlighted in bold.  The table shows the effectiveness of the proposed re-activation attack against various existing defense strategies.
> <details>
> <summary>read the caption</summary>
> Table 2: Performance (%) of backdoor re-activation attack on both white-box (WBA) and black-box (BBA) scenarios with l∞-norm bound p = 0.05 against different defenses with CIFAR-10 on PreAct-ResNet18. The best results are highlighted in boldface.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/E2odGznGim/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/E2odGznGim/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}