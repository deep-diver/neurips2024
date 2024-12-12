---
title: "Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor"
summary: "Proactive Defensive Backdoor (PDB) thwarts malicious backdoors by injecting a hidden defensive backdoor during training, suppressing attacks while maintaining model utility."
categories: []
tags: ["Machine Learning", "Deep Learning", "🏢 School of Data Science",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} cbkJBYIkID {{< /keyword >}}
{{< keyword icon="writer" >}} Shaokui Wei et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=cbkJBYIkID" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94407" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=cbkJBYIkID&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/cbkJBYIkID/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Machine learning models are vulnerable to backdoor attacks, where malicious actors manipulate training data to introduce hidden triggers that cause incorrect predictions.  Existing defense mechanisms primarily focus on identifying and removing these poisoned samples, which can be computationally expensive and unreliable.  This often leads to decreased model accuracy or increased training costs. 

This research proposes a novel defense strategy called Proactive Defensive Backdoor (PDB). Unlike traditional methods, PDB proactively injects a defensive backdoor into the model during training.  This defensive backdoor is designed to suppress malicious backdoors when a specific trigger is present while remaining undetectable to attackers.  The approach also incorporates a reversible mapping to maintain the model's performance on the original task. Extensive experiments show that PDB outperforms existing state-of-the-art defenses across various datasets and model architectures.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} PDB introduces a proactive, rather than reactive, approach to backdoor defense. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method uses a defensive backdoor to effectively counteract malicious backdoors without relying on detecting poisoned data. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Experimental results demonstrate PDB's state-of-the-art performance against various backdoor attacks. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it introduces a novel proactive defense against backdoor attacks in machine learning models**.  By shifting from reactive detection to preventative injection of a defensive backdoor, it offers a more efficient and robust solution to a critical security challenge.  The findings have significant implications for enhancing the reliability and security of AI systems across various domains and offer a new direction for future research in backdoor defense strategies.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/cbkJBYIkID/figures_3_1.jpg)

> This figure illustrates a bijective mapping function h(y) = (y+1) mod K, where K=4.  The mapping is shown using a table, illustrating how each label (0, 1, 2, 3) is mapped to a different label. This is a key component of the PDB (Proactive Defensive Backdoor) method described in the paper, allowing for the reversible mapping of target labels for the defensive backdoor.  The image also shows an example of a patched trigger being applied to an image, modifying pixel values outside the normal range to create the trigger.





![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_6_1.jpg)

> This table presents the experimental results on the CIFAR-10 dataset using the PreAct-ResNet18 model with a poisoning ratio of 5%. It compares the performance of seven different backdoor defense methods against seven different backdoor attacks.  The metrics used for comparison include Accuracy (ACC), Attack Success Rate (ASR), and Defense Effectiveness Rating (DER).  The table shows the ACC, ASR, and DER for each defense method on each attack, allowing for a comprehensive comparison of their effectiveness.





### In-depth insights


#### Proactive Backdoor
A proactive backdoor defense strategy, as the name suggests, **injects a defensive backdoor into the model during training**, rather than reacting to attacks after training. This approach utilizes the defender's advantage of controlling the training process.  **The key objective is to suppress malicious backdoors without compromising the model's utility for its intended purpose.** By strategically embedding this defensive backdoor, the system can effectively counteract malicious attempts to subvert the model's predictions using specific triggers.  A critical aspect of this technique is the **design of the defensive backdoor** itself, which needs to be reversible, inaccessible to attackers, minimally impactful on model performance, and robust to various other attack types. This proactive strategy potentially improves on traditional reactive approaches which often rely on detecting and mitigating malicious elements after the model has already been trained, thus making it **more efficient and effective**. However, potential limitations remain regarding the generalizability and robustness against sophisticated attacks that might not be detected by existing approaches, highlighting the need for further analysis and development.

#### Defense Principles
Effective backdoor defenses necessitate robust principles.  **Reversibility** is crucial; the defense should restore the original label after trigger activation.  This ensures model utility remains unaffected.  **Inaccessibility** to attackers is paramount; the defensive trigger must be cleverly designed, preventing adversaries from replicating or identifying it easily.  **Minimal impact** on performance is vital, with the defense ideally maintaining accuracy and efficiency on regular inputs.  Finally, **resistance** against diverse attacks is needed; the defense must effectively mitigate a wide range of backdoor attack methods.  These principles, when cohesively implemented, establish a strong, robust backdoor defense. Achieving these requires careful design and evaluation to ensure the model remains resilient against both known and emerging attack strategies.

#### Empirical Results
An 'Empirical Results' section in a research paper should present a comprehensive evaluation of the proposed method.  It should not only report performance metrics like accuracy, precision, and recall but also delve into the nuances.  **Statistical significance** should be clearly established, ideally with error bars or p-values.  **Comparative analysis** against state-of-the-art baselines is crucial, showcasing the method's strengths and weaknesses. The results should be presented in a clear and organized manner, often using tables and graphs. **Visualizations** can enhance understanding, particularly when comparing different methods or showing performance across various conditions.  A detailed discussion of **limitations** and potential sources of error helps ensure the results are interpreted responsibly, while **robustness analysis** (e.g. testing different data settings) is key to demonstrating generalizability.

#### Limitations & Future
A research paper's "Limitations & Future" section should critically examine the study's shortcomings.  **Data limitations** such as dataset size or bias should be frankly addressed, acknowledging their potential impact on results.  **Methodological limitations**, including assumptions made or specific techniques used, should be detailed, explaining why alternative approaches might yield different outcomes.  **Generalizability** is a crucial concern; the discussion should clarify whether findings extend beyond the specific context of the study, perhaps suggesting future avenues for testing wider applicability.  The "Future" aspect should propose concrete, actionable next steps, such as identifying datasets for further investigation, exploring modifications to the methodology, or suggesting alternative research questions.  **Focusing on limitations** demonstrates intellectual honesty and enhances the paper's credibility, while the future directions section highlights the potential for broader impact and future research endeavors.

#### Comparative Analysis
A comparative analysis section in a research paper would systematically contrast the proposed method against existing state-of-the-art approaches.  This involves a detailed examination of performance metrics across multiple datasets and experimental setups, highlighting both strengths and weaknesses.  **Key aspects** to consider include: accuracy, efficiency (training time, computational resources), robustness to various attack scenarios, and generalizability to unseen data.  The analysis should not only report quantitative results, but also offer insightful qualitative comparisons, such as explaining discrepancies and discussing the underlying reasons for superior or inferior performance.  **A robust analysis** might also explore the trade-offs between different methods, for example, a method achieving higher accuracy but requiring significantly more computational resources.  Finally, the **conclusions** drawn from the comparison must be clearly articulated, summarizing the advantages of the proposed method and acknowledging its limitations relative to existing techniques.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/cbkJBYIkID/figures_3_2.jpg)

> This figure illustrates how a defensive poisoned sample is generated.  It takes a benign image and applies a trigger by modifying pixel values outside the range of [0, 1]. A mask is used to specify which pixel values are to be modified, with white representing modification and black representing no modification. The result is a defensive poisoned sample that, when fed into the model during inference, activates the defensive backdoor.


![](https://ai-paper-reviewer.com/cbkJBYIkID/figures_4_1.jpg)

> This figure illustrates the overall pipeline of the proposed Proactive Defensive Backdoor (PDB) method. It shows how a defensive backdoor is proactively injected into the model during training to counteract malicious backdoors.  The process begins with data preparation, where both malicious and defensive poisoned datasets are created. These datasets are then used in the model training phase, where the model learns to respond correctly to both benign inputs and inputs containing the defensive trigger. During inference, the input image is embedded with the defensive trigger, allowing the model to correctly predict the label and mitigate the malicious backdoor's impact. The figure highlights the key components: malicious trigger, defensive trigger, target label mapping, and the model's response to different input types.


![](https://ai-paper-reviewer.com/cbkJBYIkID/figures_7_1.jpg)

> This figure presents a visualization of the effects of the proposed Proactive Defensive Backdoor (PDB) method on a backdoor attack.  It uses t-SNE to show how the features of the model change in the presence and absence of the defensive trigger. The Trigger Activation Change (TAC) plots show the change in neuron activation in the first and fourth blocks of the PreAct-ResNet18 model. This illustrates how PDB alters the model's behavior in response to its defensive trigger, mitigating the effect of the malicious backdoor.


![](https://ai-paper-reviewer.com/cbkJBYIkID/figures_19_1.jpg)

> This figure visualizes the impact of the proposed Proactive Defensive Backdoor (PDB) method on the BadNets attack using t-SNE and Trigger Activation Change (TAC).  The t-SNE plot shows how PDB alters the feature representation of the data, separating the poisoned samples from the benign ones.  The TAC plots (for the first and fourth blocks of the network) illustrate how the activation changes caused by the malicious backdoor are mitigated by the defensive backdoor in PDB.


![](https://ai-paper-reviewer.com/cbkJBYIkID/figures_20_1.jpg)

> The figure shows three different masks used for patched triggers in the proposed Proactive Defensive Backdoor (PDB) method.  These masks represent different patterns that can be incorporated into an image as a trigger for the defensive backdoor. The variations in trigger design are meant to explore the influence of trigger characteristics on the effectiveness of the defensive backdoor in mitigating malicious backdoor attacks.


![](https://ai-paper-reviewer.com/cbkJBYIkID/figures_21_1.jpg)

> This figure shows the results of the defense method with different strengths of augmentation against four different backdoor attacks: BadNets, Blended, WaNet, and SSBA.  The x-axis represents the strength of the augmentation, and the y-axis represents both the accuracy (ACC) and the attack success rate (ASR).  The graph shows a tradeoff; as the augmentation strength increases, the ASR decreases, indicating that stronger augmentation helps further enhance the effectiveness of the defense mechanism; however, a tradeoff between augmentation intensity and model performance is also observed.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_6_2.jpg)
> This table presents the results of the proposed Proactive Defensive Backdoor (PDB) method and five other state-of-the-art (SOTA) in-training defense methods against seven SOTA data-poisoning backdoor attack methods on the Tiny ImageNet dataset using the Vision Transformer (ViT-B-16) model architecture. The poisoning ratio is 5%.  The table displays the accuracy on benign data (ACC), attack success rate (ASR), and defense effectiveness rating (DER) for each method and attack combination.  Higher ACC and DER values, and lower ASR values, indicate better performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_8_1.jpg)
> This table presents the results of the proposed Proactive Defensive Backdoor (PDB) method and several other defense methods against backdoor attacks under different poisoning ratios (1%, 5%, 10%, 20%, and 40%).  It shows the accuracy (ACC) on benign data and the attack success rate (ASR) for each method and poisoning ratio.  It demonstrates the effectiveness of the PDB method in maintaining high accuracy while reducing the ASR across different poisoning levels.  The table highlights the resilience of PDB across varying dataset contamination levels.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_8_2.jpg)
> This table presents the training time in seconds for different defense methods and the baseline (no defense) on three datasets: GTSRB, CIFAR-10, and Tiny ImageNet.  The experiments were conducted using an RTX 4090Ti GPU against the BadNets attack with a poisoning ratio of 5%.  The table highlights the significant increase in runtime for DBD and NAB compared to the baseline and PDB, primarily due to their reliance on self-supervised and semi-supervised training techniques.  PDB demonstrates comparable runtime to the baseline, showcasing its efficiency.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_8_3.jpg)
> This table presents the results of different backdoor defense methods on the CIFAR-10 dataset using the PreAct-ResNet18 model architecture, with a poisoning ratio of 5%.  The table compares the performance of several defense methods (AC, Spectral, ABL, DBD, NAB, and the proposed PDB method) against seven different backdoor attacks (BadNets, Blended, SIG, SSBA, WaNet, BPP, and Trojan). For each defense method and each attack, the table shows the accuracy on benign data (ACC), the attack success rate (ASR), and the defense effectiveness rating (DER). Higher ACC, lower ASR, and higher DER indicate better defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_9_1.jpg)
> This table presents the results of the proposed PDB method against adaptive attacks with varying trigger sizes and poisoning ratios.  It demonstrates the robustness of PDB against attacks that try to adapt to the defensive backdoor. The table shows that PDB consistently mitigates backdoor attacks, even when the attacker adapts the size of the trigger or the amount of poisoned data.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_16_1.jpg)
> This table presents the results of different defense methods against backdoor attacks on the GTSRB dataset using the PreAct-ResNet18 model architecture.  The methods are evaluated with a 5% poisoning ratio. The table shows the accuracy on benign data (ACC), the attack success rate (ASR), and the defense effectiveness rating (DER) for each defense method against each type of backdoor attack.  A higher ACC and lower ASR indicate better defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_17_1.jpg)
> This table presents the experimental results on CIFAR-10 dataset using PreAct-ResNet18 model with a poisoning ratio of 5%. It compares the performance of different backdoor defense methods against seven different backdoor attacks. The metrics used for comparison include Accuracy on benign data (ACC), Attack Success Rate (ASR), and Defense Effectiveness Rating (DER).  Higher ACC and lower ASR values generally indicate better defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_17_2.jpg)
> This table presents the experimental results on the CIFAR-10 dataset using the PreAct-ResNet18 model with a poisoning ratio of 5%.  It compares the performance of seven different backdoor attacks against six different defense methods, including the proposed PDB method.  The results are shown in terms of Accuracy (ACC), Attack Success Rate (ASR), and Defense Effectiveness Rating (DER).  Higher ACC, lower ASR, and higher DER indicate better defense performance.  The table allows for a direct comparison of the effectiveness of various defense mechanisms against different types of backdoor attacks.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_17_3.jpg)
> This table presents the experimental results on CIFAR-10 dataset using PreAct-ResNet18 model with a poisoning ratio of 5.0%. It compares the performance of seven different backdoor attacks against six defense methods including the proposed method (PDB). The results are presented in terms of accuracy (ACC) on benign data, attack success rate (ASR), and defense effectiveness rating (DER). A higher ACC, a lower ASR, and a higher DER indicate better defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_17_4.jpg)
> This table presents the results of the proposed Proactive Defensive Backdoor (PDB) method and five state-of-the-art (SOTA) in-training defense methods against seven SOTA data-poisoning backdoor attack methods.  The results are shown for the CIFAR-10 dataset using the PreAct-ResNet18 model architecture with a 5% poisoning ratio.  The table shows the Accuracy (ACC) on benign samples, Attack Success Rate (ASR), and Defense Effectiveness Rating (DER) for each method and attack.  Higher ACC and lower ASR indicate better performance.  DER balances ACC and ASR, with higher values indicating superior defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_18_1.jpg)
> This table presents the results of different defense methods against backdoor attacks on the CIFAR-10 dataset using the VGG19-BN model architecture.  The methods were evaluated with a 5% poisoning ratio.  The table shows the accuracy (ACC) on benign data, attack success rate (ASR), and defense effectiveness rating (DER) for each defense method and each of seven different backdoor attacks.  Higher ACC and lower ASR values indicate better defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_18_2.jpg)
> This table presents the results of the proposed Proactive Defensive Backdoor (PDB) method and five state-of-the-art (SOTA) in-training defense methods against seven SOTA data-poisoning backdoor attack methods on the CIFAR-10 dataset using the PreAct-ResNet18 model.  The results are shown for Accuracy on benign data (ACC), Attack Success Rate (ASR), and Defense Effectiveness Rating (DER).  The table compares the performance of different defense methods against various backdoor attacks.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_18_3.jpg)
> This table presents the results of different defense methods against backdoor attacks on the CIFAR-10 dataset using the PreAct-ResNet18 model architecture.  The methods were evaluated at a poisoning ratio of 10.0%.  The table shows the accuracy (ACC), attack success rate (ASR), and defense effectiveness rating (DER) for each method and attack type.  It helps to compare the effectiveness of different defense mechanisms against various backdoor attacks under a higher poisoning ratio.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_19_1.jpg)
> This table presents the results of the proposed method (PDB) and other defense methods against various backdoor attacks on the Tiny ImageNet dataset using the ViT-B-16 model architecture. The results are shown for different poisoning ratios (1%, 5%, and 10%), demonstrating the effectiveness of PDB in mitigating backdoor attacks for large datasets and models.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_19_2.jpg)
> This table presents the results of experiments conducted to evaluate the performance of the proposed method (PDB) against invisible backdoor attacks and attacks with low poisoning ratios (0.1% and 0.5%).  The table shows the accuracy (ACC) and attack success rate (ASR) for both PDB and a baseline (no defense) across different attack types (BadNet, Blended, Sig, SSBA, WaNet) and trigger visibilities (Visible/Invisible). The results demonstrate PDB's effectiveness in mitigating backdoor attacks even under these challenging conditions.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_20_1.jpg)
> This table presents the experimental results on the CIFAR-10 dataset using the PreAct-ResNet18 model with a poisoning ratio of 5%. It compares the performance of the proposed Proactive Defensive Backdoor (PDB) method against several state-of-the-art (SOTA) in-training defense methods across seven different backdoor attacks.  The results are shown in terms of Accuracy on benign data (ACC), Attack Success Rate (ASR), and Defense Effectiveness Rating (DER).  Higher ACC and lower ASR values indicate better defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_20_2.jpg)
> This table presents the results of the proposed method (PDB) using different configurations of defensive triggers (masks) and target assignment strategies. It compares the accuracy (ACC) and attack success rate (ASR) of the model under various settings against several backdoor attacks. The configurations include different trigger masks (a, b, c) combined with different target assignment strategies (h1, h2). The results demonstrate the effectiveness of PDB across various defensive backdoor designs.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_21_1.jpg)
> This table presents the results of experiments conducted to evaluate the robustness of the proposed method (PDB) when the attacker uses the same target label as the defender. The experiments were performed using the PreAct-ResNet18 model on the CIFAR-10 dataset with different attacks (BadNet, Blended, SIG, BPP) and poisoning ratios (5% and 10%). The results show that the proposed method still maintains high accuracy and effectively mitigates backdoor attacks even under this scenario, showcasing its resilience against such attacks.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_21_2.jpg)
> This table presents the results of the proposed Proactive Defensive Backdoor (PDB) method compared to other state-of-the-art (SOTA) defense methods on the CIFAR-10 dataset using the PreAct-ResNet18 model architecture. The comparison is performed across seven different backdoor attacks, each with a 5% poisoning rate.  The results are shown in terms of Accuracy (ACC) on benign data, Attack Success Rate (ASR), and Defense Effectiveness Rating (DER).  Higher ACC, lower ASR, and higher DER values indicate superior defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_22_1.jpg)
> This table presents the results of different backdoor defense methods on the CIFAR-10 dataset using the PreAct-ResNet18 model architecture, with a poisoning ratio of 5%. The table includes seven different attacks (BadNets, Blended, SIG, SSBA, WaNet, BPP, and Trojan) and six defense methods (AC, DBD, Spectral, ABL, NAB, and PDB). For each attack and defense method combination, the table shows the accuracy on benign data (ACC), the attack success rate (ASR), and the defense effectiveness rating (DER).  The ACC represents the model's performance on clean samples, the ASR measures the rate at which the attack is successful, and the DER balances ACC and ASR to evaluate the trade-off between them.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_22_2.jpg)
> This table presents the experimental results on CIFAR-10 dataset using PreAct-ResNet18 model with a poisoning ratio of 5%. It compares the performance of the proposed PDB method against several state-of-the-art backdoor defense methods (AC, DBD, Spectral, ABL, NAB). The evaluation metrics include Accuracy on benign data (ACC), Attack Success Rate (ASR), and Defense Effectiveness Rating (DER).  The table shows the ACC, ASR, and DER for each defense method against seven different backdoor attacks (BadNets, Blended, SIG, SSBA, WaNet, BPP, Trojan).  Higher ACC and lower ASR values indicate better defense performance.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_23_1.jpg)
> This table presents the results of the proposed Proactive Defensive Backdoor (PDB) method and five other state-of-the-art (SOTA) in-training defense methods against seven SOTA data-poisoning backdoor attack methods on the CIFAR-10 dataset using the PreAct-ResNet18 model. The results are shown in terms of accuracy on benign data (ACC), attack success rate (ASR), and defense effectiveness rating (DER).  Each attack is performed with a 5% poisoning rate, targeting the 0th label if not specified.  The table shows the performance of different defense methods in terms of ACC, ASR, and DER for each attack.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_23_2.jpg)
> This table presents the results of the proposed PDB method (Proactive Defensive Backdoor) on the PreAct-ResNet18 model with a 5% poisoning ratio, comparing three different positions for the defensive trigger: Corner, Random, and Center. The table shows the Accuracy (ACC) and Attack Success Rate (ASR) for various backdoor attacks (BadNets, Blended, Sig, SSBA, WaNet).  It demonstrates the performance of PDB across different trigger placements.  The 'Center' placement shows relatively poor performance, indicating that the location of the defensive trigger is important for effectiveness.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_23_3.jpg)
> This table presents the results of experiments conducted using the PreAct-ResNet18 model with a 5% poisoning ratio, varying the size of the defensive trigger.  The goal was to evaluate how the size of the defensive trigger affects the performance of the PDB method against various backdoor attacks.  The table shows the Accuracy (ACC) and Attack Success Rate (ASR) for each attack under different defensive trigger sizes, demonstrating the impact of trigger size on the effectiveness of the defensive mechanism.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_24_1.jpg)
> This table presents the results of experiments conducted on the CIFAR-10 dataset using the PreAct-ResNet18 model architecture to evaluate the impact of different sampling frequencies on the performance of the proposed method, PDB.  The experiments were conducted with a 5% poisoning ratio, and the table shows the accuracy (ACC) and attack success rate (ASR) for various backdoor attacks (BadNets, Blended, SIG, SSBA, WaNet) under different sampling frequencies (1, 3, 5, 7, and 9).  The results demonstrate the effect of increasing the sampling frequency of defensive poisoned samples on the performance of the defensive backdoor, showing a trend of improved accuracy and reduced attack success rate with higher sampling frequencies.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_24_2.jpg)
> This table presents the results of the proposed PDB method on different model architectures (ResNet-18, ResNet-34, and ResNet-50).  It compares the accuracy (ACC) and attack success rate (ASR) of the models with and without the PDB defense mechanism.  The results show that PDB consistently achieves very low ASR across all models, demonstrating the effectiveness of the proposed defense.

![](https://ai-paper-reviewer.com/cbkJBYIkID/tables_25_1.jpg)
> This table compares the performance of PDB and FT-SAM on PreAct-ResNet18 against various backdoor attacks with a poisoning ratio of 5%.  It shows the accuracy (ACC) and attack success rate (ASR) for each method and attack type.  The comparison highlights the relative effectiveness of PDB in mitigating backdoor attacks, particularly demonstrating its superior performance against Blended attacks with a high blending ratio (0.2).

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cbkJBYIkID/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}