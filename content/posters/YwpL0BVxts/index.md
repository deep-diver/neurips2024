---
title: "United We Stand, Divided We Fall: Fingerprinting Deep Neural Networks via Adversarial Trajectories"
summary: "ADV-TRA uses adversarial trajectories to robustly fingerprint deep neural networks, outperforming state-of-the-art methods against various removal attacks."
categories: []
tags: ["Machine Learning", "Deep Learning", "🏢 Huazhong University of Science and Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} YwpL0BVxts {{< /keyword >}}
{{< keyword icon="writer" >}} Tianlong Xu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=YwpL0BVxts" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94664" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=YwpL0BVxts&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/YwpL0BVxts/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Deep Neural Networks (DNNs) are widely used but their intellectual property (IP) is vulnerable to theft.  Existing fingerprinting methods are often ineffective, especially against techniques designed to remove these fingerprints. These single-point methods fail because they are highly susceptible to changes in a model's decision boundary.  Such changes, introduced by attackers, can render these existing fingerprints invalid and cause a large number of false positives, mistakenly identifying innocent models as stolen. 

ADV-TRA, a novel fingerprinting scheme, uses adversarial trajectories instead of single points to characterize the model. These trajectories, built from a series of progressively adversarial samples, are much more robust to boundary changes.  The method also incorporates dynamically adjusted step sizes and generates a ‘surface trajectory’ encompassing multiple classes, enhancing uniqueness and reducing querying costs. ADV-TRA significantly outperforms state-of-the-art methods in distinguishing between infringing and innocent models on multiple datasets and against various attack types, offering a robust and efficient IP protection solution.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} ADV-TRA leverages adversarial trajectories for more robust DNN fingerprinting. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Dynamically adjusted step sizes in trajectory generation improve efficiency and uniqueness. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Superior performance against removal attacks is demonstrated with lower false positives. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on model intellectual property protection and security.  It introduces a novel, robust fingerprinting method that addresses limitations of existing single-point techniques, offering a significant advancement in the field. The superior performance against removal attacks and lower false positive rates open new avenues for research in model security and IP protection, particularly relevant in the growing field of AI model theft.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_1_1.jpg)

> This figure compares single-point fingerprinting with the proposed trajectory fingerprinting method.  When the decision boundary of a model changes (e.g., due to a removal attack), single-point samples near the boundary become invalid and lose their effectiveness for model verification. However, the trajectory-based approach is more robust; even with boundary alterations, the majority of the trajectory samples remain effective because they represent a range of adversarial examples across the boundary rather than single points.





![](https://ai-paper-reviewer.com/YwpL0BVxts/tables_6_1.jpg)

> This table presents the main experimental results on the CIFAR-10 dataset.  It compares the performance of ADV-TRA against four existing model fingerprinting techniques (IPGuard, CAE, UAP, Ours) under various removal attacks (model pruning with different pruning rates, adversarial training with different budgets, and fine-tuning with different layers and training methods). The table shows the accuracy, mutation rate (rmut), and fingerprint detection rate for both positive (infringing) and negative (innocent) models, highlighting the superior performance of the proposed ADV-TRA in accurately distinguishing between them.





### In-depth insights


#### Adv-TRA: Robustness
The robustness of Adv-TRA, a novel fingerprinting scheme for deep neural networks, is a critical aspect of its design.  **Its strength lies in leveraging adversarial trajectories**, which unlike single-point methods, offer resilience against decision boundary alterations caused by removal attacks (fine-tuning, pruning, adversarial training, and model extraction).  These trajectories, formed from a series of progressively adversarial samples, capture richer boundary information, enabling more accurate and robust model identification.  **The incorporation of dynamically adjusted step sizes** further enhances robustness by focusing on subtle details near the boundary while efficiently navigating toward it. This adaptive approach effectively reduces the querying cost and allows ADV-TRA to better tolerate greater degrees of alteration in the decision boundary. **The use of a surface trajectory**, comprised of multiple fixed-length trajectories across multiple classes, significantly mitigates the risk of false positives by generating a more holistic representation of the model's decision surface, enhancing its uniqueness and overall robustness. Through this multifaceted design, Adv-TRA demonstrably outperforms state-of-the-art methods in distinguishing between infringing and innocent models under various removal attacks.

#### Trajectory Design
Effective trajectory design is crucial for robust model fingerprinting.  The core idea revolves around generating adversarial trajectories, **not isolated single-point samples**, to characterize model decision boundaries. This approach enhances robustness against removal attacks that alter the decision boundary.  A key innovation is the use of **dynamically adjusted step sizes** during trajectory generation, ensuring that trajectories efficiently reach the decision boundary without excessive queries.  The design incorporates **fixed-length trajectories** for controllability and cost efficiency and aims to capture the global fingerprint of the model by using multiple trajectories that traverse across multiple classes, creating a **surface trajectory** for comprehensive model representation, thus greatly reducing false positives.

#### Removal Attacks
Removal attacks, aimed at circumventing model fingerprinting, pose a significant challenge to intellectual property protection of deep neural networks.  These attacks strategically modify the model's parameters or structure to invalidate previously generated fingerprints, thus rendering verification methods ineffective. **Common removal attack strategies** include fine-tuning, pruning, adversarial training, and model extraction.  Fine-tuning and retraining adjust model parameters, while pruning reduces model size, and adversarial training enhances model robustness against adversarial examples that are often used to create fingerprints. Model extraction, on the other hand, involves creating a new model that mimics the original's functionality without directly copying its parameters.  The effectiveness of each attack varies, depending on the model's architecture, training data, and the specific fingerprinting technique employed.  **Robust fingerprinting methods** must therefore incorporate strategies to mitigate the impact of these attacks, such as creating fingerprints that are less sensitive to model modifications or using multiple diverse fingerprints to increase resilience against individual attack strategies.  **Developing effective defenses** against removal attacks remains a critical area of research for securing deep learning models.

#### Global Fingerprint
The concept of a "Global Fingerprint" in the context of deep neural network (DNN) security is intriguing.  It suggests moving beyond the limitations of localized fingerprinting techniques, which are highly susceptible to adversarial attacks and model alterations. A global fingerprint would ideally capture a more holistic representation of the DNN's architecture and learned features, enabling more robust model verification. This could involve analyzing higher-level representations within the network, or employing techniques that are less sensitive to specific decision boundaries. The challenge, however, lies in designing a global fingerprint that is both unique to the model and computationally efficient. **Successfully creating a global fingerprint would represent a significant advance in DNN IP protection, offering improved resilience against sophisticated attacks and unauthorized model replication.**  However, **such a system needs careful consideration of computational costs** and the potential for false positives.  A balance must be struck between the robustness of the fingerprint and the feasibility of its implementation in real-world scenarios.  Further research into feature extraction methods, dimensionality reduction, and robust comparison techniques is crucial to realizing this vision.

#### Future Work
Future research directions stemming from this work on deep neural network fingerprinting could explore several avenues.  **Improving the efficiency of trajectory generation** is crucial, potentially through more sophisticated optimization algorithms or exploration of alternative sampling strategies to reduce the number of queries needed. **Investigating the robustness of ADV-TRA against more sophisticated removal attacks** is paramount.  The current study used four types of attacks; expanding this to include more advanced combinations and techniques, such as those leveraging generative models, would strengthen the analysis.  **Generalizing ADV-TRA to diverse DNN architectures and model types** beyond the image classification models investigated is another priority.  This would involve exploring how the trajectory approach adapts to different model structures and tasks, such as those commonly found in natural language processing or time series analysis.  Finally, **developing a more formal theoretical framework** to provide guarantees for the uniqueness and robustness of the generated fingerprints is needed, complementing the experimental evaluations with solid theoretical grounding.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_2_1.jpg)

> This figure illustrates the process of model stealing and verification. The model owner trains a source model and deploys it as a cloud service or client-sided software. The attacker attempts to steal the source model in either black-box or white-box ways. Then, the attacker can leverage removal attacks to modify the model to evade the IP infringement detection. The model owner verifies the ownership of a suspect model by querying it with a set of fingerprinting samples. Based on the output results, the owner selects a testing metric and computes it to make the final judgment.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_4_1.jpg)

> This figure illustrates the workflow of the proposed ADV-TRA model for fingerprinting DNNs.  It shows the process of generating adversarial trajectories, which starts with a base sample from one class and progressively moves toward another class, probing the decision boundary and creating fixed-length trajectories with dynamically adjusted step sizes. This results in a surface trajectory, a series of fixed-length trajectories spanning across multiple classes to capture global features. In the verification phase, this surface trajectory is used to determine if a suspect model is stolen from the source model by calculating the mutation rate.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_7_1.jpg)

> This figure shows the performance comparison of different fingerprinting methods on the CIFAR-10 dataset. The left plot (a) is a ROC curve showing the true positive rate (TPR) against the false positive rate (FPR) for each method.  The right plot (b) is a box plot that shows the distribution of fingerprint detection rates for positive and negative models for each method. The results demonstrate that ADV-TRA achieves a perfect AUC (Area Under the Curve) of 1.0, significantly outperforming other methods in distinguishing between positive (infringing) and negative (innocent) models.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_7_2.jpg)

> This figure shows the relationship between fingerprint detection rate and the number of fine-tuning epochs for both positive and negative models.  The detection rate for positive models (those subjected to fine-tuning attacks) is shown for four different fine-tuning strategies (FTLL, FTAL, RTLL, RTAL). The gray area represents the range of detection rates observed for negative models. This visualization helps in assessing the robustness of the proposed ADV-TRA fingerprinting method against fine-tuning attacks, demonstrating its effectiveness in distinguishing between genuine and fraudulent models.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_8_1.jpg)

> This figure visualizes the decision boundaries of a model under various removal attacks such as fine-tuning, pruning, and adversarial training. Each color represents a class, and changes in the decision boundary due to different attacks are clearly illustrated. The figure shows how the decision surface is affected by each attack, providing visual evidence for the effectiveness of different removal attacks in invalidating the original fingerprints.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_16_1.jpg)

> This figure visualizes the decision boundaries of a model under various removal attacks. Each color represents a different class.  The figure shows how different attacks (adversarial training with varying perturbation budgets, fine-tuning, retraining, and pruning) affect the decision boundary. The goal is to illustrate how the fingerprinting method proposed in the paper, ADV-TRA, is robust against these attacks, maintaining a consistent and distinguishable decision surface. The 'source' image shows the original decision surface, while the others show how the decision boundaries change after each attack.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_17_1.jpg)

> This figure shows the impact of varying the number of classes spanned by the trajectories on the performance metrics T1%F and T10%F.  The results are shown for CIFAR-10, CIFAR-100, and ImageNet datasets.  A trajectory spanning only two classes is equivalent to a bilateral trajectory.  Increasing the number of classes generally improves performance, though the effect is less pronounced on ImageNet, likely due to the greater complexity of the decision boundary in datasets with more classes.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_18_1.jpg)

> This figure shows the performance of ADV-TRA compared to a state-of-the-art baseline (UAP) when varying the length of the adversarial trajectories used for fingerprinting.  The box plots illustrate the distribution of detection rates for positive (legitimate) and negative (illegitimate) models.  The red line shows the AUC (Area Under the Curve) for ADV-TRA, highlighting its superior performance, particularly for longer trajectories (40-160). Shorter trajectories perform similarly to or worse than UAP.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_18_2.jpg)

> This figure shows the performance of ADV-TRA model with different brake factors and trajectory lengths. The brake factor controls the proportional relationship between the step sizes of two adjacent steps in the trajectory, while the trajectory length determines the number of samples in the trajectory. The results are shown in terms of AUC (Area Under the Curve) for CIFAR-100 and ImageNet datasets, illustrating how the combination of these two hyperparameters affects the model's ability to distinguish between positive and negative models.


![](https://ai-paper-reviewer.com/YwpL0BVxts/figures_19_1.jpg)

> This figure shows a t-SNE visualization of an adversarial trajectory generated by ADV-TRA.  The trajectory starts at a base sample (red) and progressively moves through four different classes (blue, green, orange). The size of the points indicates the proximity to the decision boundary of each class, with larger points marking the transition to a new class. The trajectory demonstrates ADV-TRA's ability to generate a smooth, continuous path through the decision boundaries of multiple classes, which is more robust to changes in those boundaries compared to single-point fingerprinting methods.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/YwpL0BVxts/tables_7_1.jpg)
> This table presents the main experimental results on the CIFAR-10 dataset, comparing the performance of ADV-TRA against four existing fingerprinting methods under various removal attacks (model pruning and adversarial training).  It shows fingerprint detection rates for both positive (infringing) and negative (innocent) models.  Higher detection rates for positive models and lower rates for negative models indicate better performance in accurately identifying IP infringement.

![](https://ai-paper-reviewer.com/YwpL0BVxts/tables_8_1.jpg)
> This table presents the main results of the CIFAR-10 experiments, comparing ADV-TRA's performance against other methods under various removal attacks (pruning and adversarial training).  It shows the fingerprint detection rates for both positive (infringing) and negative (innocent) models, highlighting ADV-TRA's superior ability to distinguish between them with high accuracy and low false positives.

![](https://ai-paper-reviewer.com/YwpL0BVxts/tables_20_1.jpg)
> This table presents the main experimental results on the CIFAR-10 dataset, comparing the performance of ADV-TRA against other state-of-the-art fingerprinting methods under various removal attacks.  It shows fingerprint detection rates for both positive (infringing) and negative (innocent) models.  Higher detection rates for positive models and lower rates for negative models indicate better performance.

![](https://ai-paper-reviewer.com/YwpL0BVxts/tables_21_1.jpg)
> This table presents the main experimental results on the CIFAR-10 dataset, comparing the performance of ADV-TRA against other state-of-the-art model fingerprinting methods under various removal attacks.  The results are shown for both positive (infringing) models and negative (innocent) models, highlighting the fingerprint detection rates, accuracy, and mutation rate (Tmut). The table demonstrates ADV-TRA's superior ability to accurately distinguish between infringing and innocent models, minimizing both false positives and false negatives.

![](https://ai-paper-reviewer.com/YwpL0BVxts/tables_21_2.jpg)
> This table presents the main experimental results on the CIFAR-10 dataset, comparing the performance of ADV-TRA with other state-of-the-art model fingerprinting methods under various removal attacks (model pruning and adversarial training).  The table shows the fingerprint detection rate for both positive (infringing) and negative (innocent) models.  A higher detection rate is desired for positive models, and a lower detection rate is desired for negative models to avoid false positives.  The results demonstrate ADV-TRA’s superior performance in distinguishing between infringing and innocent models.

![](https://ai-paper-reviewer.com/YwpL0BVxts/tables_22_1.jpg)
> This table presents the main results of the experiments conducted on the CIFAR-10 dataset.  It compares the performance of ADV-TRA against other state-of-the-art model fingerprinting methods across various removal attacks (pruning, adversarial training, and fine-tuning). The table shows fingerprint detection rates for both positive (models derived from the source model) and negative (unrelated models) models under different attack scenarios, demonstrating the effectiveness of ADV-TRA in correctly identifying infringing vs. innocent models. 

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YwpL0BVxts/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}