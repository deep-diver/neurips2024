---
title: "Samba: Severity-aware Recurrent Modeling for Cross-domain Medical Image Grading"
summary: "Samba: a novel severity-aware recurrent model, tackles cross-domain medical image grading by sequentially encoding image patches and recalibrating states using EM, significantly improving accuracy."
categories: []
tags: ["Computer Vision", "Image Classification", "🏢 Westlake University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} aIeXn5103e {{< /keyword >}}
{{< keyword icon="writer" >}} Qi Bi et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=aIeXn5103e" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94567" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=aIeXn5103e&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/aIeXn5103e/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Medical image disease grading is challenging due to the continuous progression of diseases and variations between datasets.  Existing models often struggle to generalize well to unseen target domains, leading to decreased performance.  This is particularly problematic in real-world clinical scenarios where models need to handle data from various sources and imaging techniques.



The proposed Samba method uses a **severity-aware recurrent modeling** approach to address these challenges.  It encodes image patches sequentially, using a recurrent neural network to capture the severity information of lesions.  A key innovation is the use of an **EM-based state recalibration mechanism** to map the patch embeddings to a more compact space, reducing the impact of cross-domain variations. Experimental results show that Samba outperforms existing baselines on several cross-domain medical image grading tasks, demonstrating its effectiveness in improving generalization performance.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Samba model improves cross-domain medical image grading accuracy. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Recurrent encoding of image patches effectively captures crucial lesion information. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} EM-based state recalibration mitigates cross-domain variations. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **cross-domain medical image analysis** and **disease grading**. It addresses the critical challenge of building robust models that generalize well to unseen data, a common issue in medical applications. The proposed Samba model offers a novel approach to handling this problem, **combining recurrent modeling with an EM-based state recalibration**. This work also opens **new avenues for research** in recurrent modeling and **domain generalization** techniques, particularly in medical imaging.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/aIeXn5103e/figures_1_1.jpg)

> This figure illustrates the challenges in disease grading. Subfigure (a) shows that disease progression is continuous, with variations within and between severity levels. Subfigure (b) highlights the ambiguity in image appearance, showing similar patterns across different levels and variability within the same level.  Subfigure (c) demonstrates the cross-domain clustering problem, where images from the same unseen domain tend to cluster together in the feature space, rather than by severity level, due to domain-specific variations.





![](https://ai-paper-reviewer.com/aIeXn5103e/tables_7_1.jpg)

> This table presents the performance comparison of different recurrent neural network models for the task of fatigue fracture grading. The proposed method, Samba, is compared with existing methods such as LSTM, UR-LSTM, UR-GRU, and ViT on the Fatigue Fracture Grading Benchmark. The results are presented in terms of accuracy (ACC), area under the curve (AUC), and F1-score (F1). The table highlights the significant improvement achieved by Samba over the baseline methods.





### In-depth insights


#### Severity-Aware RNN
A Severity-Aware Recurrent Neural Network (RNN) architecture for medical image grading would leverage the inherent sequential nature of RNNs to process image patches, **capturing spatial context and lesion progression**.  Unlike standard RNNs, a severity-aware model would explicitly incorporate lesion severity information into its hidden state, potentially using attention mechanisms to focus on the most severe regions. This could involve integrating lesion segmentation or classification outputs as input, or explicitly modeling severity levels within the RNN's hidden state dynamics. **The model could learn to represent disease severity in a lower dimensional space**, improving robustness and generalizability across various domains and image acquisition techniques. The training process would need careful consideration of class imbalances and the definition of severity, possibly using techniques like weighted loss functions or data augmentation to address these challenges. A well-designed severity-aware RNN promises to improve the accuracy and reliability of automated medical image grading by better handling both within- and cross-level variations in lesion appearance.

#### Cross-Domain Generalization
Cross-domain generalization, a critical aspect of machine learning, focuses on training models that generalize well to unseen domains.  **The core challenge lies in the discrepancy between the source domain (training data) and the target domain (unseen data)**.  This discrepancy can manifest in various forms, including differences in data distribution, image styles, or even annotation protocols.  Effective cross-domain generalization techniques aim to learn domain-invariant features, mitigating the negative impact of domain shifts on model performance.  **Robust methods often involve techniques such as domain adaptation, transfer learning, or data augmentation**, striving to bridge the gap between source and target domains.  **Addressing this challenge is crucial for deploying machine learning models in real-world applications**, particularly in areas with limited access to data from every possible domain of interest.  **Success in cross-domain generalization enables more reliable and flexible AI systems**, improving their applicability across diverse and potentially unpredictable scenarios.

#### EM-based Calibration
EM-based calibration, in the context of a medical image analysis model, likely refers to a technique that refines model outputs using the Expectation-Maximization (EM) algorithm.  The EM algorithm is well-suited for scenarios with incomplete or hidden data, a common situation in medical image analysis where crucial lesion information might be obscured or subtle.  **The core idea is to leverage the EM algorithm's iterative process to recalibrate the model's internal representations, improving its accuracy and robustness.**  This recalibration might involve adjusting parameters within the model or mapping the feature space to a more compact, informative one, where subtle differences are better highlighted.  The success of this method hinges on the proper modeling of the data distribution, often using a statistical model like a Gaussian Mixture Model (GMM) to capture the variability in the image data. **The EM process iteratively estimates the underlying lesion distribution based on observed data, enhancing the model's generalization ability across domains with varying imaging styles and noise levels.**  It's a valuable approach because it tackles the problem of data incompleteness, particularly important for precise medical grading where subtle details matter.  **The learnable parameters in this calibration process act as a bridge between raw model outputs and a more refined interpretation, ultimately leading to more accurate and reliable disease severity assessment.**

#### Ablation Study
An ablation study systematically removes components of a model to assess their individual contribution.  In a medical image grading context, this might involve removing parts of a recurrent neural network (RNN), such as the bidirectional layers or the EM-based recalibration module.  **By observing the impact on metrics like accuracy, AUC, and F1-score after each removal, researchers gain insight into which components are most critical for achieving high performance.**  A well-designed ablation study helps discern whether improvements stem from a specific component or from interactions between multiple parts. For instance, **it might reveal that while bidirectional processing enhances accuracy, EM-based recalibration is crucial for handling cross-domain variability and improving robustness.** The results guide future development by indicating whether to improve existing components or explore new ones. Ultimately, **a thorough ablation study strengthens the paper's findings and increases confidence in the model's design.** It demonstrates a careful investigation into the model's architecture and its contribution to the overall performance gains.

#### Future Directions
Future research directions for severity-aware recurrent modeling in cross-domain medical image grading could explore several promising avenues.  **Improving the robustness of the EM-based state recalibration** is crucial, potentially through incorporating more sophisticated techniques for handling cross-domain variations.  Another area is **exploring alternative recurrent architectures** beyond the bidirectional Mamba, such as transformers or more advanced state space models, to potentially capture long-range dependencies and complex lesion interactions more effectively.  **Investigating different loss functions** tailored to the continuous nature of disease severity could further improve grading accuracy.  Finally, a significant challenge is generalizability across diverse disease types.  **Developing a unified framework** that can adapt to various disease grading tasks with minimal retraining would be a considerable advancement in the field. This could involve leveraging transfer learning or meta-learning techniques to extract generalizable features.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/aIeXn5103e/figures_3_1.jpg)

> This figure illustrates the architecture of the proposed Samba model.  It shows the flow of image patches through four encoding stages, each containing multiple severity-aware recurrent layers.  Each Samba block uses bidirectional Mamba layers for encoding and an EM-based state recalibration module to refine feature representations using Gaussian Mixture Models and learnable severity bases, improving robustness to cross-domain variations.


![](https://ai-paper-reviewer.com/aIeXn5103e/figures_6_1.jpg)

> This figure illustrates the architecture of the Samba model. It shows how image patches are processed through multiple stages, each consisting of bidirectional Mamba layers for sequential encoding and an EM-based state recalibration module for cross-domain robustness.  The model learns to identify and represent the most severe lesions, which may be small and localized.  The final output provides the disease grade.


![](https://ai-paper-reviewer.com/aIeXn5103e/figures_14_1.jpg)

> This figure shows the impact of two hyperparameters, the number of EM algorithm iterations (T) and the method for updating severity bases, on the performance of the Samba model for fatigue fracture grading.  The top row illustrates how different values of T affect the AUC, ACC, and F1 scores when using Domain 1 as the source domain and Domain 2 as the target domain. The bottom row compares three different severity base update strategies: no update, only back propagation, and moving average, showing their effect on model performance metrics.  The results demonstrate the optimal values for T and highlight the effectiveness of moving average for updating severity bases, improving the model's generalization to unseen data.


![](https://ai-paper-reviewer.com/aIeXn5103e/figures_15_1.jpg)

> This figure visualizes the correlation matrices of patch embeddings before and after they are processed by the recurrent patch modeling module of the Samba model.  The matrices show the correlation between different image patches. The 'Before' matrices represent the correlations before processing, and the 'After' matrices show the correlations after passing through the recurrent module.  A higher correlation is represented by a redder color. The figure aims to demonstrate how the recurrent module enhances the relationships between patches that contain lesion information, facilitating the model's ability to capture relevant features for more effective disease grading.


![](https://ai-paper-reviewer.com/aIeXn5103e/figures_16_1.jpg)

> This figure visualizes the activation maps generated by the Samba model for different severity levels of diabetic retinopathy.  Each row represents a different image, and each column represents a different severity level (1-5).  The blue boxes highlight the image patches where the model activates most strongly for that particular severity level.  The figure demonstrates the model's ability to locate the relevant image regions for accurate severity classification and shows the results on the FGADR dataset (an unseen target domain).


![](https://ai-paper-reviewer.com/aIeXn5103e/figures_17_1.jpg)

> This figure illustrates the architecture of the Samba model, which consists of four encoding stages each with multiple recurrent layers.  Each stage processes image patches, using bidirectional Mamba layers to identify and track important lesion information.  An EM-based recalibration step refines the feature representations, using a Gaussian Mixture Model to model lesion distributions and learn severity-specific bases. The process aims to capture crucial lesion details and account for cross-domain variations.


![](https://ai-paper-reviewer.com/aIeXn5103e/figures_17_2.jpg)

> This figure illustrates the architecture of the Samba model, which consists of four encoding stages. Each stage processes image patches using bidirectional Mamba layers to capture decisive lesions.  An EM-based state recalibration module refines feature distribution using a Gaussian Mixture Model with learnable severity bases. The model aims to learn severity-aware representations of image patches for accurate disease grading, especially across different domains.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/aIeXn5103e/tables_7_2.jpg)
> This table presents a comparison of the performance and computational cost of VMamba-ERM and Samba on the breast cancer grading benchmark.  It shows a category-wise breakdown of accuracy (ACC), Area Under the Curve (AUC), and F1-score for each model, using Domain 1 as the source and Domain 2 as the target domain. The computational costs (GFLOPS and parameters) are also compared.

![](https://ai-paper-reviewer.com/aIeXn5103e/tables_7_3.jpg)
> This table presents the ablation study results on the proposed Samba model's components. It shows the performance impact of each component (Bi-directional State Space Modeling and EM-based State Recalibration) individually and in combination. The experiment is performed on the Fatigue Fracture Grading benchmark, using Domain 1 as the source and Domain 2 as the target domain. Evaluation metrics are Accuracy (ACC), Area Under the Curve (AUC), and F1-score, all expressed as percentages.

![](https://ai-paper-reviewer.com/aIeXn5103e/tables_7_4.jpg)
> This table presents an ablation study on the number of Gaussian Mixture Model (GMM) components (K) used in the EM-based state recalibration module of the Samba model.  The experiment is performed on the cross-domain breast cancer grading benchmark, using Domain-1 (20x magnification) as the source domain and Domain-2 (40x magnification) as the target domain. The table shows the impact of varying K on the accuracy (ACC), area under the curve (AUC), and F1-score, demonstrating how the choice of K affects the model's performance on the unseen target domain.

![](https://ai-paper-reviewer.com/aIeXn5103e/tables_9_1.jpg)
> This table compares the performance of the proposed Samba method with other state-of-the-art domain generalized diabetic retinopathy (DR) grading methods.  The comparison is performed using a single-domain generalization protocol, where one dataset serves as the source domain, and the others are treated as unseen target domains.  The evaluation metrics used are Accuracy (ACC) and F1-score, which are particularly relevant for evaluating the performance of imbalanced DR datasets. The top three results in each column are highlighted.

![](https://ai-paper-reviewer.com/aIeXn5103e/tables_15_1.jpg)
> This table shows the results of an ablation study on the number of Gaussian Mixture Model (GMM) components (K) used in the EM-based state recalibration module of the Samba model. The experiment was performed on the CAMELYON17 dataset for tumor classification. Domain-1 served as the source domain, while the remaining four domains were used as unseen target domains.  The table reports the accuracy (ACC), area under the curve (AUC), and F1-score (F1) for each domain and different values of K.

![](https://ai-paper-reviewer.com/aIeXn5103e/tables_16_1.jpg)
> This table presents a comparison of the classification performance between the baseline model (VMamba-ERM) and the proposed Samba method. The experiment was conducted on the CAMELYON17 dataset using a cross-domain setting. Domain-1 acted as the source domain, while the remaining four domains were treated as unseen target domains.  The table shows the average accuracy across the five target domains for each model and backbone type (VMama-T, VMama-S, VMama-B). The results are expressed as percentages.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/aIeXn5103e/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aIeXn5103e/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}