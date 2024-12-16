---
title: "WeiPer: OOD Detection using Weight Perturbations of Class Projections"
summary: "WeiPer enhances OOD detection by cleverly perturbing class projections, creating a richer representation that improves various existing methods and achieves state-of-the-art results."
categories: ["AI Generated", ]
tags: ["Machine Learning", "Deep Learning", "🏢 Free University of Berlin",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 8HeUvbImKT {{< /keyword >}}
{{< keyword icon="writer" >}} Maximilian Granz et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=8HeUvbImKT" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/8HeUvbImKT" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/8HeUvbImKT/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Machine learning models often struggle with out-of-distribution (OOD) data – data that differs significantly from the training data.  This can lead to incorrect predictions and unreliable results, particularly in safety-critical applications. Existing methods address this challenge by analyzing either the model's logits (outputs) or the activations of its penultimate layer. However, these methods sometimes fail to effectively distinguish OOD data, especially when it is similar to the training data. 

This paper introduces WeiPer, a novel approach to OOD detection that addresses these limitations. WeiPer works by slightly perturbing the weights of the final fully connected layer of the model, thereby creating a more informative representation of the input data.  The researchers demonstrate that WeiPer enhances existing OOD detection techniques and introduce a new method, called WeiPer+KLD, that achieves state-of-the-art results across multiple datasets. **The simplicity and general applicability of WeiPer make it a valuable tool for improving the robustness of machine learning models**.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} WeiPer improves OOD detection by perturbing class projections in the final layer, generating a richer input representation. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A novel distance-based method, WeiPer+KLD, leverages the properties of the augmented WeiPer space to improve OOD detection accuracy. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} WeiPer achieves state-of-the-art results across multiple benchmarks, particularly excelling in scenarios with OOD samples near the training distribution. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents **WeiPer**, a novel and effective method for out-of-distribution (OOD) detection that significantly improves the performance of existing methods, especially in challenging scenarios where OOD samples are close to the training distribution.  This is relevant to researchers working on improving the robustness and reliability of machine learning models, a critical concern in various applications. **The method's simplicity and versatility make it readily applicable to various models and datasets**, opening new avenues for OOD detection research.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/8HeUvbImKT/figures_1_1.jpg)

> 🔼 This figure demonstrates the effectiveness of WeiPer's weight perturbations in improving OOD detection. The left panel shows contour plots visualizing the densities of in-distribution (ID) and out-of-distribution (OOD) data projected onto two logit dimensions.  It highlights how OOD data extends into the positive cluster in a conical shape. The right panel shows the densities when projected onto the original weight vector and a perturbed one. This illustrates how a single perturbation reduces false positives, showcasing the advantage of WeiPer's approach.
> <details>
> <summary>read the caption</summary>
> Figure 1: Why random perturbations? Left: We visualize densities of CIFAR10 (ID, blue) and CIFAR100 (OOD, red) as contour plots along the two logit dimensions spanned by w₀ and w₁, zoomed in on the positive cluster of class zero. The blue axis denotes the vector associated with that class, and one of its perturbations is depicted by the turquoise line. Right: When projecting the data onto both vectors, we obtain the densities shown in the top and bottom panel, respectively. The vertical blue lines mark the 5-percentile (highest 5%) of the true ID data (CIFAR10, blue). At this decision boundary, the classifier would produce false positives in the marked dashed red tail area. A single perturbation of the class-associated vector yields already a reduction of the false positive rate (FPR) from 1.34% to 0.79%. Visually, we confirm that OOD data mostly resides close to 0, extending into the positive cluster in a particular conical shape, which is exploited by the cone of WeiPer vectors.
> </details>





![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_6_1.jpg)

> 🔼 This table presents the OOD detection results of the proposed WeiPer method and other state-of-the-art methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet-1k.  The results are reported in terms of AUROC (Area Under the Receiver Operating Characteristic Curve) and FPR95 (False Positive Rate at 95% True Positive Rate).  The table also includes ablation studies for the WeiPer method to analyze the impact of different components.  The performance of WeiPer+KLD is highlighted, showing its strengths on near-OOD datasets, especially when using ResNet backbones.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>





### In-depth insights


#### OOD Detection
Out-of-distribution (OOD) detection is a crucial aspect of robust machine learning, aiming to identify inputs that deviate significantly from the training data distribution.  **Existing methods often leverage either logit information or penultimate layer activations**, but these approaches can be limited.  This paper introduces WeiPer, a novel technique that uses **weight perturbations of class projections to create a richer representation of the input**, leading to improved OOD detection performance.  **WeiPer augments the feature space**, enabling better discrimination between in-distribution and OOD samples.  The method's effectiveness is demonstrated across multiple benchmarks, showcasing **state-of-the-art results, particularly in challenging scenarios with OOD samples positioned close to the training distribution**. WeiPer's flexibility allows for its integration with various OOD detection methods, enhancing their performance. However, the approach requires careful consideration of hyperparameters and computational cost.

#### WeiPer Method
The WeiPer method, introduced for out-of-distribution (OOD) detection, cleverly enhances existing methods by introducing perturbations to the weight vectors of a neural network's final fully connected layer.  **Instead of relying solely on the original class projections, WeiPer generates a cone of perturbed vectors around each original weight**, creating a richer representation of the input data. This simple yet effective trick significantly improves the ability to distinguish between in-distribution (ID) and OOD samples, especially in challenging scenarios where OOD data points are positioned close to the training set distribution.  **The method's strength lies in its ability to capture additional structural information from the penultimate layer**, leading to better separation of ID and OOD data.  Furthermore, WeiPer's flexibility allows its integration with various OOD detection scores, such as MSP and a novel KL-divergence based approach (WeiPer+KLD), which further leverages the augmented WeiPer space for enhanced accuracy.  **The cone of perturbed vectors is generated through carefully controlled weight perturbations**, ensuring that all perturbed vectors maintain a consistent angular deviation from the original weight vector.  This design enables the method to extract additional structural information while preserving the essential class information present in the original weights.

#### Empirical Results
The empirical results section of a research paper is crucial for validating the claims and hypotheses presented.  A strong empirical results section will thoroughly describe the experimental setup, including datasets, metrics, and baselines used. **It should clearly present the results of the experiments using visualizations like tables and graphs, emphasizing key findings that support the paper's contributions.** The discussion should interpret the results, addressing both expected and unexpected outcomes.  **A critical analysis of the results, including potential limitations and comparisons to prior work, is necessary for a robust evaluation.** It is important to show how the results support the paper's claims and also to explore any limitations or uncertainties in the data interpretation.  **Statistical significance of results should be clearly stated to ensure confidence in the findings.** Ultimately, a compelling empirical results section strengthens the overall persuasiveness and impact of the research by providing tangible evidence supporting the presented claims.

#### Limitations
The WeiPer method, while showing strong performance improvements, presents some limitations.  A significant drawback is the **increased number of hyperparameters** compared to existing methods, potentially increasing the computational cost of tuning and potentially impacting generalization.  Additionally, the improved performance comes at the cost of **higher memory consumption**, particularly noticeable when using larger WeiPer spaces to capture more of the input distribution's structure. This might restrict the applicability of WeiPer to devices with limited memory resources.  Further investigation into the effect of dimensionality and memory usage is warranted. Finally, although WeiPer demonstrates consistent improvement across benchmarks and models, the extent of its effectiveness may vary depending on the specific dataset and architecture used, highlighting the **need for further testing and analysis** across a wider range of scenarios to fully understand its generalizability and robustness.

#### Future Work
Future research directions stemming from this work could explore **extending WeiPer to other modalities beyond images**, such as text or time-series data, to assess its generalizability.  Investigating the impact of different perturbation methods and distributions on the performance of WeiPer would refine its effectiveness.  A crucial area for improvement is addressing the **computational cost of the method**. Exploring techniques to reduce memory requirements, possibly through dimensionality reduction or approximation methods, would be particularly valuable.  Finally, further theoretical analysis is needed to **establish stronger theoretical foundations for WeiPer's effectiveness**, possibly connecting it to existing theoretical frameworks of OOD detection or statistical learning theory. This would enhance the understanding of its strengths and limitations, providing insights into its performance in diverse scenarios.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/8HeUvbImKT/figures_4_1.jpg)

> 🔼 This figure illustrates the WeiPer method for out-of-distribution detection.  The left panel shows how WeiPer perturbs the weight vectors (Wfc) of the final fully connected layer by an angle controlled by the hyperparameter δ, creating r weight matrices (W1,...,Wr).  The center panel depicts the Kullback-Leibler Divergence (KLD) calculation for WeiPer+KLD, comparing the densities of the original penultimate layer activations (pz) and the perturbed activations (Pŵz). The right panel shows the calculation of the MSP (Maximum Softmax Probability) score for the WeiPer+MSP method, averaging the MSP scores across all r perturbed logit spaces.
> <details>
> <summary>read the caption</summary>
> Figure 2: WeiPer perturbs the weight vectors of Wfc by an angle controlled by δ. For each weight, we construct r perturbations resulting in r weight matrices W1, ..., Wr. KLD: For WeiPer+KLD, we treat z1, ..., zk ~ pz and w1,1z, ..., wr,cz ~ PŴz as samples of the same distribution induced by z and Wz, respectively. We approximate the densities with histograms and smooth the result with uniform kernel Tk. Afterwards, we compare the densities Tk(qz) with the mean distribution over the training samples Ez∈Ztrain(qz) for qz = pz and qz = PŴz, respectively. MSP: For a score function S on the logit space RC, we define the perturbed score SweiPer as the mean over all the perturbed logit spaces Wz. We choose S = MSP and call the resulting detector MSPWeiPer.
> </details>



![](https://ai-paper-reviewer.com/8HeUvbImKT/figures_5_1.jpg)

> 🔼 This figure visualizes the activation distributions in both the penultimate layer and the augmented WeiPer space for a ResNet18 model trained on CIFAR10.  The left pair shows the mean distributions (CIFAR10 as ID and CIFAR100 as OOD), while the right pair shows the distributions for two individual samples.  It highlights the similarity in distribution between ID samples in both spaces, contrasting with the OOD sample distribution which differs more significantly. The visualization helps demonstrate WeiPer's ability to enhance the separation of in-distribution and out-of-distribution samples.
> <details>
> <summary>read the caption</summary>
> Figure 3: Histogram of all 512 activations in the penultimate layer (left pair) and the activations in WeiPer space (right pair) of a ResNet18 trained on CIFAR10. We perturb the weight matrix 100 times to produce a 100 \* 100 = 10000-dimensional perturbed logit space. For each pair, the left panel shows the mean distribution over all samples (ID = CIFAR10, OOD = CIFAR100). The right panels show the distribution pz and pwz, respectively, for two randomly chosen samples with smoothing applied ($1 = 82 = 2).
> </details>



![](https://ai-paper-reviewer.com/8HeUvbImKT/figures_8_1.jpg)

> 🔼 This figure analyzes the impact of hyperparameters r (number of weight perturbations) and δ (angle of perturbation) on the performance of three OOD detection postprocessors (KLD, MSP, and ReAct) using CIFAR10 and ImageNet datasets.  It shows that increasing r generally improves performance while the optimal δ value varies depending on the method and dataset. The shaded areas represent the range of AUROC scores across multiple runs.
> <details>
> <summary>read the caption</summary>
> Figure 4: We investigate the effect of WeiPer hyperparameters r and δ on the performance of the three postprocessors. The left pair shows results on CIFAR10, the right pair corresponds to ImageNet (using ResNet18 for both). Models were tested using their respective near OOD datasets. The panels corresponding to δ depict AUROC performance minus the initial AUROC performance at δ = 0. The graphs show the mean over 25 runs and the shaded area around them represents the value range (min to max) over those runs. All other parameters of the methods were fixed to the optimal setting.
> </details>



![](https://ai-paper-reviewer.com/8HeUvbImKT/figures_14_1.jpg)

> 🔼 This figure demonstrates the effectiveness of random weight perturbations in improving OOD detection.  The left panel shows contour plots visualizing the density distributions of in-distribution (ID) and out-of-distribution (OOD) data projected onto two logit dimensions.  It highlights how OOD data extends into the positive cluster of ID data in a conical shape. The right panel shows the density distributions when projected onto the original and a perturbed class vector, revealing that a single perturbation significantly reduces false positives. This illustrates how WeiPer leverages this conical shape of OOD data within the ID data's distribution for improved OOD detection.
> <details>
> <summary>read the caption</summary>
> Figure 1: Why random perturbations? Left: We visualize densities of CIFAR10 (ID, blue) and CIFAR100 (OOD, red) as contour plots along the two logit dimensions spanned by w₀ and w₁, zoomed in on the positive cluster of class zero. The blue axis denotes the vector associated with that class, and one of its perturbations is depicted by the turquoise line. Right: When projecting the data onto both vectors, we obtain the densities shown in the top and bottom panel, respectively. The vertical blue lines mark the 5-percentile (highest 5%) of the true ID data (CIFAR10, blue). At this decision boundary, the classifier would produce false positives in the marked dashed red tail area. A single perturbation of the class-associated vector yields already a reduction of the false positive rate (FPR) from 1.34% to 0.79%. Visually, we confirm that OOD data mostly resides close to 0, extending into the positive cluster in a particular conical shape, which is exploited by the cone of WeiPer vectors.
> </details>



![](https://ai-paper-reviewer.com/8HeUvbImKT/figures_16_1.jpg)

> 🔼 This figure displays ablation studies on the hyperparameters of the KL divergence score function used in the WeiPer+KLD method. It shows how AUROC changes as each hyperparameter is varied individually while holding the others constant.  The plots reveal the optimal ranges for nbins, λ1, λ2, s1, and s2, highlighting the effect of each parameter on the model's performance for both CIFAR10 and ImageNet datasets.
> <details>
> <summary>read the caption</summary>
> Figure 6: KLD specific hyperparamters: We fixed the optimal hyperparameters and varied the one parameter in question by conducting 10 runs over the same fixed parameter setting on CIFAR10 and ImageNet as ID against their near OOD datasets. We report the mean and the minimum to maximum range (transparent). We set r = 5 instead of r = 100 for ImageNet to save resources. Thus the noise on the results is stronger for the ImageNet ablations. All of the parameters except the kernel sizes only have a single local maximum which indicates that they should be easy to optimize. The most important parameters seem to be the kernel sizes s1 and s2 that we use for smoothing followed by nbins. Note that s1 and s2 have a different optimum, which means it is not possible to simply choose s1 = s2 and reduce the count of hyperparameters. λ1 = 0 is the score function without the KLD specific WeiPer application. λ2 is the application of MSPWeiPer which is not beneficial for CIFAR10, but shows to be effective on ImageNet.
> </details>



![](https://ai-paper-reviewer.com/8HeUvbImKT/figures_17_1.jpg)

> 🔼 This figure visualizes the distribution of activations in both the penultimate layer and the WeiPer space of a ResNet18 model trained on CIFAR10.  The left pair shows the average distribution of activations, while the right pair shows the distribution for two individual samples. The WeiPer space, created by perturbing the weight matrix 100 times, has 1000 dimensions. The distributions are compared for both in-distribution (CIFAR10, blue) and out-of-distribution (CIFAR100, red) samples, highlighting the differences in activation patterns between the two.
> <details>
> <summary>read the caption</summary>
> Figure 3: Histogram of all 512 activations in the penultimate layer (left pair) and the activations in WeiPer space (right pair) of a ResNet18 trained on CIFAR10. We perturb the weight matrix 100 times to produce a 1000-dimensional perturbed logit space. For each pair, the left panel shows the mean distribution over all samples (ID = CIFAR10, OOD = CIFAR100). The right panels show the distribution pz and pwz, respectively, for two randomly chosen samples with smoothing applied ($1 = $2 = 2).
> </details>



![](https://ai-paper-reviewer.com/8HeUvbImKT/figures_18_1.jpg)

> 🔼 This figure visualizes the activation distributions in the penultimate layer and the WeiPer space for a ResNet18 model trained on CIFAR-10.  The left pair of histograms shows the mean distribution of activations across all samples for both in-distribution (CIFAR-10) and out-of-distribution (CIFAR-100) data in the penultimate layer. The right pair shows the distributions for two specific samples, highlighting the effect of the WeiPer transformation.  The 100 weight perturbations create a 10,000-dimensional space, and the histograms show how this transformation affects the data distributions, particularly the separation of in-distribution and out-of-distribution samples.
> <details>
> <summary>read the caption</summary>
> Figure 3: Histogram of all 512 activations in the penultimate layer (left pair) and the activations in WeiPer space (right pair) of a ResNet18 trained on CIFAR10. We perturb the weight matrix 100 times to produce a 100 \* 100 = 10000-dimensional perturbed logit space. For each pair, the left panel shows the mean distribution over all samples (ID = CIFAR10, OOD = CIFAR100). The right panels show the distribution pz and pŵz, respectively, for two randomly chosen samples with smoothing applied (s1 = s2 = 2).
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_7_1.jpg)
> 🔼 This table presents the out-of-distribution (OOD) detection results of the proposed WeiPer method and other state-of-the-art methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet.  It shows the Area Under the Receiver Operating Characteristic curve (AUROC) and the False Positive Rate at 95% True Positive Rate (FPR95) for both near and far OOD datasets.  The table highlights WeiPer's performance, especially on challenging near-OOD scenarios, and includes ablation studies to analyze the impact of different components of the method.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_8_1.jpg)
> 🔼 This table presents the OOD detection performance of WeiPer and other state-of-the-art methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet.  It shows the Area Under the Receiver Operating Characteristic curve (AUROC) and the False Positive Rate at 95% True Positive Rate (FPR95) for both 'near' and 'far' out-of-distribution (OOD) datasets.  The table highlights WeiPer's performance, particularly its strength on 'near' OOD datasets using ResNet backbones, and includes ablation studies showing results with and without certain components of the method.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_15_1.jpg)
> 🔼 This table presents the OOD detection results of the proposed WeiPer method and other state-of-the-art methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet.  The results are shown for both 'near' and 'far' out-of-distribution (OOD) datasets, indicating the proximity of the OOD data to the training distribution.  The table highlights the median AUROC (Area Under the Receiver Operating Characteristic curve) and FPR95 (False Positive Rate at 95% True Positive Rate) scores across multiple runs, demonstrating WeiPer's performance relative to existing methods. Ablation studies are also shown for CIFAR10 to analyze the effect of different components of the WeiPer method.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_15_2.jpg)
> 🔼 This table presents a comparison of the Area Under the Receiver Operating Characteristic Curve (AUROC) and False Positive Rate at 95% True Positive Rate (FPR95) for various OOD detection methods on three benchmark datasets (CIFAR10, CIFAR100, and ImageNet).  The results are broken down by whether the Out-of-Distribution (OOD) data is positioned near or far from the training set distribution.  The table highlights the performance of the proposed WeiPer method and its variations (WeiPer+MSP, WeiPer+KLD, WeiPer+ReAct)  in comparison to existing state-of-the-art techniques.  Ablation studies for the WeiPer+KLD method are shown for CIFAR10, demonstrating its effectiveness in specific scenarios.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_15_3.jpg)
> 🔼 This table presents the Area Under the Receiver Operating Characteristic Curve (AUROC) scores for the ImageNet dataset using the ResNet50 model.  The results are shown for both 'near' and 'far' out-of-distribution (OOD) datasets,  and they are broken down by different training set sizes (1k, 5k, 10k, 50k, 100k, 500k, and 1M samples).  The experiment used the optimal hyperparameters determined previously, but with a reduced number of weight perturbations (r=50) for computational reasons. This reduction in perturbations could explain any differences between these results and those reported earlier in the paper.
> <details>
> <summary>read the caption</summary>
> Table 6: AUROC results on ImageNet with ResNet50 on the near and far benchmark with different training set sizes. Each split is a random sample of the data set with each class appearing exactly as often as each other class. We chose the optimal set of hyperparameters on ImageNet, but reduced the number of repeats r to 50 instead of 100.
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_16_1.jpg)
> 🔼 This table presents the out-of-distribution (OOD) detection results of the proposed WeiPer method and several state-of-the-art methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet.  The table shows the AUROC and FPR95 scores for each method on near and far OOD datasets.  It highlights WeiPer+KLD's strong performance, particularly on near OOD data, when using ResNet backbones, while also noting its reduced performance on ViT backbones. Ablation studies on CIFAR10 using WeiPer+KLD are included for comparison.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_17_1.jpg)
> 🔼 This table presents the out-of-distribution (OOD) detection results of the proposed WeiPer method and several state-of-the-art methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet.  The table shows AUROC and FPR95 scores for both near and far OOD datasets.  It highlights WeiPer's competitive performance, especially on ResNet backbones, while noting the performance differences using ViT models.  Ablation studies comparing different WeiPer configurations are also included.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_18_1.jpg)
> 🔼 This table presents the hyperparameters used for creating the density plots shown in Figures 3, 7, and 8.  Specifically, it lists the number of bins used in the histograms (nbins), the kernel size (s) used for smoothing the density estimates, and the maximum value of the mean density of the penultimate layer activations (maxp).  These parameters are crucial for visualizing and analyzing the activation distributions in both the original and perturbed feature spaces, which are key components in understanding and evaluating the performance of the WeiPer method.
> <details>
> <summary>read the caption</summary>
> Table 9: Plotting parameters: s is the kernel size for the uniform kernel that was used for smoothing, and maxp = maxt Ez∈Ztrain [Pz](t) denotes the maximum of the mean density of the penultimate densities pt. The perturbed densities pŵz are scaled similarly.
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_19_1.jpg)
> 🔼 This table presents the out-of-distribution (OOD) detection performance of the proposed WeiPer method and several state-of-the-art methods across three benchmark datasets: CIFAR-10, CIFAR-100, and ImageNet.  The table shows AUROC and FPR95 scores for both near and far OOD datasets. It highlights WeiPer+KLD's superior performance, particularly in challenging near OOD scenarios with ResNet backbones, and also points out its relative weakness with ViT backbones.  Ablation studies comparing WeiPer+KLD with and without MSP, and with random projections are also presented for the CIFAR-10 dataset.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_19_2.jpg)
> 🔼 This table presents the out-of-distribution (OOD) detection results of WeiPer and other state-of-the-art methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet.  The table shows AUROC and FPR95 scores for each method on near and far OOD datasets. It highlights WeiPer's performance, particularly its strong results on near OOD datasets using ResNet backbones and its comparative performance on ViT backbones.  Ablation studies on CIFAR10, comparing WeiPer+KLD with and without MSP and with random projections, are also included.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_20_1.jpg)
> 🔼 This table presents the Area Under the Receiver Operating Characteristic curve (AUROC) and the False Positive Rate at 95% True Positive Rate (FPR95) for various OOD detection methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet.  The table compares WeiPer against other state-of-the-art methods, highlighting WeiPer's performance, particularly on datasets where OOD samples are similar to in-distribution samples.  Ablation studies are also included for CIFAR10, showing the performance of WeiPer+KLD with and without Maximum Softmax Probability (MSP) and with random projections.  The results show that WeiPer+KLD performs well on ResNet backbones, but its performance declines when using Vision Transformer (ViT) backbones.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_20_2.jpg)
> 🔼 This table presents the out-of-distribution (OOD) detection performance of WeiPer and other state-of-the-art methods on three benchmark datasets: CIFAR-10, CIFAR-100, and ImageNet.  The results are reported in terms of AUROC and FPR95 metrics for both near and far OOD datasets.  The table also includes ablation studies for WeiPer, showing the effect of removing specific components and using random projections instead of WeiPer's weight perturbations.  The table highlights WeiPer+KLD's strong performance, especially on near OOD datasets, but also notes its performance degradation with Vision Transformers (ViTs).
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_21_1.jpg)
> 🔼 This table presents the results of the OOD detection performance comparison between WeiPer and other state-of-the-art methods on three benchmark datasets: CIFAR10, CIFAR100, and ImageNet-1K.  The table shows AUROC and FPR95 scores for near and far OOD datasets. It highlights WeiPer's superior performance on several benchmarks, particularly those with ResNet backbones and near OOD datasets.  Ablation studies using different configurations of WeiPer are also included for comparison.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

![](https://ai-paper-reviewer.com/8HeUvbImKT/tables_22_1.jpg)
> 🔼 This table presents the Area Under the ROC Curve (AUROC) and False Positive Rate at 95% True Positive Rate (FPR95) for various OOD detection methods on CIFAR10, CIFAR100, and ImageNet datasets using ResNet and ViT backbones.  The table highlights the performance of WeiPer and its variants (WeiPer+MSP, WeiPer+ReAct, WeiPer+KLD) compared to other state-of-the-art methods.  It also shows ablation studies for WeiPer+KLD to understand the effects of removing the MSP and using random projections.
> <details>
> <summary>read the caption</summary>
> Table 2: OOD Detection results of top performing methods on the CIFAR10, CIFAR100 and ImageNet-1K benchmarks (For a comparison with every other evaluated method of OpenOOD and standard deviation over the CIFAR models, see Appendices A.5 and A.6). The top performing results for each benchmark are displayed in bold and we underline the second best result. Due to WeiPer's random nature, we report the median AUROC score over 10 different seeds. For an easy comparison, we portray the following ablations for CIFAR10 which are seperated by a line: The KLD results are the WeiPer+KLD results without MSP and RP is WeiPer+KLD with weight independent random projections drawn from a standard Gaussian. While WeiPer+KLD performs strongly especially on near datasets using ResNet backbones, its performance deteriorates with ViTs (see Section 4 for discussion).
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8HeUvbImKT/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}