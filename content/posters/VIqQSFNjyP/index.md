---
title: "Connectivity-Driven Pseudo-Labeling Makes Stronger Cross-Domain Segmenters"
summary: "SeCo: Semantic Connectivity-driven Pseudo-Labeling enhances cross-domain semantic segmentation by correcting noisy pseudo-labels at the connectivity level, improving model accuracy and robustness."
categories: ["AI Generated", ]
tags: ["Computer Vision", "Image Segmentation", "🏢 Xidian University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} VIqQSFNjyP {{< /keyword >}}
{{< keyword icon="writer" >}} Dong Zhao et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=VIqQSFNjyP" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/VIqQSFNjyP" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=VIqQSFNjyP&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/VIqQSFNjyP/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Cross-domain semantic segmentation struggles with noisy pseudo-labels, especially under significant domain shifts or when dealing with open-set classes.  Existing pixel-level selection methods fail to effectively identify and correct these speckled noises, leading to error accumulation and reduced model performance.  These issues are further exacerbated when using the Segment Anything Model (SAM) for refinement, which itself introduces uncertainties.

The proposed Semantic Connectivity-driven Pseudo-labeling (SeCo) tackles this challenge by formulating pseudo-labels at the connectivity level.  SeCo uses the SAM model for efficient interaction with pseudo-labels, categorizing semantics into "stuff" and "things" and aggregating speckled labels into semantic connectivities.  A subsequent connectivity classification task identifies and corrects noisy connectivities, leading to cleaner and more accurate pseudo-labels for self-training.  **Extensive experiments show SeCo improves the performance of existing state-of-the-art models across various cross-domain settings**, including domain adaptation, generalization, and source-free/black-box scenarios.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} SeCo improves cross-domain semantic segmentation by focusing on connectivity-level pseudo-label refinement. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} SeCo effectively filters out closed-set and open-set noise in pseudo-labels, improving accuracy. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} SeCo achieves state-of-the-art results across various cross-domain semantic segmentation tasks. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it addresses a critical challenge in cross-domain semantic segmentation: the presence of noisy pseudo-labels.  By proposing a novel method, SeCo, that leverages semantic connectivity to filter and correct noisy labels, the research significantly improves the performance of existing state-of-the-art models in various cross-domain settings, including domain adaptation and generalization, as well as source-free and black-box scenarios. **This is highly relevant to ongoing research efforts in improving the robustness and generalizability of semantic segmentation models**, offering promising new avenues for future work.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_1_1.jpg)

> 🔼 This figure compares different pseudo-labeling methods for semantic segmentation.  It highlights the effectiveness of the proposed method (SeCo) in filtering out noise from pseudo-labels, particularly addressing noise from closed-set data (due to domain shifts) and open-set data (in-the-wild data). SeCo is shown to correct speckled noise better than existing pixel-level and SAM-refined methods by aggregating pixels into semantically connected regions.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparison of (a) Pseudo-Labels (PL), (b) pixel-level PL [39], (c) SAM-refined PL [4], and (d) the proposed connectivity-level PL. The white area in the PL represents the filtered area. Our method effectively filters out and corrects closed-set noise (the orange box) induced by domain shifts, as well as open-set noise (the cyan box) in the wild data (e.g., synthesized from stable diffusion [64]).
> </details>





![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_6_1.jpg)

> 🔼 This table presents the results of experiments conducted to evaluate the effectiveness of integrating the Semantic Connectivity-driven Pseudo-labeling (SeCo) method into several state-of-the-art (SOTA) domain adaptation (DA) methods.  The experiments used the GTA5 dataset as the source domain and compared the mean Intersection over Union (mIoU) scores before and after adding SeCo.  The improvements shown highlight the positive impact of SeCo on various SOTA DA techniques. The table showcases results for both unsupervised domain adaptation (UDA) and other adaptation scenarios. 
> <details>
> <summary>read the caption</summary>
> Table 1: Performance improvement in terms of mIoU score (%) by incorporating SeCo into existing DA methods, where GTA5 serves as the source domain.
> </details>





### In-depth insights


#### Connectivity's Role
The concept of connectivity plays a crucial role in the research paper, acting as a **fundamental bridge** between low-level pixel-wise information and high-level semantic understanding.  The authors cleverly leverage connectivity to **aggregate speckled and noisy pseudo-labels**, transforming them into more robust and reliable semantic units. This approach is particularly effective in handling the challenges of cross-domain semantic segmentation where significant domain shifts introduce noise and uncertainty. By focusing on connectivity, the method effectively **mitigates the impact of pixel-level noise**, simplifying the noise localization and correction process. This is a significant departure from traditional pixel-wise methods, which often struggle with speckled noisy labels.  Furthermore, the study demonstrates that analyzing the distribution of loss at the connectivity level, rather than the pixel level, provides a more efficient means for **identifying and correcting noisy connectivities**.  The overall effect is a significant improvement in the quality of pseudo-labels, leading to stronger and more robust cross-domain segmenters, particularly in demanding scenarios like domain generalization and source-free adaptation.

#### SAM Integration
The integration of Segment Anything Model (SAM) is a **crucial component** of the proposed method.  SAM's ability to generate high-quality masks efficiently from various prompts is leveraged for **semantic connectivity aggregation**.  This is a significant departure from pixel-level approaches, which are shown to be less effective.  The paper highlights the strengths of using SAM for creating connectivity-level pseudo-labels, leading to an improved filtering out of noisy labels, especially within challenging 'stuff' and 'thing' categories.  **The integration of SAM into the PSA module is not merely a refinement step but a core part of the approach's effectiveness** in addressing the inherent limitations of pixel-level selection methods in cross-domain semantic segmentation.  **The synergy between SAM's segmentation capabilities and the connectivity-driven framework is a key innovation**, enabling the method to achieve superior accuracy and robustness, particularly under conditions of severe domain shifts and the presence of open-set noise.

#### Noise Correction
The effectiveness of pseudo-labeling in cross-domain semantic segmentation is significantly hampered by noise in the pseudo-labels, especially under severe domain shifts.  **Addressing this noise is crucial for improving the accuracy and robustness of the models.**  The paper explores different strategies for noise correction, focusing on identifying and correcting both closed-set (speckled and noisy reliable pixels) and open-set (speckled high-confidence pixels from open-set classes) noise.  Instead of pixel-level filtering, the authors propose a novel connectivity-level approach.  This involves aggregating pseudo-labels into connected semantic regions, simplifying noise localization, and then applying a connectivity correction task to refine these regions. This task helps in identifying and correcting connectivity noise, guided by the loss distribution.  The use of the Segment Anything Model (SAM) further facilitates the aggregation and boundary refinement.  **The connectivity-driven pseudo-labeling method demonstrates improved resilience against noise**, offering a more efficient and effective noise correction strategy compared to traditional pixel-level methods and other SAM-based refinements.  The approach effectively addresses the limitations of existing pixel-level methods in handling severe domain shifts and open-set noise, leading to significant improvements in performance.

#### Domain Adaptation
Domain adaptation, in the context of semantic segmentation, tackles the challenge of adapting models trained on one domain (source) to perform effectively on a different domain (target).  **This is crucial because models often underperform when faced with shifts in data distribution, image styles, or environmental conditions.**  A key technique is pseudo-labeling, where the model generates labels for the target domain, then uses these pseudo-labels for further training. However, pseudo-labeling is susceptible to noise and errors, especially under significant domain shifts.  **The paper addresses this limitation by introducing a novel connectivity-driven approach.**  Instead of focusing on pixel-level labels, which are prone to errors, it focuses on aggregating pixels into semantically connected regions, making noise correction and refinement more efficient.  This leads to more robust and accurate models in cross-domain scenarios. The proposed method also shows promise in handling more difficult adaptation tasks such as source-free and black-box adaptation, where access to the source domain is limited or entirely absent. This highlights the importance of robust and noise-resistant techniques for effective model adaptation in real-world scenarios.  **Overall, the approach showcases a promising strategy for addressing the limitations of traditional pseudo-labeling methods in domain adaptation for semantic segmentation.**

#### Future of SeCo
The future of SeCo hinges on addressing its limitations and exploring new applications.  **Improving robustness to diverse noise types** beyond closed- and open-set noise is crucial. This could involve advanced uncertainty estimation techniques or incorporating generative models for more sophisticated noise modeling. **Extending SeCo's capabilities to 3D segmentation** and other modalities (e.g., video, point clouds) would broaden its impact.  **Integration with other large language models (LLMs)** for enhanced semantic understanding and prompt generation is a promising area.  Furthermore, exploring **source-free and black-box adaptation scenarios** for real-world applications remains vital. Finally, **thorough benchmarking on a wider array of datasets** and segmentation tasks will be key for establishing SeCo's generality and effectiveness, paving the way for practical applications in autonomous driving, medical imaging, and other fields.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_2_1.jpg)

> 🔼 The figure illustrates the pipeline of the Semantic Connectivity-Driven Pseudo-labeling (SeCo) method.  It shows two main components: Pixel Semantic Aggregation (PSA) and Semantic Connectivity Correction (SCC). PSA aggregates pixel-level pseudo-labels into semantic connectivities using the Segment Anything Model (SAM), categorizing semantics into 'stuff' and 'things'. SCC then treats these connectivities as classification objects, identifying and correcting noisy connectivities through a connectivity classification task guided by loss distribution. The corrected connectivities produce high-quality pseudo-labels for further self-training.
> <details>
> <summary>read the caption</summary>
> Figure 2: The pipeline of the proposed Semantic Connectivity-Driven Pseudo-labeling (SeCo). In (a), pixel-level pseudo-labels are interactively aggregated into connectivity by SAM using the “stuff and things
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_3_1.jpg)

> 🔼 This figure compares three different methods of aggregating pseudo-labels using the Segment Anything Model (SAM).  The first method (PP-PL) uses point prompts derived from pseudo-labeled pixels. The second method (SA-PL) uses pseudo-labels to fill the connectivity of SAM.  Both of these methods result in amplified pseudo-label noise. The third method, the authors' proposed method, alleviates this problem by employing a strategy that considers both 'stuff' and 'things' categories and utilizes a combination of point and box prompts in SAM for semantic alignment.
> <details>
> <summary>read the caption</summary>
> Figure 3: Comparison of Pseudo-Label (PL) aggregation using different interactive methods with SAM [36]. Both Point Prompt-based Interaction (PP-PL) and Semantic Alignment-based Interaction (SA-PL) amplify pseudo-label noise, whereas our method alleviates this issue.
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_4_1.jpg)

> 🔼 This figure shows the loss distribution plots for semantic connectivity across three different cross-domain semantic segmentation tasks: UDA (Unsupervised Domain Adaptation) from GTA to Cityscapes, UDA from GTA to BDD100K, and SFUDA (Source-Free Unsupervised Domain Adaptation) from GTA to Cityscapes. Each plot displays two components of a bimodal Gaussian distribution, representing clean and noisy connectivities. The noisy ratio for each task is indicated in the plot. The plots demonstrate that the proposed method can effectively locate and filter out noisy connectivities based on their loss distribution.
> <details>
> <summary>read the caption</summary>
> Figure 4: The loss distribution plot of semantic connectivity on different cross-domain segmentation tasks. By establishing a bi-modal Gaussian function, noisy connectivity can be effectively located.
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_8_1.jpg)

> 🔼 This figure compares the performance of widely used pixel-level distillation methods with the proposed Semantic Connectivity Correction (SCC) method.  The comparison is done across various baselines (different semantic segmentation models) and shows that SCC, even without using the Segment Anything Model (SAM), provides significant performance gains compared to traditional distillation.
> <details>
> <summary>read the caption</summary>
> Figure 5: Comparison between widely used pixel-level distillation [90] and Semantic Connectivity Correction (SCC) without using SAM across various baselines.
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_16_1.jpg)

> 🔼 This figure shows the impact of the hyperparameters Tns and Ter on the model's performance in the GTA5 to Cityscapes domain adaptation task, using ProDA as a baseline.  The x-axis of the left graph represents the range of Tns (noise threshold), while the x-axis of the right graph represents the range of Tcr (correction threshold). The y-axis of both graphs shows the mIoU score (mean Intersection over Union), connectivity accuracy, and the number of connectivities (K). The graphs illustrate the trade-off between noise reduction and the quantity of retained connectivities. An optimal balance needs to be found to achieve the highest mIoU score.
> <details>
> <summary>read the caption</summary>
> Figure 6: Evaluation on Tns and Ter in GTA5 → Cityscapes using ProDA [90] as baseline.
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_17_1.jpg)

> 🔼 This figure compares different pseudo-labeling methods for semantic segmentation.  It shows how the proposed method (SeCo) effectively removes noise from pseudo-labels, particularly addressing closed-set noise caused by domain shifts and open-set noise present in real-world data. The comparison highlights SeCo's ability to produce cleaner and more accurate pseudo-labels compared to existing pixel-level and SAM-refined approaches.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparison of (a) Pseudo-Labels (PL), (b) pixel-level PL [39], (c) SAM-refined PL [4], and (d) the proposed connectivity-level PL. The white area in the PL represents the filtered area. Our method effectively filters out and corrects closed-set noise (the orange box) induced by domain shifts, as well as open-set noise (the cyan box) in the wild data (e.g., synthesized from stable diffusion [64]).
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_19_1.jpg)

> 🔼 This figure compares the pseudo-labels generated by four different methods on the GTA5 to BDD-100k dataset.  Method (a) shows the original pseudo-labels. Method (b) shows pixel-level pseudo-labels from a previous work ([39]). Method (c) shows pseudo-labels refined using the Segment Anything Model (SAM) from another work ([4]). Method (d) shows the pseudo-labels generated by the proposed SeCo method. The white areas in the pseudo-labels represent the parts that have been filtered out by the respective methods to remove noise and improve the quality of the pseudo-labels.
> <details>
> <summary>read the caption</summary>
> Figure 9: More visualization results of pseudo-labels from different methods on GTA5 → BDD-100k results.(a) Pseudo-Labels (PL), (b) pixel-level PL [39], (c) SAM-refined PL [4], and (d) the proposed connectivity-level PL. The white area in the PL represents the filtered area.
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_19_2.jpg)

> 🔼 This figure compares different pseudo-labeling methods for cross-domain semantic segmentation.  It shows how the proposed method (SeCo) effectively removes noise from pseudo-labels compared to pixel-level and SAM-refined methods. The noise is categorized as closed-set noise (due to domain shift) and open-set noise (from out-of-distribution data). SeCo excels at identifying and removing these noise types, resulting in cleaner pseudo-labels.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparison of (a) Pseudo-Labels (PL), (b) pixel-level PL [39], (c) SAM-refined PL [4], and (d) the proposed connectivity-level PL. The white area in the PL represents the filtered area. Our method effectively filters out and corrects closed-set noise (the orange box) induced by domain shifts, as well as open-set noise (the cyan box) in the wild data (e.g., synthesized from stable diffusion [64]).
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_20_1.jpg)

> 🔼 This figure compares four different pseudo-labeling methods for cross-domain semantic segmentation.  It shows how the proposed method (d) effectively removes noise (orange and cyan boxes) that is present in other methods. The white areas represent the parts of the images that were filtered due to noise.  The comparison includes a basic pixel-level approach (b), a SAM-refined approach (c), and the authors' proposed connectivity-level approach (d).
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparison of (a) Pseudo-Labels (PL), (b) pixel-level PL [39], (c) SAM-refined PL [4], and (d) the proposed connectivity-level PL. The white area in the PL represents the filtered area. Our method effectively filters out and corrects closed-set noise (the orange box) induced by domain shifts, as well as open-set noise (the cyan box) in the wild data (e.g., synthesized from stable diffusion [64]).
> </details>



![](https://ai-paper-reviewer.com/VIqQSFNjyP/figures_21_1.jpg)

> 🔼 This figure compares different pseudo-labeling methods for semantic segmentation.  It shows how the proposed method (SeCo) effectively removes noise from pseudo-labels, particularly noise caused by domain shifts (closed-set noise) and the presence of open-set classes in the data. The comparison includes pixel-level pseudo-labeling, SAM-refined pseudo-labeling, and the proposed connectivity-level pseudo-labeling. The white areas represent the filtered regions of pseudo-labels.
> <details>
> <summary>read the caption</summary>
> Figure 1: Comparison of (a) Pseudo-Labels (PL), (b) pixel-level PL [39], (c) SAM-refined PL [4], and (d) the proposed connectivity-level PL. The white area in the PL represents the filtered area. Our method effectively filters out and corrects closed-set noise (the orange box) induced by domain shifts, as well as open-set noise (the cyan box) in the wild data (e.g., synthesized from stable diffusion [64]).
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_6_2.jpg)
> 🔼 This table presents the results of incorporating the Semantic Connectivity-driven Pseudo-labeling (SeCo) method into several state-of-the-art domain adaptation (DA) methods.  The baseline DA methods are evaluated on the task of transferring semantic segmentation knowledge from the GTA5 dataset (source domain) to the Cityscapes dataset (target domain).  The table shows the increase in mean Intersection over Union (mIoU) score achieved by adding SeCo to each baseline method. The improvements demonstrate the effectiveness of SeCo in enhancing the performance of existing DA techniques.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance improvement in terms of mIoU score (%) by incorporating SeCo into existing DA methods, where GTA5 serves as the source domain.
> </details>

![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_7_1.jpg)
> 🔼 This table shows the improvement in mean Intersection over Union (mIoU) score achieved by integrating the Semantic Connectivity-driven Pseudo-labeling (SeCo) method with existing state-of-the-art domain generalization methods.  The results are presented for three different backbones (ResNet-101 and MiT-B5) and three different benchmark datasets (Cityscapes, BDD-100K, and Mapillary).  The table compares the performance with and without using SAM (Segment Anything Model) and with and without SeCo, illustrating the effectiveness of the proposed method in enhancing domain generalization capabilities.
> <details>
> <summary>read the caption</summary>
> Table 3: Performance improvement in terms of mIoU score (%) by incorporating SeCo into existing domain generalization methods using GTA5 as the source domain.
> </details>

![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_7_2.jpg)
> 🔼 This table compares the performance of SeCo against other state-of-the-art methods on the Open Compound domain adaptation task using the GTA5 dataset as source and the BDD-100k dataset as target.  The results are presented in terms of mean Intersection over Union (mIoU) score, broken down by weather condition (Rainy, Snowy, Cloudy, Overcast) for both the compound and open-set settings. The table highlights the improvement achieved by SeCo over existing methods.
> <details>
> <summary>read the caption</summary>
> Table 4: The comparison of performance in terms of mIoU score (%) on the Open Compoud domain adaptation task between SeCo (ours) and other state-of-the-art methods.
> </details>

![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_8_1.jpg)
> 🔼 This table presents the performance improvement achieved by integrating SeCo (Semantic Connectivity-driven Pseudo-labeling) with various state-of-the-art domain adaptation (DA) methods.  The results are shown for the GTA5 to Cityscapes unsupervised domain adaptation task, where GTA5 is the source domain and Cityscapes is the target domain. The mIoU (mean Intersection over Union) score, a common metric for evaluating semantic segmentation performance, is used to quantify the improvement.  The table demonstrates how SeCo enhances the performance of existing DA methods, highlighting its effectiveness in improving cross-domain semantic segmentation.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance improvement in terms of mIoU score (%) by incorporating SeCo into existing DA methods, where GTA5 serves as the source domain.
> </details>

![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_8_2.jpg)
> 🔼 This table presents the ablation study results for the Semantic Connectivity-Driven Pseudo-labeling (SeCo) method. It shows the impact of different components of SeCo (PSA and SCC) on the performance of different domain adaptation settings: Unsupervised Domain Adaptation (UDA), Source-Free UDA (SF-UDA), and Black-Box UDA (BB-UDA).  The results are evaluated using the mean Intersection over Union (mIoU) metric.  PSA(6) refers to a specific implementation of PSA using semantic alignment,  while the full PSA and SCC incorporate SAM (Segment Anything Model).
> <details>
> <summary>read the caption</summary>
> Table 6: Ablation experiments of SeCo under various UDA settings on GTA5 → Cityscape adaptation task. PSA: Pixel Semantic Aggregation. SCC: Semantic Connectivity Correction. PSA(6) refers to the interaction with SAM using semantic alignment [5], as shown in Fig. 3. SF-UDA: source-free UDA. BB-UDA: black-box UDA.
> </details>

![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_9_1.jpg)
> 🔼 This table presents the ablation study results on different prompt methods used with the Segment Anything Model (SAM) in the GTA → Cityscape cross-domain semantic segmentation task. It compares four different methods: the baseline without using SAM, using SAM with prompting only, using SAM with semantic alignment, using pixel semantic aggregation (PSA), and finally, using the full SeCo method (PSA+SCC). The results show the mIoU scores for each method, along with the performance differences compared to the baseline. It highlights the effectiveness of the proposed PSA and SCC components for improving performance in this task.
> <details>
> <summary>read the caption</summary>
> Table 7: Ablation studies on 'Prompting Only' (PO) and 'Semantic Alignment'(SA) across multiple tasks in GTA → Cityscape.
> </details>

![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_9_2.jpg)
> 🔼 This table compares the performance of the proposed Semantic Connectivity Correction (SCC) method against the DivideMix method on three different domain adaptation tasks: GTA5 → Cityscapes (Unsupervised Domain Adaptation), SYNTHIA → Cityscapes (Unsupervised Domain Adaptation), and GTA5 → BDD-100k (Open Compound Domain Adaptation).  For each task and method, the mean Intersection over Union (mIoU) score is reported. The table highlights the improvement achieved by SCC over DivideMix and its variants. It shows that SeCo consistently outperforms DivideMix in various cross-domain settings, demonstrating the effectiveness of the proposed method.
> <details>
> <summary>read the caption</summary>
> Table 8: Detailed comparison of our SCC and Dividemix across multiple domain adaptation tasks.
> </details>

![](https://ai-paper-reviewer.com/VIqQSFNjyP/tables_18_1.jpg)
> 🔼 This table presents the improvement in mean Intersection over Union (mIoU) score achieved by integrating the Semantic Connectivity-Driven Pseudo-Labeling (SeCo) method into several state-of-the-art (SOTA) domain adaptation (DA) methods.  The source domain used for training is GTA5, and the target domains are Cityscapes for unsupervised domain adaptation (UDA), and source-free and black-box domain adaptation tasks. The table shows the mIoU scores for each method (baseline and with SeCo) for each target domain and provides the percentage improvement brought about by SeCo. 
> <details>
> <summary>read the caption</summary>
> Table 1: Performance improvement in terms of mIoU score (%) by incorporating SeCo into existing DA methods, where GTA5 serves as the source domain.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/VIqQSFNjyP/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}