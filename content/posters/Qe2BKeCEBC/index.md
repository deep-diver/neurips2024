---
title: "Hybrid Mamba for Few-Shot Segmentation"
summary: "Hybrid Mamba Network (HMNet) boosts few-shot segmentation accuracy by efficiently fusing support and query features using a novel hybrid Mamba architecture, significantly outperforming current state-o..."
categories: []
tags: ["Computer Vision", "Image Segmentation", "🏢 Nanyang Technological University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Qe2BKeCEBC {{< /keyword >}}
{{< keyword icon="writer" >}} Qianxiong Xu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Qe2BKeCEBC" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95222" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=Qe2BKeCEBC&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/Qe2BKeCEBC/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many few-shot segmentation (FSS) methods struggle with high computational complexity due to the use of cross-attention mechanisms. Existing linear methods like Mamba, while efficient, suffer from issues like support information loss during query processing and an intra-class gap where query pixels favor self-similarity over support feature integration. These issues hinder effective use of support information for precise segmentation.

The proposed HMNet tackles these challenges with a hybrid Mamba architecture incorporating two key innovations:  **Support Recapped Mamba (SRM)** periodically re-introduces support features during query processing to prevent information loss; **Query Intercepted Mamba (QIM)** prevents unwanted interactions between query pixels, forcing them to rely more heavily on support information for improved segmentation. Extensive experiments demonstrate HMNet's superior performance on standard benchmarks.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} HMNet, a novel hybrid Mamba network, significantly improves few-shot segmentation accuracy. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The hybrid Mamba architecture efficiently fuses support and query features with linear time complexity, addressing limitations of previous methods. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} HMNet demonstrates superior performance on benchmark datasets, surpassing existing state-of-the-art methods. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents **HMNet**, a novel and efficient few-shot segmentation method that outperforms existing state-of-the-art methods.  Its use of hybrid Mamba blocks addresses key limitations of previous approaches, offering a **linear time complexity** solution.  This opens avenues for improving efficiency in few-shot learning and extends the applicability of Mamba networks to a wider range of computer vision tasks.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_1_1.jpg)

> This figure illustrates three different approaches to fusing support and query features in few-shot segmentation. (a) shows the existing cross-Mamba method, which suffers from support forgetting and intra-class gap issues. (b) and (c) present the proposed support recapped Mamba (SRM) and query intercepted Mamba (QIM), respectively, which address these issues by periodically re-scanning support features and preventing mutual interactions among query pixels.





![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_6_1.jpg)

> This table presents a comparison of the proposed HMNet model's performance with other state-of-the-art models on the PASCAL-5 dataset for few-shot segmentation.  The results are broken down by fold (50-53), showing the mean Intersection over Union (mIoU) and Foreground-Background IoU (FB-IoU) for both 1-shot and 5-shot scenarios.  The best performing model in each category is highlighted in bold.





### In-depth insights


#### Hybrid Mamba Net
The proposed Hybrid Mamba Network (HMNet) innovatively integrates the Mamba architecture into the domain of few-shot segmentation.  **HMNet tackles the limitations of existing cross-attention methods**, which suffer from quadratic complexity and the issues of support forgetting and intra-class gap.  By employing a hybrid approach with a support recapped Mamba and a query intercepted Mamba, HMNet effectively addresses these challenges.  **The support recapped Mamba periodically reintroduces support features**, preventing information loss during query processing.  Simultaneously, the **query intercepted Mamba isolates query pixels**, forcing them to leverage support features rather than relying solely on their inherent similarities. This dual strategy ensures efficient and effective fusion of support and query information.  **HMNet demonstrates a significant performance improvement over existing state-of-the-art methods**, achieving better accuracy with linear complexity, making it a practical and powerful solution for few-shot image segmentation.

#### Support Recapped
The concept of "Support Recapped" in a few-shot segmentation model addresses a critical challenge: **support forgetting**.  Standard approaches often sequentially process support and query features, leading to a gradual dilution of crucial support information within the model's hidden state.  As the query features are processed, their representation becomes dominant, overshadowing the initial support features.  A 'Support Recapped' mechanism combats this by periodically reintroducing or refreshing the support feature representation during the query processing phase.  This ensures that the model consistently maintains access to the rich information encoded in the support set, enhancing its ability to effectively segment the query image. **The frequency and method of reintroduction are crucial design parameters,** impacting the trade-off between computational cost and performance. The effectiveness of this technique highlights the importance of managing feature representations to prevent information loss, especially in resource-constrained scenarios like few-shot learning.  This method demonstrates a clear understanding of the limitations of sequential processing and provides a principled way to mitigate a major bottleneck in few-shot learning,  leading to **improved accuracy and robustness**.

#### Query Intercepted
The concept of "Query Intercepted" in a few-shot segmentation (FSS) context likely refers to a mechanism designed to **improve the utilization of support features** during the query processing phase.  Standard approaches often suffer from a support forgetting issue, where support information is gradually lost as the model processes query pixels.  A query-interception method would **actively prevent the query features from overshadowing support features** by controlling or limiting the self-interactions among query pixels. This might involve architectural modifications to enforce the integration of support information before significant query processing occurs, thus mitigating the intra-class gap problem. **The aim is to force query pixels to leverage available support features more effectively, leading to enhanced accuracy and generalization**. This technique is likely paired with a support recapping mechanism to periodically refresh support information.  This method likely involves a change in how the hidden state is managed and updated during processing to prevent information loss and better guide query pixel classification.

#### Ablation Experiments
Ablation experiments systematically remove components of a model to assess their individual contributions.  **Thoughtful ablation studies are crucial for understanding model behavior and justifying design choices.**  By isolating the impact of each component, researchers can determine which parts are essential for performance and which may be redundant or detrimental. **A well-designed ablation study should consider different combinations and orders of component removal to avoid confounding effects.** For instance, removing two interacting components simultaneously may mask the individual contributions of each. The results of ablation experiments should be presented clearly, often with quantitative metrics and sometimes with qualitative analyses such as visualization. The discussion should focus on **interpreting the results and drawing meaningful conclusions about the model's architecture and functionality.**  A successful ablation study provides strong evidence for the model's design, highlights potential areas for improvement, and enhances the overall understanding of the research contribution.

#### Future Directions
The paper's "Future Directions" section would ideally explore extending the Hybrid Mamba network (HMNet) to handle **more complex scenarios** beyond the current benchmarks.  This includes investigating its performance on datasets with significantly higher class variability and more challenging background clutter.  A crucial area for future work is improving the **computational efficiency** of QIM, ideally through CUDA implementation to accelerate inference.  Further research should focus on the theoretical underpinnings of Mamba in few-shot segmentation, particularly exploring the relationship between Mamba's parameter selection mechanism and the meta-learning nature of the task.  Addressing the **support forgetting** issue more comprehensively, perhaps through adaptive memory mechanisms, is another key area for improvement.  Finally, a thorough investigation into the **generalizability** of HMNet across diverse datasets and image modalities is necessary to assess its broader applicability.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_3_1.jpg)

> This figure illustrates the architecture of the Hybrid Mamba Network (HMNet). It shows how the query image and support image are processed through a shared backbone. The network utilizes alternating self Mamba blocks (SMB) and hybrid Mamba blocks (HMB) to capture intra- and inter-sequence dependencies. The hybrid Mamba block incorporates support recapped Mamba (SRM) and query intercepted Mamba (QIM) to mitigate the issues of support forgetting and intra-class gap.


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_4_1.jpg)

> This figure illustrates the Hybrid Mamba Block (HMB), a key component of the proposed HMNet architecture.  The HMB consists of two main parts: the Support Recapped Mamba (SRM) and the Query Intercepted Mamba (QIM).  The SRM addresses the 'support forgetting' issue by periodically rescanning support features while scanning query features. This ensures that the support information is consistently available during the query processing. The SRM splits query features into patches, downsamples support features, and arranges them alternately. Query features are scanned in parallel (Query Intercepted Mamba, QIM), preventing mutual interaction between query pixels. This forces the query pixels to effectively incorporate support features and mitigates the 'intra-class gap' issue.


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_7_1.jpg)

> This figure shows a qualitative comparison of the proposed HMNet method against the HDMNet baseline on the PASCAL-5i and COCO-20i datasets.  For each dataset, it presents several example images: the support image (providing context for the segmentation), the query image (to be segmented), the segmentation result from HDMNet, and the segmentation result from the proposed HMNet. The visual comparison highlights the superior performance of HMNet in terms of more accurately segmenting foreground objects, especially in complex scenes with clutter.


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_8_1.jpg)

> This figure illustrates the Hybrid Mamba Block (HMB) which consists of two components: the Support Recapped Mamba (SRM) and the Query Intercepted Mamba (QIM).  SRM addresses the support forgetting issue by periodically rescanning support features during the query scan, ensuring sufficient support information. QIM addresses the intra-class gap issue by preventing interactions between query pixels and forcing them to integrate support features, leading to better support information utilization. The figure visually depicts the process of feature arrangement, sequential scanning within SRM and parallel processing within QIM.


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_17_1.jpg)

> This figure details the architecture of the Self Mamba Block (SMB) used in the Hybrid Mamba Network (HMNet).  Panel (a) shows the overall structure of the SMB, which is based on an attention block and includes layer normalization (LN), a self-Mamba module, a feedforward network (FFN), and skip connections. Panel (b) illustrates the self-Mamba module itself, showing how it processes input features using layer normalization, a SILU activation function, and depthwise convolutions. Finally, panel (c) breaks down the self-selective state space model (SSM), showing how it processes the input features using separate SSMs for four different scanning directions to capture long-range dependencies.


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_17_2.jpg)

> This figure visualizes the impact of the Query Intercepted Mamba (QIM) module. The top part shows a standard Mamba approach where support features (Fs) and query features (Fq) are concatenated and scanned sequentially. The resulting enhanced query features (Fw/o QIM) achieve a cosine similarity of 46.0%. The bottom part illustrates QIM, where the support features are first processed to obtain a hidden state (Hs). This hidden state is then used to process each query feature individually in parallel, preventing interactions between query pixels. This parallel processing leads to enhanced query features (Fw/ QIM) with a higher cosine similarity of 59.0%, demonstrating the effectiveness of QIM in fusing support information.


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_18_1.jpg)

> This figure presents a qualitative comparison of the proposed HMNet model's performance against the HDMNet model on the PASCAL-5i and COCO-20i datasets.  For each dataset, several example images are shown, with the support image, query image, HDMNet segmentation result, and HMNet segmentation result displayed in separate rows. The visual comparison highlights the superior ability of the proposed HMNet model to correctly segment foreground (FG) objects and distinguish between FG and background (BG) objects compared to the HDMNet model.  The examples reveal that HMNet generally produces more accurate and complete segmentations, especially in challenging scenarios with complex backgrounds or subtle object boundaries.


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_18_2.jpg)

> This figure provides more qualitative comparison results between the proposed HMNet and HDMNet on the COCO-20 dataset.  It shows example support and query images alongside the segmentation results of each model. The images illustrate various scenarios and object categories, allowing for visual evaluation of the methods' performance.


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/figures_19_1.jpg)

> This figure shows qualitative comparison results of HDMNet and the proposed HMNet on COCO-20 dataset.  For each example, it displays the support images, query images, segmentation masks produced by HDMNet, and segmentation masks from the proposed HMNet. The results visually demonstrate that the proposed HMNet achieves better segmentation results than HDMNet, especially for complex scenes.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_7_1.jpg)
> This table compares the performance of the proposed HMNet model with other state-of-the-art models on the PASCAL-5i dataset for few-shot segmentation.  The results are broken down by fold (5i), providing the mean intersection over union (mIoU) and foreground-background IoU (FB-IoU) for both 1-shot and 5-shot scenarios.  Bold values highlight the best-performing model for each metric.

![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_7_2.jpg)
> This table presents the ablation study of different components in the proposed Hybrid Mamba Network (HMNet). It shows the impact of different modules such as Self Mamba Block (SMB), Hybrid Mamba Block (HMB) with Support Recapped Mamba (SRM) and Query Intercepted Mamba (QIM), and BAM (finetuned backbones and base class annotations).  The results are shown in terms of mean intersection over union (mIoU) scores for 5 different folds in a 1-shot setting using ResNet50 backbone.  The table demonstrates the effectiveness of each component in improving the overall segmentation performance.

![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_7_3.jpg)
> This table presents the results of an ablation study on the number of Mamba blocks used in the HMNet model.  It shows the mean Intersection over Union (mIoU), number of parameters, floating point operations (FLOPs), and frames per second (FPS) for different numbers of blocks (4, 6, 8, and 10).  The results indicate that 8 blocks provide the best balance between performance and efficiency. 

![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_14_1.jpg)
> This table compares the performance of the proposed Hybrid Mamba Network (HMNet) against other state-of-the-art methods on the PASCAL-5i dataset for few-shot segmentation.  The results are shown for both 1-shot and 5-shot settings. The table includes the mIoU (mean Intersection over Union) and FB-IoU (Foreground-Background IoU) scores for each fold and the average scores across all folds.  Bold values highlight the best performance for each metric.

![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_15_1.jpg)
> This table presents a comparison of the proposed HMNet model's performance against other state-of-the-art models on the PASCAL-5i dataset for few-shot segmentation.  The results are broken down by fold (5i) to show variability, and then averaged across all folds to provide mean mIoU and FB-IoU (foreground-background IoU) scores.  The best-performing model in each category is highlighted in bold.

![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_15_2.jpg)
> This table presents a comparison of the proposed HMNet model's performance against other state-of-the-art models on the COCO-20 dataset.  The comparison is done using two metrics: mean Intersection over Union (mIoU) and foreground-background IoU (FB-IoU), each averaged over four folds of cross-validation.  The table shows the mIoU for each fold (20<sup>i</sup>) as well as the mean and FB-IoU for both 1-shot and 5-shot settings.  The best performance in each category is highlighted in bold.

![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_16_1.jpg)
> This table compares the performance of different feature fusion methods (BAM, CyCTR, SCCAN) with the proposed HMNet method on the PASCAL-5i dataset using ResNet50 as the backbone. The 1-shot setting is used, and the results show that HMNet significantly outperforms the other methods.

![](https://ai-paper-reviewer.com/Qe2BKeCEBC/tables_16_2.jpg)
> This table presents the efficiency comparison between HDMNet and HMNet on COCO-20² with different image sizes (473x473, 633x633, 793x793, 953x953, 1113x1113).  The time (in seconds) taken for testing 4000 episodes using a single 32GB V100 GPU is shown for both methods under 1-shot setting with ResNet50 backbone.  It highlights that HMNet is significantly faster than HDMNet, especially with increasing image size.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Qe2BKeCEBC/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}