---
title: "Progressive Exploration-Conformal Learning for Sparsely Annotated Object Detection in Aerial Images"
summary: "Progressive Exploration-Conformal Learning (PECL) revolutionizes sparsely annotated object detection in aerial images by adaptively selecting high-quality pseudo-labels, overcoming limitations of exis..."
categories: []
tags: ["Computer Vision", "Object Detection", "🏢 Nanjing University of Science and Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Jzog9gvOf6 {{< /keyword >}}
{{< keyword icon="writer" >}} Zihan Lu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Jzog9gvOf6" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95684" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=Jzog9gvOf6&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/Jzog9gvOf6/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Sparsely annotated object detection (SAOD) in aerial images is challenging due to the imbalance in predicted probabilities/confidences of aerial objects.  Existing SAOD methods rely on fixed thresholding, which is insufficient for handling this imbalance. This necessitates a more robust and adaptive approach to pseudo-label selection for improved detector performance.

The paper introduces a novel Progressive Exploration-Conformal Learning (PECL) framework that addresses this challenge.  PECL uses a conformal pseudo-label explorer and a multi-clue selection evaluator to guide the adaptive selection of high-quality pseudo-labels. The framework iteratively updates the detector using these labels, creating a closed-loop system. Experiments on public datasets demonstrate that PECL significantly outperforms existing methods for SAOD in aerial imagery, showcasing its effectiveness in real-world applications where manual annotation is limited.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} PECL adaptively selects high-quality pseudo-labels, improving accuracy in sparsely annotated aerial object detection. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The conformal pseudo-label explorer and multi-clue selection evaluator work in tandem to enhance pseudo-label selection. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} PECL demonstrates superior performance compared to existing state-of-the-art methods on benchmark datasets. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in aerial object detection due to its novel approach to handling sparsely annotated data. The proposed method, PECL, offers a significant advancement by adaptively selecting high-quality pseudo-labels. This is particularly important given the high cost and time associated with manually labeling aerial images. PECL's adaptive learning framework also opens new avenues for research in semi-supervised learning and other related fields. The code release further enhances the paper's impact, facilitating broader adoption and future research building on its foundation.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/Jzog9gvOf6/figures_1_1.jpg)

> The figure shows a comparison between a global aerial image (a) and a local view of the same image (b) focusing on a sparsely annotated region.  The global view provides context, while the local view highlights the challenge of the sparsely annotated object detection (SAOD) task. Only a small subset of objects within the area of interest have been annotated, making it difficult for a model to learn effectively. Different colors represent different types of vehicles.





![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_5_1.jpg)

> This table presents a comparison of different object detection models' performance on the DOTA dataset when trained with varying amounts of labeled data (1%, 2%, 5%, and 10% label rates).  The models compared include S2A-Net, OR-CNN, and ReDet, both with and without the proposed PECL method.  The performance is measured using mean Average Precision (mAP) for oriented bounding boxes (OBB). The table allows for the assessment of PECL's effectiveness in boosting the performance of different base detectors under sparse annotation conditions.





### In-depth insights


#### SAOD in Aerial Images
Sparsely Annotated Object Detection (SAOD) in aerial images presents a unique challenge in computer vision due to the **scarcity of labeled data** and the **complexity of aerial scenes**.  Traditional object detection methods require vast amounts of labeled data, which is often expensive and time-consuming to acquire for aerial imagery.  SAOD aims to address this by leveraging a smaller number of annotated objects, combined with a larger number of unlabeled images.  This necessitates the development of techniques capable of effectively learning from limited supervision, often incorporating strategies like pseudo-labeling and semi-supervised learning.  The inherent characteristics of aerial images, such as **high variability in object scale, viewpoint, and occlusion**, pose further challenges.  Furthermore, **imbalanced class distributions** are common in aerial datasets. Effective SAOD methods must account for these issues to achieve reasonable performance. The focus on this specific domain highlights the need for algorithms robust to noisy or uncertain labels, and capable of effectively transferring knowledge from limited annotated data to a broader range of unseen aerial scenes.

#### Conformal Learning
Conformal learning is a fascinating field that offers a unique approach to uncertainty quantification in machine learning models. **Its strength lies in its ability to provide reliable prediction intervals, regardless of the underlying model's complexity or assumptions.** This is achieved by calibrating predictions through a process involving nonconformity scores that measure how unusual a data point is, relative to a reference set.  **The conformal approach allows for the incorporation of diverse data types and model architectures without sacrificing the accuracy or robustness of the uncertainty estimates.**  In the context of object detection, where the confidence scores associated with an object's presence can vary drastically due to various factors, **conformal learning can provide crucial insights into the reliability of detections, which is particularly important when dealing with limited training data or high levels of noise.** This is especially useful in domains like aerial imagery, where the sheer complexity of the scenes and the diversity of objects add significant challenges to the model's accuracy.

#### Pseudo-label Exploration
The concept of 'Pseudo-label Exploration' in the context of sparsely annotated object detection for aerial images is crucial.  It tackles the challenge of limited labeled data by intelligently identifying and utilizing high-confidence pseudo-labels from unlabeled data.  This process involves **adaptively selecting pseudo-labels**, avoiding the limitations of fixed-threshold methods that struggle with imbalanced confidence scores in aerial images.  A key aspect is the **conformal prediction framework**, which quantifies uncertainty and allows for principled selection based on confidence levels.   The method's success hinges on effectively characterizing and incorporating contextual information from the images, potentially through methods such as online clustering or other feature analysis techniques. **Multi-clue selection evaluators** provide essential feedback, optimizing the exploration strategy. This approach demonstrates a move towards more sophisticated, data-driven methods for semi-supervised learning in challenging domains. The focus is not simply on quantity of pseudo-labels, but on the **quality and reliability** of those chosen to effectively enhance the training process. This adaptive and principled exploration strategy offers significant advantages in improving object detection performance when labeled data is scarce.

#### Progressive Optimization
Progressive optimization, in the context of machine learning for object detection, particularly in sparsely annotated aerial imagery, refers to iterative refinement strategies.  The core idea is to gradually improve the model's accuracy by sequentially incorporating new information, rather than relying on a single, large training batch. This approach is particularly valuable when dealing with limited labeled data, a common challenge in aerial image analysis.  **By incrementally introducing high-confidence pseudo-labels generated from unlabeled data, progressive optimization helps to bootstrap the training process.** This is crucial because traditional approaches struggle with insufficient labeled samples.  The iterative nature allows the model to learn from increasingly complex datasets, starting with easier-to-classify instances.  **Conformal learning techniques can be integrated to provide a measure of confidence in the pseudo-labels, improving the reliability of the training data.** The iterative cycle of refining pseudo-label selection and detector training is a key component of this strategy, enabling continual adaptation.  This allows for a dynamic learning process which is well-suited to the variable nature of aerial scenes and the inherent challenges of sparse annotation.

#### Future Research
Future research directions stemming from this work on progressive exploration-conformal learning for sparsely annotated object detection in aerial images could focus on several key areas.  **Improving robustness to challenging weather conditions and varying image resolutions** would enhance real-world applicability.  Exploring alternative reward functions and exploration strategies within the reinforcement learning framework could lead to more efficient and accurate pseudo-label selection.  **Investigating the effectiveness of the proposed method on other object detection datasets** beyond DOTA and HRSC2016 would broaden the scope of the findings. Furthermore,  **in-depth analysis of the conformal prediction methodology** and its interaction with the progressive exploration aspect of the framework is warranted. This could involve exploring different non-conformity measures and examining the impact of varying levels of uncertainty on the final results.  Finally, exploring ways to integrate this approach with other semi-supervised or weakly-supervised methods could pave the way for even more efficient use of scarce annotation resources in aerial object detection.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/Jzog9gvOf6/figures_8_1.jpg)

> This figure displays the performance comparison of different reward settings at various label rates on the DOTA dataset.  The reward settings include a baseline, adding information entropy (IE), adding confidence margin (CM), adding both IE and CM, and adding IE, CM, and value binarization (VB). The label rates are 1%, 2%, 5%, and 10%. The performance is measured as mAP (mean Average Precision). The bars show that the addition of IE, CM, and VB improves the performance consistently across different label rates.


![](https://ai-paper-reviewer.com/Jzog9gvOf6/figures_8_2.jpg)

> This figure shows the training loss curves for both the conformal pseudo-label explorer and the multi-clue selection evaluator.  Two different action space sizes (ASS=1 and ASS=2) are compared.  The x-axis represents the training epoch, and the y-axis shows the loss value. The plot demonstrates that with the larger action space (ASS=2), both components converge faster and reach lower loss values, suggesting improved performance with this configuration in the sparsely annotated aerial object detection task.


![](https://ai-paper-reviewer.com/Jzog9gvOf6/figures_13_1.jpg)

> This figure shows a qualitative comparison of object detection results on three datasets (DOTA OBB, DOTA HBB, and HRSC2016 OBB) using four different methods: ground truth, ReDet (baseline), Region-based, and the proposed PECL method.  The visualization highlights the improved accuracy and precision of the PECL method, particularly in detecting smaller objects.  Each row shows results from a different dataset, while each column represents a different method, allowing for easy comparison of the performance.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_6_1.jpg)
> This table presents a comparison of the performance of several object detection methods (S2A-Net, OR-CNN, and ReDet) on the DOTA dataset for oriented bounding box (OBB) detection. The comparison is performed under different data sparsity levels (1%, 2%, 5%, and 10% label rates), reflecting the performance of the base detectors and their respective improvements when using the proposed PECL framework.  The table shows the mAP for each detector and label rate, allowing for an assessment of the effectiveness of PECL across various baselines and sparsity conditions.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_7_1.jpg)
> This table compares the performance of three different object detectors (S2A-Net, OR-CNN, and ReDet) on the horizontal bounding box (HBB) object detection task using the DOTA dataset.  The performance is evaluated at four different label rates (1%, 2%, 5%, and 10%), representing the percentage of annotated objects in the dataset.  The table shows the mean average precision (mAP) for each detector at each label rate, with and without the proposed PECL method. This allows for a comparison of the baseline detector performance and the improvement achieved by using PECL.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_7_2.jpg)
> This table presents a comparison of different object detection methods' performance on the HRSC2016 dataset.  It shows the mean Average Precision (mAP) for Oriented Bounding Boxes (OBB) and Horizontal Bounding Boxes (HBB) across various label rates (5% and 10%).  The methods compared include S2A-Net, OR-CNN, and ReDet, both with and without the proposed PECL framework. The table highlights the improvement in accuracy achieved by integrating the PECL framework for different object detection models and different box types.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_7_3.jpg)
> This table compares the performance of the proposed PECL method with other state-of-the-art (SOTA) methods on the DOTA dataset for oriented bounding box (OBB) object detection.  The comparison is specifically for the scenario where only 5% of the objects are labeled for training.  It highlights the improvement achieved by PECL over existing techniques in a low data regime.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_7_4.jpg)
> This table presents a comparison of different strategies for selecting pseudo-labels in the DOTA dataset using a 1% label rate.  The strategies involve using a pseudo-label explorer, a multi-clue selection evaluator, and an experience replay mechanism. The results are evaluated using mean Average Precision (mAP).  The table shows that using all three components yields the highest mAP.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_7_5.jpg)
> This table presents the results of an ablation study evaluating the impact of different features used in constructing the exploratory characteristic in the conformal pseudo-label explorer. The study examines four settings (I-IV) with different combinations of predicted probability, feature similarity, and confidence level.  Setting IV, incorporating all three features, achieves the highest mAP (mean Average Precision) of 63.72%, demonstrating their combined effectiveness in improving pseudo-label selection.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_12_1.jpg)
> This table presents the performance comparisons of different detector baselines after adopting the CLIP model for the OBB task on the DOTA dataset at different label rates (1%, 2%, 5%, and 10%).  It shows that while CLIP alone provides some improvement over the baseline ReDet, combining CLIP with the proposed PECL method (ReDet w/ PECL+CLIP) leads to the best performance across all label rates.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_12_2.jpg)
> This table compares the performance of the state-of-the-art (SOTA) oriented object detector, LSKNet-T, with and without the proposed Progressive Exploration-Conformal Learning (PECL) framework. The results demonstrate a significant improvement in mean Average Precision (mAP) when PECL is integrated with LSKNet-T, highlighting the effectiveness of PECL in enhancing the performance of SOTA oriented detectors for the sparsely annotated aerial object detection task on the DOTA dataset.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_12_3.jpg)
> This table compares the performance of ReDet and ReDet with PECL across various label rates (30%, 50%, 70%, and 100%). It demonstrates the effectiveness of PECL in improving object detection accuracy even with a high proportion of labeled data. Notably, the performance gain reduces as the label rate increases.

![](https://ai-paper-reviewer.com/Jzog9gvOf6/tables_14_1.jpg)
> This table shows the performance of the proposed PECL method with different numbers of prototypes (K) for each class at a 1% label rate on the DOTA dataset.  It demonstrates how the choice of K impacts the model's performance in object detection. The results suggest there is an optimal number of prototypes that balances model complexity and performance, with diminishing returns beyond a certain point.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Jzog9gvOf6/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}