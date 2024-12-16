---
title: "MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection"
summary: "MambaAD: Linear-complexity multi-class unsupervised anomaly detection using a novel Mamba-based decoder with Locality-Enhanced State Space modules."
categories: ["AI Generated", ]
tags: ["Computer Vision", "Anomaly Detection", "🏢 Zhejiang University Youtu Lab",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 8VKxTlnejE {{< /keyword >}}
{{< keyword icon="writer" >}} Haoyang He et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=8VKxTlnejE" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/8VKxTlnejE" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/8VKxTlnejE/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Industrial visual anomaly detection (AD) is crucial for enhancing efficiency and product quality in smart manufacturing. However, existing methods often struggle with multi-class settings and high computational costs.  Single-class approaches require separate models for each class, leading to increased training and memory needs. While some multi-class methods exist, they often compromise accuracy or efficiency. 



MambaAD addresses these issues by utilizing the Mamba architecture, offering superior long-range modeling and linear efficiency. It comprises a pretrained encoder and a Mamba decoder with multi-scale Locality-Enhanced State Space modules.  These modules integrate Hybrid State Space blocks and multi-kernel convolutions, enabling effective capture of both long-range dependencies and local patterns.  Extensive experiments demonstrate state-of-the-art performance on multiple datasets, highlighting its effectiveness and scalability.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} MambaAD achieves state-of-the-art performance in multi-class unsupervised anomaly detection. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The Locality-Enhanced State Space module effectively integrates global and local feature information. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} MambaAD's linear complexity makes it efficient for large-scale applications. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents **MambaAD**, a novel approach to multi-class unsupervised anomaly detection that achieves state-of-the-art performance.  It leverages the efficiency of Mamba-based models while effectively capturing both local and global patterns in data, making it suitable for various applications and datasets.  The method's linear complexity and superior performance open avenues for further research and practical applications in industrial visual anomaly detection.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/8VKxTlnejE/figures_2_1.jpg)

> 🔼 This figure compares three different anomaly detection models: RD4AD, UniAD, and the proposed MambaAD. RD4AD utilizes a CNN-based decoder to capture local context but struggles with long-range dependencies. UniAD employs a transformer-based decoder, enabling superior global modeling but suffering from quadratic complexity. In contrast, MambaAD leverages the Mamba architecture, which balances global and local modeling capabilities with linear complexity, leading to superior performance. This is reflected in its use of multi-scale features for refined prediction accuracy.
> <details>
> <summary>read the caption</summary>
> Figure 1: Compared with (a) local CNN-based RD4AD [12] and (b) global Transformer-based UniAD [47], ours MambaAD with linear complexity is capable of integrating the advantages of both global and local modeling, and multi-scale features endow it with more refined prediction accuracy.
> </details>





![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_6_1.jpg)

> 🔼 This table presents a comparison of the proposed MambaAD method with several state-of-the-art (SoTA) methods on three different datasets (MVTec-AD, VisA, and Real-IAD) for multi-class anomaly detection.  It shows the performance of each method across image-level and pixel-level evaluations, using seven metrics: AU-ROC, AP, F1_max, and AU-PRO. The table highlights the superior performance of MambaAD in achieving higher scores across all metrics and datasets, demonstrating its effectiveness for multi-class unsupervised anomaly detection.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>





### In-depth insights


#### MambaAD: A Deep Dive
MambaAD: A Deep Dive suggests a detailed exploration of the MambaAD model for multi-class unsupervised anomaly detection.  This would involve a thorough examination of its architecture, **particularly the novel Locality-Enhanced State Space (LSS) modules and Hybrid State Space (HSS) blocks**.  A deep dive would analyze how these components leverage the strengths of both CNNs and the Mamba architecture for effective long-range and local feature modeling, resulting in improved linear complexity.  The discussion should cover the **Hilbert scanning method** and its influence on global connectivity within the state space model.  Furthermore, a deep dive necessitates a comprehensive evaluation of MambaAD's performance across multiple benchmark datasets, considering both image-level and pixel-level anomaly detection metrics.  The analysis should compare MambaAD against existing state-of-the-art approaches and investigate its efficiency gains in terms of model size and computational complexity.  Finally, **a deep dive would critically assess the limitations of MambaAD**, such as potential biases in the datasets or limitations in handling highly complex anomalies. This would provide a complete understanding of its strengths and weaknesses.

#### LSS Module Design
The Locality-Enhanced State Space (LSS) module is a core component designed to synergize global and local feature extraction for superior anomaly detection.  **Its multi-scale architecture** allows the model to capture contextual information at various levels, overcoming limitations of CNN-based approaches that struggle with long-range dependencies and transformer-based methods that suffer from quadratic complexity. The integration of parallel cascaded Hybrid State Space (HSS) blocks and multi-kernel convolutions is particularly noteworthy. The HSS blocks, leveraging the efficiency of state space models, are enhanced with hybrid scanning methods and multiple scanning directions (e.g., Hilbert scans), bolstering their ability to model feature sequences effectively.  This sophisticated approach to feature encoding and decoding, combined with the incorporation of local information through multi-kernel convolutions, enables MambaAD to achieve a holistic understanding of both global patterns and fine-grained details within the input data, crucial for accurate and efficient unsupervised anomaly detection.

#### HSS Block Mechanics
The hypothetical "HSS Block Mechanics" section would delve into the detailed functioning of the Hybrid State Space (HSS) blocks, a core component of the proposed MambaAD architecture.  It would likely explain how these blocks integrate multiple scanning methods (Sweep, Scan, Z-order, Zigzag, and Hilbert) and eight scanning directions to capture both local and global feature relationships. A key focus would be on the **Hilbert scanning method**, highlighting its efficiency in encoding long-range dependencies and how it strengthens global connections within the State Space Model (SSM).  The discussion would likely involve a detailed description of the **Hybrid Scanning (HS) encoder and decoder**, explaining how they transform feature maps into sequential representations suitable for processing by the SSM. Furthermore, it should illustrate how the HSS block leverages these diverse scanning patterns to significantly improve the robustness and accuracy of feature sequence modeling.  **Mathematical formulations and diagrams** would likely be present to clarify the intricate processes involved and the integration with parallel multi-kernel convolutions for enhanced local information extraction.  The overall goal would be to demonstrate how this design choice contributes to MambaAD's superior performance on multi-class anomaly detection tasks, comparing favorably to traditional CNN and transformer-based approaches.

#### MambaAD's Limits
MambaAD, while demonstrating state-of-the-art performance in multi-class unsupervised anomaly detection, exhibits limitations.  **Its reliance on a pre-trained encoder** might limit its generalizability to datasets with significantly different distributions from the training data. The **linear complexity advantage of the Mamba architecture, while beneficial**, could still become computationally expensive when processing very high-resolution images or exceptionally long sequences. Although the Hybrid Scanning method enhances feature extraction, **the choice of scanning methods and directions might impact effectiveness**, requiring further optimization and potentially limiting its adaptability to various anomaly patterns.  Finally, **the effectiveness hinges on the balance between local and global modeling in the LSS modules**. A deeper exploration of this tradeoff and a study on more complex industrial scenarios are necessary to fully understand its limitations and room for improvement.

#### Future of MambaAD
The "Future of MambaAD" holds exciting possibilities.  **Extending MambaAD's capabilities to video anomaly detection** is a natural progression, leveraging the model's strength in handling long-range dependencies within temporal sequences.  This could involve adapting the HSS blocks to process spatiotemporal features effectively.  Furthermore, **exploring different state space models beyond Mamba** could potentially improve efficiency or modeling capacity.  Investigating other SSMs with varying architectural properties might enhance performance on specific anomaly types or datasets.  **Improving the LSS module's ability to handle complex, noisy industrial data** is also crucial.  This could involve advanced regularization techniques or incorporating noise-robust features into the architecture.  Finally, **research on model explainability and interpretability within the context of MambaAD** would enhance trust and adoption in real-world industrial settings.  Developing techniques to visualize anomaly detection processes and identify the key features driving anomaly classification would make MambaAD more transparent and user-friendly.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/8VKxTlnejE/figures_3_1.jpg)

> 🔼 This figure compares three different anomaly detection models: RD4AD, UniAD, and the proposed MambaAD. RD4AD is a local CNN-based model, UniAD is a global transformer-based model, and MambaAD integrates both global and local modeling capabilities.  The comparison highlights MambaAD's linear complexity, which allows it to handle multi-scale features for improved accuracy.
> <details>
> <summary>read the caption</summary>
> Figure 1: Compared with (a) local CNN-based RD4AD [12] and (b) global Transformer-based UniAD [47], ours MambaAD with linear complexity is capable of integrating the advantages of both global and local modeling, and multi-scale features endow it with more refined prediction accuracy.
> </details>



![](https://ai-paper-reviewer.com/8VKxTlnejE/figures_4_1.jpg)

> 🔼 This figure shows different scanning methods and directions used in the Hybrid Scanning (HS) encoder and decoder of the proposed HSS block.  Panel (a) illustrates the Hilbert scanning method, which uses 8 directions to scan the feature map, providing a comprehensive coverage of both local and global information.  Panel (b) shows four other scanning methods (Sweep, Scan, Z-Order, Zigzag) for comparison, highlighting the advantages of the Hilbert method in capturing long-range dependencies and handling various anomalous features.
> <details>
> <summary>read the caption</summary>
> Figure 3: Hybrid Scanning directions and methods. (a) The Hilbert scanning method with 8 scanning directions is used for HS Encoder and Decoder. (b) The other four scanning methods for comparison.
> </details>



![](https://ai-paper-reviewer.com/8VKxTlnejE/figures_6_1.jpg)

> 🔼 This figure compares three different anomaly detection models: RD4AD, UniAD, and the proposed MambaAD. RD4AD uses CNNs, focusing on local features; UniAD uses Transformers, focusing on global features.  MambaAD combines both approaches for a more comprehensive solution.  The illustration highlights that MambaAD leverages multi-scale features and benefits from linear complexity, indicating a potentially more efficient and accurate model.
> <details>
> <summary>read the caption</summary>
> Figure 1: Compared with (a) local CNN-based RD4AD [12] and (b) global Transformer-based UniAD [47], ours MambaAD with linear complexity is capable of integrating the advantages of both global and local modeling, and multi-scale features endow it with more refined prediction accuracy.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_7_1.jpg)
> 🔼 This table shows the incremental ablation results on the MVTec-AD and VisA datasets by adding different components to the basic Mamba model.  It demonstrates the performance improvement of each step: adding the LSS module and then the HSS block.
> <details>
> <summary>read the caption</summary>
> Table 2: Incremental Ablations.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_8_1.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD model with several state-of-the-art (SOTA) methods on three benchmark datasets for multi-class anomaly detection.  The performance is evaluated using seven metrics: AU-ROC, AP, F1_max, and AU-PRO for image-level and pixel-level evaluations. The table shows the mean average deviation (MAD) across all metrics, indicating the overall performance of each model.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_8_2.jpg)
> 🔼 This table presents a quantitative comparison of MambaAD against other state-of-the-art (SOTA) methods for multi-class anomaly detection on several benchmark datasets (MVTec-AD, VisA, and Real-IAD).  For each dataset, the table shows the performance of each method using seven metrics: AU-ROC, AP, F1_max, AU-ROC, AP, F1_max, and AU-PRO (for image-level and pixel-level anomaly detection, respectively).  The results demonstrate MambaAD's superior performance in achieving state-of-the-art results across various datasets and metrics.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_8_3.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD model against several state-of-the-art (SOTA) methods for multi-class anomaly detection.  The comparison is performed across six different datasets (MVTec-AD, VisA, Real-IAD, MVTec-3D, Uni-Medical, and COCO-AD), using seven metrics to evaluate both image-level and pixel-level performance.  The metrics used include AU-ROC, AP, F1_max, and AU-PRO.  The table shows that MambaAD achieves superior performance compared to other methods across multiple datasets.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_9_1.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD model with several state-of-the-art (SOTA) methods on three benchmark datasets for multi-class anomaly detection.  The evaluation metrics include AU-ROC, AP, F1_max, and AU-PRO, providing a comprehensive assessment of both image-level and pixel-level performance. The datasets used are MVTec-AD, VisA, and Real-IAD, each representing different characteristics and challenges in anomaly detection. The results demonstrate the superior performance of the MambaAD model across these datasets and metrics.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_13_1.jpg)
> 🔼 This table presents a comparison of the proposed MambaAD method with other state-of-the-art (SOTA) methods on six different anomaly detection datasets.  The evaluation is done using seven metrics, split into image-level and pixel-level categories.  Image-level metrics include AU-ROC, AP, and F1_max. Pixel-level metrics are AU-ROC, AP, F1_max, and AU-PRO.  The table shows the mean average deviation (MAD) of all seven metrics for each method on each dataset to provide a comprehensive comparison of performance.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_13_2.jpg)
> 🔼 This table presents the quantitative results of the proposed MambaAD method and compares it with other state-of-the-art (SOTA) methods on six different datasets for multi-class anomaly detection. The results are evaluated using seven different metrics, including image-level and pixel-level metrics. The table shows the performance of each method on each dataset for both image-level and pixel-level evaluations.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_14_1.jpg)
> 🔼 This table presents a comparison of the proposed MambaAD method with several state-of-the-art (SoTA) methods for multi-class anomaly detection on three different datasets (MVTec-AD, VisA, and Real-IAD).  It shows the performance of each method across various metrics for both image-level and pixel-level anomaly detection.  The metrics include AU-ROC, AP, F1-max, and AU-PRO. The results demonstrate the effectiveness of MambaAD in achieving superior performance compared to existing methods in most cases.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_14_2.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD model against several state-of-the-art (SOTA) methods on three different anomaly detection datasets (MVTec-AD, VisA, and Real-IAD).  The evaluation is performed in a multi-class setting and uses seven metrics to assess both image-level and pixel-level performance.  The metrics include AU-ROC, AP, F1-max, and AU-PRO, providing a comprehensive evaluation of the models' ability to detect and segment anomalies.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_14_3.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD method against several state-of-the-art (SOTA) methods on various datasets in a multi-class setting.  For each dataset (MVTec-AD, VisA, Real-IAD), image-level and pixel-level metrics (AU-ROC, AP, F1-max, AU-PRO) are reported, along with a mean average deviation (mAD) across all metrics.  The table allows for a direct comparison of performance across different methods for both image-level anomaly detection and pixel-level anomaly segmentation. 
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_15_1.jpg)
> 🔼 This table presents a quantitative comparison of MambaAD against other state-of-the-art (SOTA) methods on multiple datasets using various metrics. The metrics cover both image and pixel level evaluations for multi-class anomaly detection.  The results demonstrate the performance of MambaAD in terms of AU-ROC, AP, F1-max, and AU-PRO across different datasets.  The 'MAD' column represents the mean of all seven metrics, providing a comprehensive performance comparison across various datasets.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_15_2.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD model with several state-of-the-art (SOTA) methods for multi-class anomaly detection.  The comparison is conducted across various datasets (MVTec-AD, VisA, Real-IAD), using multiple evaluation metrics for both image-level and pixel-level performance assessment. These metrics include AU-ROC, AP, F1_max, and AU-PRO, providing a comprehensive evaluation of the model's accuracy and efficiency in handling different types of anomalies across diverse datasets.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_15_3.jpg)
> 🔼 This table presents a comparison of the proposed MambaAD model with several state-of-the-art (SOTA) methods on three different datasets (MVTec-AD, VisA, and Real-IAD) for multi-class anomaly detection.  For each dataset, it shows the performance of each method across image-level and pixel-level evaluation metrics.  Image-level metrics include AU-ROC, AP, and F1-max, while pixel-level metrics include AU-ROC, AP, F1-max, and AU-PRO. A final mAD score, representing the mean across all seven metrics, is also provided for each method and dataset, summarizing overall effectiveness.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_15_4.jpg)
> 🔼 This table presents a comparison of the proposed MambaAD method with other state-of-the-art (SOTA) methods on several benchmark datasets for multi-class anomaly detection.  It shows the performance of each method using image-level and pixel-level metrics (AU-ROC, AP, F1-max, AU-PRO, and MAD). The results demonstrate the effectiveness of MambaAD, which achieves SOTA performance in most metrics across datasets.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_16_1.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD model against several state-of-the-art (SOTA) methods for multi-class anomaly detection across various datasets (MVTec-AD, VisA, Real-IAD).  The performance is evaluated using image-level and pixel-level metrics: AU-ROC, AP, F1_max, and AU-PRO.  The 'MAD' column shows the mean of these seven metrics, providing a holistic performance summary.  The results demonstrate the superior performance of the proposed method across different datasets and evaluation metrics.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_16_2.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD model against several state-of-the-art (SOTA) methods for multi-class anomaly detection.  The evaluation is performed across six different datasets (MVTec-AD, VisA, Real-IAD, MVTec-3D, Uni-Medical, and COCO-AD), using seven metrics (AU-ROC, AP, F1_max, and AU-PRO at both image and pixel levels, plus the mean of these seven, denoted as mAD). The results show that MambaAD achieves superior performance across most datasets and metrics compared to the SOTA methods.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_17_1.jpg)
> 🔼 This table presents a comparison of the proposed MambaAD model's performance with several state-of-the-art (SoTA) methods on six different anomaly detection datasets.  The comparison is done using seven evaluation metrics, categorized into image-level and pixel-level evaluations.  Image-level metrics assess the overall accuracy of anomaly detection, while pixel-level metrics assess the accuracy of anomaly localization (segmentation). The results show that MambaAD achieves superior performance across all datasets and metrics compared to the SoTA methods, demonstrating its effectiveness in multi-class unsupervised anomaly detection.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_17_2.jpg)
> 🔼 This table presents a quantitative comparison of the proposed MambaAD method with several state-of-the-art (SoTA) methods for multi-class anomaly detection on various datasets.  It shows the performance of each method across different datasets using several image and pixel-level metrics (AU-ROC, AP, F1_max, AU-PRO, and a mean average of these metrics (mAD)).  This allows for a comprehensive evaluation of the different models' strengths and weaknesses in terms of both accuracy and efficiency across different anomaly detection scenarios.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_17_3.jpg)
> 🔼 This table presents a comparison of the proposed MambaAD method with several state-of-the-art (SOTA) methods for multi-class anomaly detection on various datasets.  It shows the performance of each method across multiple evaluation metrics, including image-level and pixel-level metrics, providing a comprehensive evaluation of the different algorithms. The metrics reported for each dataset includes AU-ROC, AP, F1_max and AU-PRO.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_18_1.jpg)
> 🔼 This table presents a comparison of the proposed MambaAD method with other state-of-the-art (SOTA) methods on several benchmark datasets for multi-class anomaly detection.  It shows the performance of each method across various image-level and pixel-level metrics, providing a comprehensive evaluation of their effectiveness in this challenging task.  The metrics include AU-ROC, AP, F1-max, and AU-PRO, offering a detailed analysis of the methods' ability to identify and segment anomalies.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

![](https://ai-paper-reviewer.com/8VKxTlnejE/tables_18_2.jpg)
> 🔼 This table presents a comparison of the proposed MambaAD method with several state-of-the-art (SoTA) methods on three different anomaly detection datasets (MVTec-AD, VisA, and Real-IAD).  For each dataset, the table shows the performance of each method across seven evaluation metrics, including image-level and pixel-level metrics.  The image-level metrics (AU-ROC, AP, and F1-max) evaluate the overall classification accuracy, while pixel-level metrics (AU-ROC, AP, F1-max, and AU-PRO) assess the accuracy of anomaly localization and segmentation. This allows for a comprehensive comparison of the methods' performance in both image-level and pixel-level anomaly detection and segmentation tasks.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative Results on different AD datasets for multi-class setting.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/8VKxTlnejE/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}