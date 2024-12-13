---
title: "Parameter Efficient Adaptation for Image Restoration with Heterogeneous Mixture-of-Experts"
summary: "AdaptIR: A novel parameter-efficient method for generalized image restoration using a heterogeneous Mixture-of-Experts (MoE) architecture that achieves superior performance and generalization."
categories: []
tags: ["Computer Vision", "Image Restoration", "🏢 Tsinghua University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} R7w68Z5iqf {{< /keyword >}}
{{< keyword icon="writer" >}} Hang Guo et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=R7w68Z5iqf" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95197" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=R7w68Z5iqf&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/R7w68Z5iqf/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current all-in-one image restoration models struggle with high computational costs and limited generalization to unseen degradations.  Existing parameter-efficient transfer learning (PETL) methods also lack generalization across varied restoration tasks due to their homogeneous representation. This paper introduces AdaptIR, an innovative solution using a Mixture-of-Experts (MoE) architecture. 



AdaptIR employs an orthogonal multi-branch design to capture diverse representation bases.  **It uses an adaptive base combination to obtain heterogeneous representation for different degradations, addressing the shortcomings of homogeneous PETL methods**.  Extensive experiments show that AdaptIR achieves stable performance on single-degradation tasks and excels in hybrid-degradation tasks, fine-tuning only a tiny fraction of parameters (0.6%) within 8 hours.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} AdaptIR significantly improves the generalization ability of image restoration models by efficiently adapting pre-trained models to various degradation types. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The heterogeneous MoE design of AdaptIR enables learning orthogonal representations for different degradations, leading to stable performance on single-degradation tasks and superior performance on hybrid-degradation tasks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} AdaptIR's parameter efficiency makes it significantly faster and cheaper to train than existing all-in-one methods, addressing scalability issues in generalized image restoration. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in image restoration because it introduces a novel parameter-efficient method, AdaptIR, that significantly improves generalization ability across various degradation types.  **AdaptIR addresses the limitations of existing all-in-one methods and current parameter-efficient transfer learning (PETL) techniques**, offering a more efficient and effective approach.  Its innovative heterogeneous Mixture-of-Experts (MoE) design opens new avenues for research in PETL and generalized image restoration, paving the way for more robust and adaptable models.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_1_1.jpg)

> This figure shows the results of applying existing Parameter-Efficient Transfer Learning (PETL) methods to image restoration tasks.  Panels (a) and (b) demonstrate the instability of PETL methods on single degradation tasks, showing inconsistent performance across different types of image degradation. Panel (c) shows that the performance of these methods further deteriorates when dealing with hybrid degradations, which necessitates heterogeneous representations. Panels (d) and (e) present Fourier analysis of the Adapter method and the proposed AdaptIR method, respectively, illustrating the key difference in their feature representations. The Adapter method shows homogeneous frequency characteristics regardless of the degradation type, while the AdaptIR method exhibits adaptive and heterogeneous features, demonstrating its ability to handle diverse degradations more effectively.





![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_6_1.jpg)

> This table presents a quantitative comparison of different methods for image restoration on hybrid degradation tasks.  The tasks involve restoring images degraded by a combination of low resolution and either noise or JPEG compression. The table shows the PSNR and SSIM metrics for various methods on different benchmark datasets (Set5, Set14, BSDS100, Urban100, Manga109), along with the number of trainable parameters used by each method. The best and second-best results for each metric and dataset are highlighted in red and blue, respectively. This allows for a direct comparison of the performance and efficiency of each approach.





### In-depth insights


#### AdaptIR: MoE Approach
The proposed AdaptIR employs a Mixture-of-Experts (MoE) architecture for parameter-efficient adaptation in image restoration. This approach is particularly noteworthy for tackling the challenge of generalization across diverse degradation types.  Instead of training a monolithic all-in-one model, AdaptIR leverages a pre-trained backbone and adds a lightweight MoE module. This module is composed of multiple branches designed to learn orthogonal feature representations capturing local spatial, global spatial, and channel information, addressing the limitations of homogeneous representations in existing PETL methods.  **The key innovation lies in the heterogeneous representation learning**, enabling the model to effectively adapt to unseen degradations and hybrid scenarios.  **An adaptive base combination mechanism ensures that the model can flexibly utilize these diverse feature representations for different degradation types**, leading to superior performance compared to other parameter-efficient transfer learning (PETL) methods and all-in-one approaches.  The results demonstrate AdaptIR's effectiveness in handling both single and hybrid-degradation tasks with minimal parameter tuning, showcasing its efficiency and generalization capabilities.

#### Heterogeneous Learning
Heterogeneous learning tackles the challenge of handling diverse data types or modalities, unlike homogeneous learning which focuses on a single type.  In the context of image restoration, this means adapting to various degradation types (noise, blur, rain streaks).  **A key advantage is improved generalization**, allowing a model trained on multiple degradations to handle unseen combinations effectively.  **However, heterogeneous learning presents complexities**. Designing models capable of efficiently processing and integrating diverse information requires careful consideration of representation learning and model architectures.  **Mixture-of-Experts (MoE) models** are a prime example of a strategy addressing this, by allocating specialized sub-models to different data types.  The success of heterogeneous learning hinges on achieving **effective representation learning** for each data type and a robust mechanism for integrating those diverse representations to generate a unified outcome.  The goal is a model that is **both efficient and powerful** in handling varied image restoration tasks, outperforming single-task models or limited generalization approaches.

#### Hybrid Degradation Tests
Hybrid degradation tests in image restoration research are crucial for evaluating the **generalization ability** of models.  Unlike single-degradation tests, which assess performance on isolated image imperfections (e.g., blur, noise), hybrid tests evaluate how well a model handles multiple, simultaneous degradations. This is more realistic as real-world images often suffer from combined impairments.  **Robust performance** on hybrid degradation showcases a model's ability to disentangle and address various forms of corruption effectively, going beyond memorizing specific patterns associated with single degradations. The design of effective hybrid tests requires careful consideration of degradation combinations to ensure they are both **challenging and representative** of real-world scenarios. A diverse range of hybrid test sets, combining different types and levels of noise, blur, compression artifacts, etc., is needed to ensure the model's capability is thoroughly assessed. The results from hybrid tests provide a more comprehensive measure of a restoration model's overall effectiveness and its readiness for real-world applications.

#### PETL for Restoration
Parameter-Efficient Transfer Learning (PETL) offers a promising avenue for adapting pre-trained image restoration models to new, unseen degradation types, thus **improving generalization** and reducing computational costs.  The core idea is to fine-tune only a small subset of parameters in a pre-trained backbone, rather than retraining the entire model.  However, direct application of existing PETL methods faces challenges.  Many methods yield **homogeneous representations** across diverse tasks, hindering effective adaptation to heterogeneous degradations where diverse, task-specific features are needed.  Therefore, a key focus for future research should be on developing new PETL techniques that can learn more **heterogeneous representations** and adaptively combine them for optimal performance across different image restoration tasks. This could involve exploring novel architectural designs or training strategies that promote the learning of more diverse and task-specific feature representations.  The goal is to find the sweet spot between parameter efficiency and effective generalization for broader application in image restoration.

#### Future Work: Enhancements
Enhancing the AdaptIR model for future work involves several key areas.  **Improving the efficiency** of the heterogeneous mixture-of-experts (MoE) architecture is crucial, potentially through exploring more efficient attention mechanisms or employing techniques like sparse MoEs.  **Expanding the model's capability** to handle a broader range of image degradations, including those not encountered during training, is essential.  This could involve incorporating more robust feature extraction methods or self-supervised learning techniques.  **Addressing the limitations** revealed in the paper, such as the homogeneous representation issue in certain PETL methods, warrants further research, perhaps through exploring alternative heterogeneous representation learning strategies.  Finally, **evaluating AdaptIR on larger and more diverse datasets** will be necessary to confirm the model's generalization ability and robustness across different types of image degradation.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_3_1.jpg)

> The figure illustrates the AdaptIR architecture and its integration into a pre-trained transformer-based image restoration model. AdaptIR consists of three parallel branches: Local Interaction Module (LIM), Frequency Affine Module (FAM), and Channel Gating Module (CGM).  These modules extract local spatial, global spatial, and channel features, respectively. An Adaptive Feature Ensemble combines these features to generate a heterogeneous representation for different degradation types. This combined representation is then added to the output of the frozen MLP, adapting the pre-trained model to downstream tasks without extensive retraining. The figure highlights the modular and parallel design of AdaptIR, emphasizing its parameter efficiency.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_7_1.jpg)

> This figure shows the Fourier analysis of the output features from the Local Interaction Module (LIM) and the Frequency Affine Module (FAM). The Fourier transform is used to analyze the frequency characteristics of the features, which provides insights into the type of spatial information captured by each module.  The LIM focuses on high-frequency local texture details, while the FAM captures low-frequency global spatial information. This orthogonal representation is a key aspect of the AdaptIR model's design.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_7_2.jpg)

> This figure visualizes the channel activation values generated by the Channel Gating Module (CGM) in the AdaptIR model. The CGM is a component designed to capture channel interactions and perform channel selection. The bar chart shows the activation values for each channel, indicating the relative importance assigned to each channel by the CGM for a particular task. Channels with higher activation values are considered more important for the task. The distribution of activations across channels indicates that the CGM adaptively learns to select salient channels relevant to specific degradations and tasks. This adaptive channel selection is a key feature of AdaptIR’s ability to learn heterogeneous representations for different image restoration tasks.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_8_1.jpg)

> The figure shows the scalability of different PETL methods (AdaptIR, Adapter, LoRA, FacT, and Pretrain) on two hybrid degradation tasks (LR4&Noise30 and LR4&JPEG30) across different datasets (Urban100 and Manga109). The x-axis represents the number of trainable parameters (in millions), and the y-axis represents the PSNR (Peak Signal-to-Noise Ratio) in dB. The plot demonstrates how the performance of each method varies with the number of trainable parameters. AdaptIR consistently outperforms other methods across different parameter settings, showcasing its superior efficiency and scalability.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_12_1.jpg)

> This figure shows the architectures of six different parameter-efficient transfer learning (PETL) methods.  (a) VPT prepends learnable prompt tokens to the input of a transformer layer. (b) Adapter employs a bottleneck structure with an intermediate GELU activation. (c) LoRA uses low-rank matrices to approximate incremental weights in projection layers. (d) AdaptFormer inserts a module before the second LayerNorm, using a parallel insertion. (e) SSF uses learnable scale and shift factors to modulate frozen features. (f) FacT tensorizes ViT and uses low-rank approximations.  The figure provides a visual comparison of how these different PETL methods modify pre-trained transformer models to adapt to new tasks with minimal parameter changes.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_16_1.jpg)

> This figure shows the frequency characteristics of features from the three branches of a classic Mixture-of-Experts (MoE) model when dealing with hybrid degradations (SR4&DN30 and SR4&JPEG30).  It also presents Fourier analyses of the LoRA and FacT models, highlighting the homogeneous representation learned by these methods across different tasks.  The contrast between the classic MoE and these other PETL methods emphasizes the AdaptIR's ability to learn heterogeneous representations, which are crucial for adapting to various restoration tasks.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_16_2.jpg)

> This figure shows the results of experiments comparing the performance of existing PETL methods and the proposed AdaptIR on different image restoration tasks.  Subfigures (a) and (b) demonstrate that applying existing PETL methods directly to single-degradation tasks can result in unstable performance. Subfigure (c) shows that these methods perform poorly on hybrid-degradation tasks, which demand heterogeneous representations. Subfigures (d) and (e) use Fourier analysis to visualize the frequency characteristics of feature representations learned by Adapter and AdaptIR, respectively.  AdaptIR is shown to learn heterogeneous, degradation-specific representations, while Adapter's representations are homogeneous across different degradations.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_17_1.jpg)

> This figure shows that directly applying existing Parameter-Efficient Transfer Learning (PETL) methods to image restoration results in unstable performance for single degradation tasks and suboptimal performance for hybrid degradation tasks, which require heterogeneous representation.  The Fourier analysis reveals that Adapter, a common PETL method, uses homogeneous frequency representation across different degradations, while AdaptIR, the proposed method, adaptively learns degradation-specific heterogeneous representations.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_17_2.jpg)

> This figure visualizes the distribution of feature responses from the three branches (Local Interaction Module, Frequency Affine Module, and Channel Gating Module) of AdaptIR across different image restoration tasks. Each task is represented by a separate subplot, and each subplot shows three histograms representing the feature response densities of the three branches. The figure aims to demonstrate that AdaptIR learns heterogeneous representations by adaptively weighting the contributions of the three branches based on the specific task, highlighting the model's ability to capture task-specific features.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_18_1.jpg)

> This figure shows a visual comparison of different PETL methods on the hybrid degradation task LR4&Noise30 (low-resolution x4 and noise with σ=30).  It highlights the superior performance of the proposed AdaptIR method compared to others (VPT, LORA, Adapter, FacT) in restoring image details and overall quality. The red boxes point to specific regions for a closer comparison. The Appendix contains additional visualization.


![](https://ai-paper-reviewer.com/R7w68Z5iqf/figures_18_2.jpg)

> This figure demonstrates the limitations of existing Parameter-Efficient Transfer Learning (PETL) methods for image restoration.  Panels (a) and (b) show that these methods, when directly applied, lead to unstable performance on single-degradation tasks (low-light enhancement and heavy rain streak removal). Panel (c) highlights their suboptimal performance on hybrid degradations, which demand heterogeneous representations.  Panels (d) and (e) use Fourier analysis to visualize the frequency characteristics of Adapter (a common PETL method) and the proposed AdaptIR method.  The visualization shows that Adapter produces homogeneous frequency representations across different tasks, while AdaptIR learns task-specific, heterogeneous representations.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_7_1.jpg)
> This table compares the performance of AdaptIR against two other all-in-one image restoration methods (AirNet and PromptIR) on single image restoration tasks.  It shows the number of parameters, training time, GPU memory usage, PSNR, and SSIM for each method on light deraining and denoising tasks using different datasets. The training time for AdaptIR only reflects the time spent fine-tuning the pre-trained model for each task, excluding the initial pre-training phase.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_7_2.jpg)
> This table compares the proposed AdaptIR method with two other all-in-one image restoration methods (AirNet and PromptIR) across multiple image restoration tasks.  It shows the number of parameters, GPU memory usage, training time, and performance metrics (PSNR/SSIM) for light deraining, denoising (σ=25), and denoising (σ=30) tasks.  The results highlight AdaptIR's efficiency in achieving superior or comparable performance with significantly fewer parameters and training time compared to the other all-in-one methods.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_8_1.jpg)
> This table presents a quantitative comparison of different parameter-efficient transfer learning (PETL) methods and a fully fine-tuned model on hybrid degradation restoration tasks.  The tasks involve restoring images degraded by a combination of low-resolution and either Gaussian noise or JPEG compression artifacts.  The table shows the PSNR and SSIM scores for each method on five different benchmark datasets (Set5, Set14, BSDS100, Urban100, and Manga109). The number of trainable parameters (#param) for each method is also provided.  The best and second-best results for each dataset are highlighted in red and blue, respectively, to help visualize the performance differences between methods.  This data helps assess the effectiveness and efficiency of different PETL approaches in handling complex, multiple-degradation scenarios.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_8_2.jpg)
> This table presents the ablation study results on the impact of different design choices in AdaptIR on PSNR(dB).  The baseline setting uses depth-separable projection in LIM and FAM, along with channel-spatial orthogonal modeling. The table compares the baseline against several variations, including removing the adaptive feature ensemble, removing depth-separable operations in LIM or FAM, and removing both the CGM and depth-separable operations. The results highlight the contribution of each component to the overall performance.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_8_3.jpg)
> This table presents the ablation study results on the performance of AdaptIR with different components (LIM, FAM, and CGM). The results show that using all three components leads to the best performance on PSNR across different datasets (Set5, Set14, and Urban100). Removing any single component or combination of components significantly reduces the performance, indicating their importance for achieving the best results.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_8_4.jpg)
> This table presents the ablation study on different insertion positions (MLP or Attention) and forms (parallel or sequential) of the AdaptIR module.  The results, measured in PSNR(dB), show the impact of these design choices on performance across three benchmark datasets (Set5, Set14, and Urban100). The goal is to determine the optimal location and method for integrating the AdaptIR module for the best performance gains.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_13_1.jpg)
> This table compares AdaptIR with other prompt-based methods on denoising and deraining tasks.  It shows that AdaptIR achieves comparable or better performance with significantly lower adaptation costs and faster adaptation time.  The table highlights AdaptIR's efficiency and effectiveness compared to other prompt-based approaches.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_14_1.jpg)
> This table presents a quantitative comparison of different parameter-efficient transfer learning (PETL) methods and a fully fine-tuned model for image restoration on hybrid degradation tasks.  The hybrid degradation tasks involve combining low-resolution with either additive Gaussian noise or JPEG compression artifacts.  The table shows the PSNR and SSIM scores achieved by each method on five standard image restoration datasets (Set5, Set14, BSDS100, Urban100, Manga109).  The best and second-best performing methods for each dataset are highlighted in red and blue, respectively.  The number of trainable parameters (#param) for each method is also included, demonstrating the parameter efficiency of the PETL approaches compared to the fully fine-tuned model.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_14_2.jpg)
> This table presents a quantitative comparison of different methods for image restoration on hybrid degradation tasks.  It shows the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) metrics for various methods, including the proposed AdaptIR, on two types of hybrid degradation: low-resolution images with added noise (LR4&Noise30) and low-resolution images with JPEG compression artifacts (LR4&JPEG30).  The results are shown for different datasets (Set5, Set14, BSDS100, Urban100, Manga109) and the number of trainable parameters (#param) used by each method is included. The best and second-best performing methods are highlighted in red and blue respectively, indicating AdaptIR's superior performance.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_14_3.jpg)
> This table presents a quantitative comparison of different parameter-efficient transfer learning (PETL) methods and a fully fine-tuned model on two hybrid image degradation tasks: LR4&Noise30 (low-resolution x4 and noise with σ=30) and LR4&JPEG30 (low-resolution x4 and JPEG compression with quality factor q=30).  The metrics used for evaluation are PSNR and SSIM on standard image restoration datasets (Set5, Set14, BSDS100, Urban100, and Manga109). The table highlights the best and second-best performing methods for each dataset and degradation type, indicating the superiority of the proposed AdaptIR method in terms of both PSNR and SSIM, especially when dealing with limited parameters.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_14_4.jpg)
> This table presents a quantitative comparison of different parameter-efficient transfer learning (PETL) methods and a fully fine-tuned model on two hybrid degradation tasks: LR4&Noise30 (low-resolution x4 and noise with σ=30) and LR4&JPEG30 (low-resolution x4 and JPEG compression with quality factor q=30).  The metrics used for evaluation are PSNR and SSIM on five standard datasets (Set5, Set14, BSDS100, Urban100, Manga109).  The table highlights the best and second-best performing methods for each dataset and degradation type, indicating the superior performance of the proposed AdaptIR method in handling hybrid degradation.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_15_1.jpg)
> This table presents a quantitative comparison of different parameter-efficient transfer learning (PETL) methods and a full fine-tuning approach on hybrid image degradation restoration tasks.  The table shows PSNR and SSIM scores for various methods on different benchmark datasets (Set5, Set14, BSDS100, Urban100, Manga109) under two types of hybrid degradation: LR4&Noise30 (low resolution and noise) and LR4&JPEG30 (low resolution and JPEG compression).  The number of trainable parameters (#param) for each method is also indicated. The best and second-best results for each metric and dataset are highlighted in red and blue, respectively. The purpose of the table is to demonstrate the superiority of the proposed AdaptIR method, which achieves better performance with significantly fewer parameters than other methods, particularly on hybrid degradation tasks.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_15_2.jpg)
> This table presents a quantitative comparison of different parameter-efficient transfer learning (PETL) methods and a fully fine-tuned model on two hybrid degradation tasks: LR4&Noise30 (x4 low-resolution and noise with σ=30) and LR4&JPEG30 (x4 low-resolution and JPEG compression with quality factor q=30).  The metrics used for evaluation are PSNR and SSIM on several benchmark datasets (Set5, Set14, BSDS100, Urban100, Manga109).  The table highlights the best and second-best performing methods for each task and dataset, indicating the superior performance of the proposed AdaptIR method, particularly in achieving stability and high performance in hybrid degradation scenarios.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_15_3.jpg)
> This table compares the performance of different parameter-efficient transfer learning (PETL) methods on a real-world denoising task using the SIDD dataset.  The methods compared include AdaptFor, LoRA, Adapter, FacT, MoE, and the proposed AdaptIR. The table shows the number of parameters (#param) for each method and the peak signal-to-noise ratio (PSNR) achieved.  AdaptIR demonstrates superior performance compared to other PETL methods.

![](https://ai-paper-reviewer.com/R7w68Z5iqf/tables_18_1.jpg)
> This table presents a quantitative comparison of different methods for image restoration on hybrid degradation tasks, specifically LR4&Noise30 and LR4&JPEG30.  The metrics used for comparison are PSNR and SSIM, calculated for various datasets (Set5, Set14, BSDS100, Urban100, and Manga109). The table also includes the number of trainable parameters (#param) for each method. The best and second-best results for each metric and dataset are highlighted in red and blue respectively, allowing for easy comparison of the different methods' performance.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R7w68Z5iqf/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}