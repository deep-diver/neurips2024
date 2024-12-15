---
title: "FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner"
summary: "FlowTurbo: Blazing-fast, high-quality flow-based image generation via a velocity refiner!"
categories: []
tags: ["Computer Vision", "Image Generation", "🏢 Tsinghua University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 1jG5ngXVs3 {{< /keyword >}}
{{< keyword icon="writer" >}} Wenliang Zhao et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=1jG5ngXVs3" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/96854" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/papers/2409.18128" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=1jG5ngXVs3&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/1jG5ngXVs3/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Flow-based generative models offer a promising alternative to diffusion models for image generation, but their sampling process is often slow. This paper tackles this issue by introducing FlowTurbo, a framework designed to accelerate the sampling process while maintaining or even improving the quality of generated images.  The core problem is that existing samplers for flow-based models rely on traditional numerical methods that are not very efficient.

FlowTurbo's key innovation is a lightweight velocity refiner that leverages the observation that the velocity predictions in flow-based models become stable during the sampling process. This allows the model to estimate the velocity more efficiently.  Additionally, FlowTurbo introduces several other techniques such as a pseudo corrector and sample-aware compilation to further speed up the process.  The results demonstrate that FlowTurbo can significantly accelerate various tasks like image generation and editing, achieving real-time performance with state-of-the-art image quality.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} FlowTurbo accelerates flow-based image generation by 53.1%–58.3% (class-conditional) and 29.8%–38.5% (text-to-image). {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A lightweight velocity refiner improves sampling efficiency without sacrificing quality. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Real-time image generation is achieved, establishing a new state-of-the-art. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in image generation because **it introduces FlowTurbo, a novel framework that significantly accelerates flow-based generative models**, achieving real-time performance. This addresses a key limitation of flow-based models and opens new avenues for real-time applications like image editing and inpainting.  Its innovative velocity refiner and other optimizations are valuable contributions to the field, paving the way for more efficient and versatile generative models.  The results showing real-time generation with high image quality (low FID score) are particularly impactful.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/1jG5ngXVs3/figures_2_1.jpg)

> This figure visualizes the stability of velocity predictions in flow-based models compared to diffusion models during sampling.  By plotting the curvature of the sampling trajectories, it demonstrates that flow-based models exhibit significantly more stable velocity predictions, suggesting the potential for optimization by using a lightweight velocity estimation model.





![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_6_1.jpg)

> This table presents the main results of the FlowTurbo method, comparing its performance against the baseline Heun's method for both class-conditional and text-to-image generation tasks.  It shows the FID (Fréchet Inception Distance) scores, latency (in milliseconds per image), and FLOPs (floating point operations) for different configurations of FlowTurbo, indicating improvements in speed and image quality compared to the baseline.





### In-depth insights


#### FlowTurbo's Speedup
FlowTurbo significantly accelerates flow-based image generation by leveraging the inherent stability of velocity predictions during sampling.  **A lightweight velocity refiner, trained in under 6 hours, effectively replaces computationally expensive velocity predictors at specific sampling steps.** This, combined with the pseudo corrector and sample-aware compilation techniques, yields an impressive speedup. Results demonstrate **acceleration ratios of 53.1%-58.3% for class-conditional generation and 29.8%-38.5% for text-to-image generation**.  This efficiency gain is particularly noteworthy, enabling **real-time image generation with a state-of-the-art FID of 2.12 on ImageNet at 100 ms/img.**  The speedup isn't solely due to fewer model evaluations; the integration of Heun's method, pseudo-correction, and sample-aware compilation further optimizes the sampling process, establishing a new standard for flow-based generative model performance.

#### Velocity Refiner
The Velocity Refiner, a core component of FlowTurbo, addresses the computational cost of iterative sampling in flow-based generative models.  **Its key innovation is leveraging the observation that velocity predictions stabilize during sampling.** This stability allows for the training of a lightweight, efficient model—the refiner—to regress the velocity offset at each step. This contrasts with traditional methods, which repeatedly evaluate the full velocity prediction model. By replacing the original, computationally expensive model with the refiner at specific steps, FlowTurbo significantly accelerates generation without sacrificing image quality. **This velocity refinement strategy is thus the engine driving FlowTurbo's speed improvements**, establishing a new state-of-the-art in real-time image generation.

#### Pseudo Corrector
The proposed 'Pseudo Corrector' method offers a computationally efficient approach to enhance the sampling speed in flow-based generative models without significantly compromising the accuracy. By cleverly reusing the velocity prediction from the previous sampling step, it effectively halves the number of model evaluations per step, thus accelerating the generation process.  This is a **significant improvement** over traditional methods such as the Heun method, which requires two model evaluations per step. The **pseudo corrector maintains the same convergence order** as the Heun method, demonstrating its effectiveness in improving efficiency without sacrificing accuracy.  The authors provide a rigorous proof of this convergence, demonstrating that the method achieves the same local truncation error and global convergence order as Heun's method. This makes the pseudo corrector a **valuable contribution**, providing a practical approach to accelerate sampling in flow-based models while maintaining high-quality outputs and opening the door to real-time generation capabilities.

#### Sampling Stability
The concept of 'sampling stability' in the context of generative models, particularly flow-based models, centers on the **consistency and predictability of the sampling trajectory**.  Unlike diffusion models where the noise progressively diminishes, flow-based models aim to learn a velocity field guiding samples along a more direct path from a prior distribution to the data distribution.  Sampling stability, therefore, refers to the **degree to which this velocity field remains consistent during the sampling process.** A highly stable velocity field allows for significant computational gains. **Consistent velocity estimates** mean that computationally expensive model evaluations can be reduced or even replaced with simpler estimations as the sampling progresses, thus enabling faster inference.  Conversely, a highly unstable velocity field would necessitate frequent model evaluations to maintain accuracy, undermining the efficiency advantages of flow-based models.  **Achieving high sampling stability** is key to the success of FlowTurbo, as it enables the creation of a lightweight velocity refiner that accelerates the process while preserving image quality.  **Further research into optimizing the learning process** for flow-based models to enhance sampling stability would yield even more efficient and high-quality image generation.

#### Future of Flows
The "Future of Flows" in generative modeling is bright, with **significant potential for real-time applications** currently hindered by computational constraints.  Flow-based models offer advantages over diffusion models in terms of sampling efficiency and theoretical elegance, but practical implementations have lagged.  Recent advancements, like the velocity-refiner approach in FlowTurbo, highlight the potential for substantial speedups, paving the way for **real-time image generation and editing**.  However, challenges remain: **further improvement of sampling speed and quality**, especially at high resolutions, is needed.  Moreover,  **research into efficient training methods and architectural innovations** tailored to the unique properties of flows is crucial.  Exploring the potential of flows beyond image generation, towards other modalities like video and 3D, is a promising avenue.  Finally, addressing ethical concerns regarding malicious use of generated content via robust safeguards remains a critical aspect of the future of flow-based models.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/1jG5ngXVs3/figures_4_1.jpg)

> This figure illustrates the FlowTurbo framework, highlighting its key components: a lightweight velocity refiner for efficient velocity estimation, a pseudo corrector to reduce model evaluations, and sample-aware compilation for further speedup.  It shows how these components are integrated to accelerate the sampling process in flow-based generative models.


![](https://ai-paper-reviewer.com/1jG5ngXVs3/figures_7_1.jpg)

> The figure visualizes the stability of velocity predictions (vθ) in flow-based models compared to noise predictions (εθ) in diffusion models during sampling.  The curvature of the sampling trajectory is used as a measure of stability. Flow-based models show significantly more stable velocity predictions, suggesting that a simpler, less computationally expensive model could be used to estimate velocity, thereby speeding up generation.


![](https://ai-paper-reviewer.com/1jG5ngXVs3/figures_9_1.jpg)

> The figure visualizes the curvature of sampling trajectories for various diffusion and flow-based models.  It demonstrates that the velocity predictions in flow-based models exhibit greater stability compared to diffusion models during the sampling process, suggesting that a lightweight velocity estimation model could improve the sampling efficiency of flow-based models. 


![](https://ai-paper-reviewer.com/1jG5ngXVs3/figures_19_1.jpg)

> The figure visualizes the stability of velocity predictions in flow-based models compared to diffusion models during sampling.  It plots the curvature of the sampling trajectory for several different models, showing that flow-based models exhibit much more stable velocity predictions, which is the key insight motivating the proposed FlowTurbo framework. This stability allows for the use of a lightweight velocity refiner, leading to faster sampling.


![](https://ai-paper-reviewer.com/1jG5ngXVs3/figures_20_1.jpg)

> The figure visualizes the stability of velocity predictions in flow-based models compared to diffusion models during sampling.  It plots the curvature of the sampling trajectory for several models, showing that flow-based models exhibit greater stability, motivating the use of a lightweight velocity refiner for faster sampling.


![](https://ai-paper-reviewer.com/1jG5ngXVs3/figures_20_2.jpg)

> The figure visualizes the stability of velocity predictions in flow-based models compared to diffusion models during sampling.  By plotting the curvature of the sampling trajectories, it demonstrates that flow-based models exhibit more stable velocity predictions, which suggests the possibility of using a lightweight model to estimate velocity and thus accelerate the sampling process.


![](https://ai-paper-reviewer.com/1jG5ngXVs3/figures_21_1.jpg)

> The figure visualizes the stability of velocity predictions in flow-based models compared to diffusion models during sampling.  It shows that the curvature of the sampling trajectory is significantly lower (smoother) for flow-based models, indicating more stable velocity predictions. This observation supports the use of a lightweight velocity refiner in FlowTurbo to accelerate the sampling process.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_6_2.jpg)
> This table compares the performance of FlowTurbo against other state-of-the-art models on ImageNet class-conditional image generation. The metrics used for comparison are sampling speed (latency in ms/img), FID score (Fréchet Inception Distance, lower is better), Inception Score (IS, higher is better), Precision, and Recall.  The results show that FlowTurbo achieves superior performance in both speed and image quality compared to existing methods.

![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_8_1.jpg)
> This table presents the ablation study of the FlowTurbo framework. It shows the impact of adding each component (velocity refiner, pseudo corrector, and sample-aware compilation) on the FID score and latency.  It also analyzes the effect of different ranges for the hyperparameter ∆t and the impact of changing the number of velocity refiners and pseudo correctors.

![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_8_2.jpg)
> This table presents the ablation study of the FlowTurbo framework. It shows the impact of adding each component of FlowTurbo (lightweight velocity refiner, pseudo corrector, and sample-aware compilation) to a baseline Heun's method.  The study also explores the effects of different hyperparameters such as the time step size (Δt) and number of velocity refiners and pseudo correctors on FID and inference latency. The results demonstrate the effectiveness of each component and the optimal hyperparameter settings for achieving real-time image generation with high-quality.

![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_8_3.jpg)
> This table presents an ablation study on the order of applying different sampling blocks (Heun's method, pseudo corrector, and velocity refiner) within the FlowTurbo framework for class-conditional image generation.  By testing different sequences of these blocks, the authors demonstrate the optimal ordering (H, P, R) for achieving the best balance between sampling speed and image quality.

![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_8_4.jpg)
> This table presents an ablation study on the order of applying different sampling blocks within the FlowTurbo framework.  It compares four configurations: two with two repeated blocks ( [H1P1R1]x2 and H2P2R2 ), and two with three repeated blocks ( [H1P1R1]x3 and H3P3R3 ). The results show that applying the blocks in the order H, P, and then R provides the best trade-off between FID (sampling quality) and latency (sampling speed).

![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_8_5.jpg)
> This table shows the ablation study of different architectures for the velocity refiner. Two different architectures are compared: SiT-S (a smaller version of SiT-XL) and a single block of SiT-XL.  The results demonstrate that using a block of SiT-XL as the refiner yields slightly better FID scores compared to SiT-S.

![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_17_1.jpg)
> This table presents an ablation study on the number of velocity refiners used in the FlowTurbo framework.  By varying the number of refiners while keeping other hyperparameters constant, the authors observe that there's an optimal number of refiners that minimizes the Fréchet Inception Distance (FID) score, a metric indicating image quality.  The table shows the FID and inference latency for different configurations, highlighting the trade-off between speed and quality.

![](https://ai-paper-reviewer.com/1jG5ngXVs3/tables_18_1.jpg)
> This table compares the performance of FlowTurbo against other state-of-the-art methods for text-to-image generation.  It focuses on the trade-off between sampling speed (latency) and image quality (FID).  The results show that FlowTurbo achieves a favorable balance, providing competitive FID scores with significantly faster generation times than other diffusion models that use 15 steps of DPM-Solver++.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/1jG5ngXVs3/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}