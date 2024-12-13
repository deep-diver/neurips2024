---
title: "DisC-GS: Discontinuity-aware Gaussian Splatting"
summary: "DisC-GS enhances Gaussian Splatting for real-time novel view synthesis by accurately rendering image discontinuities and boundaries, improving visual quality."
categories: []
tags: ["Computer Vision", "3D Vision", "🏢 Lancaster University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} ScbmEmtsH5 {{< /keyword >}}
{{< keyword icon="writer" >}} Haoxuan Qu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=ScbmEmtsH5" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95092" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=ScbmEmtsH5&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/ScbmEmtsH5/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Gaussian Splatting, while efficient for novel view synthesis, struggles with accurately rendering sharp boundaries and discontinuities due to the continuous nature of Gaussian distributions.  This leads to blurry edges and reduced image quality in complex scenes, hindering real-world applications.  The existing methods tried to tackle this issue but often compromise on rendering speed or quality.

DisC-GS, a novel framework, directly addresses this limitation. It introduces a "pre-scissoring" step to segment Gaussian distributions along boundaries represented by Bézier curves before rendering.  A Bézier-boundary gradient approximation strategy maintains the differentiability of the process, enabling effective loss calculation during training.  Extensive experiments demonstrate DisC-GS's efficacy in rendering high-quality images with sharp boundaries in real-time, surpassing existing methods in various evaluation metrics.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} DisC-GS improves Gaussian Splatting by adding discontinuity-aware rendering, resulting in higher image quality. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A Bézier-boundary gradient approximation strategy maintains differentiability during training, despite the introduction of discontinuities. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The method shows consistent performance improvements across various datasets and 2D/3D Gaussian splatting implementations. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because **it addresses a critical limitation of Gaussian Splatting**, a leading technique in novel view synthesis. By enabling discontinuity-aware rendering, it significantly improves image quality, particularly in scenes with sharp boundaries. This opens avenues for more realistic and efficient 3D scene rendering, impacting various fields like autonomous driving and virtual reality.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/ScbmEmtsH5/figures_1_1.jpg)

> This figure illustrates the limitations of Gaussian splatting in rendering discontinuities and boundaries in images.  (a) shows a ground truth image with a boundary map generated using the Canny algorithm. (b) demonstrates how continuous Gaussian distributions inevitably 'pass over' hard boundaries, causing blurring. (c) compares images rendered with and without the proposed DisC-GS framework, highlighting the improvement in boundary accuracy achieved by DisC-GS.





![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_9_1.jpg)

> This table presents a quantitative comparison of the proposed DisC-GS method against several state-of-the-art novel view synthesis methods.  The comparison is done across three benchmark datasets: Tanks&Temples, Mip-NeRF360, and Deep Blending. The evaluation metrics used are SSIM (Structural Similarity Index), PSNR (Peak Signal-to-Noise Ratio), and LPIPS (Learned Perceptual Image Patch Similarity). Higher values for SSIM and PSNR indicate better image quality, while a lower LPIPS score indicates better perceptual similarity to the ground truth.





### In-depth insights


#### Discontinuity Issue
The core issue revolves around Gaussian Splatting's inability to accurately render sharp discontinuities and boundaries in images due to the inherent smoothness of Gaussian distributions.  **Gaussian functions, being continuous, inevitably blend across edges**, leading to blurry representations of sharp features.  This limitation significantly impacts the visual quality of novel view synthesis, particularly noticeable in scenes with many fine details or distinct objects. The paper highlights this as a major drawback, arguing that the continuous nature of Gaussian distributions fundamentally limits their ability to represent the discrete nature of scene discontinuities.  **Addressing this discontinuity issue is crucial for achieving photorealistic rendering.**  The authors propose a solution to enhance Gaussian Splatting by incorporating Bézier curves to precisely define boundaries and implement a discontinuity-aware blending mechanism.  The solution maintains differentiability during backpropagation by using a Bézier boundary gradient approximation, ensuring the continuous nature of the loss function during optimization. This allows the model to learn the discontinuities explicitly, enabling the generation of high-quality images.

#### Bezier Boundary
The concept of "Bezier Boundary" in the context of a computer graphics research paper likely refers to a method of using Bézier curves to define and represent boundaries within a scene or image.  **Bézier curves are parametric curves defined by control points**, offering flexibility in shaping smooth or sharp contours.  In this application, they provide a powerful way to model complex, irregular boundaries that are difficult to represent with simpler geometric primitives.  The use of Bézier curves for boundaries offers several advantages. Firstly, **their parametric nature allows for smooth transitions** between different sections of a boundary.  Secondly, **they provide a mathematical framework for precise boundary manipulation and calculations**, enabling operations such as boundary intersection, smoothing and clipping.  Furthermore, the ability to control the curve using only a small number of control points makes the approach computationally efficient, particularly beneficial for real-time rendering scenarios. The key insight is that the introduction of Bézier curves allows for discontinuity-aware rendering, making it possible to represent and render sharp edges, discontinuities, and complex shapes more accurately. This addresses a common limitation in techniques such as Gaussian splatting, which struggles with rendering hard boundaries.  In the paper, **Bézier curves likely play a role in ‘pre-scissoring’ Gaussian distributions**, effectively cutting off parts of the distribution that would otherwise spill over the boundaries, leading to more accurate and visually appealing rendering.

#### Gradient Approx
The heading 'Gradient Approx' likely refers to a method for approximating gradients within a neural network framework, specifically addressing the challenge of discontinuity-aware rendering.  **The core problem is maintaining differentiability** during training when introducing discontinuities, as standard gradient descent methods struggle with non-differentiable functions.  The proposed approximation technique likely involves **smoothing or replacing the discontinuous parts** of the function with differentiable counterparts. This could be achieved using techniques like Bézier curves, which are inherently smooth and easily differentiable, to represent boundaries, making the gradient calculation smoother and more stable.  **A key insight is the trade-off between accuracy and differentiability**:  a precise representation of discontinuities may compromise differentiability, while a smoothed approximation sacrifices some accuracy but allows for effective gradient-based optimization.  The success of this approach hinges on finding a suitable balance, ensuring that the approximation remains sufficiently accurate for the task while still enabling stable training. **The effectiveness would be demonstrated empirically**, showing improved rendering quality despite using an approximation method.

#### 3D Gaussian Ext
The heading '3D Gaussian Ext' likely refers to an extension or enhancement of the 3D Gaussian Splatting technique used for novel view synthesis.  This likely involves improving the representation or rendering of 3D scenes using Gaussian distributions. Potential improvements could include **handling discontinuities and boundaries more effectively**, addressing limitations of the original method, perhaps through new algorithms or data structures for representing the Gaussians. It may also involve **enhanced rendering speed or quality**, such as through optimization techniques or more advanced sampling strategies.  Further enhancements could focus on improving the accuracy and efficiency of the representation itself, enabling the handling of more complex scenes or higher resolutions.  **Bézier curves or other shape approximations may play a role**, allowing for more precise boundary representation within the Gaussian framework.

#### Future Works
Future research directions stemming from this paper could explore several avenues. **Improving efficiency** is crucial; while the method demonstrates efficacy, further optimization for real-time performance on larger, more complex scenes is needed.  **Extending the approach** to handle various types of discontinuities and boundaries beyond those presented would broaden its applicability.  Investigating the **impact of different Bézier curve parameterizations** or alternative boundary representation techniques on accuracy and computational cost is warranted.  Finally, a thorough **comparative analysis against a wider range of state-of-the-art techniques** across diverse datasets, with a focus on quantitative and qualitative metrics like PSNR, SSIM, and LPIPS, would strengthen the claims and provide a clearer picture of the method's true capabilities.  **Addressing potential limitations**, like the handling of highly textured or reflective surfaces, should be addressed in future iterations.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/ScbmEmtsH5/figures_4_1.jpg)

> This figure illustrates the process of discontinuity-aware rendering for a single Gaussian distribution.  It shows how additional control points (red and purple) are introduced to define Bézier curves that mask parts of the Gaussian distribution, ensuring that discontinuities are accurately represented in the final rendered image.


![](https://ai-paper-reviewer.com/ScbmEmtsH5/figures_15_1.jpg)

> This figure presents a comparison of images rendered using 2D Gaussian Splatting with and without the proposed DisC-GS method.  The images showcase the improved rendering of discontinuities and boundaries by DisC-GS.  Each row shows the same scene rendered from a slightly different viewpoint, highlighting the accuracy of boundary and discontinuity rendering achieved with DisC-GS.


![](https://ai-paper-reviewer.com/ScbmEmtsH5/figures_16_1.jpg)

> This figure presents a comparison of images rendered using 2D Gaussian Splatting with and without the DisC-GS framework.  Each row shows the same scene rendered from a slightly different viewpoint. The left column shows results from standard 2D Gaussian Splatting, demonstrating blurriness and inaccuracies at discontinuities and boundaries. The middle column presents the same scenes rendered with the DisC-GS method, highlighting significantly improved accuracy at boundaries and discontinuities. The rightmost column displays the ground truth images for comparison.  The highlighted areas in red boxes show regions where the improvement from using DisC-GS is most visually apparent.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_9_2.jpg)
> This table presents a quantitative comparison of the proposed DisC-GS method against several state-of-the-art novel view synthesis methods.  The comparison is performed across three different datasets: Tanks & Temples, Mip-NeRF360, and Deep Blending.  Evaluation metrics used are SSIM (Structural Similarity Index), PSNR (Peak Signal-to-Noise Ratio), and LPIPS (Learned Perceptual Image Patch Similarity). Higher SSIM and PSNR values, and lower LPIPS values, indicate better performance. The table demonstrates the superior performance of DisC-GS across all three datasets and metrics.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_9_3.jpg)
> This table presents the ablation study on the impact of the number of control points used to define each Bézier curve in the DisC-GS framework.  The results (SSIM, PSNR, LPIPS) show that using 4 control points per curve yields the best performance, suggesting a balance between accuracy and computational complexity.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_13_1.jpg)
> This table presents the performance comparison between the baseline method (2D Gaussian Splatting) and the proposed DisC-GS method in terms of MaskedSSIM.  The evaluation is performed separately on two types of image areas: boundary-rich areas and boundary-sparse areas. Boundary-rich areas are those that contain numerous discontinuities and boundaries, while boundary-sparse areas are those with fewer such features. The results show that DisC-GS significantly outperforms the baseline in boundary-rich areas, indicating its effectiveness in handling images with complex boundaries.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_13_2.jpg)
> This table presents the results of an ablation study comparing two variations of the DisC-GS framework. The first variation introduces different numbers of Bézier curves for different Gaussian distributions in the 3D scene representation, while the second variation uses the same number (M) of Bézier curves for each Gaussian.  The results (SSIM, PSNR, LPIPS) show that using a consistent number (M) of curves per Gaussian yields slightly better performance than the variant with varying numbers of curves per Gaussian.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_13_3.jpg)
> This table compares the image sharpness of images rendered by the baseline method (2D Gaussian Splatting) and the proposed DisC-GS method.  Image sharpness is measured using an energy gradient function, as described in the paper.  The results show that DisC-GS produces significantly sharper images than the baseline, highlighting its ability to render sharp boundaries more accurately.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_14_1.jpg)
> This table compares the performance of the proposed DisC-GS framework with and without considering both sides of the Bézier curve when determining if the indicator function needs modification. The results show that considering both sides leads to better performance, indicated by higher SSIM and PSNR, and lower LPIPS.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_14_2.jpg)
> This ablation study analyzes the impact of the small numbers (epsilon, epsilon1, and epsilon2) introduced in the Bézier-boundary gradient approximation strategy on the overall performance of the DisC-GS framework.  The table compares the performance metrics (SSIM, PSNR, LPIPS) obtained with and without these small numbers, demonstrating their contribution to improved rendering accuracy and stability.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_14_3.jpg)
> This table presents the ablation study on the impact of the number of Bézier curves per Gaussian (M) on the performance of the proposed DisC-GS framework. The results show that using 3 Bézier curves per Gaussian yields optimal performance, achieving the highest SSIM and PSNR scores while having the lowest LPIPS score.  The results suggest a good robustness to this hyperparameter, as performance remains high even with variations in the number of curves.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_14_4.jpg)
> This table presents the ablation study on the initial learning rate for the newly introduced attribute Ccurve in the DisC-GS framework. The results show SSIM, PSNR, and LPIPS scores for four different values of lrcurve (1e-4, 2e-4, 5e-4, and 1e-3).  The values are consistent, indicating robustness of the framework to this hyperparameter.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_14_5.jpg)
> This table compares the proposed DisC-GS method with several state-of-the-art novel view synthesis methods on three benchmark datasets: Tanks&Temples, Mip-NeRF360, and Deep Blending.  The performance is evaluated using three metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). Higher PSNR and SSIM values and lower LPIPS values indicate better visual quality.

![](https://ai-paper-reviewer.com/ScbmEmtsH5/tables_18_1.jpg)
> This table presents a quantitative comparison of the proposed DisC-GS method against several state-of-the-art novel view synthesis methods.  The comparison is performed across three benchmark datasets: Tanks&Temples, Mip-NeRF360, and Deep Blending.  Three metrics are used for evaluation: Structural Similarity Index Measure (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Learned Perceptual Image Patch Similarity (LPIPS). Higher SSIM and PSNR values, and lower LPIPS values indicate better performance. The table demonstrates the superior performance of DisC-GS across all three datasets and metrics.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ScbmEmtsH5/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}