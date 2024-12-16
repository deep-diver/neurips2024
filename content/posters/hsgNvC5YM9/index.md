---
title: "Constant Acceleration Flow"
summary: "Constant Acceleration Flow (CAF) drastically accelerates image generation in diffusion models by leveraging a constant acceleration equation, outperforming state-of-the-art methods in both speed and q..."
categories: ["AI Generated", ]
tags: ["Machine Learning", "Deep Learning", "🏢 Korea University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} hsgNvC5YM9 {{< /keyword >}}
{{< keyword icon="writer" >}} Dogyun Park et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=hsgNvC5YM9" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/hsgNvC5YM9" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/papers/2411.00322" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=hsgNvC5YM9&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/hsgNvC5YM9/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many generative models, such as diffusion models, are slow and computationally expensive due to their multi-step generation process.  Recent models try to speed up the process by simplifying the flow trajectories, but they still face limitations in accurately learning straight trajectories.  This leads to suboptimal performance, especially for few-step generation.



The authors introduce a novel framework called Constant Acceleration Flow (CAF) to overcome these limitations. **CAF uses a simple constant acceleration equation, adding acceleration as a learnable variable to enhance the estimation of the ODE flow.** The study also proposes initial velocity conditioning and a reflow process to improve estimation accuracy. Experimental results show that CAF significantly outperforms existing methods on various datasets, achieving superior performance in both one-step and few-step generation, and demonstrating improved coupling preservation and inversion capabilities.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} CAF improves the speed and accuracy of image generation in diffusion models. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} CAF introduces acceleration as a learnable variable in the ODE framework, leading to more expressive and accurate flow estimation. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} CAF outperforms state-of-the-art methods on benchmark datasets, demonstrating its effectiveness in practical applications. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in generative modeling and diffusion models because it introduces a novel framework that significantly speeds up the generation process while maintaining high image quality.  **The Constant Acceleration Flow (CAF) addresses limitations of existing methods by introducing acceleration as a learnable parameter, improving accuracy and efficiency, and opening avenues for new research in efficient ODE flow estimation.** Its superior performance on benchmark datasets highlights its potential for real-world applications.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_0_1.jpg)

> 🔼 This figure compares the sampling trajectories of Rectified Flow and Constant Acceleration Flow (CAF) to illustrate the importance of Initial Velocity Conditioning (IVC) in addressing the flow crossing problem.  Rectified Flow, due to its constant velocity assumption, struggles with accurate trajectory estimation near intersecting flows, leading to curved paths. In contrast, CAF leverages IVC to successfully learn the ground-truth straight trajectories by reducing the ambiguity at the intersection points.
> <details>
> <summary>read the caption</summary>
> Figure 1: Initial Velocity Conditioning (IVC). We illustrate the importance of IVC to address the flow crossing problem, which hinders the learning of straight ODE trajectories during training. In Fig. 1a, Rectified flow suffers from approximation errors at the overlapping point xt (where x = x²), resulting in curved sampling trajectories due to flow crossing. Conversely, Fig. 1b demonstrates that CAF, utilizing IVC, successfully estimates ground-truth trajectories by minimizing the ambiguity at xt.
> </details>





![](https://ai-paper-reviewer.com/hsgNvC5YM9/tables_7_1.jpg)

> 🔼 This table presents a comparison of the proposed Constant Acceleration Flow (CAF) model's performance against various other generative models on the CIFAR-10 dataset.  The models are evaluated based on FID (Fréchet Inception Distance) and IS (Inception Score), which are common metrics for assessing the quality of generated images. Lower FID values and higher IS values indicate better performance. The table also includes a comparison of conditional generation, where models are trained to generate images from specific classes within the dataset. The 'Rec' column represents the recall metric, which measures the diversity of generated images.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance on CIFAR-10.
> </details>





### In-depth insights


#### Constant Accel Flow
The concept of "Constant Accel Flow" suggests a novel approach to modeling probability flows in generative models, specifically addressing limitations of existing methods like Rectified Flow.  Instead of assuming constant velocity, **Constant Accel Flow introduces acceleration as an additional learnable parameter**, allowing for more nuanced and accurate representation of complex trajectories between data points.  This approach shows promise in improving the accuracy of ODE flow estimation, leading to better performance in few-step generation and enhanced coupling preservation.  The addition of **initial velocity conditioning and a reflow process further refines trajectory learning**, mitigating issues like flow crossing and improving the overall quality of generated samples.  The framework's effectiveness is demonstrated empirically across various datasets, suggesting its potential to accelerate and enhance the quality of image generation in generative models.

#### Flow Crossing Fix
Addressing flow crossing is crucial for the success of diffusion models. **Flow crossing, where trajectories of interpolants intersect, introduces approximation errors** and hinders learning of straight ODE trajectories.  This problem is typically tackled using a reflow procedure, which iteratively straightens the trajectories by rewiring the flow to build deterministic data couplings. However, **constant velocity modeling, as often employed in Rectified Flow, has limitations** in few-step generation.  **The effectiveness of reflow hinges on the ability to accurately approximate the underlying velocity field**, which is challenged by complex couplings.  Innovative approaches that leverage additional learnable parameters, such as acceleration, or employ sophisticated velocity conditioning strategies, could potentially mitigate the shortcomings of traditional reflow mechanisms and significantly improve the generation quality and speed of diffusion models.  **Addressing the limitations of reflow requires further investigation into more expressive flow models and robust techniques for learning intricate couplings.**

#### Synthetic & Real Data
A robust evaluation of any machine learning model necessitates a multifaceted approach encompassing both synthetic and real-world datasets.  **Synthetic data**, carefully crafted to exhibit specific characteristics, offers unparalleled control over variables, enabling a focused investigation of model behavior under diverse conditions.  This allows for precise measurements of performance and facilitates a deeper understanding of underlying mechanisms.  **Real-world datasets**, conversely, provide a more realistic testbed, reflecting the complexities and nuances inherent in real-world scenarios. While they lack the granular control of synthetic data, their ability to capture the true range of variability and unexpected phenomena is essential for verifying the generalizability of the model's performance. The combination of synthetic and real datasets empowers a holistic evaluation, unveiling both the strengths and limitations of the approach, leading to a more comprehensive and dependable assessment.

#### CAF Ablation Study
A CAF ablation study systematically investigates the contribution of each component within the Constant Acceleration Flow (CAF) model.  By selectively disabling features like **initial velocity conditioning (IVC)** or the **reflow process**, researchers can isolate their individual effects on performance metrics such as FID scores and flow straightness. This approach helps determine which model components are crucial for achieving high-quality image generation and which may be redundant. **Results might show IVC significantly improves accuracy by reducing ambiguity in flow estimation, while reflow primarily enhances coupling preservation.**  A comprehensive analysis allows for refinement of the CAF model and offers insights into the optimal balance between model complexity and performance.  The ablation study strengthens the paper by providing evidence-based justification for design choices and highlighting the core contributions of the CAF framework.

#### Future Directions
Future research could explore **enhanced acceleration modeling techniques** that go beyond the constant acceleration assumption, perhaps incorporating adaptive or learned acceleration profiles for increased expressiveness and accuracy.  Investigating **alternative ODE solvers** beyond the Euler method could further improve sampling efficiency and accuracy. The effectiveness of CAF in higher dimensional spaces and on more complex datasets should also be explored.  **Combining CAF with other generative model techniques**, such as diffusion models, could lead to novel hybrid approaches with improved performance.  Finally,  a thorough investigation into the **generalizability and robustness of CAF** across various tasks and datasets would provide valuable insights into its broader applicability and limitations.  This would strengthen the practical impact and establish the long-term value of CAF.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_1_1.jpg)

> 🔼 This figure compares Rectified Flow and Constant Acceleration Flow (CAF) in terms of handling the flow crossing problem.  Rectified flow, due to its constant velocity assumption, struggles with accuracy when trajectories intersect (flow crossing).  This results in curved sampling paths. In contrast, CAF uses initial velocity conditioning (IVC) and learns an acceleration model, leading to more accurate estimations of straight ODE trajectories and the avoidance of flow crossing.
> <details>
> <summary>read the caption</summary>
> Figure 1: Initial Velocity Conditioning (IVC). We illustrate the importance of IVC to address the flow crossing problem, which hinders the learning of straight ODE trajectories during training. In Fig. 1a, Rectified flow suffers from approximation errors at the overlapping point  xt (where x1 = x2), resulting in curved sampling trajectories due to flow crossing. Conversely, Fig. 1b demonstrates that CAF, utilizing IVC, successfully estimates ground-truth trajectories by minimizing the ambiguity at xt.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_1_2.jpg)

> 🔼 This figure compares Rectified Flow and Constant Acceleration Flow (CAF) in addressing the flow crossing problem during training.  Rectified Flow, using only constant velocity, suffers from approximation errors leading to curved trajectories at points where trajectories intersect.  CAF incorporates Initial Velocity Conditioning (IVC), which helps it accurately learn straight trajectories by minimizing ambiguity at the intersection point and reducing approximation errors.
> <details>
> <summary>read the caption</summary>
> Figure 1: Initial Velocity Conditioning (IVC). We illustrate the importance of IVC to address the flow crossing problem, which hinders the learning of straight ODE trajectories during training. In Fig. 1a, Rectified flow suffers from approximation errors at the overlapping point  xt (where x1 = x2t), resulting in curved sampling trajectories due to flow crossing. Conversely, Fig. 1b demonstrates that CAF, utilizing IVC, successfully estimates ground-truth trajectories by minimizing the ambiguity at xt.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_2_1.jpg)

> 🔼 This figure compares the performance of 2-Rectified Flow and Constant Acceleration Flow (CAF) on 2D synthetic datasets.  It shows that CAF generates samples that more closely match the target distribution (π₁) than 2-Rectified Flow, particularly when using a negative acceleration (h=2).  The figure illustrates both the generated samples and the sampling trajectories for different values of the hyperparameter *h*, which controls the initial velocity and acceleration.
> <details>
> <summary>read the caption</summary>
> Figure 6: Experiments on various 2D synthetic dataset. We compare results between 2-Rectified Flow and our Constant Acceleration Flow (CAF) on 2D synthetic data. π₀ (blue) and π₁ (green) are source and target distributions parameterized by Gaussian mixture models. The generated samples (orange) from CAF form a more similar distribution as the target distribution π₁.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_3_1.jpg)

> 🔼 This figure visualizes how the sampling trajectories of the Constant Acceleration Flow (CAF) model change based on different values of hyperparameter 'h'. The hyperparameter 'h' controls the initial velocity and acceleration, impacting the trajectory's shape.  The figure shows trajectories for three different 'h' values (h=0, h=1, h=2), demonstrating how the acceleration affects the path taken from the initial distribution (π0) to the target distribution (π1).
> <details>
> <summary>read the caption</summary>
> Figure 3: Sampling trajectories of CAF with different h. The sampling trajectories of CAF are displayed for different values of h, which determines the initial velocity and acceleration.  π0 and π1 are mixtures of Gaussian distributions. We sample across sampling steps of N = 7 to show how sampling trajectories change with h.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_8_1.jpg)

> 🔼 This figure compares the image generation quality of 2-Rectified Flow and Constant Acceleration Flow (CAF) on the CIFAR-10 dataset.  Two different numbers of sampling steps (N=1 and N=10) are used. The results show that CAF produces images with more vivid colors and finer details compared to 2-Rectified Flow, highlighting the improvement in image quality achieved by CAF.
> <details>
> <summary>read the caption</summary>
> Figure 4: Qualitative results on CIFAR-10. We compare the quality of generated images from 2-Rectified flow and CAF (Ours) with N = 1 and 10. Each image x₁ is generated from the same x0 for both models. CAF generates more vivid images with intricate details than 2-RF for both N.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_9_1.jpg)

> 🔼 This figure demonstrates the effectiveness of CAF in preserving couplings and generating accurate images compared to Rectified Flow (RF).  Panel (a) shows that while RF's sampling trajectories deviate from the intended path due to flow crossing, CAF maintains accurate trajectories by preserving the coupling between data points. Panel (b) further illustrates this by visually comparing image generation results (using LPIPS scores to quantify perceptual similarity). CAF produces more accurate and visually similar results.
> <details>
> <summary>read the caption</summary>
> Figure 5: Experiments for coupling preservation. (a) We plot the sampling trajectories during training where their interpolation paths I are crossed. Due to the flow crossing, RF (top) rewires the coupling, whereas CAF (bottom) preserves the coupling of training data. (b) CAF accurately generates target images from the given noise (e.g., a car from the car noise), while RF often fails (e.g., a frog from the car noise). LPIPS [52] values are in parentheses.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_17_1.jpg)

> 🔼 This figure compares the performance of 2-Rectified Flow and CAF on 2D synthetic datasets.  The source and target distributions (π₀ and π₁) are shown in blue and green, respectively, and are parameterized using Gaussian mixture models. The generated samples from CAF (orange) are shown to more closely resemble the target distribution (π₁) compared to 2-Rectified Flow.  Different subplots illustrate the results and sampling trajectories with varying values of the hyperparameter *h*, which controls the initial velocity and acceleration in CAF. 
> <details>
> <summary>read the caption</summary>
> Figure 6: Experiments on various 2D synthetic dataset. We compare results between 2-Rectified Flow and our Constant Acceleration Flow (CAF) on 2D synthetic data. π₀ (blue) and π₁ (green) are source and target distributions parameterized by Gaussian mixture models. The generated samples (orange) from CAF form a more similar distribution as the target distribution π₁.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_18_1.jpg)

> 🔼 This figure provides additional qualitative results to support the claim that CAF outperforms Rectified Flow in preserving the coupling between the input noise (x0) and the target image (x1).  It shows several examples of image generation from CIFAR-10 where the target image is generated from a given noise using both CAF and Rectified Flow. CAF demonstrates improved accuracy in generating images that closely match the ground truth, highlighting its superior performance in coupling preservation.
> <details>
> <summary>read the caption</summary>
> Figure 7: Additional visualizations of coupling preservation on CIFAR-10. CAF accurately generates target images (x1) from the given noise (x0), while Rectified Flow often fails to preserve coupling of x0 and x1.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_19_1.jpg)

> 🔼 This figure compares the image generation quality of 2-Rectified flow and Constant Acceleration Flow (CAF) on the CIFAR-10 dataset.  Two different numbers of sampling steps (N=1 and N=10) were used.  The results show that CAF generates images with more vivid colors and finer details than 2-Rectified flow, demonstrating the superiority of CAF in image generation.
> <details>
> <summary>read the caption</summary>
> Figure 4: Qualitative results on CIFAR-10. We compare the quality of generated images from 2-Rectified flow and CAF (Ours) with N = 1 and 10. Each image x₁ is generated from the same x0 for both models. CAF generates more vivid images with intricate details than 2-RF for both N.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_20_1.jpg)

> 🔼 This figure shows the results of conditional image generation on the CIFAR-10 dataset using the proposed Constant Acceleration Flow (CAF) method.  Different rows represent different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Within each row, the left column shows the generated image with only one sampling step, the middle column shows the result with 10 steps, and the rightmost column with 50 steps.  The purpose is to demonstrate that CAF generates high-quality images even with a small number of sampling steps, maintaining consistency across various generation lengths.
> <details>
> <summary>read the caption</summary>
> Figure 9: Qualitative results on conditional generation (CIFAR-10). We illustrate generating images with varying sampling steps, demonstrating consistency quality even for a one-step generation.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_21_1.jpg)

> 🔼 This figure compares the qualitative results of unconditional image generation on the CIFAR-10 dataset between two models: 2-Rectified Flow and Constant Acceleration Flow (CAF). Both models are distilled versions which means they have been trained to mimic the output of a larger, pre-trained model.  The comparison is shown for both one-step and ten-step generation.
> <details>
> <summary>read the caption</summary>
> Figure 10: Comparisons on unconditional generation (CIFAR-10). We compare distilled model from 2-Rectified Flow (2-RF+Distill+GAN) and CAF (CAF+Distill+GAN) with qualitative results.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_22_1.jpg)

> 🔼 This figure shows the qualitative results of unconditional image generation using the Constant Acceleration Flow (CAF) model with different values of the hyperparameter *h*.  The hyperparameter *h* scales the initial velocity and affects the acceleration of the flow.  The figure demonstrates that the CAF model generates high-quality images across a range of *h* values, suggesting robustness to the specific choice of this parameter.
> <details>
> <summary>read the caption</summary>
> Figure 11: Unconditional generation for different h on CIFAR-10. We display qualitative results of CAF for different values of h, indicating that our framework is robust to the choice of h.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_23_1.jpg)

> 🔼 This figure compares the reconstruction results using inversion between the proposed CAF model and the baseline Rectified Flow (RF) model.  The ground truth images are shown in (a).  (b) shows the reconstruction using the CAF model with one step, achieving PSNR=46.68 and LPIPS=0.007. (c) shows the reconstruction result of the Rectified Flow model, with one step, showing PSNR=29.33 and LPIPS=0.204.  This illustrates CAF's superior reconstruction performance.
> <details>
> <summary>read the caption</summary>
> Figure 12: Reconstruction results using inversion.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_23_2.jpg)

> 🔼 This figure compares the reconstruction results of the proposed CAF model and the baseline Rectified Flow (RF) model. The figure shows that CAF generates images with higher PSNR and lower LPIPS values compared to RF, indicating that CAF achieves better reconstruction quality.
> <details>
> <summary>read the caption</summary>
> Figure 12: Reconstruction results using inversion.
> </details>



![](https://ai-paper-reviewer.com/hsgNvC5YM9/figures_24_1.jpg)

> 🔼 This figure shows a grid of images generated by the Constant Acceleration Flow (CAF) model on the ImageNet 64x64 dataset.  The model was used with a single sampling step (N=1), achieving a Fréchet Inception Distance (FID) score of 1.69, indicating high-quality image generation. The images demonstrate the model's ability to generate diverse and realistic images across various classes from ImageNet.
> <details>
> <summary>read the caption</summary>
> Figure 14: Qualitative results on conditional generation for ImageNet 64×64 (N = 1, FID=1.69).
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/hsgNvC5YM9/tables_8_1.jpg)
> 🔼 This table compares the flow straightness, measured by the Normalized Flow Straightness Score (NFSS), between 2-Rectified Flow and CAF (ours) on 2D synthetic and CIFAR-10 datasets.  Lower NFSS values indicate straighter trajectories. The results demonstrate that CAF achieves superior flow straightness compared to 2-Rectified Flow.
> <details>
> <summary>read the caption</summary>
> Table 4: Flow straightness comparison.
> </details>

![](https://ai-paper-reviewer.com/hsgNvC5YM9/tables_8_2.jpg)
> 🔼 This table presents the ablation study results on CIFAR-10 with one sampling step (N=1). It shows the impact of different components of the proposed Constant Acceleration Flow (CAF) model on the FID score.  The components include constant acceleration, initial velocity conditioning, and a reflow procedure. Each row represents a different configuration, combining the presence or absence of these elements, and the corresponding FID score is reported. This helps in understanding the contribution of each component to the model's overall performance.
> <details>
> <summary>read the caption</summary>
> Table 5: Ablation study on CIFAR-10 (N = 1).
> </details>

![](https://ai-paper-reviewer.com/hsgNvC5YM9/tables_15_1.jpg)
> 🔼 This table presents a comparison of reconstruction error metrics for different models on a specific task (likely image reconstruction).  Lower LPIPS (Learned Perceptual Image Patch Similarity) and higher PSNR (Peak Signal-to-Noise Ratio) values indicate better reconstruction quality.  The table shows that CAF (Constant Acceleration Flow) and CAF (+GAN) significantly outperform other methods (CM, CTM, 2-RF) demonstrating superior performance in reconstruction.
> <details>
> <summary>read the caption</summary>
> Table 6: Reconstruction error.
> </details>

![](https://ai-paper-reviewer.com/hsgNvC5YM9/tables_15_2.jpg)
> 🔼 This table presents the results of a box inpainting task, comparing different models.  The models are evaluated based on FID (Fréchet Inception Distance), which is a metric used to assess the quality of generated images.  Lower FID indicates better image quality.  The NFE (number of function evaluations) column shows the computational cost of each model.  CM, CTM, and EDM represent baseline methods. 2-RF stands for 2-Rectified Flow.  The CAF (ours) and CAF (+GAN) (ours) results indicate the performance of the proposed Constant Acceleration Flow model, with and without Generative Adversarial Networks (GANs), respectively.
> <details>
> <summary>read the caption</summary>
> Table 7: Box inpainting.
> </details>

![](https://ai-paper-reviewer.com/hsgNvC5YM9/tables_15_3.jpg)
> 🔼 This table compares the performance of the proposed Constant Acceleration Flow (CAF) model with the AGM model.  The comparison highlights key differences in acceleration type (constant vs. time-varying), the existence of a closed-form solution for ODE, the use of a reflow procedure for velocity, and the resulting FID scores on the CIFAR-10 dataset.  Lower FID scores indicate better performance.
> <details>
> <summary>read the caption</summary>
> Table 8: Comparison between AGM and CAF.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hsgNvC5YM9/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}