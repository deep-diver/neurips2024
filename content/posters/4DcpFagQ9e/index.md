---
title: "Score Distillation via Reparametrized DDIM"
summary: "Researchers improved 3D shape generation from 2D diffusion models by showing that existing Score Distillation Sampling is a reparameterized version of DDIM and fixing its high-variance noise issue via..."
categories: []
tags: ["Computer Vision", "Image Generation", "🏢 MIT",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 4DcpFagQ9e {{< /keyword >}}
{{< keyword icon="writer" >}} Artem Lukoianov et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=4DcpFagQ9e" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/96682" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/papers/2405.15891" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=4DcpFagQ9e&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/4DcpFagQ9e/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current 3D shape generation methods based on 2D diffusion models, like Score Distillation Sampling (SDS), often produce cartoonish, over-smoothed results.  This is because SDS introduces excessive noise variance, unlike the more controlled noise process in Denoising Diffusion Implicit Models (DDIM). This discrepancy in noise handling leads to significant quality differences between 2D and 3D outputs.

This paper provides a novel perspective, demonstrating that SDS is essentially a reparameterized version of DDIM.  Leveraging this insight, the authors introduce Score Distillation via Inversion (SDI), a refined method that corrects the noise approximation problem in SDS. By using DDIM inversion to estimate the noise, SDI significantly improves 3D generation quality, achieving results comparable to or better than state-of-the-art techniques, all without the need for additional training or multi-view supervision.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Score Distillation Sampling (SDS) can be understood as a high-variance version of DDIM. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Inverting DDIM within SDS update steps significantly improves 3D generation quality. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The proposed Score Distillation via Inversion (SDI) method achieves better or similar 3D generation quality compared to state-of-the-art methods without additional training. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in 3D generation because **it addresses the critical issue of over-smoothed and unrealistic shapes produced by current Score Distillation Sampling (SDS) methods.** By providing a theoretical understanding and practical improvements to SDS, it bridges the gap between 2D and 3D generation quality.  This opens avenues for developing higher-quality 3D content generation methods and offers insights into improving the efficiency of existing techniques. The findings could lead to advancements in various applications such as computer graphics, virtual reality, and robotics.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_0_1.jpg)

> This figure demonstrates the differences in image and 3D shape generation between DDIM, SDS, and the proposed method (SDI).  DDIM (a) produces a high-quality 2D image, but SDS (b) applied to the same model results in a blurry image. In 3D shape generation, SDS (d) creates over-smoothed and unrealistic shapes, while the SDI method (e) significantly improves the quality and detail, showing closer alignment to the quality of DDIM in 2D (c). The figure highlights how SDI addresses the limitations of SDS in both 2D and 3D generation.





![](https://ai-paper-reviewer.com/4DcpFagQ9e/tables_7_1.jpg)

> This table presents a quantitative comparison of the proposed SDI method against several state-of-the-art baselines for text-to-3D generation.  The metrics used for comparison include CLIP Score (measuring alignment between generated 3D shapes and text prompts), CLIP IQA (assessing image quality, sharpness, and realism), ImageReward (imitating human preferences), and the percentage of generation failures (diverged). The table also provides the average generation time and VRAM usage for each method. This allows for a comprehensive evaluation of the performance and efficiency of different approaches to text-to-3D generation.





### In-depth insights


#### DDIM-SDS Bridge
The core idea of a 'DDIM-SDS Bridge' would be to formally connect the Denoising Diffusion Implicit Models (DDIM) and Score Distillation Sampling (SDS) methods.  DDIM is a powerful image generation technique, while SDS adapts it for 3D shape creation using 2D diffusion models as guidance.  A key insight would be to show how the image guidance used in SDS can be viewed as a reparameterization of DDIM's denoising process. **This reveals a fundamental connection: SDS is essentially a high-variance version of DDIM with a differently-sampled noise term.** The excessive noise in SDS causes over-smoothing in 3D shapes. A 'bridge' would propose a method to improve the noise approximation in SDS, bringing its generative process closer to the more controlled and higher-quality process of DDIM. **This improved approximation would ideally lead to 3D shapes that have the fidelity and realism of DDIM's 2D image generation, reducing over-smoothing and preserving fine detail.** The resulting algorithm would inherit the efficiency and scalability of SDS while enjoying enhanced generation quality, with no requirement for additional network training.

#### SDI Algorithm
The Score Distillation via Inversion (SDI) algorithm offers a refined approach to 3D shape generation using 2D diffusion models.  **SDI addresses the over-smoothing and lack of detail often observed in existing Score Distillation Sampling (SDS) methods by directly connecting SDS updates to the velocity field of Denoising Diffusion Implicit Models (DDIM).**  This connection reveals that the core difference between SDS and DDIM lies in noise handling; SDS introduces excessive variance through independent noise sampling at each step, while DDIM utilizes a more sophisticated, noise-predictive approach.  **SDI rectifies this by using DDIM inversion to obtain a better approximation of the conditional noise for each SDS step**, significantly improving 3D generation quality.  The method's effectiveness is demonstrated empirically, outperforming other state-of-the-art methods without the need for additional training or multi-view supervision.  **Key to SDI's success is its principled approach which directly tackles the core issue of noisy guidance in SDS, leading to more realistic and detailed 3D shapes.**  While SDI shows promising results, future work could address remaining challenges in 3D consistency and resolving content drift between views.

#### Noise Inversion
The concept of 'noise inversion' in the context of diffusion models, particularly relevant to 3D shape generation, involves reversing the noise addition process to recover cleaner representations.  **This is crucial because the standard Score Distillation Sampling (SDS) method adds noise randomly at each step, leading to over-smoothing and a loss of high-frequency details in the generated 3D shapes.**  Noise inversion, as explored in techniques like Score Distillation via Inversion (SDI), aims to address this by inferring or estimating the noise term more intelligently.  Instead of using random noise, it leverages the pre-trained 2D diffusion model to predict and correct the noise in each step, thus maintaining consistency and improving the generation quality. **The process effectively aligns the generative process in 3D with the refined process in 2D, leading to significantly better results.** This refinement is analogous to conditioning the noise on prior predictions to keep the trajectory coherent.  **This subtle yet powerful change allows for recovering higher-frequency details and avoiding over-saturation, resulting in more realistic and detailed 3D models.**

#### 3D Generation
The 3D generation methods explored in this research paper address the limitations of current approaches, which often result in **over-smoothed or cartoonish shapes**.  The core idea is to leverage pre-trained 2D diffusion models for 3D shape generation, thus avoiding the need for large 3D training datasets.  The authors demonstrate that a better understanding of the underlying diffusion process, specifically the relationship between the image guidance used in Score Distillation Sampling (SDS) and Denoising Diffusion Implicit Models (DDIM), enables significant improvements. **A key contribution is the Score Distillation via Inversion (SDI) method**, which replaces the random noise sampling in SDS with a more informed noise approximation derived from DDIM inversion.  This results in significantly improved 3D generation quality, particularly with enhanced detail preservation and a reduction in over-smoothing.  Experiments compare SDI to state-of-the-art methods, showing competitive or better results without the need for additional training or multi-view supervision.  The findings offer **valuable insights into the relationship between 2D and 3D asset generation** within the context of diffusion models.

#### Future Work
The paper's "Future Work" section suggests several promising avenues.  **Improving 3D consistency** between generated views is crucial, potentially through incorporating pre-trained depth or normal estimators.  Addressing **content drift** across views could involve stronger view conditioning or leveraging multi-view supervision or video generation models.  The inherent limitations of relying on 2D diffusion models for 3D generation are acknowledged, suggesting the exploration of alternative 3D generative approaches.  **Reducing reliance on prompt augmentation** and refining the noise inference process are also key focus areas.  Finally, investigating techniques to mitigate the propagation of biases inherent in the 2D diffusion models to the 3D generation process, and enhancing the **diversity** and quality of the generated assets are mentioned as directions for future research.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_1_1.jpg)

> This figure shows a comparison between DDIM and SDS for both 2D and 3D image generation.  DDIM produces high-quality images, while SDS results in blurry 2D images and over-smoothed, unrealistic 3D shapes. The authors' proposed method improves the quality of SDS, making its results closer to those of DDIM.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_3_1.jpg)

> This figure shows the effect of different classifier-free guidance (CFG) values on the quality of images generated by Stable Diffusion 2.1.  With low CFG values (1 and 5), the generated images lack detail and appear somewhat washed out, indicating that the model is not fully utilizing the prompt. As the CFG value increases (10 and 30), the images become sharper and more detailed, reflecting a more faithful interpretation of the prompt. However, at very high CFG values (100), the images become over-saturated and lose their natural appearance, suggesting that excessive guidance can negatively impact image quality.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_4_1.jpg)

> This figure visually explains the relationship between Score Distillation Sampling (SDS) and Denoising Diffusion Implicit Models (DDIM). The left side shows how noisy images, NeRF representations, and single-step denoised images change over time in SDS and DDIM.  The right side illustrates how DDIM moves towards a denoised image using a change of variables to better understand the SDS process.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_5_1.jpg)

> This figure illustrates the Score Distillation via Inversion (SDI) algorithm.  The process begins by rendering a random view of a 3D shape.  DDIM inversion is then applied to reduce the noise level in the image to 't'. A pre-trained diffusion model further denoises the image to a level of 't-τ'. Finally, this denoised image is used to update the 3D shape via backpropagation, improving its quality iteratively.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_6_1.jpg)

> This figure visually explains the evolution of variables during Score Distillation and its relation to DDIM. The left side shows how noisy images, NeRF representations in 3D, and single-step denoised images change over time in SDS. The right side illustrates how each DDIM step progresses towards a denoised image, highlighting the connection between DDIM and the proposed process on xo(t).


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_7_1.jpg)

> This figure compares the image generation quality of different methods. (a) shows the high-quality image generated by DDIM. (b) shows a blurry image generated by SDS in 2D. (c) shows that by modifying the noise term in SDS to agree with DDIM, a similar image quality to (a) is achieved. (d) shows that SDS produces over-saturated and simplified 3D shapes. (e) shows that by the same modification of the noise term in SDS as in (c), higher-quality 3D shapes are generated.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_7_2.jpg)

> This figure shows the ablation study of the proposed improvements on top of SDS. Starting from Dreamfusion with CFG 7.5, it incrementally adds higher NeRF rendering resolution (64 × 64 to 512 × 512), linear schedule on t, and DDIM inversion. The results clearly demonstrate that the main improvement in quality comes from the inferred noise.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_8_1.jpg)

> This figure demonstrates the difference in image generation quality between DDIM and SDS, both in 2D and 3D.  DDIM produces sharp, high-quality images, while SDS produces blurry images in 2D and over-smoothed, less detailed shapes in 3D. The authors propose a method to improve SDS by modifying its noise term to be consistent with DDIM, resulting in better image quality in 2D and significantly improved 3D shape generation.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_8_2.jpg)

> This figure shows an ablation study comparing different strategies for DDIM inversion in the context of Score Distillation via Inversion (SDI). The left panel shows the mean squared error (MSE) in equation (8) of the paper for each strategy at different noise levels (t). The right panel presents qualitative results obtained using those strategies, showing the generated images for a given prompt. This is useful for understanding the impact of different strategies for inferring noise during the denoising process. 


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_9_1.jpg)

> This figure compares the image generation quality of different methods.  It shows that DDIM produces high-quality images. However, using Score Distillation Sampling (SDS) with the same diffusion model leads to blurry images in 2D and over-smoothed, unrealistic shapes in 3D.  The authors' proposed method improves the quality, resulting in images closer to the quality of DDIM in 2D and significantly better quality in 3D. 


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_15_1.jpg)

> This figure compares the image generation quality of DDIM, SDS, and the proposed SDI method.  It showcases how SDS produces blurry 2D images and oversmoothed 3D shapes compared to DDIM.  The authors' SDI method aims to address these shortcomings by improving the noise approximation within SDS, resulting in better agreement with DDIM's quality for both 2D and 3D generation.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_16_1.jpg)

> This figure shows an ablation study comparing the results of different modifications applied to the Interval Score Matching (ISM) algorithm.  The main change is replacing the random noise sampling process in ISM with the DDIM inversion method proposed by the authors, resulting in improved 3D image generation quality. Other changes include a change of the range of the time variable, a linear annealing schedule for the time variable and the use of conditional inversion. Each variation is illustrated with images of generated hamburgers.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_16_2.jpg)

> This figure demonstrates the results of different methods for image and 3D shape generation.  (a) shows high-quality images generated by DDIM. (b) shows the blurry results when using Score Distillation Sampling (SDS) for 2D image generation. (d) shows the over-saturated and simplified 3D shapes produced by SDS.  Finally, (c) and (e) show the improved results achieved by the proposed method in 2D and 3D, respectively, demonstrating a closer match to the quality of the original DDIM model.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_17_1.jpg)

> This figure demonstrates the differences in image and 3D shape generation between DDIM, SDS, and the proposed SDI method.  DDIM produces high-quality images. SDS, when applied to 2D image generation, results in blurry images, and in 3D, produces over-smoothed and low-detail shapes. The authors' method (SDI) significantly improves the quality of 3D shape generation by modifying the noise term in the SDS algorithm to align with DDIM.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_17_2.jpg)

> This figure demonstrates the difference in image and 3D shape generation between DDIM and SDS.  DDIM produces high-quality images, whereas SDS leads to blurry 2D images and over-smoothed, unrealistic 3D shapes. The authors' proposed method, by modifying the noise term in SDS to match DDIM, significantly improves the quality of both 2D and 3D generation, closing the gap between the two methods.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_18_1.jpg)

> This figure shows a comparison of image generation results between DDIM and SDS, highlighting the over-smoothing effect of SDS in both 2D and 3D image generation. The authors' proposed method (ours) is shown to produce results closer in quality to the DDIM model in 2D and significantly improve the 3D generation quality compared to standard SDS.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_19_1.jpg)

> This figure visualizes the evolution of variables during Score Distillation and DDIM processes.  The left side shows how noisy images, 3D NeRF representations, and single-step denoised images change over time in both 2D and 3D generation. The right side illustrates how each DDIM step moves towards a denoised image, and how this can be represented as a change of variables.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_19_2.jpg)

> This figure demonstrates the difference between DDIM and SDS in 2D and 3D image/shape generation.  DDIM, a 2D diffusion model, produces high-quality images.  However, when used with SDS (Score Distillation Sampling) for 3D shape generation, the results are blurry and over-smoothed in 2D and over-saturated and simplified in 3D. The authors' proposed method improves the quality of 3D shape generation by modifying the noise term in SDS to match DDIM.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_20_1.jpg)

> This figure shows a comparison between DDIM, SDS, and the proposed SDI method for 2D and 3D image/shape generation.  It highlights how DDIM produces high-quality images, while SDS results in blurry 2D images and over-smoothed 3D shapes. The authors' method (SDI) aims to improve upon SDS by modifying its noise term to match DDIM, leading to better quality in both 2D and 3D.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_20_2.jpg)

> This figure demonstrates the differences in image and 3D shape generation between DDIM, SDS, and the proposed SDI method.  DDIM generates high-quality images, while SDS produces blurry 2D images and over-smoothed 3D shapes. The authors' method (SDI) improves upon SDS, generating higher-quality 2D images and more detailed 3D shapes, closer in quality to those produced by DDIM.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_20_3.jpg)

> This figure demonstrates the differences between DDIM, SDS, and the proposed method (SDI) in both 2D and 3D image generation.  It highlights how DDIM produces high-quality images, while SDS results in blurry 2D and over-smoothed 3D outputs.  The authors' SDI method aims to address these issues by modifying the noise term in SDS to align with DDIM, thus improving the quality of both 2D and 3D generated assets.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_21_1.jpg)

> This figure demonstrates the results of different methods for image and 3D shape generation using diffusion models.  It compares the output of DDIM (a high-quality image generator), standard Score Distillation Sampling (SDS) applied to DDIM (resulting in blurry 2D images and over-smoothed 3D shapes), and the proposed method (SDI) which significantly improves upon SDS, producing higher-quality 2D images and much more detailed 3D shapes.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_22_1.jpg)

> This figure compares the image generation quality between DDIM and SDS, both in 2D and 3D. It highlights that while DDIM generates high-quality images, SDS produces blurry images in 2D and over-smoothed shapes in 3D. The authors' proposed method improves the quality of SDS results, making them closer to that of DDIM.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_23_1.jpg)

> This figure compares the image generation quality between DDIM, SDS, and the proposed SDI method in both 2D and 3D settings. It highlights how DDIM produces high-quality images, while SDS results in blurry 2D images and over-smoothed 3D shapes. The authors' SDI method aims to improve the quality by modifying the noise term in SDS to match DDIM, resulting in better quality in both 2D and 3D.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_23_2.jpg)

> This figure compares the image generation quality of DDIM, SDS, and the proposed method (SDI).  It shows that DDIM produces high-quality 2D images, while SDS produces blurry 2D images and over-smoothed 3D shapes.  The proposed method, SDI, significantly improves the quality of 3D shape generation by better matching the noise term to DDIM.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_24_1.jpg)

> This figure demonstrates the difference in image and 3D shape generation quality between DDIM, SDS, and the proposed method (SDI).  It shows that DDIM produces high-quality 2D images, but SDS results in blurry 2D images and oversmoothed 3D shapes. In contrast, SDI significantly improves the quality of 3D generation by using a modified noise term.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_24_2.jpg)

> This figure compares the image generation quality of DDIM and SDS in both 2D and 3D settings.  It demonstrates that while DDIM generates high-quality images, SDS produces blurry 2D images and over-smoothed 3D shapes. The authors' proposed method improves the quality of SDS by modifying the noise term to match DDIM, resulting in significantly better 3D shape generation.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_24_3.jpg)

> This figure shows a comparison of image generation using different methods. (a) shows high-quality images generated by DDIM. (b) shows blurry images generated by SDS using DDIM as the base model for 2D generation. (d) shows over-smoothed and simplified 3D shapes also produced by SDS. (c) shows improved 2D image generation using the proposed method, which closely matches the quality of DDIM. (e) shows significantly improved 3D shape generation using the proposed method, which preserves higher-frequency detail and reduces over-smoothing.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_24_4.jpg)

> This figure compares the image generation quality between DDIM and SDS in both 2D and 3D.  It shows how DDIM produces high-quality images, whereas SDS results in blurry (2D) and over-smoothed (3D) outputs. The authors' proposed method, by modifying the noise term in SDS, improves the quality to match that of DDIM in 2D and significantly enhances 3D generation.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_24_5.jpg)

> This figure demonstrates the results of different methods for generating images and 3D shapes. It shows that the proposed method improves image quality and 3D shape generation significantly compared to the existing Score Distillation Sampling method. The images generated by the proposed method match the quality of those generated by the original DDIM model.  The figure visualizes the differences in 2D and 3D generation outcomes, highlighting the over-smoothing and over-saturation issues present in previous methods.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_24_6.jpg)

> This figure demonstrates the shortcomings of Score Distillation Sampling (SDS) in generating high-quality 2D and 3D images compared to DDIM.  The authors show how their proposed method improves the quality of SDS by changing the way noise is handled, resulting in clearer and more detailed images.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_25_1.jpg)

> This figure compares the image generation quality of DDIM, SDS, and the proposed method (SDI) in both 2D and 3D settings.  It highlights how DDIM produces high-quality images, while SDS results in blurry 2D images and over-smoothed 3D shapes.  The authors' SDI method addresses these issues, achieving better quality comparable to DDIM in 2D and significantly improved results in 3D.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_26_1.jpg)

> This figure compares the 3D model generation results of several different methods, including Dreamfusion, NFSD, ProlificDreamer, ISM, HiFA, and the authors' proposed method SDI.  Two different prompts were used to generate the 3D models: 'An ice cream sundae' and 'A 3D model of an adorable cottage with a thatched roof'. The figure visually demonstrates the differences in the quality and detail of the generated 3D models across the various approaches.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_26_2.jpg)

> This figure compares the 3D generation results of the proposed method (Ours) with several other state-of-the-art methods, including Dreamfusion, NFSD, ProlificDreamer, ISM, and HiFA.  Two different prompts were used to generate the 3D models: 'An ice cream sundae' and 'A 3D model of an adorable cottage with a thatched roof.' The figure visually demonstrates the relative quality and detail of the 3D models generated by each method.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_26_3.jpg)

> This figure shows a comparison of image generation quality between DDIM, SDS, and the proposed SDI method.  It demonstrates that while DDIM produces high-quality 2D images, SDS results in blurry 2D images and over-smoothed 3D shapes.  The authors' method, SDI, improves the quality of both 2D and 3D generations by modifying the noise term in the SDS algorithm to match DDIM.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_27_1.jpg)

> This figure compares the 3D model generation results of different methods, including Dreamfusion, NFSD, ProlificDreamer, ISM, HiFA, and the authors' proposed method (Ours). Two prompts were used for the comparison, namely, “An ice cream sundae” and “A 3D model of an adorable cottage with a thatched roof”.  The figure visually demonstrates the differences in the quality and detail of 3D models generated by each method. The authors' method aims to improve upon existing methods by generating higher-quality 3D models with improved details and textures.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_27_2.jpg)

> This figure shows a comparison of image generation results between DDIM and SDS, both in 2D and 3D. It highlights the limitations of SDS in producing blurry 2D images and over-smoothed 3D shapes compared to the high-quality results of DDIM. The authors propose a modification to SDS that addresses these issues, resulting in improved image quality closer to that of DDIM. The figure visually demonstrates the effectiveness of their proposed method.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_27_3.jpg)

> This figure shows a comparison between DDIM and SDS for 2D and 3D image generation.  DDIM produces high-quality images, while SDS results in blurry 2D images and over-smoothed, unrealistic 3D shapes. The authors' proposed method improves SDS results to be closer to DDIM's quality for both 2D and 3D generation.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_27_4.jpg)

> This figure shows a comparison of image generation using DDIM and SDS, highlighting the over-smoothing effect of SDS in both 2D and 3D. The authors' proposed method is shown to improve the quality of 3D shape generation significantly.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_27_5.jpg)

> This figure compares the image generation quality of DDIM, SDS, and the proposed SDI method in both 2D and 3D settings.  It showcases how DDIM produces high-quality images, while SDS results in blurry 2D images and over-smoothed 3D shapes. The authors' method (SDI) significantly improves upon the quality of SDS by better matching DDIM's results.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_27_6.jpg)

> This figure demonstrates the results of different image generation methods.  (a) shows high-quality 2D image generation using DDIM. (b) shows blurry 2D results using Score Distillation Sampling (SDS). (c) shows improved 2D results using the proposed method. (d) shows over-saturated and simplified 3D shapes using SDS. (e) shows improved 3D results using the proposed method. The proposed method aims to improve the quality of 3D shape generation by modifying the noise term in SDS to match DDIM.


![](https://ai-paper-reviewer.com/4DcpFagQ9e/figures_27_7.jpg)

> This figure compares the image generation quality of different methods.  (a) shows high-quality images generated by DDIM. (b) shows blurry images generated by SDS using DDIM as the base model in 2D. (c) shows that the proposed algorithm improves the quality of the 2D image generation, closer to that of DDIM. (d) shows that SDS produces over-saturated and simplified shapes in 3D. (e) shows that the proposed algorithm significantly improves 3D generation, addressing the over-smoothing and enhancing details.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/4DcpFagQ9e/tables_13_1.jpg)
> This table presents a quantitative comparison of the proposed method (SDI) against several state-of-the-art baselines for text-to-3D generation.  The evaluation metrics include CLIP Score (measuring alignment between generated images and text prompts), and CLIP IQA (assessing image quality, sharpness, and realism).  The table also provides additional metrics such as the percentage of failed generations, generation time, and VRAM usage for each method.

![](https://ai-paper-reviewer.com/4DcpFagQ9e/tables_16_1.jpg)
> This table quantitatively compares the performance of the proposed Score Distillation via Inversion (SDI) method against Interval Score Matching (ISM) using five key metrics: CLIP Score, CLIP IQA (quality, sharpness, real), and ImageReward.  The comparison is conducted with 5000 steps of each method.  Lower values are better for ImageReward and higher values are better for the remaining metrics. The results indicate similar performance between the two methods, but SDI shows slightly better results in certain metrics.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/4DcpFagQ9e/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}