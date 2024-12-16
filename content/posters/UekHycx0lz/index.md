---
title: "DreamSteerer: Enhancing Source Image Conditioned Editability using Personalized Diffusion Models"
summary: "DreamSteerer enhances source image-conditioned editability in personalized diffusion models via a novel Editability Driven Score Distillation objective and mode shifting regularization, achieving sign..."
categories: ["AI Generated", ]
tags: ["Computer Vision", "Image Generation", "🏢 Australian National University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} UekHycx0lz {{< /keyword >}}
{{< keyword icon="writer" >}} Zhengyang Yu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=UekHycx0lz" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/UekHycx0lz" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/papers/2410.11208" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=UekHycx0lz&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/UekHycx0lz/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current text-to-image (T2I) personalization methods struggle with precise image editing using personalized concepts, often resulting in unsatisfactory editability.  This is mainly due to the limited scope of reference images used during personalization, leading to poor adaptation to source image layouts and a lack of data diversity that causes the model to collapse to the patterns learned from limited reference images.  These issues necessitate higher level of editability than what personalized generation requires.

DreamSteerer tackles this challenge by introducing a novel plug-in method that enhances the source image-conditioned editability. **It employs Editability Driven Score Distillation (EDSD) and mode-shifting regularization to improve editing fidelity and avoid mode trapping**.  The method also includes modifications to the Delta Denoising Score framework that allow for high-fidelity editing. Extensive experiments demonstrate that DreamSteerer significantly improves editability across different T2I personalization baselines while remaining computationally efficient.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} DreamSteerer significantly improves source image-conditioned editability in personalized diffusion models. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Editability Driven Score Distillation (EDSD) and mode shifting regularization are key components for enhancing editing fidelity. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} DreamSteerer is a plug-and-play method compatible with various T2I personalization baselines, improving efficiency and providing better real-world image editing results. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in image editing and personalization.  It addresses the limitations of existing methods by significantly improving the editability of personalized diffusion models.  **The proposed DreamSteerer framework is versatile, compatible with different personalization baselines, and computationally efficient.** This opens exciting new avenues for high-fidelity image manipulation using personalized concepts, impacting various applications like creative content generation and digital art.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_1_1.jpg)

> 🔼 This figure shows several examples of image editing using different personalization methods (Textual Inversion, DreamBooth, and Custom Diffusion) and compares the results with and without the proposed DreamSteerer method.  Each row represents a different editing task, with the source image to be edited displayed next to reference images that provide the desired personal concept.  The results demonstrate that DreamSteerer leads to significantly improved editing fidelity, especially when the source and reference images have significant structural differences. The model is able to adapt to the source image while preserving the learned appearance of the personal concept.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>





![](https://ai-paper-reviewer.com/UekHycx0lz/tables_7_1.jpg)

> 🔼 This table compares the performance of DreamSteerer against three different personalization baselines (Textual Inversion, DreamBooth, and Custom Diffusion) across multiple metrics.  The metrics evaluate semantic similarity (using CLIP), perceptual similarity (using LPIPS), and structural similarity (using SSIM and MS-SSIM).  It also includes image quality assessment scores (Topiq, Musiq, LIQE). The results demonstrate DreamSteerer's improved performance over the baselines across all metrics.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison with different baselines (DreamSteerer uses the same model as baseline).
> </details>





### In-depth insights


#### Personalized Editability
The concept of "Personalized Editability" in the context of text-to-image diffusion models signifies a significant advancement in image manipulation. It goes beyond merely generating images from text prompts; it allows users to **edit existing images** using concepts they've personally taught the model. This personalization is crucial because it offers a level of **control and precision** that transcends the limitations of generic text prompts.  The challenge lies in effectively integrating personalization with editing frameworks, ensuring the edited image preserves the original's structure and background while seamlessly incorporating the new personalized elements.  **High-fidelity local editing**, a key goal, requires addressing mode trapping issues where edits become blurry or blend excessively with pre-existing features in the source image.  Success depends on innovative techniques, such as the Editability Driven Score Distillation (EDSD) approach introduced in the paper, which uses a novel loss function to improve the personalized model's ability to handle edits. Further enhancements, like spatial feature guided sampling, are critical to overcome mode trapping, ensuring that edits align both with user intent and the source image's structure.  The ultimate aim is to enhance the **flexibility and expressiveness** of image manipulation by creating a highly user-centric and adaptable image editing experience.

#### EDSD & Mode Shift
The proposed DreamSteerer method introduces **Editability Driven Score Distillation (EDSD)** to enhance source image-conditioned editability in personalized diffusion models.  EDSD directly optimizes the personalized model parameters to align the score estimations of the pre-trained and personalized models, leading to improved editability. However, a mode trapping issue is identified where the generated images fall between the reference images and the source image.  To address this, DreamSteerer incorporates a **mode-shifting regularization** using spatial feature guided sampling.  This technique leverages the spatial awareness of UNet attention features to guide the sampling process, shifting the model's distribution and preventing mode collapse. By combining EDSD and mode-shifting regularization, DreamSteerer achieves significant improvement in editing fidelity while maintaining the structural layout of the source image, effectively enhancing personalized editing capabilities.

#### Source Score Bias
Source score bias, within the context of personalized image editing using diffusion models, arises from **discrepancies in the score estimations** between the pre-trained and personalized models.  The personalized model, fine-tuned on a limited dataset of user-specified images, may develop a **biased representation of the source image category**. This bias manifests as a tendency to favor attributes or styles present in the reference images, even when these attributes are not present in the source image being edited. The result is a lack of editability where the generated image may exhibit distorted features or deviate from the desired editing direction.  **Techniques like source score correction** are crucial to mitigate this effect.  Correcting this bias often involves using the pre-trained model's score estimations to guide the personalized model, ensuring the edits remain true to the source image's original structure and overall appearance while accurately integrating personalized concepts.  The choice of score function and the approach to integration significantly impact the success of this correction, highlighting the importance of carefully addressing source score bias for effective personalized image manipulation.

#### Ablation Study Results
Ablation studies systematically remove components of a model to assess their individual contributions.  In the context of a research paper, the 'Ablation Study Results' section would present the performance metrics for each variant, showing how each removed component impacts the overall system.  **Key insights typically focus on the relative importance of different modules or techniques.** For instance, a significant drop in performance after removing a specific component would highlight its critical role. Conversely, minimal change indicates the component's lesser importance or possible redundancy.  **Careful analysis of these results guides design decisions**, highlighting essential parts to retain and areas where improvements can be made. The section would likely include tables or figures comparing performance, along with statistical significance tests.  **A well-executed ablation study strengthens the paper's claims**, providing clear evidence of each component's effectiveness.  The writing should emphasize the relative contribution of each part, and how the results provide a better understanding of model functioning.

#### Future Work
Future research could explore several promising avenues. **Extending DreamSteerer's capabilities to handle more complex edits**, such as those involving significant subject pose or viewpoint changes, would be a valuable improvement.  Investigating the impact of different personalization methods and their respective strengths and weaknesses in relation to DreamSteerer's performance could reveal important insights.  **Developing more sophisticated regularization techniques** to further mitigate mode collapse and enhance edit fidelity is also crucial. Additionally, the computational efficiency of DreamSteerer could be improved, especially for high-resolution images or complex edits, through optimizations and potentially the exploration of alternative architectures. Finally, **a more comprehensive evaluation on a larger and more diverse dataset** is necessary to solidify the claims and establish the method's generalizability.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/UekHycx0lz/figures_4_1.jpg)

> 🔼 This figure demonstrates the effectiveness of DreamSteerer in enhancing the editability of personalized diffusion models.  It shows several examples of image editing using different personalization methods (Textual Inversion, DreamBooth, and Custom Diffusion), where DreamSteerer consistently produces higher-fidelity results, particularly when the source and reference images have significant structural differences.  DreamSteerer adapts to the source image layout while preserving the appearance learned from the personalized concept.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_5_1.jpg)

> 🔼 This figure illustrates the overall framework of the DreamSteerer method. It shows how the method enhances source image conditioned editability by using an Editability Driven Score Distillation (EDSD) objective and a mode shifting regularization with spatial feature guided sampling. The EDSD objective aligns the score estimations of the pre-trained and personalized models, and the mode shifting regularization alleviates mode trapping issues.
> <details>
> <summary>read the caption</summary>
> Figure 3: Overall framework of DreamSteerer (the gradient flows are illustrated with dashed lines).
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_6_1.jpg)

> 🔼 This figure shows the impact of different regularization techniques on image editing and generation using a DreamBooth model.  The source image depicts a cat sitting next to a mirror, which is then edited using various methods.  The image illustrates the baseline results without any regularization and the improvements observed after applying mode shifting regularization and spatial feature guided sampling.  The goal is to demonstrate how these techniques enhance the fidelity and consistency of the edits, specifically ensuring the edited image maintains a similar structure and background to the original source image while incorporating the desired changes.
> <details>
> <summary>read the caption</summary>
> Figure 4: The effect of different regularization strategies on the editing and generation results of a DreamBooth baseline. The source prompt is 'a photo of a cat sitting next to a mirror'.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_7_1.jpg)

> 🔼 This figure showcases the results of DreamSteerer applied to several source images, comparing its performance to other methods (Textual Inversion, DreamBooth, and Custom Diffusion). It demonstrates that DreamSteerer effectively enhances the editability of source images conditioned on the learned concepts from personalized diffusion models, resulting in improved editing fidelity and natural adaptation to the source image layout even when significant structural differences exist between the source and reference images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_8_1.jpg)

> 🔼 This figure presents an ablation study to demonstrate the effectiveness of the proposed methods, EDSD and Mode Shifting Regularization.  The left half shows the results when EDSD is not used, and the right half demonstrates results without Mode Shifting Regularization. By comparing the results with and without each component, the individual contributions of EDSD and Mode Shifting to the overall performance of DreamSteerer are highlighted. This helps to understand their impact on editability, image fidelity, and structural preservation.
> <details>
> <summary>read the caption</summary>
> Figure 6: Ablation study on EDSD and Mode Shifting Regularization.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_9_1.jpg)

> 🔼 This figure shows several examples of image editing using different methods: Textual Inversion, DreamBooth, and a custom diffusion model. The 'Ours' column represents the results obtained using the proposed DreamSteerer method. The results demonstrate that DreamSteerer significantly improves the editing fidelity, especially when the source and reference images have significant structural differences.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_16_1.jpg)

> 🔼 This figure shows the results of using different Jacobian omitting strategies in the Editability Driven Score Distillation (EDSD) method. The results demonstrate that setting the Jacobian to -I leads to significantly better results with natural adaptation to the layout of the source image, while setting the Jacobian to I tends to destroy the structural layout and background of the source image. This indicates that setting the Jacobian to I maximizes the discrepancy between personalized and source model score estimations.
> <details>
> <summary>read the caption</summary>
> Figure 8: Results with different Jacobian omitting strategy.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_17_1.jpg)

> 🔼 This figure showcases the results of DreamSteerer, a novel method for improving the source image conditioned editability of personalized diffusion models.  It presents several examples comparing the results of DreamSteerer with existing textual inversion, DreamBooth, and custom diffusion models.  In each example, a source image and several reference images are provided; DreamSteerer successfully adapts the style and appearance from the reference images to the source image, often with better fidelity and naturalness than the baselines, even when there are significant structural differences between the source and reference images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_17_2.jpg)

> 🔼 This figure shows a comparison of image editing results using Custom-Edit and Custom-Edit enhanced with DreamSteerer.  It demonstrates DreamSteerer's ability to improve editing fidelity and natural adaptation to the source image layout, particularly when challenging structural differences exist between the source and reference images.  The results showcase how DreamSteerer enhances the quality and accuracy of the editing process, providing more realistic and aligned results compared to the base Custom-Edit method.
> <details>
> <summary>read the caption</summary>
> Figure 10: Comparison with Custom-Edit.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_18_1.jpg)

> 🔼 This figure showcases the results of DreamSteerer applied to different source images and personalized concepts.  Each row shows the reference images used for personalization, the original source image, and then the results of three different editing methods: Textual Inversion, DreamBooth, and a custom diffusion model.  DreamSteerer significantly improves the editing results by maintaining a high fidelity to the source image while incorporating the desired personalized concept. The results highlight DreamSteerer's ability to handle challenging scenarios where there are significant structural differences between the source and reference images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_18_2.jpg)

> 🔼 This figure shows an ablation study comparing different regularization strategies used in the DreamSteerer method for image editing with personalized diffusion models.  The baseline model is DreamBooth.  The source image is a photo of a cat next to a mirror. Four scenarios are displayed: (a) Baseline results without any mode shifting or additional regularization; (b) Results without mode shifting but with the EDSD method; (c) Results with mode shifting but without EDSD; (d) Results with both EDSD and mode shifting.  The figure demonstrates how each approach affects the fidelity and accuracy of the generated image in terms of incorporating the personalized concept and preserving the structure of the source image.
> <details>
> <summary>read the caption</summary>
> Figure 4: The effect of different regularization strategies on the editing and generation results of a DreamBooth baseline. The source prompt is 'a photo of a cat sitting next to a mirror'.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_19_1.jpg)

> 🔼 This figure shows the results of DreamSteerer applied to several source images using different personalization methods, including Textual Inversion, DreamBooth, and Custom Diffusion.  The results demonstrate that DreamSteerer successfully enhances the editability of these baselines, producing high-fidelity edits that closely match the reference images while preserving the structure and background of the source image, even in challenging scenarios with significant structural differences between the source and reference images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_20_1.jpg)

> 🔼 This figure shows the ablation study of the proposed method, DreamSteerer. The left part shows the comparison of the results with and without Editability Driven Score Distillation (EDSD). The right part shows the comparison of the results with and without mode shifting regularization.  Each row represents a different editing task, showcasing the source image, the reference image(s), and the results using various combinations of the proposed components.  The results highlight the individual and combined effects of EDSD and mode shifting on improving image editability.
> <details>
> <summary>read the caption</summary>
> Figure 6: Ablation study on EDSD and Mode Shifting Regularization.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_20_2.jpg)

> 🔼 This figure demonstrates the impact of different components of the DreamSteerer method on image editing using a DreamBooth baseline. The top row shows the edited images, comparing the results obtained with only SDS (score distillation sampling), DDS (delta denoising score) with and without source score bias correction, and DDS-S (modified DDS) with and without EDSD (editability driven score distillation) and mode shifting regularization. The bottom row displays the editing direction vectors, visualizing how much each pixel is changed in the process. Brown color indicates no changes in the pixel, suggesting the efficacy of DreamSteerer's approach to enhance source image conditioned editability.
> <details>
> <summary>read the caption</summary>
> Figure 5: Illustration on the effect of the proposed components on editing with a DreamBooth baseline (1st row shows the editing results; 2nd row shows the editing directions, where brown means zero).
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_20_3.jpg)

> 🔼 This figure shows several examples of image editing using DreamSteerer and compares it with other methods (Textual Inversion, DreamBooth, and Custom Diffusion).  Each row displays a reference image set used for personalization, the source image to be edited, and the results from different editing methods. The results demonstrate that DreamSteerer is better able to maintain the overall structure of the source image while incorporating the personalized concept, especially in cases with significant structural differences between the source and reference images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_21_1.jpg)

> 🔼 This figure demonstrates the capability of DreamSteerer to enhance the source image conditioned editability using existing personalized diffusion models.  It shows several examples of image editing where a personalized concept (e.g., a specific cat or dog) is applied to a source image.  Even when the source and reference images have significant structural differences, DreamSteerer successfully adapts, preserving the source image's structure while incorporating the personalized concept with high fidelity.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_22_1.jpg)

> 🔼 This figure shows several examples of image editing using DreamSteerer.  Each row presents the reference images used for personalization, the source image to be edited, and the results of editing with three different methods: Textual Inversion, DreamBooth, and a custom diffusion model.  The results demonstrate DreamSteerer's ability to successfully integrate personalized concepts while preserving the structure and background of the source image, even when there are significant structural differences between the source and reference images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_23_1.jpg)

> 🔼 This figure shows several examples of image editing using DreamSteerer, a method that enhances the editability of personalized diffusion models. Each row shows the reference images used for personalization, the source image to be edited, and the results of editing using Textual Inversion, DreamBooth, and Custom Diffusion baselines with and without DreamSteerer.  DreamSteerer demonstrates improved editing fidelity, especially when there are significant structural differences between the source and reference images. It allows for the adaptation of the edited image to the source while preserving the appearance of the personalized concept.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_24_1.jpg)

> 🔼 This figure showcases the results of DreamSteerer applied to several source images alongside the results of existing methods (Textual Inversion, DreamBooth, and Custom Diffusion).  For each row, a source image is edited using a reference image to incorporate a specific personalized concept (e.g., a cat statue, a brown cat). DreamSteerer significantly improves editing fidelity compared to existing methods, especially when source and reference images have significant structural differences. It adapts to the source image layout while incorporating the desired visual concept.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_24_2.jpg)

> 🔼 This figure shows an ablation study on the effect of different regularization strategies for image editing using a personalized diffusion model. Specifically, it compares the results of using a DreamBooth baseline with and without mode shifting regularization and spatial feature guided sampling.  The results demonstrate that incorporating mode shifting regularization and spatial feature guided sampling significantly improves the editability of the personalized diffusion model, enabling the generation of images that preserve the structural layout of the source image while successfully incorporating the learned personal concept. This highlights the importance of these regularization techniques in achieving high-fidelity editing results with personalized diffusion models.
> <details>
> <summary>read the caption</summary>
> Figure 4: The effect of different regularization strategies on the editing and generation results of a DreamBooth baseline. The source prompt is 'a photo of a cat sitting next to a mirror'.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_25_1.jpg)

> 🔼 This figure demonstrates the effectiveness of DreamSteerer in enhancing the editability of source images when using personalized diffusion models.  It showcases several editing examples where the source image is modified based on a reference image and a text prompt. The results highlight DreamSteerer's ability to maintain the appearance learned from the personalized concept while adapting to the structure of the source image, even when there are significant differences between the source and reference images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_26_1.jpg)

> 🔼 This figure showcases example results of DreamSteerer applied to various source images and personalized concepts.  Each row represents an edit, comparing results from Textual Inversion, DreamBooth and a Custom Diffusion model (baselines) to the results produced by the same models enhanced with DreamSteerer.  The results demonstrate how DreamSteerer improves the fidelity and naturalness of the edits, especially when there's a significant structural difference between the reference images and the image being edited. It shows that DreamSteerer can seamlessly integrate the learned appearance of the personalized concepts with the structure of the source images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_26_2.jpg)

> 🔼 This figure visualizes the averaged cross-attention maps from the UNet encoder and decoder of a diffusion probabilistic model at different resolutions. The visualization focuses on the 'astronaut' token within the prompt, 'an astronaut riding a horse'.  The maps highlight the areas of the image that the model focuses on when processing that specific token, demonstrating the model's attention mechanism at different resolution levels.
> <details>
> <summary>read the caption</summary>
> Figure 22: Visualization on averaged cross-attention maps of the DPM UNet encoder and decoder at different resolutions corresponding to 'astronaut' token in the prompt 'an astronaut riding a horse'.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_26_3.jpg)

> 🔼 This figure shows several examples of image editing using DreamSteerer and other methods (Textual Inversion, DreamBooth, Custom Diffusion).  For each example, the top row shows the reference images used for personalization. The second row displays the original image to be edited. The following columns display the results of editing that image with different methods.  The results demonstrate DreamSteerer's ability to maintain the appearance of personalized concepts while adapting to the source image structure.  This is particularly noticeable in challenging cases where source and reference images have significant differences in composition or structure.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_27_1.jpg)

> 🔼 This figure shows several examples of image editing using DreamSteerer.  Each row presents a different image editing task. The 'Reference' column displays the reference images used to personalize the diffusion model. The 'Source' column shows the original image to be edited. The remaining columns demonstrate the editing results obtained using different methods: Textual Inversion, DreamBooth, and a custom diffusion model. DreamSteerer consistently produces higher-fidelity edits, seamlessly integrating the personalized concept into the source image while preserving its overall structure and background. The results highlight DreamSteerer's ability to handle even significant structural differences between the source and reference images.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



![](https://ai-paper-reviewer.com/UekHycx0lz/figures_27_2.jpg)

> 🔼 This figure demonstrates the effectiveness of DreamSteerer in enhancing the editability of source images when using existing text-to-image (T2I) personalization methods.  It showcases several examples where DreamSteerer successfully integrates personalized concepts into source images, even when there are substantial structural differences between the source and reference images.  The results highlight DreamSteerer's ability to maintain the appearance of the personalized concept while adapting to the structure of the source image.
> <details>
> <summary>read the caption</summary>
> Figure 1: DreamSteerer enables efficient editability enhancement for a source image with any existing T2I personalization models, leading to significantly improved editing fidelity in various challenging scenarios. When the structural difference between source and reference images are significant, it can naturally adapt to the source while maintaining the appearance learned from the personal concept.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/UekHycx0lz/tables_8_1.jpg)
> 🔼 This table presents the results of an ablation study conducted to evaluate the impact of two key components of the DreamSteerer method: Editability Driven Score Distillation (EDSD) and Mode Shifting Regularization.  The study uses three different personalization baselines (Textual Inversion, DreamBooth, and Custom Diffusion). For each baseline, the table shows the performance metrics (CLIP-I, LPIPS, SSIM, MS-SSIM) under different configurations: with both EDSD and Mode Shifting, with only EDSD, with only Mode Shifting, and with neither. The best and second-best results for each metric and baseline are highlighted, demonstrating the individual and combined contributions of EDSD and Mode Shifting to improving the editability of personalized diffusion models.
> <details>
> <summary>read the caption</summary>
> Table 2: Ablation study on EDSD and Mode Shifting, the best and second best results are highlighted.
> </details>

![](https://ai-paper-reviewer.com/UekHycx0lz/tables_9_1.jpg)
> 🔼 This table compares the performance of DreamSteerer against three different personalization baselines: Textual Inversion, DreamBooth, and Custom Diffusion.  For each baseline, the table shows the results obtained using the baseline model alone and then again using DreamSteerer as a plugin. The metrics used for comparison are CLIP-I (semantic similarity), LPIPS (perceptual similarity), SSIM (structural similarity), and MS-SSIM (multi-scale structural similarity).  The table highlights that DreamSteerer significantly improves the results across all three baselines and all metrics, demonstrating its effectiveness in enhancing source image conditioned editability.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison with different baselines (DreamSteerer uses the same model as baseline).
> </details>

![](https://ai-paper-reviewer.com/UekHycx0lz/tables_18_1.jpg)
> 🔼 This table compares the performance of DreamSteerer against the Custom Edit baseline.  The metrics used are CLIP similarity (CLIP B/32 and CLIP L/14), LPIPS (AlexNet and VGG), SSIM, MS-SSIM, and three image quality assessment (IQA) metrics (Topiq, Musiq, LIQE).  It shows that DreamSteerer achieves comparable performance in CLIP scores, lower LPIPS scores (indicating better perceptual similarity), and similar SSIM and MS-SSIM scores (indicating similar structural similarity). The slight differences in IQA metrics might be due to minor variations in image quality that are not captured by the other metrics.
> <details>
> <summary>read the caption</summary>
> Table 4: Comparison with Custom-Edit.
> </details>

![](https://ai-paper-reviewer.com/UekHycx0lz/tables_27_1.jpg)
> 🔼 This table compares the performance of DreamSteerer against three different personalization baselines: Textual Inversion, DreamBooth, and Custom Diffusion.  For each baseline, it shows the results using several metrics: CLIP-I (higher is better for semantic similarity), LPIPS (lower is better for perceptual similarity), SSIM, MS-SSIM (both higher is better for structural similarity), and three image quality assessment metrics (Topiq, Musiq, and LIQE; higher is better).  DreamSteerer consistently outperforms the baselines across all metrics, demonstrating its effectiveness in enhancing the source image conditioned editability.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison with different baselines (DreamSteerer uses the same model as baseline).
> </details>

![](https://ai-paper-reviewer.com/UekHycx0lz/tables_28_1.jpg)
> 🔼 This table compares the performance of DreamSteerer against three different personalization baselines: Textual Inversion, DreamBooth, and Custom Diffusion.  The results are evaluated using various metrics including CLIP-I (higher is better, measuring semantic similarity), LPIPS (lower is better, measuring perceptual similarity), SSIM, and MS-SSIM (higher is better, measuring structural similarity), along with three image quality assessment scores: AlexNet, VGG and IQA (Topiq, Musiq, LIQE).  DreamSteerer consistently outperforms the baselines across all metrics, showcasing significant improvement in both perceptual and structural alignment with the source image, particularly in challenging editing scenarios.  This highlights the effectiveness of the proposed method in enhancing the editability of personalized diffusion models.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison with different baselines (DreamSteerer uses the same model as baseline).
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/UekHycx0lz/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UekHycx0lz/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}