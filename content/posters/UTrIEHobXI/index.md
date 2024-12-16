---
title: "Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images"
summary: "Geometry Cloak embeds invisible perturbations in images to thwart AI-based 3D reconstruction, forcing the AI to generate identifiable patterns that act as watermarks to assert copyright."
categories: ["AI Generated", ]
tags: ["Computer Vision", "3D Vision", "🏢 Hong Kong Baptist University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} UTrIEHobXI {{< /keyword >}}
{{< keyword icon="writer" >}} Qi Song et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=UTrIEHobXI" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/UTrIEHobXI" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=UTrIEHobXI&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/UTrIEHobXI/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

The rise of AI-powered single-image 3D reconstruction methods poses a significant threat to copyright protection. Malicious users can easily generate unauthorized 3D models from copyrighted images, causing substantial financial and reputational damage to copyright holders. Existing methods like digital watermarking are not effective in preventing the creation of unauthorized 3D models from single images. This paper proposes a novel image protection approach called "Geometry Cloak." It addresses the vulnerabilities of AI-based 3D reconstruction by embedding carefully crafted perturbations directly into the images before they are fed to the AI model. These perturbations, which are imperceptible to the human eye, act as a "cloak" that forces the AI to fail 3D reconstruction in a specific manner, revealing a customized watermark that acts as evidence of ownership.  Unlike conventional adversarial attacks that aim to merely degrade output quality, Geometry Cloak introduces a controlled failure mode. This enables copyright holders to easily verify ownership over any attempted 3D reconstructions.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel 'geometry cloak' method effectively prevents unauthorized 3D model generation from copyrighted images using AI. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method embeds invisible perturbations that reveal an identifiable pattern when the AI attempts 3D reconstruction. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Extensive experiments validate the approach's effectiveness, demonstrating the method's ability to safeguard digital assets. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it introduces a novel approach to protect copyrighted images from unauthorized 3D reconstruction using AI.  **It addresses a critical issue in the field of digital asset protection**, providing a much-needed solution for safeguarding visual content. The proposed method's efficacy could greatly impact various industries, **setting a new standard for copyright protection in a rapidly evolving technological landscape**.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_1_1.jpg)

> 🔼 This figure illustrates three scenarios for 3D model reconstruction from images. (a) shows the unprotected image easily reconstructed into 3D models by malicious users with TGS. (b) shows the Digital Watermarking method which embeds messages into the image, but the message cannot be retrieved from novel views. (c) introduces the proposed Geometry Cloak method which embeds invisible perturbations into images, forcing TGS to produce an identifiable pattern (watermark) upon reconstruction, allowing copyright holders to claim ownership.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of our scenario. (a) Images without protection. Images can be easily reconstructed into 3D models by malicious users with TGS [52], posing a threat to the copyright of the image owner. (b) Digital Watermarking offers a solution by embedding copyright messages into the view-image before 3D reconstruction. However, the embedded message cannot be extracted from novel rendered views. (c) Geometry Cloak. Our geometry cloak utilizes the disturbance-prone components of TGS, achieving view-specific watermark embedding. Our method can compromise the unauthorized reconstructed 3D model while providing a verifiable pattern for copyright claim.
> </details>





![](https://ai-paper-reviewer.com/UTrIEHobXI/tables_7_1.jpg)

> 🔼 This table presents a quantitative comparison of four different perturbation strategies against a non-perturbed baseline.  The strategies are Gaussian noise, adversarial image perturbation, geometry cloaking without a target pattern, and the proposed geometry cloaking method.  The comparison is made using image quality metrics (PSNR, SSIM, LPIPS) and a geometry quality metric (Chamfer Distance). Results are shown for three different perturbation budgets (epsilon = 2, 4, and 8) across two datasets (Omni3D and GSO).
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative comparison of perturbation strategies. We present the outcomes of four distinct perturbation strategies compared to the non-perturbed results. These strategies include Gaussian noise (random Gaussian noise), Adversarial image (perturbing image features), geometry cloak without target, and geometry cloak. We evaluate the image quality metrics (PSNR/SSIM/LPIPS) and geometry quality metric (Chamfer Distance, CD) on the Omni3D [45] and GSO [7] datasets at perturbation budgets of e = 2, 4, 8.
> </details>





### In-depth insights


#### Geometry Cloak
The concept of "Geometry Cloak" presents a novel approach to safeguarding copyrighted images from unauthorized 3D reconstruction using techniques like Triplane Gaussian Splatting (TGS).  It cleverly leverages the vulnerabilities within TGS by embedding **invisible geometry perturbations**, or a "cloak," directly into the image. These perturbations aren't random; they're carefully designed to force TGS, upon attempting 3D reconstruction, to generate a specific, identifiable pattern acting as a watermark. This watermark serves as irrefutable proof of ownership, allowing copyright holders to easily assert their rights. Unlike typical adversarial attacks that merely degrade output quality, this method ensures a controlled failure of the 3D reconstruction process, making it a far more robust and reliable copyright protection strategy. The effectiveness of the geometry cloak is experimentally verified in the paper, demonstrating its ability to disrupt the reconstruction process while remaining imperceptible to the human eye.  **View-specific Projected Gradient Descent** is utilized to optimize the embedded perturbations to enhance the visibility of the watermark in specific views of the 3D model. The research highlights the **uniqueness** of targeting the geometry features within TGS, as opposed to simpler image-based perturbations, for improved robustness against attacks.  This makes "Geometry Cloak" a promising solution for protecting 3D assets derived from copyrighted 2D images in the age of advanced single-view 3D reconstruction.

#### TGS Vulnerabilities
Triplane Gaussian Splatting (TGS) shows vulnerability to **adversarial attacks**, particularly those manipulating the **geometry features** rather than image features.  **Directly perturbing the point cloud** within the TGS framework proves more effective than manipulating image features, as the point cloud is crucial for 3D reconstruction and less robust to alterations. The paper proposes a **geometry cloaking** technique which leverages this vulnerability. By introducing carefully crafted perturbations in the geometry domain, TGS is forced to reconstruct a 3D model containing an identifiable pattern or watermark, thereby enabling copyright protection. This method's **view-specific PGD** optimization ensures the embedded message is revealed only under specific viewpoints, strengthening its effectiveness against unauthorized 3D model generation.  The success relies on understanding TGS's internal workings and targeting its weaknesses for specific and verifiable outcomes, highlighting the importance of considering both image and geometry components for robust protection against TGS exploitation.

#### PGD Optimization
**Projected Gradient Descent (PGD)** optimization, in the context of adversarial attacks and image cloaking, is a powerful iterative method to subtly alter input images (introducing a 'geometry cloak') to manipulate the output of a target model, like Triplane Gaussian Splatting (TGS).  The core idea is to minimize the distance between the model's output (a 3D reconstruction) and a predefined 'target pattern' by iteratively updating the image perturbations. This process cleverly exploits the vulnerability of the TGS model to specific types of perturbations by making the reconstructed 3D model reveal a watermark. **The 'view-specific' nature of the PGD** addresses the robustness of the TGS model to general disturbances. This optimization is crucial because simple adversarial attacks often fail to produce consistently identifiable patterns; PGD helps guide the perturbation to create the desired, verifiable watermark that only appears in specific views, making copyright infringement easily detectable. **The choice of an appropriate distance metric** (e.g., Chamfer Distance) is important for effectively measuring the similarity between the perturbed reconstruction and the target pattern.  The success of the method relies on the delicate balance between the strength of the perturbation (to reliably induce the target pattern) and the imperceptibility of the alterations to the original image.  **Careful consideration of hyperparameters** such as the perturbation budget (epsilon) and learning rate is necessary to achieve optimal results.

#### 3D Model Protection
3D model protection is a crucial area of research, given the ease with which high-quality 3D models can be generated from images using techniques like Triplane Gaussian Splatting (TGS).  The paper highlights the vulnerability of copyrighted images to unauthorized 3D reconstruction and proposes a novel solution: **geometry cloaking**. This technique embeds invisible perturbations into images before they're processed by TGS, forcing the algorithm to generate 3D models with identifiable patterns—essentially watermarks—that can be used to verify ownership. Unlike previous methods focused on degrading output quality, geometry cloaking actively guides the reconstruction process towards a specific, verifiable result. **View-specific Projected Gradient Descent (PGD)** is employed to iteratively optimize these perturbations.  The effectiveness of this approach is demonstrated through experiments, showcasing the ability to induce specific, identifiable patterns while maintaining the image's visual fidelity.  The approach targets the inherent vulnerabilities of point cloud representation within TGS, making it more effective than simply perturbing image features. The overall impact emphasizes the need for proactive measures to secure intellectual property rights in the context of increasingly sophisticated single-image 3D reconstruction methods.

#### Future Directions
Future research could explore **more sophisticated geometry cloaking techniques**, potentially leveraging advancements in generative models or adversarial machine learning to create even more robust and undetectable perturbations.  Investigating **different types of watermarking**, beyond simple geometric patterns,  could enhance the method's resilience against tampering or removal. For example, integrating robust, view-invariant watermarks could enhance reliability.  Furthermore, expanding the work to encompass other single-view 3D reconstruction methods beyond TGS, and assessing the effectiveness of geometry cloaking in various domains including video and multi-view scenarios, would significantly broaden its applicability.  **Evaluating the impact of different compression and editing techniques** on the cloaked images is important to understand the practical limitations of the approach and to determine ways to improve its robustness. Finally, thoroughly addressing the ethical and legal considerations surrounding image protection techniques and their potential for misuse is critical.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_4_1.jpg)

> 🔼 This figure illustrates the proposed geometry cloak method. It shows how the method works by embedding invisible perturbations into input images for TGS, inducing the 3D reconstruction process to fail in a specific way. The core TGS representation is shown with its explicit point cloud and implicit triplane feature field. The target patterns are shown and how view-specific PGD iteratively optimizes the reconstructed point cloud to achieve the desired characteristics. 
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall of our proposed method. We propose to induce the 3D reconstruction process with our geometry cloak. (a) The core representation of TGS [52] includes an explicit point cloud and an implicit triplane-based feature field. The features of the novel view image are extracted through the coordinates in the point cloud. (b) The target patterns (Section 4.1) are designed to induce the final reconstruction result. (c) In order to make the reconstruction result show some distinguishable characteristics, we use projected gradient descent (PGD) [28] to iteratively optimize the reconstructed point cloud so that it has consistent characteristics with the target point cloud (Section 4.2).
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_5_1.jpg)

> 🔼 This figure illustrates two methods for creating target geometry patterns used in the Geometry Cloak technique.  Pre-defined patterns are simple: alphanumeric characters are converted directly into 2D point clouds.  Customized patterns offer more flexibility; users extract the point cloud from an image, then modify it (using tools like instructP2P or Meshlab) to create a custom pattern.
> <details>
> <summary>read the caption</summary>
> Figure 3: Two different target geometry patterns. (1) Pre-defined patterns: we directly convert alphanumeric characters into a 2D point cloud as watermarks. (2) Customized patterns: In E1, we first extract the point cloud of the image that needs to be protected. In E2, we edit the acquired point cloud through text-guided methods like instructP2P [48] or open-source software meshlab [5].
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_5_2.jpg)

> 🔼 This figure illustrates the proposed Geometry Cloak method's workflow.  It breaks down the process into three phases: (a) Core TGS representation showing the point cloud and triplane features; (b) The addition of a geometry cloak to the original image, targeting a specific pattern for the reconstruction; (c) Use of Projected Gradient Descent (PGD) to iteratively optimize the geometry cloak, ensuring the reconstructed point cloud matches the desired pattern. The overall goal is to force a specific, identifiable pattern in any unauthorized 3D reconstruction.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall of our proposed method. We propose to induce the 3D reconstruction process with our geometry cloak. (a) The core representation of TGS [52] includes an explicit point cloud and an implicit triplane-based feature field. The features of the novel view image are extracted through the coordinates in the point cloud. (b) The target patterns (Section 4.1) are designed to induce the final reconstruction result. (c) In order to make the reconstruction result show some distinguishable characteristics, we use projected gradient descent (PGD) [28] to iteratively optimize the reconstructed point cloud so that it has consistent characteristics with the target point cloud (Section 4.2).
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_7_1.jpg)

> 🔼 This figure shows a comparison of 3D reconstruction results using different perturbation methods.  The 'Input View' shows the original image.  The 'Reconstructed' row displays the results after applying various perturbations: no perturbation, Gaussian noise, adversarial image attacks, and the proposed geometry cloaking methods (with and without a target pattern).  The results illustrate the impact of each method on the 3D reconstruction, particularly highlighting the effectiveness of geometry cloaking in disrupting the reconstruction process due to its effect on perturbation-prone geometry features.
> <details>
> <summary>read the caption</summary>
> Figure 5: Qualitative reconstructed results with different perturbing strategies. Compare to Gauss. noise and Adv. image, our method can significantly affect the reconstructed results, indicating the explicit geometry features are perturbation-prone during 3D reconstruction.
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_8_1.jpg)

> 🔼 This figure illustrates two different ways to create target geometry patterns for the Geometry Cloak method.  Pre-defined patterns use directly converted alphanumeric characters into 2D point clouds as watermarks. Customized patterns allow users more control, letting them extract and then modify the point cloud from images they want to protect, using either text-guided methods or software like Meshlab.
> <details>
> <summary>read the caption</summary>
> Figure 3: Two different target geometry patterns. (1) Pre-defined patterns: we directly convert alphanumeric characters into a 2D point cloud as watermarks. (2) Customized patterns: In E1, we first extract the point cloud of the image that needs to be protected. In E2, we edit the acquired point cloud through text-guided methods like instructP2P [48] or open-source software meshlab [5].
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_13_1.jpg)

> 🔼 This figure illustrates the proposed geometry cloak method.  It shows the core components of Triplane Gaussian Splatting (TGS), how a geometry cloak is embedded, and how projected gradient descent is used to iteratively optimize the reconstructed point cloud to match a target pattern for watermarking.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall of our proposed method. We propose to induce the 3D reconstruction process with our geometry cloak. (a) The core representation of TGS [52] includes an explicit point cloud and an implicit triplane-based feature field. The features of the novel view image are extracted through the coordinates in the point cloud. (b) The target patterns (Section 4.1) are designed to induce the final reconstruction result. (c) In order to make the reconstruction result show some distinguishable characteristics, we use projected gradient descent (PGD) [28] to iteratively optimize the reconstructed point cloud so that it has consistent characteristics with the target point cloud (Section 4.2).
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_14_1.jpg)

> 🔼 This figure illustrates the proposed geometry cloaking method. It shows the core components of Triplane Gaussian Splatting (TGS), the process of embedding a geometry cloak, and the use of Projected Gradient Descent (PGD) to optimize the cloak for a specific outcome.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall of our proposed method. We propose to induce the 3D reconstruction process with our geometry cloak. (a) The core representation of TGS [52] includes an explicit point cloud and an implicit triplane-based feature field. The features of the novel view image are extracted through the coordinates in the point cloud. (b) The target patterns (Section 4.1) are designed to induce the final reconstruction result. (c) In order to make the reconstruction result show some distinguishable characteristics, we use projected gradient descent (PGD) [28] to iteratively optimize the reconstructed point cloud so that it has consistent characteristics with the target point cloud (Section 4.2).
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_15_1.jpg)

> 🔼 This figure showcases two methods for creating target geometry patterns to be embedded in images as geometry cloaks.  Pre-defined patterns directly convert alphanumeric characters into 2D point clouds, acting as simple watermarks. Customized patterns allow for more nuanced control, beginning with extracting the point cloud of an image to be protected, and then refining this using tools like instructP2P or MeshLab to customize the point cloud, potentially embedding more context-specific information into the pattern.
> <details>
> <summary>read the caption</summary>
> Figure 3: Two different target geometry patterns. (1) Pre-defined patterns: we directly convert alphanumeric characters into a 2D point cloud as watermarks. (2) Customized patterns: In E1, we first extract the point cloud of the image that needs to be protected. In E2, we edit the acquired point cloud through text-guided methods like instructP2P [48] or open-source software meshlab [5].
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_15_2.jpg)

> 🔼 This figure shows qualitative results of using two different types of target geometry patterns in the Geometry Cloak method.  The first uses predefined alphanumeric characters as watermarks, which are visible from specific viewpoints in the reconstructed 3D model. The second uses customized patterns where users can select parts of the image to protect; only those selected parts will fail to reconstruct properly in the 3D model.
> <details>
> <summary>read the caption</summary>
> Figure 6: Qualitative results of two different target geometry patterns. (a) Pre-defined patterns: The letters “A” and “X” are used as watermark messages. The embedded watermark can be effectively observed from a certain perspective. (b) Customized patterns: Users can selectively control the parts that need protection, causing the 3D reconstruction of corresponding parts to fail. More qualitative experimental results are provided in the Appendix.
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_16_1.jpg)

> 🔼 This figure compares the 3D reconstruction results from single images using different methods: a baseline (no perturbation), random Gaussian noise, adversarial attacks on image features, and the proposed geometry cloak.  The geometry cloak significantly alters the reconstructed results by targeting the point cloud's geometry, unlike the other methods that produce relatively minor changes. This highlights the vulnerability of the point cloud in the reconstruction process and demonstrates the effectiveness of the geometry cloak in disrupting unauthorized 3D model generation.
> <details>
> <summary>read the caption</summary>
> Figure 5: Qualitative reconstructed results with different perturbing strategies. Compare to Gauss. noise and Adv. image, our method can significantly affect the reconstructed results, indicating the explicit geometry features are perturbation-prone during 3D reconstruction.
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_16_2.jpg)

> 🔼 This figure illustrates the proposed Geometry Cloak method.  Panel (a) shows the core components of Triplane Gaussian Splatting (TGS): an explicit point cloud and an implicit triplane feature field. Panel (b) demonstrates the pre-defined patterns used to induce a specific failure mode in the TGS reconstruction. Panel (c) details the iterative optimization process using projected gradient descent (PGD) to embed the target patterns into the reconstructed point cloud, making the resulting 3D model reveal an identifiable pattern.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall of our proposed method. We propose to induce the 3D reconstruction process with our geometry cloak. (a) The core representation of TGS [52] includes an explicit point cloud and an implicit triplane-based feature field. The features of the novel view image are extracted through the coordinates in the point cloud. (b) The target patterns (Section 4.1) are designed to induce the final reconstruction result. (c) In order to make the reconstruction result show some distinguishable characteristics, we use projected gradient descent (PGD) [28] to iteratively optimize the reconstructed point cloud so that it has consistent characteristics with the target point cloud (Section 4.2).
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_16_3.jpg)

> 🔼 This figure compares the results of 3D reconstruction using different perturbation strategies: Gaussian noise, adversarial attack on image features, geometry cloak without target, and the proposed geometry cloak.  The results demonstrate that the proposed method is significantly more effective at disrupting the 3D reconstruction process, highlighting the vulnerability of geometry features in TGS to adversarial attacks.
> <details>
> <summary>read the caption</summary>
> Figure 5: Qualitative reconstructed results with different perturbing strategies. Compare to Gauss. noise and Adv. image, our method can significantly affect the reconstructed results, indicating the explicit geometry features are perturbation-prone during 3D reconstruction.
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_16_4.jpg)

> 🔼 This figure illustrates the overall proposed method. It shows three phases: (a) The core representation of the Triplane Gaussian Splatting (TGS) model, which consists of an explicit point cloud and an implicit triplane-based feature field. (b) The protection phase where a geometry cloak is embedded into the image, aiming to induce a specific pattern in the 3D reconstruction result. (c) The unauthorized reconstruction phase, in which malicious users attempt to reconstruct the 3D model from the cloaked image, revealing the embedded pattern.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall of our proposed method. We propose to induce the 3D reconstruction process with our geometry cloak. (a) The core representation of TGS [52] includes an explicit point cloud and an implicit triplane-based feature field. The features of the novel view image are extracted through the coordinates in the point cloud. (b) The target patterns (Section 4.1) are designed to induce the final reconstruction result. (c) In order to make the reconstruction result show some distinguishable characteristics, we use projected gradient descent (PGD) [28] to iteratively optimize the reconstructed point cloud so that it has consistent characteristics with the target point cloud (Section 4.2).
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_17_1.jpg)

> 🔼 This figure illustrates the proposed Geometry Cloak method. It shows the core components of Triplane Gaussian Splatting (TGS), the process of embedding the geometry cloak, and the use of view-specific Projected Gradient Descent (PGD) to optimize the reconstructed point cloud to reveal a specific pattern. The process involves generating a cloaked image that forces TGS to produce a 3D model revealing a pre-defined watermark.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall of our proposed method. We propose to induce the 3D reconstruction process with our geometry cloak. (a) The core representation of TGS [52] includes an explicit point cloud and an implicit triplane-based feature field. The features of the novel view image are extracted through the coordinates in the point cloud. (b) The target patterns (Section 4.1) are designed to induce the final reconstruction result. (c) In order to make the reconstruction result show some distinguishable characteristics, we use projected gradient descent (PGD) [28] to iteratively optimize the reconstructed point cloud so that it has consistent characteristics with the target point cloud (Section 4.2).
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_17_2.jpg)

> 🔼 This figure illustrates the proposed Geometry Cloak method.  It shows the core components of Triplane Gaussian Splatting (TGS), the embedding of a geometry cloak (invisible perturbation) into an image before it's processed by TGS, and the use of Projected Gradient Descent (PGD) to optimize the cloak for a specific, identifiable outcome in the 3D reconstruction.  The goal is for the 3D model generated from the cloaked image to reveal a custom watermark, proving unauthorized reconstruction.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall of our proposed method. We propose to induce the 3D reconstruction process with our geometry cloak. (a) The core representation of TGS [52] includes an explicit point cloud and an implicit triplane-based feature field. The features of the novel view image are extracted through the coordinates in the point cloud. (b) The target patterns (Section 4.1) are designed to induce the final reconstruction result. (c) In order to make the reconstruction result show some distinguishable characteristics, we use projected gradient descent (PGD) [28] to iteratively optimize the reconstructed point cloud so that it has consistent characteristics with the target point cloud (Section 4.2).
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_17_3.jpg)

> 🔼 This figure shows qualitative results of using two different types of target geometry patterns for the geometry cloaking method. The first example uses pre-defined patterns (alphanumeric characters) as watermarks.  The watermark is visible from certain viewpoints of the 3D reconstruction. The second example utilizes customized patterns, where the user can choose which parts of the object's point cloud to modify. This allows for more targeted protection, resulting in the failure of 3D reconstruction of only the specified parts.  The appendix contains additional qualitative results.
> <details>
> <summary>read the caption</summary>
> Figure 6: Qualitative results of two different target geometry patterns. (a) Pre-defined patterns: The letters 'A' and 'X' are used as watermark messages. The embedded watermark can be effectively observed from a certain perspective. (b) Customized patterns: Users can selectively control the parts that need protection, causing the 3D reconstruction of corresponding parts to fail. More qualitative experimental results are provided in the Appendix.
> </details>



![](https://ai-paper-reviewer.com/UTrIEHobXI/figures_17_4.jpg)

> 🔼 This figure shows qualitative results of applying the geometry cloak method using two different types of target geometry patterns: pre-defined and customized. The pre-defined patterns use alphanumeric characters as watermarks, which are visible from specific viewpoints after 3D reconstruction using TGS.  The customized patterns allow users to select which parts of an image to protect, leading to failures in reconstructing those selected parts in the 3D model.  The Appendix contains additional qualitative results.
> <details>
> <summary>read the caption</summary>
> Figure 6: Qualitative results of two different target geometry patterns. (a) Pre-defined patterns: The letters “A” and “X” are used as watermark messages. The embedded watermark can be effectively observed from a certain perspective. (b) Customized patterns: Users can selectively control the parts that need protection, causing the 3D reconstruction of corresponding parts to fail. More qualitative experimental results are provided in the Appendix.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/UTrIEHobXI/tables_13_1.jpg)
> 🔼 This table compares four different perturbation strategies against a non-perturbed baseline for reconstructing 3D models from images.  The strategies are Gaussian noise, adversarial attacks on image features, a geometry cloak without a target pattern, and the proposed geometry cloak method.  The table shows the impact of each perturbation on image quality metrics (PSNR, SSIM, LPIPS) and the Chamfer Distance (CD) for 3D geometry quality across two datasets (Omni3D and GSO) at various perturbation budget levels.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative comparison of perturbation strategies We present the outcomes of four distinct perturbation strategies compared to the non-perturbed results. These strategies include Gaussian noise (random Gaussian noise), Adversarial image (perturbing image features), geometry cloak without target, and geometry cloak. We evaluate the image quality metrics (PSNR/SSIM/LPIPS) and geometry quality metric (Chamfer Distance, CD) on the Omni3D [45] and GSO [7] datasets at perturbation budgets of e = 2, 4, 8.
> </details>

![](https://ai-paper-reviewer.com/UTrIEHobXI/tables_14_1.jpg)
> 🔼 This table compares four different perturbation strategies against a non-perturbed baseline. The strategies are Gaussian noise, adversarial image perturbations, geometry cloaking without a target pattern, and geometry cloaking with a target pattern.  The table evaluates the impact of each perturbation strategy on both image quality (using PSNR, SSIM, and LPIPS) and 3D geometry quality (using Chamfer Distance). The results are shown for three different perturbation budgets (epsilon = 2, 4, and 8) across two datasets (Omni3D and GSO).
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative comparison of perturbation strategies. We present the outcomes of four distinct perturbation strategies compared to the non-perturbed results. These strategies include Gaussian noise (random Gaussian noise), Adversarial image (perturbing image features), geometry cloak without target, and geometry cloak. We evaluate the image quality metrics (PSNR/SSIM/LPIPS) and geometry quality metric (Chamfer Distance, CD) on the Omni3D [45] and GSO [7] datasets at perturbation budgets of  = 2, 4, 8.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/UTrIEHobXI/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}