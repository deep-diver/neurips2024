---
title: "ReFIR: Grounding Large Restoration Models with Retrieval Augmentation"
summary: "ReFIR enhances Large Restoration Models' accuracy by incorporating retrieved images as external knowledge, mitigating hallucination without retraining."
categories: []
tags: ["Computer Vision", "Image Generation", "🏢 Tsinghua University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} iFKmFUxQDh {{< /keyword >}}
{{< keyword icon="writer" >}} Hang Guo et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=iFKmFUxQDh" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94016" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=iFKmFUxQDh&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/iFKmFUxQDh/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Large Restoration Models (LRMs) have significantly improved image restoration. However, they often suffer from "hallucination," generating incorrect details, particularly with severe degradation. This is due to LRMs' heavy reliance on limited internal knowledge. Existing solutions, such as increasing model size, can be computationally expensive. 



This paper introduces ReFIR, a novel retrieval-augmented framework. ReFIR leverages retrieved high-quality images as external knowledge to enhance LRMs.  It uses a nearest-neighbor lookup to find relevant reference images and injects their textures into the LRM's detail restoration stage. **ReFIR is training-free and adaptable to different LRMs, effectively mitigating hallucination and significantly improving both fidelity and realism of restoration results.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} ReFIR improves the fidelity of large restoration models by using retrieved images as additional external knowledge. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} ReFIR is a training-free method applicable to various large restoration models, addressing the hallucination challenge. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The study offers insights into the internal mechanisms of LRMs, paving the way for advancements in image restoration. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it introduces a novel and effective approach to improve the fidelity of large restoration models (LRMs) without requiring additional training.  **This addresses a significant challenge in image restoration** where LRMs often produce unrealistic details (hallucination), especially with severely degraded inputs. The proposed method, ReFIR, is adaptable to various LRMs, making it widely applicable. This work also offers insights into the internal mechanisms of LRMs and opens up new avenues for future research in improving LRM performance and addressing the hallucination issue.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_1_1.jpg)

> This figure shows a comparison of image restoration results between existing Large Restoration Models (LRMs) and the proposed ReFIR method.  The top row shows an example where the existing LRMs (SeeSR and SUPIR) fail to accurately restore details of a dome, resulting in hallucination (incorrect or unrealistic details). The bottom row shows a similar issue with the restoration of a building's columns. In contrast, the ReFIR method, when combined with the existing LRMs, produces more faithful and realistic restorations, demonstrating its ability to mitigate the hallucination problem.





![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_6_1.jpg)

> This table compares the performance of ReFIR against other state-of-the-art methods on two real-world image super-resolution datasets, CUFED5 and WR-SR.  The metrics used include PSNR, SSIM, LPIPS, NIQE, and FID, assessing both fidelity and perceptual quality.  The results show that ReFIR consistently improves upon existing methods across all metrics on both datasets.





### In-depth insights


#### LRM Hallucination
The concept of "LRM Hallucination" highlights a critical limitation of Large Restoration Models (LRMs).  Analogous to the hallucinations observed in Large Language Models (LLMs), LRMs sometimes generate unrealistic or inaccurate details in restored images, especially when dealing with severely degraded inputs. This phenomenon stems from the **limited internal knowledge** embedded within the model's weights; LRMs, lacking sufficient information, produce plausible-sounding but ultimately incorrect outputs.  **Addressing this necessitates augmenting the model's knowledge base**, perhaps by incorporating external data sources, training on more diverse datasets, or improving the model's ability to identify and handle its own uncertainty.  The paper suggests a retrieval-augmented approach as one solution, a technique analogous to retrieval-augmented generation methods used in LLMs, allowing for the integration of external, relevant data to improve accuracy and fidelity of the restoration process.  The success of such methods hinges on the **effective retrieval of suitable reference images** and the **skillful integration of that information** without negatively impacting the LRM's existing capabilities.  Therefore, future research should focus on developing more robust and efficient techniques for both external knowledge retrieval and integration to mitigate LRM hallucinations and enhance image restoration quality.

#### ReFIR Framework
The ReFIR framework presents a novel approach to enhance large restoration models (LRMs) by incorporating external knowledge via retrieval augmentation.  This method directly addresses the **hallucination problem** often seen in LRMs, where severe degradations lead to unrealistic or inaccurate image restorations.  Instead of relying solely on internal model knowledge, ReFIR leverages a retriever to locate content-relevant, high-quality images from an external database. These images then serve as references, guiding the LRM towards a more faithful reconstruction.  **ReFIR's key innovation** lies in the cross-image injection mechanism, intelligently integrating retrieved image features into the LRM's restoration process, specifically targeting the detail texture restoration stage. This approach is particularly noteworthy for its **training-free nature**, adapting to various LRMs without requiring additional training or parameter adjustments.  The framework demonstrates **significant improvements in both fidelity and realism** of restored images, especially in challenging scenarios.  The proposed architecture is adaptable and shows promise in tackling the limitations of current LRMs.

#### Cross-Image Injection
The proposed 'Cross-Image Injection' method is a **novel approach** to enhance Large Restoration Models (LRMs) by integrating external knowledge from retrieved high-quality images.  Instead of solely relying on internal model parameters, this technique leverages **high-fidelity textures** from reference images to augment the LRM's ability to generate detailed and realistic restorations, especially crucial for severely degraded images where hallucination is a common problem.  The key innovation lies in the careful injection of these external textures during the detail texture restoration phase of the LRM's workflow, identified through a probing analysis of the model's internal mechanisms. This **targeted injection** avoids disrupting the overall structure reconstruction while effectively enriching the generated image with high-quality, contextually relevant details.  Furthermore, the method incorporates a **spatial adaptive gating mechanism** to address spatial misalignments between the input low-quality image and the reference image, selectively fusing the external textures with the LRM's internal predictions only where appropriate.  Finally, a **distribution alignment technique** ensures a smooth and consistent integration, mitigating potential domain-shift issues between the original low-quality image and the external reference.  This combined approach yields highly faithful and realistic restoration results without requiring additional training for the LRM.

#### Retrieval Augmentation
Retrieval augmentation, in the context of large restoration models (LRMs), presents a powerful technique to address the inherent limitations of these models, particularly their propensity for hallucinations. By **retrieving relevant high-quality images** from an external database, LRMs are provided with additional contextual knowledge that transcends their internal model weights.  This external knowledge helps to **ground the restoration process**, ensuring that generated details remain faithful to the original scene even in cases of severe degradation.  The strategy is particularly effective for scenarios where the LRM lacks sufficient internal knowledge about specific scene textures or content.  **The integration is often seamless**, requiring minimal or no retraining of the LRM itself, making it a highly adaptable and versatile approach. The key to successful retrieval augmentation is the design of an efficient retrieval system and a mechanism for effectively incorporating the retrieved information into the model's decision-making process, perhaps by employing cross-image injection or other sophisticated attention mechanisms. While computationally more expensive than solely relying on internal knowledge, retrieval augmentation offers a significant improvement in fidelity and realism, ultimately leading to more robust and dependable LRM performance.

#### Future of ReFIR
The future of ReFIR hinges on addressing its current limitations and exploring its potential extensions.  **Improving the efficiency of the retrieval process** is crucial, perhaps by employing more advanced indexing techniques or exploring faster similarity search methods.  **Reducing reliance on high-quality reference images** is another key area.  Exploring techniques that allow ReFIR to effectively utilize lower-quality or even synthetic reference images could significantly broaden its applicability.  **Development of training-free adaptation methods** that allow ReFIR to quickly adapt to new LRMs without manual intervention is also important, enhancing its versatility.  Finally, **investigating the application of ReFIR to other image restoration tasks** beyond those explored in the paper (like video restoration or other modalities) and exploring integration with other models, such as GANs, will open up exciting new avenues of research.  Ultimately, the future success of ReFIR depends on further refining its core mechanisms and expanding its capabilities to meet the evolving demands of the image restoration field.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_2_1.jpg)

> This figure visualizes the working mechanism of Large Restoration Models (LRMs) using PCA and Fourier analysis. The left side shows the top three principal components of latent features extracted from the self-attention layers of the ControlNet and UNet decoder using PCA. The right side presents the power spectrum of those latent features using Fourier analysis.  This helps illustrate how LRMs process image information, distinguishing structure reconstruction from detail texture restoration.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_3_1.jpg)

> This figure illustrates the ReFIR framework, which consists of two main stages: 1) Reference Image Retrieval, where a retriever (R) searches for content-relevant images from a high-quality image database (D) given a low-quality input image; 2) High-fidelity Image Restoration, where a large restoration model uses the retrieved images (IR) to restore the high-quality image, leveraging additional external knowledge.  The framework is designed to be adaptable to various existing large restoration models (LRMs) without requiring additional training.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_4_1.jpg)

> This figure illustrates the cross-image injection module, a key component of the ReFIR framework. It shows how the model leverages both the target restoration chain (CT) and the source reference chain (Cs), which share the same model weights.  The source reference chain processes retrieved high-quality images, while the target restoration chain processes the low-quality image to be restored. The cross-image injection module facilitates the transfer of high-quality texture information from Cs to CT through separate attention mechanisms. This attention mechanism is designed to overcome the domain preference problem and efficiently use external knowledge from the retrieved image, thus enhancing the restoration quality and reducing hallucination.  Spatial adaptive gating is used to handle spatial misalignment between the low-quality and high-quality images, ensuring consistent texture integration.  Distribution alignment then refines the fused features to minimize any discrepancies between the two chains.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_6_1.jpg)

> This figure shows the results of image restoration using different methods. The first column shows the reference image, the second column shows the ground truth image, and the following columns show the results obtained using different large restoration models (LRMs). The last column shows the results of the proposed method, ReFIR, which incorporates retrieved images as external knowledge to improve the fidelity of the restoration results. The figure highlights how existing LRMs struggle with severe degradations, resulting in hallucinatory outputs that differ from the original scene. In contrast, ReFIR successfully addresses this issue by leveraging external knowledge and generating results faithful to the original scene.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_7_1.jpg)

> This figure shows a comparison of image restoration results using existing Large Restoration Models (LRMs) and the proposed ReFIR method. The first column shows a reference image. The second column shows the ground truth image. Subsequent columns illustrate results from various LRMs (SeeSR, SUPIR) and the proposed ReFIR method.  The results demonstrate that the existing LRMs struggle to faithfully reproduce details in severely degraded images (hallucination), while ReFIR, by incorporating external knowledge from retrieved images, significantly improves the fidelity of restoration without any additional training.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_8_1.jpg)

> This figure illustrates the ReFIR framework, which consists of two main stages. First, the retriever (R) searches a high-quality image database (D) for content-relevant images to the given low-quality image (LQ). These retrieved images (IR) then serve as external knowledge to improve the restoration process.  The second stage uses a large restoration model (LRM) to restore the high-fidelity (HQ) image using the retrieved images as a condition.  This process is training-free and adaptable to various LRMs.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_8_2.jpg)

> This figure shows the normalized attention scores obtained by averaging all samples and time steps.  The plot illustrates the attention scores when using in-domain low-quality (LQ) images versus cross-domain high-quality (HQ) images.  The graph visually demonstrates the effect of domain preference, where the model shows a stronger tendency to attend to features within the same domain (LQ) even when cross-domain (HQ) information would be more beneficial. An equilibrium line is included as a reference.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_9_1.jpg)

> This figure shows a comparison of image restoration results between existing Large Restoration Models (LRMs) and the proposed ReFIR method. The first column shows the reference images, the second column shows the ground truth, and the remaining columns show results from different LRMs (SeeSR, SUPIR) and the ReFIR method. The ReFIR method significantly improves the fidelity of the restored images, especially in challenging cases where the input image has severe degradation. This demonstrates that the ReFIR method can effectively mitigate the hallucination problem of LRMs by incorporating additional external knowledge.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_9_2.jpg)

> This figure illustrates how the ReFIR framework integrates with existing Large Restoration Models (LRMs).  It shows the latent manifold, representing the model's internal representation of the image. The current latent (Xt) is influenced by two forces: one from the LRM's internal knowledge (a force based on the model's pre-trained knowledge and parameters), and another from the external knowledge provided by the retrieved reference image (the force from the retrieved image). These two forces interact to modify the latent representation, resulting in the 'Finally Modified Xt-1', which guides the restoration process towards a more faithful and high-fidelity result.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_15_1.jpg)

> This figure shows a comparison between the results of existing Large Restoration Models (LRMs) and the proposed ReFIR method on several images.  The existing LRMs struggle with severe degradations, producing unrealistic details (hallucinations) that are inconsistent with the ground truth. In contrast, ReFIR, by incorporating external knowledge from retrieved images, produces restorations that are significantly more faithful to the original scene and visually realistic.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_18_1.jpg)

> This figure visualizes the working mechanism of Large Restoration Models (LRMs) using Principal Component Analysis (PCA) and Fourier analysis. The left side shows PCA visualization of latent features extracted from the self-attention layers of the ControlNet and UNet decoder, highlighting the information processed in each stage.  The right side displays the power spectrum of these features using Fourier analysis, demonstrating how frequency components change during the image restoration process. Appendix H provides further visualizations.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_19_1.jpg)

> This figure visualizes the attention maps generated during the cross-image injection process in ReFIR.  It shows how query pixels from the low-quality image (target restoration chain) attend to relevant regions in the retrieved high-quality image (source reference chain). The visualization demonstrates that ReFIR effectively leverages high-quality texture information from the retrieved image to guide the restoration of the low-quality image, mitigating the hallucination problem.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_20_1.jpg)

> This figure shows a comparison of image restoration results between existing Large Restoration Models (LRMs) and the proposed ReFIR method.  The leftmost column shows a reference image.  The next column shows the ground truth.  The following columns illustrate results from SeeSR, SUPIR, and the proposed ReFIR method.  The figure highlights the hallucination problem in LRMs, where restored images may contain details that are inconsistent with the original scene, and how ReFIR improves restoration fidelity by incorporating external knowledge from retrieved reference images without any retraining.


![](https://ai-paper-reviewer.com/iFKmFUxQDh/figures_21_1.jpg)

> This figure demonstrates the effectiveness of the proposed ReFIR method. It shows that existing Large Restoration Models (LRMs) often suffer from hallucination—producing unrealistic details in severely degraded images.  The ReFIR method, however, is shown to improve the restoration by incorporating external knowledge (retrieved images) without additional training, resulting in more faithful and realistic results.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_7_1.jpg)
> This table presents a quantitative comparison of the proposed ReFIR method against state-of-the-art methods on the RealPhoto dataset, which contains real-world degraded images without ground truth.  The comparison uses several no-reference image quality assessment metrics, including NIQE (Natural Image Quality Evaluator), MUSIQ (Multi-scale Image Quality Transformer), and CLIPIQA (CLIP Image Quality Analyzer).  The results show that ReFIR consistently improves performance across all metrics compared to the baseline methods (SeeSR and SUPIR).

![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_7_2.jpg)
> This table compares the model complexity (number of parameters, GPU memory, and inference time) of the SeeSR and SUPIR models before and after integrating the ReFIR framework.  The comparison highlights the computational overhead introduced by ReFIR while demonstrating the minimal increase in the number of parameters.  The experiment used a single 80G NVIDIA A100 GPU and an input image resolution of 2048x2048 pixels.

![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_8_1.jpg)
> This table presents the ablation study results on the effectiveness of different components used in the cross-image injection part of the ReFIR framework. It shows the performance (PSNR↑, SSIM↑, NIQE↓, FID↓)  with different combinations of separate attention (SA), spatial adaptive gating (SG), and distribution alignment (DA) techniques.  The results demonstrate that all three components contribute positively to the performance, with the combination of all three yielding the best results.

![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_8_2.jpg)
> This table presents ablation study results on different positions of cross-image injection within the ReFIR framework. It compares the performance (PSNR↑, SSIM↑, NIQE↓, FID↓) when the cross-image injection is applied to different parts of the model: only the encoder, only the decoder, both encoder and decoder, and no injection (baseline). The results indicate the optimal performance is achieved by applying the injection to the decoder only, highlighting its crucial role in detail texture restoration and the importance of targeting this specific stage for external knowledge integration.

![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_9_1.jpg)
> This table presents an ablation study on the impact of different types of reference images on the restoration performance. It compares using no reference image (NoRef), using the ground truth high-quality image (HQRef), using a bicubic upsampling of the low-quality image (SelfRef), and using a randomly selected high-quality image (Random). The results are compared to a baseline (Baseline) which represents the original method without the retrieval augmentation. The metrics used to evaluate the performance are PSNR, SSIM, LPIPS, NIQE, and FID.

![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_14_1.jpg)
> This table presents the performance results of all low-resolution (LR) images using fallback strategies in extreme conditions, where relevant high-quality reference images may be scarce or unavailable.  The results are compared across different scenarios: using the original LRM without reference images (origin_lrm), generating reference images using the BLIP model and StableDiffusion (gen_ref), adaptively selecting the better result between gen_ref and ReFIR (ada_gen_ref), and finally, using the proposed ReFIR method. The metrics used are NIQE (lower is better), MUSIQ (higher is better), and CLIPIQA (higher is better).

![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_15_1.jpg)
> This table presents the results of experiments conducted to assess the impact of using multiple retrieved images in the image restoration process.  It shows how using multiple reference images affects performance metrics (PSNR, SSIM, LPIPS, NIQE, FID) and computational cost (GPU memory usage and inference time). The results are compared across different numbers of retrieved images (NoRef, OneRef, TwoRef) on an A100 GPU.  The table provides quantitative insights into the trade-offs between improved restoration quality and increased computational demand when using multiple references.

![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_16_1.jpg)
> This table presents the ablation study results for different cross-image injection designs. It compares the performance of three methods: replacing the original self-attention results from CT with corresponding latent in Cs, concatenating KT and Ks and using the proposed separate attention method. The results are evaluated based on PSNR↑, SSIM↑, NIQE↓, and FID↓ metrics, showing the superior performance of the proposed separate attention approach.

![](https://ai-paper-reviewer.com/iFKmFUxQDh/tables_17_1.jpg)
> This table presents a quantitative comparison of the proposed ReFIR method against other state-of-the-art methods for real-world image super-resolution.  It shows the performance improvements achieved by ReFIR across various metrics, including PSNR, SSIM, LPIPS, NIQE, and FID. The improvements demonstrate the effectiveness of ReFIR in improving both the fidelity and perceptual quality of the restored images.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/iFKmFUxQDh/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}