---
title: "RAW: A Robust and Agile Plug-and-Play Watermark Framework for AI-Generated Images with Provable Guarantees"
summary: "RAW: A novel watermark framework ensures the authenticity of AI-generated images by embedding learnable watermarks directly into the image data, providing provable guarantees even under adversarial at..."
categories: []
tags: ["Computer Vision", "Image Generation", "🏢 University of Minnesota",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} ogaeChzbKu {{< /keyword >}}
{{< keyword icon="writer" >}} Xun Xian et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=ogaeChzbKu" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93609" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=ogaeChzbKu&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/ogaeChzbKu/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

The rise of AI image generation tools raises serious concerns about copyright infringement and misuse of generated content. Current watermarking techniques either embed fixed codes in latent representations (limiting adaptability and real-time applications) or are vulnerable to attacks.  This leads to a need for robust and efficient watermarking techniques that can provide provable guarantees and be adaptable across different AI models.



This paper introduces RAW, a novel watermarking framework that directly injects learnable watermarks into the image data.  By jointly training a classifier with the watermark, it can effectively detect the presence of a watermark with provable guarantees on false positives.  RAW boasts significant advantages in speed (30-200x faster than existing methods) while maintaining image quality and robustness against attacks.  The framework's model-agnostic nature enhances its flexibility, and the provable security makes it highly appealing for real-world application.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} RAW provides a robust and efficient watermarking framework for AI-generated images. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The framework offers provable guarantees on the false positive rate, even when facing adversarial attacks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} RAW demonstrates significantly improved watermark encoding speed and detection performance compared to existing methods. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in AI and computer vision due to its timely address of intellectual property concerns in AI-generated content.  The **provable guarantees** offered by the RAW framework are significant, along with its **efficiency and adaptability**. This opens avenues for robust watermarking techniques across diverse generative models and facilitates further research in secure and reliable content authentication.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/ogaeChzbKu/figures_9_1.jpg)

> This figure shows four subfigures, each presenting a different aspect of the RAW watermarking framework's performance. (a) shows the relationship between the false positive rate (FPR) and the alpha (α) parameter, demonstrating the effectiveness of the framework in controlling FPRs.  (b) shows how the average AUROC improves as the training sample size increases for both MS-COCO and DB-Diffusion datasets. (c) and (d) illustrate the tradeoff between watermark robustness and image quality (measured by FID) by varying the watermark strength parameters c1 and c2. The results indicate a balance can be achieved between strong watermark detection and preserving image quality.





![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_1_1.jpg)

> This table compares several watermarking methods based on three key features: whether they are model-agnostic (meaning they work on images generated by any AI model), whether they can be deployed on-the-fly (meaning the watermark can be added to an image immediately after it is generated), and whether they offer provable guarantees on the False Positive Rate (FPR) even if someone tries to remove the watermark.  A checkmark indicates that the method satisfies the feature, an X indicates it does not, and a slash (/) means it is sometimes true, depending on how the method is used.





### In-depth insights


#### AI Watermarking
AI watermarking is a rapidly evolving field focused on embedding imperceptible digital signatures into AI-generated content, such as images and videos.  **The primary goal is to deter unauthorized copying and distribution**, while maintaining the quality of the original content.  Unlike traditional watermarking techniques, AI watermarking must address the unique challenges posed by generative models and their capacity to manipulate or remove watermarks.  **Robustness against adversarial attacks is a critical concern**, requiring advanced techniques to ensure watermark persistence even after image modifications.  Methods often involve embedding the watermark in latent spaces of the generative model, or directly within the image data, requiring **joint training of both the watermark insertion and detection processes**.  While significant progress has been made, **achieving a balance between robustness, invisibility, and computational efficiency remains a key challenge.** The development of more sophisticated and secure methods, alongside explorations into provable guarantees on detection performance, is essential to the future of AI watermarking.

#### RAW Framework
The RAW framework, a robust and agile plug-and-play watermarking system, presents a novel approach to safeguarding AI-generated images.  **Departing from traditional encoder-decoder methods**, RAW directly embeds learnable watermarks into the image data itself, avoiding the limitations of manipulating latent representations. This method enhances compatibility with various generative architectures and enables on-the-fly watermark injection. The framework's clever integration of advanced smoothing techniques ensures **provable guarantees on the false positive rate**, even under adversarial attacks.  Empirical evidence demonstrates significant improvements in watermark encoding speed and detection performance, significantly outperforming existing techniques while maintaining high image quality.  **This plug-and-play nature** makes RAW highly accessible and adaptable for diverse applications and users, strengthening its real-world applicability and making it a promising advancement in protecting intellectual property in the age of generative AI.

#### Provably Robust
The concept of "Provably Robust" in the context of watermarking AI-generated images is crucial.  It signifies a watermarking system's **resistance to adversarial attacks** aimed at removing or altering the embedded watermark.  A provably robust system offers mathematical guarantees, often expressed as bounds on false-positive rates (FPRs), even under various image manipulations or intentional attacks. This is a significant advancement over traditional methods which only empirically demonstrate robustness.  **Mathematical guarantees** provide higher confidence that the system will correctly identify watermarked images in real-world scenarios.  The methods used to achieve provable robustness may involve techniques such as randomized smoothing or conformal prediction, offering theoretical frameworks to analyze the system's behavior under uncertainty and attacks.  The robustness is not absolute, often depending on specific assumptions about the nature and strength of adversarial attacks.  **The balance between provable guarantees and real-world effectiveness is a key challenge.**  While provable robustness is desirable, computationally expensive methods might reduce real-time applicability.  Therefore, a practical system should consider the trade-off between strong mathematical guarantees and efficient implementation.

#### Real-time Watermarking
Real-time watermarking presents a significant challenge in image processing, demanding techniques that embed and detect watermarks without noticeable latency.  **The trade-off between speed and robustness is crucial**; faster methods might compromise security against sophisticated attacks or image manipulations.  Achieving real-time performance necessitates optimized algorithms, potentially leveraging hardware acceleration or specialized architectures like GPUs.   **Model-agnostic approaches are preferred** for real-time applications as they avoid the constraints of model-specific methods which may require access to model internals, therefore limiting the broad applicability.  **Provable guarantees on the false positive rate are paramount**, particularly under adversarial conditions, to ensure reliability and trustworthiness.  The research focus shifts towards efficient embedding schemes, advanced detection mechanisms that are both rapid and resilient, and mathematically sound techniques for verifying the watermark's presence without introducing computational bottlenecks.  The development of such robust and high-speed watermarking systems opens the door for wide-spread use cases within copyright protection, authentication and provenance tracking of AI-generated content.

#### Future Directions
Future research directions stemming from this robust watermarking framework (RAW) could explore several promising avenues. **Improving the efficiency of watermark embedding and detection** for large-scale deployments is crucial.  This involves optimizing algorithms for speed and scalability, particularly when dealing with high-resolution images and videos.  A second area focuses on **enhancing the provable guarantees on false positive rates (FPRs)**. This could involve exploring new theoretical bounds or refining existing methods.  Furthermore, **expanding the applicability of RAW to diverse generative models and media types** (beyond images) would broaden its impact.  Finally, investigating the resilience of RAW against increasingly sophisticated adversarial attacks and exploring new defensive strategies is of paramount importance.  **Developing more effective methods for verifying the provenance of AI-generated content** and mitigating the potential misuse of watermarks warrants further investigation. Overall, a thoughtful exploration of these research directions will strengthen the capabilities of this watermarking framework and improve AI-generated content security.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/ogaeChzbKu/figures_17_1.jpg)

> This figure demonstrates the overall flow of the proposed RAW framework and its differences from encoder-decoder based methods. In RAW, watermarks are directly introduced and injected into images and are jointly trained with the watermark classifier.  The top panel (a) shows RAW, where frequency and spatial watermarks are added to the image, and a classifier is used to determine if the image is watermarked or not. The bottom panel (b) shows the traditional encoder-decoder method, where a watermark is encoded into the image by the encoder, and the decoder is used to extract it for comparison. 


![](https://ai-paper-reviewer.com/ogaeChzbKu/figures_18_1.jpg)

> This figure shows the effects of joint training and the inclusion of spatial watermarks in the RAW watermarking framework.  The left chart compares the training loss and test accuracy when using only a model versus a model with a watermark. The right chart shows the effects of using only frequency domain watermarks versus using both frequency and spatial domain watermarks.  The results indicate improved performance with joint training and the addition of spatial watermarks, suggesting their importance for robust watermarking.


![](https://ai-paper-reviewer.com/ogaeChzbKu/figures_21_1.jpg)

> This figure shows three examples of images that have been watermarked using the RAW method. The top row displays the original images, while the middle row shows the same images after watermarking.  The bottom row presents a magnified view of the pixel differences between the original and watermarked versions. Visually, the differences are subtle, highlighting the invisibility of the watermarking technique while it preserves image quality.


![](https://ai-paper-reviewer.com/ogaeChzbKu/figures_21_2.jpg)

> This figure shows three example images: an original image (top row), its RAW-watermarked version (middle row), and the pixel-level difference between the original and watermarked images (bottom row). The pixel differences are amplified 4x for better visualization. The images demonstrate that the watermarking process is almost imperceptible and maintains the image quality.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_7_1.jpg)
> This table summarizes the main results of the proposed RAW watermarking framework and compares its performance with other state-of-the-art methods.  It shows the AUROC (Area Under the Receiver Operating Characteristic curve) scores for both benign conditions and under various adversarial attacks (Ad-ROC). Encoding speed (in seconds per image) is also compared across methods.  Finally, image quality is assessed using Fréchet Inception Distance (FID) and CLIP scores.

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_7_2.jpg)
> This table presents the Area Under the Receiver Operating Characteristic (AUROC) scores for various watermarking methods under nine different image manipulations or attacks. These manipulations include common image distortions like rotation, cropping, blurring, and adding noise, as well as three adversarial attacks designed to remove watermarks.  The results show how robust each method is against these manipulations, with higher AUROC scores indicating better robustness.  The table is separated by datasets (MS-COCO and DBDiffusion).

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_8_1.jpg)
> This table summarizes the main results of the RAW watermarking framework, comparing its performance to other methods. It shows the Area Under the ROC Curve (AUROC) for both normal conditions ('N-ROC') and under nine different adversarial attacks ('Ad-ROC').  The encoding speed is also provided, showcasing the efficiency of RAW in embedding watermarks.  The table also includes the Fréchet Inception Distance (FID) and CLIP scores which are used to measure the quality of the watermarked images.  The lower the FID, the better the image quality.  CLIP score indicates how well generated images align with the text prompt used.

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_9_1.jpg)
> This table summarizes the main results of the RAW watermarking framework, comparing its performance to other state-of-the-art methods across two datasets (MS-COCO and DBdiffusion).  It shows the AUROC (Area Under the ROC Curve) scores for both normal conditions and under nine different adversarial attacks or manipulations.  Encoding speed (in seconds per image) is also compared, demonstrating RAW's efficiency.  Finally, Fréchet Inception Distance (FID) and CLIP scores are included to assess image quality.

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_15_1.jpg)
> This table summarizes the main experimental results, comparing the performance of RAW against other watermarking methods.  It shows the Area Under the Receiver Operating Characteristic (AUROC) scores for both normal conditions ('N-ROC') and under nine different types of image manipulations or attacks ('Ad-ROC').  Additionally, it provides the speed of watermark embedding (in seconds per image) for each method.  Lower encoding speed values are better.

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_17_1.jpg)
> This table summarizes the main results of the RAW watermarking framework, comparing its performance with other state-of-the-art methods across two datasets (MS-COCO and DBdiffusion).  It shows the Area Under the Receiver Operating Characteristic (AUROC) scores under both normal conditions ('N-ROC') and after applying nine different image manipulations/adversarial attacks ('Ad-ROC').  Additionally, it presents the time taken for watermark injection (encoding speed) per image, demonstrating RAW's efficiency.

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_18_1.jpg)
> This table presents Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) values for watermarked images generated using different watermarking methods on the MS-COCO and DBdiffusion datasets.  Higher PSNR and SSIM values indicate better image quality after watermarking.  The results show that RAW achieves comparable PSNR to StegaStamp but lower than DwtDctSvd and RivaGAN, while maintaining good SSIM scores.

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_19_1.jpg)
> This table presents the Area Under the Receiver Operating Characteristic (AUROC) scores for different model architectures used in the watermark detection system.  It shows performance both without any image manipulations (AUROC (Ben)) and with nine different manipulations or adversarial attacks (AUROC (Adv)). The goal is to compare the robustness and accuracy of the detection system using various model architectures.

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_20_1.jpg)
> This table presents the results of an ablation study on the effect of varying the size of the watermarked training dataset used to fine-tune a pre-trained watermarking and verification model. The results are shown for both the MS-COCO and DBDiffusion datasets, separating performance under benign conditions (AUROC (Ben)) from that under nine distinct image manipulations and adversarial attacks (AUROC (Adv)). The table demonstrates that satisfactory performance can be achieved with a reasonably small training dataset.

![](https://ai-paper-reviewer.com/ogaeChzbKu/tables_20_2.jpg)
> This table presents the Area Under the Receiver Operating Characteristic (AUROC) scores for watermark detection.  It compares the performance with benign images (no attacks or manipulations) and with images subjected to nine types of attacks and manipulations.  The results are broken down by the dataset used (MS-COCO and DBdiffusion) and the generative model used (SDXL and BriXL).

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/ogaeChzbKu/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}