---
title: "Attack-Resilient Image Watermarking Using Stable Diffusion"
summary: "ZoDiac: a novel image watermarking framework leveraging pre-trained stable diffusion models for robust, invisible watermarks resistant to state-of-the-art attacks."
categories: []
tags: ["Computer Vision", "Image Generation", "🏢 University of Massachusetts Amherst",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} e6KrSouGHJ {{< /keyword >}}
{{< keyword icon="writer" >}} Lijun Zhang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=e6KrSouGHJ" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94294" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/papers/2401.04247" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=e6KrSouGHJ&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/e6KrSouGHJ/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

The rise of powerful generative models like Stable Diffusion poses a significant threat to traditional image watermarking techniques. Existing methods struggle to withstand attacks that leverage this technology, leading to a critical need for more robust solutions.  This vulnerability necessitates reliable methods to distinguish between AI-generated and human-created content for content protection and authenticity verification. 



ZoDiac offers a groundbreaking solution by embedding watermarks into the latent space of images using a pre-trained stable diffusion model.  This approach provides **high invisibility and exceptional robustness against sophisticated attacks, including those using stable diffusion**.  The method's efficacy is validated on multiple benchmarks, showcasing superior performance compared to existing techniques.  Furthermore, **ZoDiac is open-source**, fostering broader adoption and further research in the field.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} ZoDiac uses pre-trained stable diffusion models to embed watermarks in the latent space, making them highly robust to various attacks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} ZoDiac outperforms existing watermarking methods in robustness against state-of-the-art attacks (detection rate above 98%, false positive rate below 6.4%). {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} ZoDiac is open-source, readily accessible, and demonstrates the potential of stable diffusion for robust watermarking. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it introduces a novel, robust watermarking method that addresses the critical issue of image authenticity in the era of advanced generative AI.**  Existing methods are easily circumvented by these models, so ZoDiac's resilience is a significant advancement for digital rights management and content verification.  Its open-source nature further enhances its impact on the field.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_1_1.jpg)

> This figure compares the watermark detection rates (WDR) of six existing watermarking methods (DwtDct, DwtDctSvd, RivaGAN, SSL, CIN, StegaStamp) and the proposed ZoDiac method before and after a stable diffusion-based attack.  The bar chart shows a significant drop in WDR for all existing methods after the attack, highlighting their vulnerability. In contrast, ZoDiac maintains a high WDR, demonstrating its robustness.  The included images illustrate that ZoDiac's watermarks are visually imperceptible.





![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_6_1.jpg)

> This table presents a comparison of different image watermarking methods' performance before and after various attacks.  It shows the image quality (PSNR, SSIM, LPIPS) and watermark detection rate (WDR) for each method across three benchmark datasets (MS-COCO, DiffusionDB, WikiArt).  The attacks include individual image manipulations (brightness, contrast, JPEG compression, etc.) and combined attacks.  The table highlights ZoDiac's superior robustness and high watermark detection rates compared to existing methods.





### In-depth insights


#### ZoDiac Framework
The ZoDiac framework presents a novel approach to image watermarking, leveraging the power of pre-trained stable diffusion models.  Its core innovation lies in embedding watermarks **directly into the latent space** of images, rather than the pixel domain. This offers several advantages. Firstly, it ensures **high invisibility**, as modifications occur at a level imperceptible to the human eye. Secondly, this method proves remarkably **robust against sophisticated attacks**, including those employing stable diffusion itself, which were previously able to remove traditional watermarks.  The framework cleverly utilizes the diffusion process itself as a form of defense mechanism, enhancing watermark persistence even under strong attacks.  Finally, ZoDiac's reliance on pre-trained models significantly reduces the need for extensive training, making it **efficient and readily deployable**. The overall effect is a watermarking technique that achieves superior robustness and invisibility compared to previous methods, opening new possibilities for digital image authentication and copyright protection.

#### Watermark Encoding
Watermark encoding is a crucial step in any digital watermarking system, aiming to embed information invisibly and robustly within a digital image.  **ZoDiac's approach is particularly novel**, leveraging the latent space of a pre-trained stable diffusion model. This is significant because it **shifts the watermarking process away from the pixel domain** to a higher-level representation that's more resilient to attacks. By encoding in the latent space, ZoDiac takes advantage of the inherent structure of the diffusion model, making the watermark extremely difficult to remove without also significantly degrading the image quality. The specific method of encoding, **injecting the watermark into the Fourier space of the latent vector**, further enhances robustness by focusing on low-frequency components that are less sensitive to image manipulation and distortion.   The **iterative optimization process** ensures both invisibility and robustness, while mixing the watermarked and original images at the end further refines image quality.  This sophisticated process stands in stark contrast to traditional watermarking approaches that focus on manipulating pixel values directly, often proving vulnerable to modern generative AI attacks. The choice of the Fourier space and the adaptive enhancement process are **key contributors** to ZoDiac's superior robustness and resilience against a range of attacks.

#### Attack Resilience
The research paper's focus on "attack resilience" is crucial in the context of digital watermarking, especially given the rise of powerful image generation techniques.  The authors demonstrate that existing watermarking methods are vulnerable to sophisticated attacks, particularly those leveraging generative AI models like Stable Diffusion.  **ZoDiac's resilience stems from its unique approach of embedding watermarks in the latent space of a pre-trained Stable Diffusion model.** This is a key innovation, as it leverages the model's inherent denoising properties as a defense mechanism against attacks. The results show that ZoDiac significantly outperforms state-of-the-art methods in terms of watermark detection rate even under strong attacks, making it a **robust and promising solution** for protecting digital images in an era of advanced image manipulation.  However, **limitations exist**, particularly in scenarios involving rotation attacks, highlighting the need for further research to enhance its robustness against a wider range of attacks and attack combinations.

#### Ablation Studies
Ablation studies systematically remove components of a model or system to assess their individual contributions and understand the model's behavior.  In a research paper, this section would rigorously evaluate the impact of removing or modifying different aspects of the proposed approach.  For example, if the work focuses on a novel watermarking technique, ablation studies might involve removing specific components of the watermark embedding or detection processes to see how each part affects the overall performance. This would involve comparing results across multiple variations and providing a detailed analysis of the performance changes, including metrics such as detection rates, false positive rates, and image quality.  **Key insights often emerge from observing which components significantly impact performance and which have minimal effect.** This helps to understand the critical aspects of the technique, validate design choices, and guide future improvements.  **Furthermore, ablation studies demonstrate the robustness of the proposed approach by showing it maintains its effectiveness even with some component alterations.** It's crucial that ablation studies be comprehensive and well-designed to gain meaningful insights into the model and to effectively support the paper's claims.

#### Future Work
Future research directions stemming from this work could explore extending ZoDiac's capabilities to handle **higher-dimensional data** like videos.  Investigating more sophisticated watermark embedding techniques within the latent space, perhaps leveraging **adversarial training** or more complex watermark patterns, could significantly improve robustness against advanced attacks.  Another crucial area is to develop more efficient watermark detection methods, potentially through **optimized Fourier transforms** or the incorporation of deep learning for improved accuracy and speed.  Finally, a **thorough analysis** of the trade-offs between watermark robustness, image quality, and computational cost should be conducted to guide the design of practical watermarking systems for real-world applications.  **Addressing the issue of rotation invariance** directly within the ZoDiac framework is also key to improve its overall practicality and resilience.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_3_1.jpg)

> This figure illustrates the ZoDiac framework, which consists of watermark embedding and detection phases.  The embedding phase involves initializing a latent vector from the input image, encoding a watermark into this vector, and then adaptively enhancing the resulting image to maintain quality. The detection phase uses DDIM inversion, Fourier transform, and a statistical test to detect the presence of the watermark.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_8_1.jpg)

> This figure compares the watermark detection rate (WDR) of six existing watermarking methods (DwtDct, DwtDctSvd [9], RivaGAN [48], SSL [16], CIN [25], StegaStamp [35]) and ZoDiac before and after a stable diffusion-based watermark removal attack [51].  The bar chart shows a significant drop in WDR for existing methods after the attack, while ZoDiac maintains a high WDR. The included images illustrate that ZoDiac's watermarks are perceptually invisible.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_8_2.jpg)

> This figure shows the robustness of ZoDiac watermarking method across three different pre-trained stable diffusion models under various attacks.  The x-axis represents the watermarked image quality (SSIM), and the y-axis shows the watermark detection rate (WDR). Each line represents a different attack ('None', 'Rotation', 'Zhao23', 'All w/o Rotation'), and different line styles represent different stable diffusion models ('2-1-base', 'v1-4', 'xl-base-1.0').  The figure demonstrates the consistency of ZoDiac's performance across various models and attack types, maintaining relatively high detection rates even with reduced image quality.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_13_1.jpg)

> This figure illustrates the workflow of the ZoDiac watermarking framework.  The embedding phase consists of three steps: initializing a latent vector representation of the input image using DDIM inversion, encoding the watermark into the latent vector's frequency domain using FFT, and then enhancing the image quality through a mixing process. The detection phase involves inverting the watermarked image back to the latent vector space, performing an FFT, and using a statistical test (non-central χ2 distribution) to detect the watermark.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_15_1.jpg)

> This figure compares the watermark detection rate (WDR) of six existing watermarking methods (DwtDct, DwtDctSvd, RivaGAN, SSL, CIN, StegaStamp) with ZoDiac, before and after a stable diffusion-based watermark removal attack.  The results show that ZoDiac significantly outperforms existing methods, maintaining a high WDR even after the attack.  The included example images illustrate that ZoDiac's watermarks are visually imperceptible.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_15_2.jpg)

> This figure compares the watermark detection rate (WDR) of six existing watermarking methods (DwtDct, DwtDctSvd, RivaGAN, SSL, CIN, StegaStamp) and ZoDiac before and after a stable diffusion-based attack.  The graph shows a significant drop in WDR for the existing methods after the attack, while ZoDiac maintains a high detection rate. The images illustrate that ZoDiac's watermarks are visually imperceptible.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_16_1.jpg)

> This figure compares the watermark detection rate (WDR) of six existing watermarking methods (DwtDct, DwtDctSvd, RivaGAN, SSL, CIN, StegaStamp) with ZoDiac, both before and after a stable diffusion-based watermark removal attack.  It visually demonstrates that ZoDiac significantly outperforms existing methods in terms of robustness to this attack.  The included example images highlight ZoDiac's invisibility.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_17_1.jpg)

> The figure compares the watermark detection rate (WDR) of six existing watermarking methods and the proposed ZoDiac method before and after a stable diffusion-based watermark removal attack.  It visually demonstrates ZoDiac's superior robustness against this attack. The example images show that the ZoDiac watermarks are nearly imperceptible to the human eye.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_20_1.jpg)

> This figure compares the visual results of watermarking images using two different methods: ZoDiac and ZoDiac without Fourier Transform (ZoDiac w/o FT).  The images demonstrate that ZoDiac, which injects watermarks into the frequency domain, maintains much better visual quality compared to ZoDiac w/o FT, which operates in the spatial domain.  The comparison highlights the importance of the frequency domain for maintaining image quality in watermarking.


![](https://ai-paper-reviewer.com/e6KrSouGHJ/figures_22_1.jpg)

> This figure compares the watermark detection rate (WDR) of six existing watermarking methods (DwtDct, DwtDctSvd, RivaGAN, SSL, CIN, StegaStamp) with the proposed ZoDiac method, before and after a stable diffusion-based watermark removal attack.  It demonstrates the superior robustness of ZoDiac. The two example images visually show that the ZoDiac watermarks are imperceptible.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_8_1.jpg)
> This table presents the results of an ablation study on the impact of different denoising steps (number of iterations in the denoising process) during image watermarking using ZoDiac. It shows how the image quality (measured by PSNR) and watermark detection rate (WDR) are affected by varying the denoising steps.  A denoising step of 0 indicates using only the image autoencoder without the diffusion process.  The table helps to understand the contribution of the diffusion model itself to the robustness of the watermarking process against attacks.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_8_2.jpg)
> This table presents a quantitative comparison of ZoDiac against six other watermarking methods across three benchmark datasets (MS-COCO, DiffusionDB, WikiArt) under eleven different attacks (ten individual plus two composite).  The table shows image quality metrics (PSNR, SSIM, LPIPS) and watermark detection rates (WDR) before and after each attack.  The gray highlights indicate methods achieving maximum or near-maximum WDR for each attack.  ZoDiac consistently demonstrates superior performance.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_9_1.jpg)
> This table presents a comprehensive evaluation of image quality and watermark robustness for ZoDiac and several baseline methods across three datasets (MS-COCO, DiffusionDB, WikiArt) under various attacks (individual and composite).  It shows ZoDiac's superior performance in terms of Watermark Detection Rate (WDR) while maintaining good image quality (PSNR, SSIM, LPIPS).

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_14_1.jpg)
> This table presents a comparison of image quality and watermark robustness for several watermarking methods across three datasets.  Image quality is measured using PSNR, SSIM, and LPIPS. Watermark robustness is measured by the Watermark Detection Rate (WDR) before and after applying ten different attacks individually, and two composite attacks (all attacks and all attacks excluding rotation).  The table highlights the best performing methods for each attack scenario and shows that ZoDiac consistently performs very well, particularly against the most advanced attack.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_17_1.jpg)
> This table shows the False Positive Rate (FPR) for six existing watermarking methods (DwtDct, DwtDctSvd, RivaGAN, SSL, CIN, StegaStamp) on the MS-COCO dataset.  The FPR represents the percentage of non-watermarked images incorrectly identified as watermarked. Lower FPR values indicate better performance.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_17_2.jpg)
> This table presents a quantitative evaluation of different watermarking methods' performance.  It shows image quality metrics (PSNR, SSIM, LPIPS) before and after various attacks, including individual attacks and combinations.  The watermark detection rate (WDR) is used to measure the robustness of each method against each attack. The table highlights ZoDiac's superior performance, particularly against the most challenging attacks.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_18_1.jpg)
> This table presents a comprehensive evaluation of image quality and watermark robustness for ZoDiac and six other watermarking methods across three datasets (MS-COCO, DiffusionDB, WikiArt).  It assesses performance before and after ten individual attacks and two composite attacks ('All' and 'All w/o Rotation').  Image quality is measured using PSNR, SSIM, and LPIPS. Watermark robustness is measured using the Watermark Detection Rate (WDR).  The table highlights the methods with the highest WDR for each attack, showing ZoDiac's superior performance.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_18_2.jpg)
> This table presents a comparison of different watermarking methods' performance before and after various attacks.  It shows image quality metrics (PSNR, SSIM, LPIPS) and watermark detection rates (WDR). The gray shading highlights top-performing methods, emphasizing ZoDiac's superior robustness.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_19_1.jpg)
> This table presents a comparison of image quality and watermark robustness of different watermarking methods before and after various attacks on three benchmark datasets.  Image quality is measured by PSNR, SSIM, and LPIPS.  Robustness is measured by the watermark detection rate (WDR) under ten individual attacks and two composite attacks. The table highlights ZoDiac's superior performance, particularly against advanced attacks.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_19_2.jpg)
> This table presents a comparison of image quality (PSNR, SSIM, LPIPS) and watermark robustness (WDR) for ZoDiac and six other watermarking methods across three datasets (MS-COCO, DiffusionDB, WikiArt).  It evaluates performance before and after ten individual attacks, and two composite attacks ('All' and 'All w/o Rotation').  Gray highlighting indicates top-performing methods. ZoDiac consistently shows high WDR and is within 2% of the maximum for most attacks, significantly outperforming other methods, especially against the Zhao23 attack.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_19_3.jpg)
> This table presents the performance of different watermarking methods against the Gaussian blur attack with varying kernel sizes.  It shows the PSNR (Peak Signal-to-Noise Ratio), the watermark detection rate (WDR), and the false positive rate (FPR) for each method under different attack strengths.  The results highlight the robustness of ZoDiac against this specific attack.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_19_4.jpg)
> This table presents a comprehensive evaluation of different watermarking methods' performance under various attacks. It compares the image quality (PSNR, SSIM, LPIPS) and watermark detection rate (WDR) before and after applying 10 individual attacks and 2 composite attacks (all attacks and all attacks except rotation). The results show that ZoDiac outperforms existing methods, especially when facing advanced attacks like Zhao23.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_19_5.jpg)
> This table presents a comprehensive evaluation of image quality and watermark robustness of ZoDiac and other watermarking methods.  It shows the PSNR, SSIM, and LPIPS scores of watermarked images before and after various attacks (including brightness/contrast adjustments, JPEG compression, noise addition, blurring, denoising, and state-of-the-art generative AI-based attacks).  The Watermark Detection Rate (WDR) and False Positive Rate (FPR) are also reported for each method and attack. Gray highlights indicate top-performing methods.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_19_6.jpg)
> This table presents a comprehensive comparison of ZoDiac against six other watermarking methods across three benchmark datasets (MS-COCO, DiffusionDB, WikiArt). It evaluates the performance of each method before and after applying ten different attacks (including brightness/contrast adjustments, JPEG compression, Gaussian noise/blur, BM3D denoising, VAE-based compression, and a stable diffusion-based attack).  The table highlights image quality metrics (PSNR, SSIM, LPIPS) and watermark detection rate (WDR) and false positive rate (FPR) to demonstrate ZoDiac's superior robustness against a wide range of attacks.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_20_1.jpg)
> This table presents a comparison of image quality (PSNR, SSIM, LPIPS) and watermark robustness (WDR) for different watermarking methods across three datasets (MS-COCO, DiffusionDB, WikiArt) before and after applying various attacks, including individual and composite attacks. The gray highlighting indicates methods with top WDRs, showing ZoDiac's superior performance across most attacks.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_20_2.jpg)
> This table shows ZoDiac's watermark detection rates under two composite attacks. The 'All w/o' attack combines all individual attacks except rotation, while 'All w/o Rev.' reverses the order of those attacks.  The results illustrate the impact of attack order on the watermark's robustness, showing a slight decrease in detection rates when the attack order is reversed.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_21_1.jpg)
> This table presents a comprehensive evaluation of image quality and watermark robustness for ZoDiac and six other watermarking methods across three datasets (MS-COCO, DiffusionDB, WikiArt).  It compares the methods' performance before and after ten individual attacks and two combined attacks (All attacks and All attacks without rotation).  The metrics used include PSNR, SSIM, LPIPS, and Watermark Detection Rate (WDR). The table highlights ZoDiac's superior performance, especially against the most advanced attack (Zhao23).

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_21_2.jpg)
> This table presents the results of a pipeline-aware attack experiment, where the attacker has varying levels of knowledge about the watermarking method.  It shows the Watermark Detection Rate (WDR) for both the original watermark and a newly injected watermark under four conditions:  1. Attacker knows both the model weights and watermark injection settings. 2. Attacker knows only the watermark injection settings. 3. Attacker knows only the model weights. 4. Attacker knows neither.  The results demonstrate that ZoDiac is resilient to attacks with partial knowledge but vulnerable to a full pipeline-aware attack.

![](https://ai-paper-reviewer.com/e6KrSouGHJ/tables_22_1.jpg)
> This table presents a comparison of image quality and watermark robustness (WDR) for different watermarking methods before and after various attacks. The attacks include individual attacks (brightness, contrast, JPEG compression, Gaussian noise, Gaussian blur, BM3D denoising, Bmshj18, Cheng20, Zhao23, rotation) and two composite attacks ('All' and 'All w/o Rotation').  Higher PSNR and SSIM values indicate better image quality, while lower LPIPS indicates better perceptual similarity. Higher WDR indicates better watermark robustness. ZoDiac outperforms other methods in most scenarios.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/e6KrSouGHJ/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}