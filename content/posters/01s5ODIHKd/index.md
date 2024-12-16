---
title: "FreqMark: Invisible Image Watermarking via Frequency Based Optimization in Latent Space"
summary: "FreqMark: Robust invisible image watermarking via latent frequency space optimization, resisting regeneration attacks and achieving >90% bit accuracy with high image quality."
categories: ["AI Generated", ]
tags: ["Computer Vision", "Image Generation", "🏢 University of Science and Technology of China",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 01s5ODIHKd {{< /keyword >}}
{{< keyword icon="writer" >}} YiYang Guo et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=01s5ODIHKd" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/01s5ODIHKd" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/01s5ODIHKd/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Existing image watermarking techniques struggle with regeneration attacks, where AI models are used to remove the watermark without significantly affecting image quality.  These methods often embed watermarks in either the image's frequency or latent space, but each approach has vulnerabilities: frequency domain methods are susceptible to regeneration attacks, while latent space methods are vulnerable to noise.  This creates a need for a more robust and resilient watermarking technique.



FreqMark tackles this issue by embedding the watermark in the latent *frequency* space of the image using a variational autoencoder (VAE). This approach combines the strengths of both frequency and latent space methods.  **The results show FreqMark significantly outperforms existing methods in robustness, especially against regeneration attacks, while maintaining high image quality and offering flexibility in bit encoding.** The authors demonstrate high bit accuracy (over 90%) even under various attacks, highlighting the method's effectiveness and potential for real-world applications.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} FreqMark offers superior robustness against regeneration attacks compared to existing methods. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method allows a flexible trade-off between image quality and watermark robustness. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} FreqMark achieves high bit accuracy (>90%) even under various attack scenarios. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it introduces a novel and robust watermarking method, FreqMark, that significantly improves the security of digital images against sophisticated attacks.  **Its focus on latent frequency space optimization offers a new avenue for watermarking research, addressing the limitations of traditional methods.** This work is particularly timely due to the increasing prevalence of AI-generated content and the need for effective mechanisms to protect intellectual property and authenticity.  **FreqMark's flexible trade-off between image quality and robustness opens new possibilities for various watermarking applications across different domains.**

------
#### Visual Insights



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_1_1.jpg)

> 🔼 This figure compares the robustness of three different watermarking approaches against Gaussian noise and regeneration attacks.  The leftmost panel shows watermarking in the image frequency domain, which is resistant to Gaussian noise but vulnerable to regeneration attacks. The middle panel illustrates watermarking in the image's latent space. This approach improves resistance to regeneration attacks but is more susceptible to Gaussian noise. The rightmost panel showcases the FreqMark method, which embeds the watermark in the latent frequency space, providing strong robustness against both Gaussian noise and regeneration attacks.
> <details>
> <summary>read the caption</summary>
> Figure 1: The robustness of different watermark encoding positions. Left: Encoding in image frequency space resists Gaussian noise but is vulnerable to regeneration attacks. Middle: Encoding in image latent space enhances resistance to regeneration attacks but introduces vulnerabilities to Gaussian noise. Right: FreqMark encodes latent frequency space in the image, achieving a strong defense against regeneration and traditional attacks.
> </details>





![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_6_1.jpg)

> 🔼 This table presents a comparison of the performance of four different watermarking methods (DwtDctSvd, SSL Watermark, Stable Signature, and FreqMark) on two datasets (ImageNet and DiffusionDB).  The performance is evaluated based on PSNR and SSIM values for image quality, and bit accuracy under various attacks (none, brightness, contrast, JPEG compression, Gaussian blur, Gaussian noise, VAE-B, VAE-C, and diffusion regeneration attacks). The table shows the average performance and standard deviation (±std) for each method and attack scenario.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of different watermarking methods on ImageNet and DiffusionDB.
> </details>





### In-depth insights


#### Latent Space Opt.
The heading 'Latent Space Opt.' strongly suggests a research focus on optimizing processes within a latent space.  This implies the use of **latent variable models**, likely **autoencoders** or **variational autoencoders (VAEs)**, to represent data in a lower-dimensional space. The optimization likely aims to improve various aspects of the latent representation, such as **reducing dimensionality**, enhancing **data separability**, or improving the **generalizability** of a model trained using that representation.  Specific optimization techniques could involve gradient-based methods, evolutionary algorithms, or other strategies. The context of the paper will determine the precise goals.  **Robustness** and **efficiency** of the optimization process are likely important considerations, especially given the potential for real-world applications. The ultimate aim is to leverage the advantages of a latent space representation for enhanced performance or understanding in a specific domain.  Depending on the field, this could relate to image processing, natural language processing, or other areas where latent space modeling is prevalent.

#### FreqMark Robustness
FreqMark's robustness is a key strength, demonstrated by its resilience against various attacks.  **The method's success stems from embedding watermarks within the latent frequency space of images**, a strategy that proves more resistant to regeneration attacks than traditional frequency or latent space methods.  **Experimental results showcase high bit accuracy (exceeding 90% for a 48-bit message) even under diverse attacks**, including Gaussian noise, JPEG compression, brightness/contrast adjustments, and even sophisticated VAE regeneration attacks. This robustness is further enhanced by an augmentation strategy employed during training, which makes the watermarks more resilient to a wider range of manipulations.  **The flexible nature of FreqMark, allowing for trade-offs between image quality and bit accuracy**, contributes to its overall robustness and adaptability for different application needs. However, while robust, FreqMark's performance against certain specific spatial transformations, such as rotation and cropping, might need further improvement.

#### VAE-based Encoding
A VAE-based encoding scheme in a research paper likely involves using a Variational Autoencoder (VAE) to represent images in a lower-dimensional latent space.  This is advantageous for several reasons.  First, it significantly reduces the dimensionality of the data, making subsequent processing and watermark embedding computationally more efficient. Second, the **latent space often captures meaningful semantic features of the images**, which can improve the robustness of the watermark against various attacks.  A well-trained VAE can reconstruct high-quality images from the compressed latent representation, ensuring that the watermarking process minimally impacts image quality.  The choice of the VAE architecture and training method will affect the quality of the latent space representation, and therefore, the overall performance of the watermarking scheme.  **Careful consideration of these factors is crucial to achieving a balance between embedding efficiency, robustness, and image quality**.  The specific approach to embedding information within the latent space, whether directly modifying the latent vectors or using some other transformation, would also be a key aspect of the technique.

#### Regeneration Attacks
Regeneration attacks pose a significant threat to image watermarking, as they leverage generative models to remove watermarks by reconstructing the image.  **Existing frequency-domain methods are particularly vulnerable**, failing to withstand such sophisticated attacks. These attacks highlight a critical limitation: the focus on embedding watermarks in readily accessible image domains, which are easier for generative models to manipulate.  **Methods embedding watermarks in the latent space of images show promise**, as it's more resilient to manipulation by generative models, but are still susceptible to other forms of attack.  **A hybrid approach**, combining frequency and latent space techniques, as explored in some recent research, may prove a more robust solution.  This strategy aims to leverage the resilience of latent space against regeneration attacks while maintaining the benefits of frequency methods against other image manipulations.  The challenge lies in **finding the optimal balance** between embedding strength, image quality, and robustness against all types of attacks, a key area for further research and improvement.

#### Future Enhancements
Future enhancements for this watermarking technique could explore several avenues.  **Improving robustness against various attacks** remains a priority, especially focusing on advanced generative model-based attacks and sophisticated adversarial techniques.  **Expanding the payload capacity** while maintaining image quality is crucial for wider applicability.  Investigating alternative encoding mechanisms beyond frequency-domain optimization in latent space, perhaps leveraging novel deep learning architectures, could unlock further improvements.  **Addressing computational efficiency** is important for practical implementation, particularly for real-time or high-throughput watermarking scenarios.  Finally, **rigorous testing and validation across diverse datasets** and application domains would enhance the generalizability and practical utility of the method.  Further investigation into perceptual quality metrics could fine-tune the watermarking process for truly imperceptible embedding.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_3_1.jpg)

> 🔼 This figure illustrates the FreqMark watermarking process.  The encoding stage uses a pre-trained Variational Autoencoder (VAE) to map an image to a latent frequency space.  A perturbation (δm) is added to this latent space, encoding the watermark message. This is then passed through the VAE decoder to produce the watermarked image. The decoding stage uses a pre-trained image encoder, extracting features from the watermarked image and using a comparison with predefined vectors to reveal the hidden message.  Noise (e1 and e2) is introduced during training to enhance robustness.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overview of FreqMark. Encoding: FreqMark employs a pre-trained VAE model to encode watermarks within the latent frequency space of the image.  e1 and e2 are Gaussian noise perturbations added during training. All networks are fixed and only perturbation δm is trained. Decoding: FreqMark utilizes a pre-trained image encoder to extract features from the image and extracts the watermark by comparing this feature against predefined directions.
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_5_1.jpg)

> 🔼 This figure shows example images from the ImageNet and DiffusionDB datasets that have been watermarked using the FreqMark method.  The top row shows the original images, the middle row shows the same images after watermark embedding, and the bottom row shows the pixel-level difference between the original and watermarked images (amplified for visibility).  The caption highlights that the watermarks are 48 bits long and resistant to various attacks.
> <details>
> <summary>read the caption</summary>
> Figure 3: Examples of watermarked images. The first three columns are from ImageNet [16], and the others are generated by the prompts from DiffusionDB [49]. These watermarked images have 48-bit messages and are robust to various attacks. Top: origin image. Middle: watermarked image. Bottom: pixel difference (amplified by a factor of 10 to enhance the visual effect).
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_6_1.jpg)

> 🔼 This figure shows the correlation matrices for each bit of a decoded message (left) and a random message (right).  The diagonal line represents the perfect correlation of a bit with itself. The rest of the matrix shows the correlation between different bits. The decoded message shows more structured correlation than the random message, illustrating the dependency between bits in the watermarked image. This is important because independent bits are easier to break during attacks.
> <details>
> <summary>read the caption</summary>
> Figure 4: The correlation matrix of each bit of the decoded message from the vanilla images and the random message.
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_7_1.jpg)

> 🔼 This figure compares the visual results of watermark embedding using different methods. The first column shows the original images, and the following columns show the watermarked images with different embedding strategies.  The methods compared include embedding in the pixel space, the frequency domain of the pixel space, the latent space, and the frequency domain of the latent space. The last column highlights the difference between the original and watermarked images, amplified for better visualization. The figure demonstrates the impact of embedding location on image quality and watermark invisibility. The 'Latent Frequency' method, proposed by the authors, aims to achieve a balance between robustness and image quality.
> <details>
> <summary>read the caption</summary>
> Figure 5. Watermarked images under different optimization locations. We compared four distinct optimization objectives for watermark embedding, including the image pixel domain, the frequency domain of the image pixel, the image latent space, and the frequency domain of the image latent space (ours). The difference after watermarking addition is amplified by a factor of 10.
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_9_1.jpg)

> 🔼 This figure illustrates the FreqMark watermarking method.  The encoding process uses a pre-trained Variational Autoencoder (VAE) to embed the watermark in the latent frequency space of the input image.  Only the perturbation (δm) is trained during this process, keeping the encoder, decoder, and other components fixed. The decoding process uses a pre-trained image encoder to extract features from the watermarked image, and compares these to pre-defined directional vectors to recover the hidden message. Gaussian noise (e1 and e2) is added during training to enhance robustness.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overview of FreqMark. Encoding: FreqMark employs a pre-trained VAE model to encode watermarks within the latent frequency space of the image.  e1 and e2 are Gaussian noise perturbations added during training. All networks are fixed and only perturbation δm is trained. Decoding: FreqMark utilizes a pre-trained image encoder to extract features from the image and extracts the watermark by comparing this feature against predefined directions.
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_15_1.jpg)

> 🔼 The figure consists of two sub-figures. The left sub-figure shows the relationship between training steps and watermark robustness under various attacks. The right sub-figure shows how PSNR and SSIM change with increasing training steps.  The graph shows that the watermark robustness improves with increased training steps, while image quality (as measured by PSNR and SSIM) remains relatively stable after initial fluctuations.
> <details>
> <summary>read the caption</summary>
> Figure 7: The relationship between the training steps, watermark robustness, and image quality.
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_15_2.jpg)

> 🔼 This figure shows the receiver operating characteristic (ROC) curves for the FreqMark watermarking method under various attacks, including brightness, contrast, JPEG compression, Gaussian blur, Gaussian noise, and VAE-B, VAE-C, and Diffusion regeneration attacks. The curves are plotted for two datasets: DiffusionDB and ImageNet.  The x-axis represents the false positive rate (FPR), while the y-axis represents the true positive rate (TPR).  Each curve shows the trade-off between the FPR and TPR for a specific attack, demonstrating the robustness of FreqMark under various attacks.  The different curves represent different attacks on the watermarked image. Ideally, the curves should be high in the y-direction and low in the x-direction (high TPR, low FPR).
> <details>
> <summary>read the caption</summary>
> Figure 8: The TPR/FPR curve under various attacks in two datasets.
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_16_1.jpg)

> 🔼 This figure showcases examples of images watermarked using the proposed FreqMark method.  It demonstrates the imperceptibility of the watermarking process by visually comparing the original images, the watermarked images, and an amplified version of the pixel differences. The dataset used to source the images includes ImageNet and DiffusionDB, highlighting the method's versatility.
> <details>
> <summary>read the caption</summary>
> Figure 3: Examples of watermarked images. The first three columns are from ImageNet [16], and the others are generated by the prompts from DiffusionDB [49]. These watermarked images have 48-bit messages and are robust to various attacks. Top: origin image. Middle: watermarked image. Bottom: pixel difference (amplified by a factor of 10 to enhance the visual effect).
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_17_1.jpg)

> 🔼 This figure shows examples of images watermarked using the FreqMark method.  It demonstrates the invisibility of the watermark by comparing the original image to the watermarked version, showing only minor visual differences (amplified for visibility).  The images come from two datasets: ImageNet and DiffusionDB, highlighting the method's versatility across different image types and sources.  The robustness of the watermark to attacks is also implied by the minimal changes.
> <details>
> <summary>read the caption</summary>
> Figure 3: Examples of watermarked images. The first three columns are from ImageNet [16], and the others are generated by the prompts from DiffusionDB [49]. These watermarked images have 48-bit messages and are robust to various attacks. Top: origin image. Middle: watermarked image. Bottom: pixel difference (amplified by a factor of 10 to enhance the visual effect).
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_17_2.jpg)

> 🔼 This figure showcases examples of images watermarked using the proposed FreqMark method.  It demonstrates the imperceptibility of the watermarking process by comparing original images to their watermarked counterparts and showing the difference (amplified for visibility). The images are from the ImageNet and DiffusionDB datasets.  The caption highlights the robustness of the watermarks to various attacks and that they encode a 48-bit message.
> <details>
> <summary>read the caption</summary>
> Figure 3: Examples of watermarked images. The first three columns are from ImageNet [16], and the others are generated by the prompts from DiffusionDB [49]. These watermarked images have 48-bit messages and are robust to various attacks. Top: origin image. Middle: watermarked image. Bottom: pixel difference (amplified by a factor of 10 to enhance the visual effect).
> </details>



![](https://ai-paper-reviewer.com/01s5ODIHKd/figures_18_1.jpg)

> 🔼 This figure shows several examples of images watermarked using the FreqMark method. The top row displays the original images, the middle row shows the same images after watermarking, and the bottom row highlights the pixel-level differences between the original and watermarked images (amplified for better visibility).  The images are sourced from both the ImageNet and DiffusionDB datasets, demonstrating the technique's effectiveness across different image types. The watermarks are 48 bits long and the images demonstrate robustness against several attacks.
> <details>
> <summary>read the caption</summary>
> Figure 3: Examples of watermarked images. The first three columns are from ImageNet [16], and the others are generated by the prompts from DiffusionDB [49]. These watermarked images have 48-bit messages and are robust to various attacks. Top: origin image. Middle: watermarked image. Bottom: pixel difference (amplified by a factor of 10 to enhance the visual effect).
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_6_2.jpg)
> 🔼 This table compares the image quality (PSNR and SSIM) of images reconstructed by a Variational Autoencoder (VAE) and the watermarked images produced by FreqMark.  It shows that FreqMark maintains good image quality compared to the VAE reconstruction, demonstrating that the watermark embedding process does not significantly degrade the image quality.
> <details>
> <summary>read the caption</summary>
> Table 2: Comparison of image quality between VAE and FreqMark.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_7_1.jpg)
> 🔼 This table compares the performance of FreqMark against three other watermarking methods (DwtDctSvd, SSL Watermark, and Stable Signature) across various metrics on ImageNet and DiffusionDB datasets.  The metrics include PSNR, SSIM, and bit accuracy under different attacks (none, brightness, contrast, JPEG compression, Gaussian blur, Gaussian noise, VAE-B, VAE-C, and diffusion regeneration).  The table allows for a quantitative comparison of the robustness and image quality trade-offs of each method.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of different watermarking methods on ImageNet and DiffusionDB.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_7_2.jpg)
> 🔼 This table presents the results of experiments evaluating the robustness of the FreqMark watermarking method against diffusion attacks of varying intensities.  It shows the bit accuracy and PSNR (Peak Signal-to-Noise Ratio, a measure of image quality) for different numbers of diffusion steps.  The number of diffusion steps is a parameter controlling the strength of the attack; higher numbers mean stronger attacks.  The data illustrates how the watermark's accuracy and image quality degrade as the diffusion attack becomes more intense.
> <details>
> <summary>read the caption</summary>
> Table 4: Performance under Different Diffusion Steps.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_8_1.jpg)
> 🔼 This table presents the performance of the FreqMark watermarking method under two types of attacks: VAE attack (in the latent frequency domain) and Gaussian noise attack (in the pixel frequency domain).  The results are shown in terms of Peak Signal-to-Noise Ratio (PSNR) and bit accuracy for different attack strengths.  The PSNR measures image quality, while bit accuracy reflects the success rate of watermark extraction after the attacks.
> <details>
> <summary>read the caption</summary>
> Table 5: Performance under VAE Attack in Latent FFT Domain and Gaussian Noise Disruption in Pixel FFT Domain.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_8_2.jpg)
> 🔼 This table presents the results of adversarial attacks targeting the latent representations of watermarked images.  It shows the bit accuracy and true positive rate (TPR) at a 0.1% false positive rate (FPR) under different attack strengths (eps).  The attack strength is measured as the maximum L-infinity distance between the original image and the adversarial image.
> <details>
> <summary>read the caption</summary>
> Table 6: Performance under Different Adversarial Attack Strength.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_9_1.jpg)
> 🔼 This table compares the performance of FreqMark against three other watermarking methods (DwtDctSvd, SSL Watermark, and Stable Signature) across various attacks (brightness, contrast, JPEG compression, Gaussian blur, Gaussian noise, VAE-B, VAE-C, and diffusion).  The metrics used are PSNR, SSIM, and bit accuracy for both ImageNet and DiffusionDB datasets. The results show FreqMark's superior robustness and accuracy across various attacks, especially against VAE-based and diffusion regeneration attacks.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of different watermarking methods on ImageNet and DiffusionDB.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_14_1.jpg)
> 🔼 This table presents a comparison of different watermarking methods, namely DwtDctSvd, SSL Watermark, Stable Signature, and FreqMark, across various metrics.  The metrics evaluated include PSNR and SSIM for image quality, and bit accuracy under different attacks (brightness, contrast, JPEG compression, Gaussian blur, Gaussian noise, VAE-B, VAE-C, and Diffusion attacks) for watermark robustness. The comparison is performed on two datasets, ImageNet and DiffusionDB, demonstrating the performance of each method under various conditions and attacks. 
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of different watermarking methods on ImageNet and DiffusionDB.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_15_1.jpg)
> 🔼 This table compares the performance of different watermarking methods (DwtDctSvd, SSL Watermark, Stable Signature, and FreqMark) on two datasets, ImageNet and DiffusionDB.  For each method, it provides the average PSNR and SSIM scores, as well as the bit accuracy under various attacks (brightness change, contrast change, JPEG compression, Gaussian blur, Gaussian noise, VAE-B attack, VAE-C attack, and Diffusion attack).  The standard deviation (±std) of the results is also included. This allows for a comprehensive evaluation of the robustness and image quality trade-offs across different watermarking techniques.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of different watermarking methods on ImageNet and DiffusionDB.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_16_1.jpg)
> 🔼 This table compares the performance of different watermarking methods, including DwtDctSvd, SSL Watermark, Stable Signature, and FreqMark, across various metrics such as PSNR, SSIM, and bit accuracy.  The performance is evaluated on two datasets: ImageNet and DiffusionDB, and under various attack scenarios including brightness, contrast, JPEG compression, Gaussian blur, Gaussian noise, and VAE and diffusion regeneration attacks.  The average bit accuracy under these conditions is a key metric. This allows for a comprehensive comparison of the robustness and quality trade-offs for each method.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of different watermarking methods on ImageNet and DiffusionDB.
> </details>

![](https://ai-paper-reviewer.com/01s5ODIHKd/tables_16_2.jpg)
> 🔼 This table compares the performance of different watermarking methods (DwtDctSvd, SSL Watermark, Stable Signature, and FreqMark) on two datasets, ImageNet and DiffusionDB.  The comparison is based on several metrics: PSNR, SSIM, and bit accuracy under various attacks (brightness, contrast, JPEG compression, Gaussian blur, Gaussian noise, VAE-B, VAE-C, and Diffusion regeneration attacks).  The table shows the average performance and standard deviation for each metric and attack type, allowing for a comprehensive assessment of robustness and image quality.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of different watermarking methods on ImageNet and DiffusionDB.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/01s5ODIHKd/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}