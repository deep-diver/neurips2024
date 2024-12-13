---
title: "Can Simple Averaging Defeat Modern Watermarks?"
summary: "Simple averaging of watermarked images reveals hidden patterns, enabling watermark removal and forgery, thus highlighting the vulnerability of content-agnostic watermarking methods."
categories: []
tags: ["Computer Vision", "Image Generation", "🏢 National University of Singapore",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} X2G7LA7Av9 {{< /keyword >}}
{{< keyword icon="writer" >}} Pei Yang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=X2G7LA7Av9" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94798" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=X2G7LA7Av9&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/X2G7LA7Av9/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Digital watermarking protects copyrights and source identification of images. Existing methods, especially content-agnostic ones (embedding fixed patterns), are vulnerable to attacks that remove watermarks with minimal distortion.  These attacks often exploit the embedded patterns to remove or forge watermarks, threatening image authenticity and copyright protection. This is a significant issue as content-agnostic methods are commonly used due to their simplicity and computational efficiency. 

This research introduces a novel steganalysis attack that uses simple averaging of multiple watermarked images to reveal the underlying watermark pattern.  This method is effective against various content-agnostic watermarking techniques, even under black-box conditions (without knowing the algorithm's details). The researchers demonstrated the attack's effectiveness on several watermarking methods and propose security guidelines advocating for content-adaptive methods and multi-key assignments to enhance resilience against steganalysis vulnerabilities. 

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Content-agnostic watermarking methods are vulnerable to steganalysis attacks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Simple averaging can reveal watermark patterns, enabling removal and forgery. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Content-adaptive methods are more resilient to steganalysis. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in digital watermarking and image security because it reveals a critical vulnerability in many existing watermarking techniques.  **The findings challenge the commonly held assumptions about the robustness of content-agnostic watermarking methods and highlight the need for improved security evaluations.** This opens avenues for developing more secure watermarking strategies resilient to steganalysis attacks and strengthens the overall security of digital media in the era of generative AI.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_3_1.jpg)

> This figure illustrates the process of watermark steganalysis, a method to extract, remove, or forge watermarks.  The top half shows the greybox scenario, where the adversary has access to both watermarked and clean images. These images are averaged separately, and then subtracted to reveal the watermark pattern (δw). This pattern is then used to remove the watermark from a watermarked image by subtracting it from the image.  The bottom half shows the blackbox setting, where the adversary only has access to watermarked images and a collection of unrelated clean images.  The process is similar, with unrelated clean images being used instead of paired clean images for the extraction of watermark pattern. The extracted watermark pattern is then used to remove the watermark, or even create a forged watermark on a clean image by adding it to the clean image.





![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_20_1.jpg)

> This table presents the performance of 12 watermarking algorithms (6 content-agnostic and 6 content-adaptive) under a steganalysis-based watermark removal attack.  It shows the Area Under the Curve (AUC), bit accuracy, and detection accuracy of each method before and after the attack, along with the Peak Signal-to-Noise Ratio (PSNR) as a measure of the image quality degradation after removal.  The results demonstrate the vulnerability of content-agnostic watermarks to steganalysis compared to the robustness of content-adaptive watermarks.





### In-depth insights


#### Watermark Averaging
Watermark averaging, as a novel steganalysis technique, reveals vulnerabilities in content-agnostic watermarking methods. By averaging multiple watermarked images, the underlying watermark pattern emerges, enabling its extraction.  This **extracted pattern facilitates effective watermark removal and even forgery**. The approach's effectiveness is demonstrated across various watermarking algorithms, highlighting the threat posed by simple averaging. The findings underscore the **importance of designing content-adaptive watermarking strategies**, as they exhibit greater resilience against this steganalysis technique.  Furthermore, the research explores the implications for watermark security, advocating for security evaluations against steganalysis, and suggesting **multi-key assignments as a potential mitigation strategy**.  The study's impact extends beyond image watermarks to include other multimedia forms, underscoring the broader significance of steganalysis in protecting digital content.

#### Steganalysis Attacks
Steganalysis attacks target the hidden information embedded within digital media using watermarking techniques.  These attacks aim to detect, extract, or even remove the embedded data without significantly impacting the perceived quality of the media. **Content-agnostic watermarking**, which embeds fixed patterns regardless of content, proves especially vulnerable.  **Steganalysis exploits the inherent patterns** introduced by such methods.  This could involve statistical analysis to identify deviations from expected patterns or the use of machine learning models trained to distinguish watermarked from unwatermarked data.  The efficacy of steganalysis varies widely depending on the robustness of the watermarking technique used, the amount of data embedded, and the sophistication of the attack.  **Simple averaging of multiple watermarked images** has been demonstrated as an effective black-box steganalysis attack against specific techniques, revealing the underlying pattern and enabling watermark removal or forgery. The development of more robust and secure watermarking methods that are resilient to advanced steganalysis techniques is crucial, particularly in the age of generative AI.

#### Content Agnostic Risks
Content-agnostic watermarking, while computationally efficient, introduces significant security risks.  **The core vulnerability stems from embedding fixed patterns independent of image content.** This predictability allows adversaries to employ steganalysis techniques, effectively revealing and removing the watermark with minimal perceptual distortion.  **Averaging multiple watermarked images reveals the underlying pattern**, facilitating both watermark removal and forgery. This threat is amplified by the fact that many existing content-agnostic methods are highly susceptible to these attacks, despite claims of robustness against traditional distortions.  **The reliance on fixed patterns fundamentally weakens their security**, highlighting a critical need for a shift toward content-adaptive approaches that dynamically adjust watermark placement and strength based on the image content itself. This adaptive approach offers superior robustness against steganalysis, though it comes at the cost of increased computational complexity.

#### Security Guidelines
The proposed security guidelines emphasize the critical need for **robustness against steganalysis** in watermarking techniques.  The authors advocate for shifting towards **content-adaptive methods**, which dynamically adjust watermark placement and strength based on image content, thereby enhancing resilience against analytical attacks.  While acknowledging that content-agnostic techniques offer computational advantages, the paper highlights their vulnerability and suggests mitigating strategies, such as **assigning multiple watermarks per user**, to enhance security. The guidelines underscore the importance of performing security evaluations against steganalysis during the development and design phases, emphasizing a proactive security approach for watermarking algorithms.  **Multi-key assignments** and thorough testing are presented as key recommendations for improving the security and robustness of watermarking schemes in the age of increasingly sophisticated attacks.  Ultimately, the guidelines advocate for a paradigm shift in watermarking design, moving away from easily exploitable patterns towards more secure content-adaptive approaches.

#### Future of Watermarking
The future of watermarking hinges on addressing the vulnerabilities exposed by steganalysis.  **Content-adaptive techniques**, which embed watermarks dynamically based on image content, are crucial for improved robustness against sophisticated attacks.  **Multi-key assignments** offer a temporary mitigation strategy for content-agnostic methods, but these are ultimately insufficient for long-term security.  Further research should focus on developing new algorithms that are inherently resistant to steganalysis, potentially integrating advanced techniques from areas like AI security and cryptography.  **Security evaluations** against steganalysis must become standard practice for all new watermarking methods.  The development of efficient and effective counter-steganalysis methods will also be essential to maintaining the integrity of digital watermarks.  Finally,  **establishing comprehensive guidelines and standards** for watermarking techniques, promoting both robustness and security against evolving threats, is key to ensuring the long-term viability of this important technology.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_5_1.jpg)

> The figure shows the performance of watermark detectors under steganalysis-based removal for both content-agnostic and content-adaptive watermarking methods.  It compares the Area Under the Curve (AUC), bit accuracy, and detection accuracy of various watermarking algorithms before and after the removal process.  Additionally, the Peak Signal-to-Noise Ratio (PSNR) is shown to illustrate the impact on image quality. This illustrates the effectiveness of the steganalysis attack against content-agnostic watermarks and the robustness of content-adaptive methods.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_6_1.jpg)

> This figure visualizes the watermark patterns extracted from Tree-Ring watermarked images using different methods and settings. The top section shows patterns extracted from the DDIM-inverted latent space without subtracting the clean image.  The first and second rows present the Fourier transforms of these patterns. The bottom section shows patterns extracted from image space, with the left side using paired (greybox) images and the right side using unpaired (blackbox) images. These patterns are compared to the ground truth pattern to illustrate the effectiveness of the steganalysis technique in identifying and extracting the watermark pattern even in a blackbox setting.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_6_2.jpg)

> This figure compares the performance of steganalysis-based watermark removal against traditional distortion-based attacks in terms of preserving image quality. The x-axis represents different image quality metrics (PSNR, SSIM, LPIPS, and SIFID), while the y-axis represents the Area Under the Curve (AUC) of the Tree-Ring watermark detector. The blue dots represent the steganalysis-based method; the other colors represent distortion attacks.  The plot shows that steganalysis achieves high AUC values with comparatively better image quality than the other methods. This indicates the effectiveness of steganalysis in removing Tree-Ring watermarks with minimal perceptual distortion.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_7_1.jpg)

> This figure shows the effectiveness of the proposed steganalysis-based watermark removal and forgery methods against Tree-Ring watermarks.  The top half displays histograms showing the distribution of distances between watermarked images and watermark-removed images for different numbers of averaged images.  The bottom half shows similar histograms for forged watermarks, illustrating how the method alters clean images to mimic watermarked ones.  The red dashed lines indicate the thresholds for successful watermark detection (at 1% False Positive Rate). The plots reveal that as more images are averaged, the distribution of distances for the watermark removal (top) shifts toward higher distances from the true watermarked images; simultaneously, the distances for forgery (bottom) shift toward lower distances, making it easier to remove and forge the watermarks.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_7_2.jpg)

> This figure shows the performance of different watermarking methods when a steganalysis-based watermark removal technique is applied.  It compares content-agnostic and content-adaptive methods, showing how the content-agnostic methods are significantly more vulnerable to the attack.  The effectiveness of the watermark removal is measured by AUC, bit accuracy, and detection accuracy of the watermark detector. Image quality after the removal is measured using PSNR (Peak Signal-to-Noise Ratio).  The figure illustrates that the content-adaptive methods maintain high detection accuracy, while the content-agnostic methods experience substantial performance degradation.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_8_1.jpg)

> This figure shows the results of applying steganalysis-based watermark removal to two audio watermarking methods: AudioSeal and WavMark. The plots illustrate the detection accuracy (Det Acc for AudioSeal and Bit Acc for WavMark) and the changes in audio quality (SI-SNR) after removal.  The 'NR' points represent the baseline performance without any watermark removal.  The x-axis shows the number of audio segments averaged during pattern extraction (n).  The figure demonstrates the effectiveness of the steganalysis approach in degrading the performance of both methods, especially as the number of averaged segments increases.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_13_1.jpg)

> This figure visualizes the impact of different levels of steganalysis-based watermark removal and image distortions.  It showcases how increasing the strength of the extracted watermark pattern (through multiplication) affects the image.  The distortions are blurring, gaussian noise, and brightness manipulation, demonstrating the robustness of the Tree-Ring algorithm to some distortions, but showing the effectiveness of the proposed steganalysis attack.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_15_1.jpg)

> This figure shows a visualization of watermark patterns extracted from various content-agnostic watermarking methods. Each row corresponds to a different method (Tree-Ring, RingID, RAWatermark, DwtDctSvd, RoSteALS, and Gaussian Shading).  The columns represent the number of watermarked images averaged during pattern extraction (5, 10, 50, 100, 500, 1000, 5000). For each method, there are two columns representing the pattern extraction under blackbox (left) and greybox (right) settings.  The patterns extracted are adaptively normalized before visualization, making it easier to compare the patterns across different methods and averaging levels. The visual patterns highlight the distinct characteristics of watermarks embedded by different methods and show how these patterns change with different amounts of averaging. This visual information supports the quantitative analysis of watermark removal vulnerability in the paper.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_16_1.jpg)

> This figure visualizes the patterns extracted from various content-adaptive watermarking methods under both greybox and blackbox settings.  Each row represents a different watermarking algorithm (Stable Signature, WmAdapter, RivaGAN, SSL, HiDDeN, and DwtDct).  Within each row, the images show the extracted patterns for different numbers of watermarked images used in the averaging process (from 5 to 5000).  The patterns are normalized for easier comparison and visualization.  The figure aims to show that content-adaptive watermarking techniques tend to produce less discernible patterns than content-agnostic ones (as illustrated in a previous figure).


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_17_1.jpg)

> This figure illustrates the process of watermark removal and forgery using a simple linear assumption.  It shows how averaging paired (greybox setting) or unpaired (blackbox setting) images, and then subtracting the averages, allows the adversary to extract a watermark pattern. This pattern can be used to remove the watermark from watermarked images or forge a watermark onto clean images.  The greybox setting assumes access to paired clean and watermarked images, while the blackbox setting does not require this.  The figure highlights the core concept of the steganalysis approach presented in the paper.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_18_1.jpg)

> This figure visualizes the results of watermark removal from content-agnostic watermarking methods.  It shows images from several different watermarking methods (Tree-Ring, RingID, RAWatermark, DwtDctSvd, RoSteALS, and Gaussian Shading) before and after watermark removal.  The number of images averaged during the watermark extraction process (n) is varied (5, 10, 50, 100, 500, 5000), illustrating how the effectiveness of removal and the resulting image quality change with n. Both greybox (paired images) and blackbox (unpaired images) settings are displayed.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_19_1.jpg)

> This figure shows the results of watermark removal on content-adaptive watermarking methods.  Each row represents a different watermarking method (Stable Signature, WmAdapter, RivaGAN, SSL, HiDDeN, DwtDct), with separate columns for blackbox and greybox settings.  Each column shows the original image followed by the results after watermark removal with increasing numbers of averaged images used to extract the watermark pattern (5, 10, 50, 100, 500, 5000). This visualizes the effect of the averaging-based attack on the different methods' robustness, showing that the visual impact of removal increases with the number of images averaged.


![](https://ai-paper-reviewer.com/X2G7LA7Av9/figures_20_1.jpg)

> This figure shows the performance of 12 different watermarking methods (6 content-agnostic and 6 content-adaptive) when their watermarks are removed using the proposed steganalysis-based method.  The performance is measured by AUC, Bit Accuracy, and Detection Accuracy.  The plots also show PSNR to indicate the visual quality after watermark removal.  The results demonstrate that content-agnostic watermarking methods are more vulnerable to the proposed attack than content-adaptive methods.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_21_1.jpg)
> This table presents the decoding accuracy of the RingID watermarking algorithm under various steganalysis-based removal attacks.  It shows the performance (decoding accuracy) with different numbers of images averaged during the pattern extraction process (n=5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000).  Additionally, it provides corresponding image quality metrics (PSNR, SSIM, LPIPS, SIFID) to assess the impact of watermark removal on image quality under both blackbox and greybox settings.  NRmv represents the performance without any removal attempt. The table highlights the trade-off between effective watermark removal and image quality degradation.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_21_2.jpg)
> This table presents the results of watermark removal experiments using a steganalysis-based approach.  It shows the performance (AUC, Bit Accuracy, Detection Accuracy) of several watermarking methods, both content-agnostic and content-adaptive, under different numbers of images used in the averaging process for watermark extraction. The impact of the removal on image quality (PSNR) is also shown, to demonstrate that successful removal is possible without large image distortions.  This helps understand the vulnerability of content-agnostic methods to steganalysis.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_21_3.jpg)
> This table presents the performance (bit accuracy) of the DwtDctSvd watermarking algorithm under the steganalysis-based removal attack.  It shows the bit accuracy, along with image quality metrics (PSNR, SSIM, LPIPS, and SIFID) for both blackbox and greybox settings. The number of images averaged (n) during pattern extraction varies from 5 to 5000, allowing an analysis of the trade-off between watermark removal effectiveness and image quality degradation.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_21_4.jpg)
> This table presents the performance of various watermark detectors after applying a steganalysis-based watermark removal technique.  The performance is measured using AUC, bit accuracy, and detection accuracy.  It also shows the Peak Signal-to-Noise Ratio (PSNR) to assess the impact of the removal on image quality.  The table categorizes watermarking methods as either content-agnostic or content-adaptive, highlighting the different responses to the removal technique.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_22_1.jpg)
> This table presents the performance of 12 watermarking methods (6 content-agnostic and 6 content-adaptive) under a steganalysis-based watermark removal attack.  For each method, it shows the AUC (Area Under the Curve) for watermark detection, bit accuracy, and detection accuracy. The impact of the attack on image quality is measured using PSNR (Peak Signal-to-Noise Ratio). The table compares the performance of content-agnostic and content-adaptive methods, demonstrating the vulnerability of the former to steganalysis.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_22_2.jpg)
> This table shows the performance of various watermark detectors (both content-agnostic and content-adaptive) under the steganalysis-based removal attack.  For each method, the Area Under the Curve (AUC) of the watermark detector, bit accuracy, and detection accuracy are presented.  PSNR (Peak Signal-to-Noise Ratio) is also included to measure the image quality degradation after the watermark removal. The number of images averaged (n) during the pattern extraction process is varied across the experiments.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_22_3.jpg)
> This table presents the performance (bit accuracy) of the Stable Signature (SSL) watermarking method under a steganalysis-based removal attack. It shows the bit accuracy, PSNR, SSIM, LPIPS, and SIFID values for both blackbox and greybox settings at different numbers of averaged images (n).  NRmv indicates the performance without removal. Lower bit accuracy suggests more successful watermark removal, while the image quality metrics measure the distortion introduced by the removal process.  Lower PSNR, SSIM indicate greater image degradation, while higher LPIPS, and SIFID indicate larger perceptual differences.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_22_4.jpg)
> This table presents the results of a Tree-Ring watermark's detection accuracy at a 1% false positive rate (FPR).  It shows the performance of watermark removal and forgery techniques under various conditions.  'NRmv' signifies that no watermark removal was attempted. The number of watermarked images averaged during pattern extraction, and the techniques of removal and forgery are indicated, illustrating the effect of averaging on watermark removal and forgery success.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_23_1.jpg)
> This table presents the results of applying steganalysis-based watermark removal on the DwtDctSvd watermarking method.  It shows the bit accuracy of watermark detection after removal, along with several image quality metrics (PSNR, SSIM, LPIPS, and SIFID). The data is separated into blackbox (no access to original clean images) and greybox (access to original clean images) settings, and shows the performance for different numbers of images averaged during the pattern extraction process.  The results illustrate the trade-off between watermark removal effectiveness and image quality degradation.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_23_2.jpg)
> This table presents the results of watermark removal experiments using steganalysis.  It shows the performance (AUC, Bit Accuracy, Detection Accuracy) of 12 different watermarking methods (6 content-agnostic and 6 content-adaptive) after applying a steganalysis-based removal technique.  The effect of the number of images averaged during pattern extraction (n) on both the watermark removal effectiveness and the resulting image quality (PSNR) is also demonstrated.  Content-agnostic methods show vulnerability to the attack while content-adaptive methods are robust.

![](https://ai-paper-reviewer.com/X2G7LA7Av9/tables_23_3.jpg)
> This table presents the performance of various watermarking methods under a steganalysis-based removal attack.  It shows the Area Under the Curve (AUC), bit accuracy, and detection accuracy of watermark detectors before and after applying the attack.  Additionally, the Peak Signal-to-Noise Ratio (PSNR) is shown, indicating the image quality degradation resulting from the attack. The table compares content-agnostic (vulnerable) and content-adaptive (robust) watermarking methods.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/X2G7LA7Av9/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}