---
title: "Virtual Scanning: Unsupervised Non-line-of-sight Imaging from Irregularly Undersampled Transients"
summary: "Unsupervised learning framework enables high-fidelity non-line-of-sight (NLOS) imaging from irregularly undersampled transients, surpassing state-of-the-art methods in speed and robustness."
categories: []
tags: ["Computer Vision", "Image Generation", "🏢 Tianjin University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} R4IBZrSF5d {{< /keyword >}}
{{< keyword icon="writer" >}} Xingyu Cui et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=R4IBZrSF5d" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95200" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=R4IBZrSF5d&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/R4IBZrSF5d/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Non-line-of-sight (NLOS) imaging aims to reconstruct hidden scenes using light reflected from an intermediary surface. Current methods often require dense, regularly sampled data from this surface, limiting their applicability to real-world scenarios with complex surfaces. This constraint is addressed in this paper. 

The paper introduces a novel unsupervised learning-based framework that leverages irregularly undersampled transients.  **It employs a virtual scanning process to address the inherent ambiguity in reconstructing the albedo volume and uses a SURE-based denoiser for robustness to noise.** This method significantly improves reconstruction quality and speed while greatly reducing the need for large, precisely-aligned datasets.  Extensive experiments on simulated and real-world data demonstrate its superior performance and generalization capabilities compared to existing techniques.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} An unsupervised learning-based framework for NLOS imaging from irregularly undersampled transients (IUT) is proposed, eliminating the need for expensive paired data. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A virtual scanning process and SURE-based denoiser significantly improve the robustness and quality of NLOS image reconstruction. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The proposed method achieves superior reconstruction fidelity and speed compared to state-of-the-art techniques, validated through extensive experiments. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is significant because it presents a novel unsupervised learning framework for non-line-of-sight (NLOS) imaging, a field crucial for various applications like autonomous driving and robotics.  **It overcomes the limitations of existing methods that rely on dense, regularly sampled data by effectively reconstructing hidden scenes from irregularly undersampled transients.** This advance is crucial for real-world scenarios with complex, unstructured environments. The paper's proposed virtual scanning process enhances reconstruction accuracy, and a SURE-based denoiser improves robustness to noise. These contributions could greatly enhance the practicality and performance of NLOS imaging systems.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_1_1.jpg)

> The figure illustrates an active confocal NLOS (non-line-of-sight) imaging system with an irregular relay surface.  A laser and detector are focused on a relay surface, pulses emitted by the laser reflect off the surface to illuminate a hidden scene. The detector captures photons bouncing back from the scene towards the relay surface, as transients.  The yellow points on the relay surface indicate scannable points used to collect the transient data. Blue points indicate unscannable areas on the irregular surface.  An example of undersampled transient data is also displayed (left).





![](https://ai-paper-reviewer.com/R4IBZrSF5d/tables_8_1.jpg)

> This table compares the inference time of different NLOS reconstruction methods (LCT, FK, RSD, NeTF, USM, CC-SOCR, and the proposed method) using both CPU and GPU.  The inference time is averaged across 16 irregularly undersampled transients (IUT) with a size of 128x128x256.  The results demonstrate that the proposed method achieves significantly faster inference time compared to other methods, particularly CC-SOCR which requires significantly longer time.





### In-depth insights


#### NLOS Imaging IUT
NLOS (Non-Line-of-Sight) imaging aims to reconstruct hidden scenes by using indirect measurements.  The IUT (Irregularly Undersampled Transients) approach presents a significant challenge to traditional NLOS methods, which typically assume dense and regularly sampled data. **IUT leads to significant ill-posedness**, making accurate reconstruction difficult.  This necessitates novel techniques that can handle incomplete and irregularly distributed data, which is a major advance in the field.  Existing supervised learning-based solutions require substantial paired datasets of ground truth and measurements, which are expensive and difficult to obtain.  **Unsupervised learning offers a compelling alternative**, as it can learn implicit priors directly from the data without the need for explicit ground truth.  However, unsupervised methods face ambiguity in inferring albedo volumes.  **Advanced techniques like virtual scanning are needed to overcome such ambiguity and improve reconstruction quality.**  Overall, IUT is a crucial and challenging area in NLOS imaging, requiring innovative methodologies, especially in the context of unsupervised learning.

#### Virtual Scan Method
The core idea behind the virtual scan method is to **mitigate the limitations of traditional NLOS imaging**, which often requires dense and regular scans across a large relay surface.  This method cleverly addresses the challenges of irregularly undersampled transients (IUT) by **virtually simulating scans** on a complete relay surface, even when only sparse real measurements are available. This virtual process leverages a neural network trained on complete data, allowing the network to learn both range and null space components, which are crucial for high-quality reconstruction in NLOS imaging. In essence, the virtual scan method **acts as a powerful regularization technique** to overcome ill-posedness and ensure accurate albedo reconstruction from limited measurements. It is a significant step towards more robust and practical NLOS imaging in real-world settings, enabling applications where acquiring dense data is difficult or impossible.

#### SURE Denoising
The SURE (Stein's Unbiased Risk Estimator) denoising technique offers a powerful approach to noise reduction in low-light imaging scenarios, particularly valuable in non-line-of-sight (NLOS) imaging where photon counts are often limited.  **It leverages a data-driven approach, learning implicit priors from the noisy data itself without requiring paired clean/noisy data**, a significant advantage when such pairs are difficult to acquire. By minimizing the SURE loss function, the denoiser enhances the robustness of the NLOS reconstruction process.  **Crucially, SURE denoising incorporates a physics-guided component**, modeling the low-photon time-resolved detector's characteristics, resulting in improved noise reduction in the presence of Poisson noise and dark counts commonly found in such systems. This results in cleaner, higher fidelity reconstructions, improving the overall quality of the NLOS image and reducing artifacts resulting from noise.

#### Unsupervised Learning
Unsupervised learning in the context of non-line-of-sight (NLOS) imaging addresses the challenge of reconstructing hidden scenes from limited and noisy data without relying on paired training data.  **This is crucial because obtaining perfectly aligned ground truth albedo volumes for training is often difficult and expensive.**  The approach leverages the inherent structure in the NLOS measurement process to learn implicit priors. By training on irregularly undersampled transients, the model learns to reconstruct 3D albedo volumes using a virtual scanning technique, effectively learning from both range and null spaces.  **This allows for reconstruction of high-quality images even with incomplete data, while addressing the limitations of supervised methods reliant on large paired datasets.** This unsupervised technique demonstrates impressive robustness and generalizability across various relay surfaces and real-world scenarios.  **The incorporation of a physics-guided denoiser further enhances robustness to noise common in low-photon imaging.** Overall, the approach offers a promising avenue for more practical and scalable NLOS imaging systems by drastically reducing the dependence on painstaking data acquisition and annotation.

#### Future NLOS Research
Future research in non-line-of-sight (NLOS) imaging should prioritize addressing the limitations of current techniques.  **Improving robustness to noise and variations in real-world environments** is crucial, particularly in low-light conditions.  This could involve developing more sophisticated denoising algorithms or exploring alternative sensing modalities.  **Addressing the computational cost** of existing methods is also critical for practical applications.  This might be accomplished by developing faster inference algorithms or leveraging more efficient hardware.  **Extending NLOS imaging to more complex scenarios**, such as imaging through scattering media or dynamic environments, is another significant area for future work.  Furthermore, the development of **unsupervised or weakly-supervised learning approaches** would significantly reduce the reliance on large, labeled datasets which are currently difficult and expensive to obtain.  Finally, exploring new applications and **integrating NLOS imaging with other sensing modalities** (such as LiDAR or radar) could open up many new opportunities and capabilities.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_3_1.jpg)

> This figure illustrates the pipeline of the unsupervised framework proposed in the paper.  It shows the SURE-based denoiser (a), which is a pre-processing step to reduce noise in the input. The virtual scanning reconstruction network (VSRNet) is the main component (b), which reconstructs the 3D albedo volume. Relay surfaces used for training are shown in (c), highlighting the irregular nature of the surfaces and how scanning points are selected. The virtual scanning process (d) is a key part of the proposed method, creating virtual scans to improve reconstruction by accounting for the null space (areas of the image that cannot be directly observed).


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_4_1.jpg)

> This figure uses a toy example to illustrate the concept of range-null space decomposition in non-line-of-sight (NLOS) reconstruction and how the proposed virtual scanning method enhances the reconstruction by recovering null-space components.  (a) shows the decomposition of the true albedo volume (ρ) into range-space (Dr(ρ)) and null-space (Dn(ρ)) components using the first observation operator (H1). (b) demonstrates that using only the measurement consistency loss, the reconstruction may not accurately capture the true albedo due to the ambiguity in the null space. (c) shows that by introducing a virtual scanning process, the network can learn from both range and null spaces, leading to a more accurate reconstruction.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_6_1.jpg)

> This figure compares the reconstruction results of the 'bunny' model using different NLOS methods under various irregularly undersampled relay surfaces.  The results demonstrate the effectiveness of the proposed method in recovering geometric structures and achieving the highest PSNR (Peak Signal-to-Noise Ratio) values among compared methods.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_7_1.jpg)

> This figure displays the reconstruction results of several different methods on publicly available real-world datasets using various irregular relay surfaces. Each row represents a different dataset with different relay surface patterns. The columns represent different reconstruction methods, including LCT, FK, RSD, NeTF, USM, CC-SOCR, and the proposed method in the paper. The ground truth images of each scene are shown in the last column, offering a visual comparison of the performance of each algorithm.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_8_1.jpg)

> This figure compares the reconstruction results of different methods on a 'bunny' shaped object, using various irregularly sampled relay surfaces.  The intensity of each reconstruction is normalized.  PSNR (Peak Signal-to-Noise Ratio) values are provided for quantitative comparison of each method's reconstruction quality against the ground truth image.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_9_1.jpg)

> This figure presents ablation studies comparing the performance of the proposed method with and without the virtual scanning process and the SURE-based denoiser.  Subfigure (a) shows reconstructions using two different relay surfaces (Window 6 and Window 4) comparing the proposed method (Ours) with and without virtual scanning (VS).  It demonstrates that the virtual scanning process significantly improves reconstruction quality. Subfigure (b) shows reconstructions using the same relay surfaces, comparing the proposed method (Ours) with and without the SURE-based denoiser.  This subfigure highlights the impact of the denoiser in improving the robustness of the method to noise.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_13_1.jpg)

> This figure shows the architecture of the P-Unet network used for denoising irregularly undersampled transients. The network is based on 3D partial convolution, which uses a sampling matrix to differentiate between valid and invalid voxels, effectively capturing spatial information from the transients. The network also includes instance normalization layers, which are insensitive to input distribution, to enhance generalization. The network has an encoder-decoder structure with skip connections, and the kernel sizes of the 3D convolutions are indicated by the blue arrows.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_14_1.jpg)

> This figure demonstrates the reconstruction results of a 'bunny' model using different relay surface patterns.  The intensity of each reconstructed image is normalized from 0 to 1 for easier comparison, and the peak signal-to-noise ratio (PSNR) in decibels (dB) is provided as a quantitative evaluation metric for each reconstruction. This shows the impact of the relay surface on the quality of the NLOS reconstruction.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_16_1.jpg)

> The figure shows the experimental setup of the NLOS imaging system and reconstruction results with different sampling rates.  (a) is a top view of the setup, illustrating the laser source, detector, relay surface, and the hidden object with different light paths shown. (b) shows the maximum intensity projections of the transient signals and the corresponding reconstructions from the proposed method, highlighting the effect of different sampling rates on reconstruction quality.  The green boxes highlight the improved reconstruction quality using the proposed method in comparison to the direct measurement.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_18_1.jpg)

> This figure presents the reconstruction results from three publicly available real-world datasets ([10-12]) using different methods, including LCT, FK, RSD, NeTF, USM, CC-SOCR, and the proposed method. Each row represents a different relay surface pattern, allowing for a comparison of the methods' performance under various scenarios. The ground truth images are provided for reference. The results show that the proposed method achieves superior reconstruction quality compared to other methods across diverse relay surface patterns.


![](https://ai-paper-reviewer.com/R4IBZrSF5d/figures_19_1.jpg)

> This figure shows the reconstruction results of several real-world datasets ([10-12]) using different irregular relay surface patterns.  Each row represents a different dataset and relay surface pattern, comparing different NLOS reconstruction methods (LCT, FK, RSD, NeTF, USM, CC-SOCR, and the proposed method). The ground truth images are shown on the far right for comparison. This figure demonstrates the performance of each method across various real-world scenarios with irregularly sampled data.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/R4IBZrSF5d/tables_17_1.jpg)
> This table shows the results of an ablation study on the hyperparameter epsilon (ɛ) used in the SURE-based denoiser.  Different values of ɛ were tested, and the resulting Peak Signal-to-Noise Ratio (PSNR) and its variance are reported.  The PSNR measures the quality of the denoised images, with higher values indicating better denoising performance. The variance indicates the consistency of the results for each value of ɛ.

![](https://ai-paper-reviewer.com/R4IBZrSF5d/tables_17_2.jpg)
> This table shows the results of an ablation study on the hyperparameter β, which controls the balance between the measurement consistency loss and the virtual scanning loss in the proposed unsupervised NLOS imaging framework.  The PSNR (Peak Signal-to-Noise Ratio) is used to measure the quality of the 3D albedo volume reconstruction.  Different values of β were tested, and the table shows the mean PSNR and its variance for each value. The optimal value of β is determined based on maximizing PSNR.

![](https://ai-paper-reviewer.com/R4IBZrSF5d/tables_17_3.jpg)
> This table presents the results of an ablation study on the impact of varying the intervals of relay surfaces used during training on the performance of the proposed NLOS imaging method. The performance metric used is the Peak Signal-to-Noise Ratio (PSNR), calculated as the average and standard deviation across multiple trials. The table shows that using a wider range of intervals ([4, 8, 12, 16, 20]) during training leads to better reconstruction performance (higher average PSNR) compared to using smaller or fewer intervals.

![](https://ai-paper-reviewer.com/R4IBZrSF5d/tables_18_1.jpg)
> This table shows the results of an ablation study on the impact of the number of rotations of relay surfaces during training on the reconstruction performance, measured by PSNR.  The study varied the number of rotations (10, 20, 30, 40, 50, 60) and measured the mean and variance of the PSNR for each rotation condition. The results indicate an optimal number of rotations around 40, beyond which the performance starts to decrease.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/R4IBZrSF5d/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}