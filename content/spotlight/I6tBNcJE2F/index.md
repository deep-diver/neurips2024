---
title: "Real-world Image Dehazing with Coherence-based Pseudo Labeling and Cooperative Unfolding Network"
summary: "CORUN-Colabator: a novel cooperative unfolding network and coherence-based label generator achieves state-of-the-art real-world image dehazing by effectively integrating physical knowledge and generat..."
categories: []
tags: ["Computer Vision", "Image Generation", "🏢 Tsinghua University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} I6tBNcJE2F {{< /keyword >}}
{{< keyword icon="writer" >}} Chengyu Fang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=I6tBNcJE2F" target="_self" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95790" target="_self" >}}
↗ NeurIPS Homepage
{{< /button >}}


<audio controls>
    <source src="https://ai-paper-reviewer.com/I6tBNcJE2F/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Real-world image dehazing is notoriously difficult due to complex haze distributions and limited paired training data. Existing deep learning methods struggle to accurately model real haze or generalize well from synthetic data.  This leads to color distortion, detail blurring, and poor performance on real-world images. 

The researchers tackle this challenge with two key innovations: 1) **CORUN**, a cooperative unfolding network that jointly models atmospheric scattering and image scenes to effectively restore image details. 2) **Colabator**, a coherence-based label generator that iteratively creates high-quality pseudo labels for training, addressing the lack of real-world paired data. Experiments demonstrate that CORUN with Colabator achieves state-of-the-art results in real-world image dehazing and improves the performance of downstream tasks.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel cooperative unfolding network (CORUN) effectively integrates physical knowledge into deep learning for improved real-world image dehazing. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A coherence-based label generator (Colabator) improves model generalizability by generating high-quality pseudo labels from real-world data. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The proposed method achieves state-of-the-art performance on benchmark datasets and enhances downstream tasks like object detection. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents a novel approach to real-world image dehazing, a challenging problem with significant real-world applications.  **The cooperative unfolding network and coherence-based label generator offer improvements in accuracy and generalization, advancing the state-of-the-art**. This work opens new avenues for research in image restoration and related fields, such as object detection and semantic segmentation, improving the performance of downstream tasks. The proposed framework's plug-and-play nature allows for integration with other dehazing methods, further enhancing their capabilities. 

------
#### Visual Insights



![](https://ai-paper-reviewer.com/I6tBNcJE2F/figures_0_1.jpg)

> This figure shows a comparison of the results obtained by several state-of-the-art image dehazing methods on a set of real-world hazy images.  The methods include DGUN, PDN, and CORUN. The results demonstrate that CORUN outperforms the other methods in terms of detail preservation and color correction, especially when combined with the Colabator framework. The '+' suffix indicates methods enhanced by the Colabator framework, showcasing improved generalization capabilities for haze removal and color correction.





![](https://ai-paper-reviewer.com/I6tBNcJE2F/tables_6_1.jpg)

> This table presents a quantitative comparison of the proposed method with several state-of-the-art dehazing techniques on the RTTS dataset.  The metrics used to evaluate the performance are FADE (lower is better), BRISQUE (lower is better), and NIMA (higher is better).  The table highlights the best and second-best performing methods for each metric in red and blue, respectively, showcasing the superiority of the proposed approach.





### In-depth insights


#### RID's Challenges
Real-world image dehazing (RID) presents significant challenges due to the **complex and unpredictable nature of real-world haze**. Unlike synthetic haze, real haze exhibits variations in density, color, and distribution, making it difficult to model accurately.  **Data scarcity** is another major hurdle; paired datasets of hazy and corresponding clear images are limited, hindering the training of robust deep learning models. The **lack of ground truth** for real-world hazy images further complicates evaluation, as determining the true 'clear' image remains subjective.  Additionally,  **generalization** to unseen haze conditions and scene types is crucial, but existing models often struggle to achieve it, exhibiting performance degradation on real-world images beyond the training distribution. Finally, **computational cost** remains a significant barrier; the intricate nature of RID requires high-capacity hardware and extensive training time, limiting accessibility for many researchers.

#### CORUN's Design
CORUN's design is a novel cooperative unfolding network architecture for real-world image dehazing.  It cleverly integrates physical knowledge from the atmospheric scattering model (ASM) into a deep learning framework.  This is achieved through a cooperative modeling approach involving transmission and scene gradient descent modules, which jointly model atmospheric scattering and the underlying image scene.  **The cooperative aspect allows for a more accurate and detailed representation of the hazy image and its constituent components**.  The iterative nature of the unfolding network mirrors the traditional ASM iterative solution, providing a natural link between physics-based models and data-driven deep learning. Further enhancing its effectiveness is the incorporation of coherence losses (global and local) **to ensure physically plausible outputs and minimize overfitting**. This focus on coherence is critical, particularly given the lack of large-scale paired training data in real-world dehazing scenarios.

#### Colabator: Label Gen
Colabator, a novel label generation framework, is designed to address the challenge of limited real-world paired data in real-world image dehazing (RID).  It leverages an iterative mean-teacher approach, where a teacher network generates high-quality pseudo-labels for training a student network. **Colabator's key strength lies in its ability to iteratively refine the label pool by incorporating only high-quality pseudo labels**, improving the overall label quality.  The selection process involves a compound image quality assessment, prioritizing labels with both global and local coherence, ensuring visually appealing and distortion-free results.  **This method greatly enhances the performance and generalization ability of the student network**.  Furthermore, **Colabator's dynamic label pool effectively addresses overfitting issues**, common with traditional pseudo-label methods.  The integration of Colabator with CORUN, a cooperative unfolding network, demonstrates state-of-the-art results in RID, highlighting its significant impact in improving network training and ultimately, dehazing performance.

#### Future RID Research
Future research in real-world image dehazing (RID) should prioritize addressing the limitations of current models in handling complex, real-world haze scenarios.  **Improving the accuracy of haze distribution modeling** is crucial, moving beyond simplified assumptions to capture the intricate variability of atmospheric scattering.  This requires **more robust datasets** that encompass a wider range of haze types, weather conditions, and scene complexities, ideally coupled with advanced techniques for data augmentation.  Another key area is **enhancing the generalization capabilities of models**, allowing them to adapt to unseen scenarios.  Investigating novel network architectures, incorporating physical priors effectively, and developing advanced regularization methods are promising avenues to explore. Furthermore, research should focus on the **integration of RID with other low-level vision tasks**, such as deblurring and denoising, to create a unified framework for tackling multiple image degradations.  Exploring the use of **multi-modal approaches** that incorporate information beyond RGB images, such as depth or spectral data, could significantly improve accuracy and robustness.  Finally, a deeper understanding of **perceptual quality assessment** is needed to better guide model development and evaluation, ensuring that generated images are not only technically superior but also visually appealing.

#### CORUN Limitations
The CORUN model, while demonstrating state-of-the-art performance in real-world image dehazing, exhibits limitations primarily concerning **severe haze density** and **texture detail preservation**.  Its struggles with extremely hazy images, resulting in low-quality texture details, highlight a need for further advancements in handling scenes where information is significantly obscured.  **Limited generalizability** to other image degradation types, such as blurring and low light, also restricts its applicability beyond haze removal.  **Overfitting** is another area of concern, indicating that the model's reliance on physical information might limit its capacity for learning complex, non-linear haze patterns.  Future work could focus on integrating additional modalities to improve robustness and expand its capabilities beyond haze removal.  Addressing these limitations through techniques like generative modeling and incorporating complementary information could improve overall performance.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/I6tBNcJE2F/figures_2_1.jpg)

> This figure illustrates the architecture of the Cooperative Unfolding Network (CORUN) proposed in the paper.  CORUN is a deep unfolding network that cooperatively models atmospheric scattering and image scenes. The figure shows the network's structure, including the Transmission and Scene Gradient Descent Modules (TGDM and SGDM) and the Cooperative Proximal Mapping Modules (T-CPMM and S-CPMM) at each stage.  The details of TGDM and SGDM are also shown, illustrating how they iteratively refine the transmission map and dehazed image.  The figure uses various symbols to represent different operations within the network, such as element-wise summation, subtraction, multiplication, dot product, etc. This visual representation helps readers understand how CORUN works.


![](https://ai-paper-reviewer.com/I6tBNcJE2F/figures_4_1.jpg)

> This figure illustrates the pipeline of the proposed Coherence-based label generator (Colabator). It shows a two-stage process: pre-training and fine-tuning.  During pre-training, a dehazing network is trained on synthetic hazy and ground truth images.  Fine-tuning utilizes a mean-teacher framework where a teacher network generates pseudo-labels for real-world hazy images. These labels are then used to train a student network.  Colabator leverages a combination of strong and weak augmentations, a CLIP-based image quality assessment, and an optimal label pool to generate high-quality pseudo-labels, addressing the scarcity of real-world paired data for training.


![](https://ai-paper-reviewer.com/I6tBNcJE2F/figures_6_1.jpg)

> This figure compares the performance of several state-of-the-art real-world image dehazing methods on the RTTS dataset.  Each row shows a hazy input image followed by the dehazed results from PDN, DAD, PSD, D4, DGUN, RIDCP, and the proposed CORUN method.  The comparison highlights the differences in how well each method handles various types of haze, detail preservation, and color accuracy.


![](https://ai-paper-reviewer.com/I6tBNcJE2F/figures_7_1.jpg)

> This figure compares the performance of various state-of-the-art dehazing methods on the RTTS dataset.  It shows the original hazy image alongside the results produced by each method, allowing for a visual comparison of their effectiveness in terms of haze removal, color correction, and detail preservation. Zooming in is recommended to better appreciate the differences in detail.


![](https://ai-paper-reviewer.com/I6tBNcJE2F/figures_8_1.jpg)

> This figure compares the results of various state-of-the-art image dehazing methods on the RTTS dataset.  Each row represents a different hazy image and its dehazed counterparts from several methods, showing the performance of each method in restoring details and colors. The 'Ours' column shows the results obtained by the proposed method (CORUN) in the paper, highlighting its improved detail restoration and color correction compared to other methods.


![](https://ai-paper-reviewer.com/I6tBNcJE2F/figures_9_1.jpg)

> This figure compares the results of different dehazing methods on the RTTS dataset.  Each row shows a hazy input image and the results from various methods: PDN, DAD, PSD, D4, DGUN, RIDCP, and the proposed method (Ours).  The results show that the proposed method generally produces more visually appealing and detailed outputs, preserving finer details and colours more faithfully than other methods, particularly in complex scenes.


![](https://ai-paper-reviewer.com/I6tBNcJE2F/figures_16_1.jpg)

> This figure compares the results of several state-of-the-art real-world image dehazing methods on the RTTS dataset.  It shows a series of hazy images and their corresponding dehazed versions produced by different methods.  The goal is to illustrate the visual quality improvements achieved by the proposed CORUN method. The highlighted regions in the image comparison call attention to the differences between the different methods, such as the level of detail preserved, color accuracy, and the removal of haze.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/I6tBNcJE2F/tables_7_1.jpg)
> This table presents the quantitative results of the proposed Colabator framework on the RTTS dataset. It demonstrates the improvement achieved by incorporating Colabator into two different dehazing networks: DGUN and CORUN. The metrics used for evaluation include FADE, BRISQUE, and NIMA.  The results highlight the effectiveness of Colabator in enhancing the generalizability and performance of both networks, especially in terms of reducing haze artifacts and improving overall image quality.

![](https://ai-paper-reviewer.com/I6tBNcJE2F/tables_7_2.jpg)
> This table presents a quantitative comparison of different image dehazing methods on the RTTS dataset.  Three metrics are used for evaluation: FADE (lower is better), BRISQUE (lower is better), and NIMA (higher is better).  The table shows the performance of various methods, including the proposed method, highlighting the best and second-best results for each metric.

![](https://ai-paper-reviewer.com/I6tBNcJE2F/tables_8_1.jpg)
> This table presents a quantitative comparison of the proposed method against other state-of-the-art dehazing methods on the RTTS dataset.  The metrics used for comparison are FADE, BRISQUE, and NIMA.  Lower values for FADE and BRISQUE indicate better performance, while a higher value for NIMA indicates better performance. Red and blue highlight the top two performing methods for each metric.

![](https://ai-paper-reviewer.com/I6tBNcJE2F/tables_17_1.jpg)
> This table presents a quantitative comparison of the proposed method (Ours) against several state-of-the-art dehazing methods on the RTTS dataset.  Metrics used for comparison include FADE (Haze Density), BRISQUE (Image Quality), and NIMA (Aesthetic Quality). The best and second-best results for each metric are highlighted in red and blue, respectively. This table demonstrates the superior performance of the proposed method in terms of both quantitative metrics and visual quality.

![](https://ai-paper-reviewer.com/I6tBNcJE2F/tables_17_2.jpg)
> This table presents a quantitative comparison of the proposed method (CORUN) against other state-of-the-art real-world image dehazing methods on the RTTS dataset.  Three metrics are used for evaluation: FADE (haze density), BRISQUE (image quality), and NIMA (aesthetic quality).  Lower FADE and BRISQUE scores and a higher NIMA score indicate better performance.  The best and second-best results for each metric are highlighted in red and blue, respectively. This demonstrates the superior quantitative performance of CORUN.

![](https://ai-paper-reviewer.com/I6tBNcJE2F/tables_17_3.jpg)
> This table presents a quantitative comparison of the proposed method with several state-of-the-art dehazing methods on the RTTS dataset.  The metrics used for evaluation are: FADE (lower is better), BRISQUE (lower is better), and NIMA (higher is better).  The table highlights the superior performance of the proposed method by indicating the best and second-best results in red and blue, respectively. This demonstrates the effectiveness of the proposed approach in achieving better overall dehazing quality compared to other methods.

![](https://ai-paper-reviewer.com/I6tBNcJE2F/tables_17_4.jpg)
> This table presents a quantitative comparison of the proposed method (Ours) against several state-of-the-art real-world image dehazing methods on the RTTS dataset.  The comparison uses three metrics: FADE (haze density), BRISQUE (image quality), and NIMA (aesthetic quality). Lower scores for FADE and BRISQUE are better, while higher scores for NIMA are better. The table highlights that the proposed method achieves the best performance according to all three metrics, outperforming the second-best method significantly.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/I6tBNcJE2F/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}