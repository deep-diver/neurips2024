---
title: "DarkSAM: Fooling Segment Anything Model to Segment Nothing"
summary: "DarkSAM, a novel prompt-free attack, renders the Segment Anything Model incapable of segmenting objects across diverse images, highlighting its vulnerability to universal adversarial perturbations."
categories: ["AI Generated", ]
tags: ["Computer Vision", "Image Segmentation", "🏢 Huazhong University of Science and Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 0o9E8AsFgW {{< /keyword >}}
{{< keyword icon="writer" >}} Ziqi Zhou et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=0o9E8AsFgW" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/0o9E8AsFgW" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/0o9E8AsFgW/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

The Segment Anything Model (SAM) is a powerful image segmentation model known for its impressive generalization capabilities. However, its vulnerability to adversarial attacks has not been thoroughly investigated.  This paper focuses on this issue, highlighting the need for more robust segmentation models in real-world applications where adversarial attacks are a concern.  Existing adversarial attacks are often tailored to specific inputs or rely on labels, which are not applicable to SAM's label-free mask output.



To address this, the researchers developed DarkSAM, a novel, prompt-free universal adversarial attack.  DarkSAM cleverly employs both spatial and frequency domain manipulation to disrupt crucial object features, effectively fooling SAM into "segmenting nothing." The method demonstrated high success rates and transferability across various datasets, significantly impacting the reliability of SAM.  The findings emphasize the importance of considering adversarial robustness when designing and deploying image segmentation models.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} DarkSAM is the first prompt-free universal attack framework against the Segment Anything Model (SAM). {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} DarkSAM uses a hybrid spatial-frequency attack to effectively fool SAM by disrupting crucial object features in both spatial and frequency domains. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Experimental results demonstrate DarkSAM's strong attack capability and transferability across different datasets and SAM variants. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because **it reveals a critical vulnerability in the widely used Segment Anything Model (SAM)**.  This vulnerability could have significant implications for various applications that rely on SAM for accurate image segmentation, potentially opening up new avenues for adversarial attacks and necessitating the development of more robust models.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_1_1.jpg)

> 🔼 This figure illustrates the process of attacking the Segment Anything Model (SAM) using a Universal Adversarial Perturbation (UAP).  A benign example image is shown, which is then modified by adding a UAP, creating an adversarial example. This adversarial example is fed into SAM, along with a prompt, and the output is 'Segment Nothing', a black mask indicating the failure to segment any objects. This shows the effectiveness of the UAP in fooling the SAM model.
> <details>
> <summary>read the caption</summary>
> Figure 1: Illustration of fooling SAM using UAP
> </details>





![](https://ai-paper-reviewer.com/0o9E8AsFgW/tables_6_1.jpg)

> 🔼 This table presents the mean Intersection over Union (mIoU) scores achieved by DarkSAM under various settings.  It compares the performance of the Segment Anything Model (SAM) and its variants (HQ-SAM, PerSAM) on four different datasets (ADE20K, MS-COCO, Cityscapes, SA-1B) using three different prompt types (point, box, and segment everything).  The gray shaded cells show the clean (unattacked) mIoU scores, while the other cells show the mIoU scores after applying DarkSAM's universal adversarial perturbation (UAP). The bold values highlight the best performance in each case.
> <details>
> <summary>read the caption</summary>
> Table 1: The mIoU (%) of DarkSAM under different settings. Values covered by gray denote the clean mIoU, others denote adversarial mIoU. ADE20K, MS-COCO, CITYSCAPES abbreviated as ADE, COCO, CITY, respectively. Bolded values indicate the best results.
> </details>





### In-depth insights


#### DarkSAM: Universal Attack
The conceptual heading 'DarkSAM: Universal Attack' suggests a research focus on creating a robust and broadly applicable adversarial attack against the Segment Anything Model (SAM).  A universal attack, unlike targeted attacks, aims to fool SAM regardless of the specific image or prompt.  This implies DarkSAM likely uses a single, carefully designed perturbation to consistently disrupt SAM's segmentation capabilities across diverse inputs. The research likely explores the vulnerabilities of SAM, potentially highlighting its susceptibility to malicious manipulation.  **The 'Dark' element might indicate the attack's stealth or effectiveness in circumventing SAM's defenses.** The success of a universal attack would underscore significant security concerns for SAM and similar models, especially in applications where reliable and trustworthy segmentation is crucial, such as autonomous vehicles or medical imaging. **DarkSAM's universality would also imply a high level of transferability**, meaning that the attack's effectiveness likely generalizes to various input images and other variations of the SAM model.  The study probably includes extensive experimental evaluations demonstrating DarkSAM's effectiveness across different datasets and prompts.  **The details of the attack method itself likely represent a novel approach**, potentially employing strategies from both spatial and frequency domains to generate a potent adversarial perturbation.

#### Hybrid Attack Strategy
A hybrid attack strategy, in the context of adversarial machine learning, likely combines multiple attack methods to enhance effectiveness against a target model.  This approach recognizes that a single attack vector might be insufficient to overcome a robust model's defenses.  **Combining spatial and frequency domain manipulations**, for example, could exploit vulnerabilities in both the image's semantic content and its texture information, leading to a more potent and successful attack. The strength lies in the synergy; one attack method could create weaknesses that are then exploited by another. This **multi-pronged approach** increases the chances of bypassing multiple layers of defense mechanisms and achieving a higher attack success rate.  Furthermore, a hybrid strategy might leverage **transferability**, where an attack developed for one dataset or model effectively generalizes to others.  **Prompt-free** attacks, for instance, are particularly effective as they don't rely on specific user inputs, increasing the attack's applicability in real-world scenarios where user interactions are unpredictable.  The development and evaluation of such hybrid attacks require sophisticated techniques that analyze different attack aspects and measure overall impact across various datasets and models.

#### SAM Vulnerability
The Segment Anything Model (SAM) demonstrates impressive generalization capabilities, but its vulnerability to adversarial attacks remains a crucial concern.  A key vulnerability lies in SAM's reliance on both image features and prompts.  **Adversarial attacks can exploit this dual dependency**, manipulating either the image or the prompt (or both) to cause SAM to misinterpret the input and fail to segment objects correctly.  **Universal adversarial perturbations (UAPs)** are especially concerning, as these single perturbations can fool SAM across a range of images and prompts, highlighting a significant weakness in its robustness.  The research suggests that understanding the ways in which spatial and frequency features are processed within SAM is vital to developing effective countermeasures.  **Spatial attacks** disrupt object semantics, while **frequency attacks** target high-frequency components (texture information), further deceiving the model.  **Defenses** against these attacks should focus on robust feature extraction and prompt processing, as simple image pre-processing might be insufficient to mitigate the sophisticated adversarial manipulations employed.

#### Transferability Study
A transferability study in the context of adversarial attacks on a model like Segment Anything Model (SAM) would explore how well an attack designed for one set of images generalizes to other, unseen images.  **High transferability** suggests a robust and effective attack, potentially highlighting vulnerabilities in the model's core architecture. Researchers would test the attack's success rate on various datasets, image types, and perhaps even against different versions or variants of the same model.  **Factors influencing transferability** could include the diversity of datasets (natural images vs. medical scans), the types of prompts used (points, boxes, masks), and whether the adversarial perturbation is carefully tailored to specific images or applies broadly.  The study should also compare against other adversarial attack methods to determine DarkSAM's relative efficacy and robustness. **A successful transferability study** could demonstrate that the attack's impact isn't limited to specific data sets but reveals broader vulnerabilities, possibly implying a need for improved model robustness or more sophisticated defense mechanisms.

#### Ablation Experiments
Ablation experiments systematically remove components of a model to assess their individual contributions.  In this context, it would involve removing or deactivating parts of the proposed attack framework (e.g., spatial attack, frequency attack, or specific components within them) to determine their relative importance in the overall attack success.  **Analyzing the results allows researchers to understand the relative importance of each component**, pinpointing crucial aspects and highlighting potential weaknesses.  **The results could indicate if one component is significantly more effective than others, or if all are necessary for optimal performance.**  A thoughtful analysis would also consider potential interactions between components, such as whether the frequency attack enhances the spatial attack or if they are independent.  Such a detailed investigation would allow for **a more refined and targeted design** of future attack strategies by clarifying which features must be preserved or improved for optimal impact, and which might be redundant or less impactful.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_3_1.jpg)

> 🔼 This figure illustrates the shadow target strategy used in DarkSAM.  It shows how multiple prompts are used on benign example images to obtain multiple segmentation masks from SAM. These individual masks are then merged to create a single 'blueprint' mask that represents the key semantic features of the image. This blueprint mask is then used as the attack target for generating the universal adversarial perturbation (UAP). This addresses the challenge of dealing with varying inputs (both images and prompts) in the prompt-guided image segmentation model SAM by creating a stable and consistent target for the attack, irrespective of the actual prompt used.
> <details>
> <summary>read the caption</summary>
> Figure 2: Illustration of the proposed shadow target strategy
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_4_1.jpg)

> 🔼 This figure illustrates the DarkSAM framework, which consists of a spatial attack and a frequency attack. The spatial attack involves semantically decoupling the foreground and background of an image and then scrambling SAM's decision by destroying the features of the foreground and background. The frequency attack enhances the attack effectiveness by distorting the high-frequency components (texture information) of the image while maintaining consistency in their low-frequency components (shape information).  Both attacks use a single UAP to mislead SAM into incorrectly segmenting images, regardless of the prompts.
> <details>
> <summary>read the caption</summary>
> Figure 3: The framework of DarkSAM
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_7_1.jpg)

> 🔼 This figure shows the qualitative results of applying DarkSAM attacks to the Segment Anything Model (SAM). It compares the segmentation results of benign images with those of adversarial examples created using point and box prompts, under three different prompting modes (point, box, and segment everything). The results are shown for four different datasets: ADE20K, MS-COCO, Cityscapes, and SA-1B. The figure demonstrates how DarkSAM successfully fools SAM into producing incorrect segmentations.
> <details>
> <summary>read the caption</summary>
> Figure 4: Visualizations of SAM segmentation results for adversarial examples across four datasets. The first four columns and the middle four columns display the segmentation results for point and box prompts, respectively. The last three columns show results under the segment everything mode for benign examples, as well as adversarial examples created using point and box prompts, respectively.
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_8_1.jpg)

> 🔼 This figure illustrates the framework of DarkSAM, a novel prompt-free hybrid spatial-frequency universal adversarial attack against prompt-guided image segmentation models. It shows how DarkSAM works in both the spatial and frequency domains. In the spatial domain, DarkSAM divides the SAM output into foreground and background, then scrambles SAM's decision by destroying the features of both foreground and background. In the frequency domain, it enhances attack effectiveness by distorting the high-frequency components of the image while maintaining consistency in the low-frequency components.  The combined spatial and frequency attacks aim to fool SAM into not segmenting any objects.
> <details>
> <summary>read the caption</summary>
> Figure 3: The framework of DarkSAM
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_9_1.jpg)

> 🔼 This figure compares the performance of DarkSAM against other UAP methods (UAP, UAPGD, SSP, SegPGD, and Attack-SAM) and various prompt types (point, box, and all).  The visualizations show the segmentation masks generated by SAM for benign and adversarial examples.  The plots in (a) to (e) show the ablation study results, demonstrating the effectiveness of different modules, varying prompt numbers, perturbation budget, and thresholds on the attack performance.
> <details>
> <summary>read the caption</summary>
> Figure 6: Visualizations of the comparison study
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_14_1.jpg)

> 🔼 This figure shows the qualitative results of the DarkSAM attack against the SAM model on four datasets.  The top row displays the original images with point prompts used for the segmentation. The middle row shows the segmentation masks produced by SAM from the original, unmodified images.  The bottom row shows the segmentation masks produced by SAM after the DarkSAM attack has been applied. The results demonstrate the effectiveness of the DarkSAM attack in preventing the SAM model from correctly segmenting objects in the images across various prompt types.
> <details>
> <summary>read the caption</summary>
> Figure 4: Visualizations of SAM segmentation results for adversarial examples across four datasets. The first four columns and the middle four columns display the segmentation results for point and box prompts, respectively. The last three columns show results under the segment everything mode for benign examples, as well as adversarial examples created using point and box prompts, respectively.
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_15_1.jpg)

> 🔼 This figure shows the results of applying the Segment Anything Model (SAM) to both benign and adversarial examples from four different datasets.  The top row shows the original images and the bottom two rows show the segmentation masks generated by SAM for the benign images (middle row) and the adversarial images (bottom row) using point and box prompts. The figure demonstrates the effectiveness of DarkSAM in fooling SAM into producing incorrect segmentations.
> <details>
> <summary>read the caption</summary>
> Figure 4: Visualizations of SAM segmentation results for adversarial examples across four datasets. The first four columns and the middle four columns display the segmentation results for point and box prompts, respectively. The last three columns show results under the segment everything mode for benign examples, as well as adversarial examples created using point and box prompts, respectively.
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_15_2.jpg)

> 🔼 This figure shows the results of applying the SAM model to both benign and adversarial examples from four datasets. It compares the segmentation results obtained using point and box prompts, and also shows the results for the 'segment everything' mode, which is a special prompt that aims to segment the entire image.  The comparison highlights how the adversarial examples are able to fool SAM into producing incorrect segmentation masks, illustrating the effectiveness of the DarkSAM attack.
> <details>
> <summary>read the caption</summary>
> Figure 4: Visualizations of SAM segmentation results for adversarial examples across four datasets. The first four columns and the middle four columns display the segmentation results for point and box prompts, respectively. The last three columns show results under the segment everything mode for benign examples, as well as adversarial examples created using point and box prompts, respectively.
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_16_1.jpg)

> 🔼 This figure shows a qualitative comparison of SAM's segmentation results on benign and adversarial examples from four different datasets (ADE20K, COCO, Cityscapes, SA-1B).  Three prompt types are used (point, box, and 'segment everything').  The goal is to visually demonstrate DarkSAM's effectiveness in fooling SAM into producing incorrect segmentations regardless of the prompt type.
> <details>
> <summary>read the caption</summary>
> Figure 4: Visualizations of SAM segmentation results for adversarial examples across four datasets. The first four columns and the middle four columns display the segmentation results for point and box prompts, respectively. The last three columns show results under the segment everything mode for benign examples, as well as adversarial examples created using point and box prompts, respectively.
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_16_2.jpg)

> 🔼 This figure illustrates the DarkSAM framework, which consists of two main attack components: a semantic decoupling-based spatial attack and a texture distortion-based frequency attack.  The spatial attack manipulates the foreground and background features of an image to mislead the SAM model. The frequency attack distorts high-frequency components while preserving low-frequency components, aiming to further confuse the model's segmentation capabilities.  The combined effect of both attacks renders SAM incapable of performing object segmentation across a range of inputs.
> <details>
> <summary>read the caption</summary>
> Figure 3: The framework of DarkSAM
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_16_3.jpg)

> 🔼 This figure shows a qualitative comparison of SAM's segmentation performance on benign and adversarial examples across four datasets (ADE20K, MS-COCO, Cityscapes, SA-1B) and three prompt types (point, box, and 'segment everything').  The results visually demonstrate the effectiveness of DarkSAM in fooling SAM, causing it to fail to segment the objects in the adversarial examples. The 'segment everything' mode shows how DarkSAM affects the model's ability to segment everything within the image.
> <details>
> <summary>read the caption</summary>
> Figure 4: Visualizations of SAM segmentation results for adversarial examples across four datasets. The first four columns and the middle four columns display the segmentation results for point and box prompts, respectively. The last three columns show results under the segment everything mode for benign examples, as well as adversarial examples created using point and box prompts, respectively.
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_17_1.jpg)

> 🔼 This figure shows a qualitative comparison of the SAM model's performance on benign and adversarial examples across four datasets (ADE20K, MS-COCO, Cityscapes, SA-1B).  It demonstrates the impact of DarkSAM's attack on SAM's segmentation ability using different types of prompts (point, box, and segment everything). The top row presents the original images with prompts, and the middle and bottom rows show the segmentation results for benign and adversarial images, respectively. The results illustrate how DarkSAM effectively fools the SAM model into generating incorrect segmentations.
> <details>
> <summary>read the caption</summary>
> Figure 4: Visualizations of SAM segmentation results for adversarial examples across four datasets. The first four columns and the middle four columns display the segmentation results for point and box prompts, respectively. The last three columns show results under the segment everything mode for benign examples, as well as adversarial examples created using point and box prompts, respectively.
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_17_2.jpg)

> 🔼 This figure shows the ablation study results about the effect of random seeds on DarkSAM's performance.  The ASR (Attack Success Rate) is plotted against the number of random seeds used for both point prompts (P2P) and box prompts (B2B).  The results indicate DarkSAM maintains consistent performance regardless of the random seed used.
> <details>
> <summary>read the caption</summary>
> Figure A8: The results (%) of ablation study about random seeds
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_18_1.jpg)

> 🔼 This figure shows the results of applying the Segment Anything Model (SAM) to both benign and adversarial examples from four different datasets.  It demonstrates the impact of DarkSAM's attack on SAM's ability to segment images under various prompt types (point, box, and 'segment everything').  The visual comparison between benign and adversarial results highlights DarkSAM's effectiveness in causing SAM to fail in object segmentation.
> <details>
> <summary>read the caption</summary>
> Figure 4: Visualizations of SAM segmentation results for adversarial examples across four datasets. The first four columns and the middle four columns display the segmentation results for point and box prompts, respectively. The last three columns show results under the segment everything mode for benign examples, as well as adversarial examples created using point and box prompts, respectively.
> </details>



![](https://ai-paper-reviewer.com/0o9E8AsFgW/figures_19_1.jpg)

> 🔼 This figure shows qualitative results of DarkSAM attack using point prompts on SAM-L (Segment Anything Model with ViT-L backbone) under point prompts.  It presents a comparison between benign examples (correctly segmented by SAM-L) and adversarial examples generated by DarkSAM (incorrectly segmented by SAM-L). The results visualize how DarkSAM successfully fools SAM-L, causing it to produce incorrect segmentation masks across various images. 
> <details>
> <summary>read the caption</summary>
> Figure A1: Qualitative results of the DarkSAM using point prompts on SAM-L under point prompts
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/0o9E8AsFgW/tables_7_1.jpg)
> 🔼 This table presents the results of a study on the transferability of adversarial attacks against the Segment Anything Model (SAM).  It shows the Attack Success Rate (ASR) for different prompt types (point and box) and across different datasets. The ASR indicates how often the adversarial attack successfully prevents SAM from segmenting objects.  The table highlights the ability of a single adversarial perturbation to fool SAM across various prompt types and datasets.
> <details>
> <summary>read the caption</summary>
> Table 2: The ASR (%) of the cross-prompt transferability study on SAM. “BOX → POINT” indicates that adversarial examples created using box are tested in point mode. Others stand the same meaning.
> </details>

![](https://ai-paper-reviewer.com/0o9E8AsFgW/tables_8_1.jpg)
> 🔼 This table compares the attack success rate (ASR) of DarkSAM against other universal adversarial perturbation (UAP) methods and a sample-wise attack method.  The ASR is calculated for four datasets (ADE20K, COCO, Cityscapes, and SA-1B) and two prompt types (point and box).  DarkSAM significantly outperforms all other methods, demonstrating its effectiveness in fooling the Segment Anything Model (SAM). The '*' indicates that the attack method did not work at all.
> <details>
> <summary>read the caption</summary>
> Table 3: The ASR (%) of comparison study
> </details>

![](https://ai-paper-reviewer.com/0o9E8AsFgW/tables_15_1.jpg)
> 🔼 This table presents the mean Intersection over Union (mIoU) results of DarkSAM's attack against MobileSAM, a variant of the Segment Anything Model.  It shows the mIoU for clean images (no attack) and images subjected to DarkSAM's attack, broken down by dataset (ADE20K, MS-COCO, CITYSCAPES, and SA-1B) and whether point or box prompts were used.  Higher values indicate better segmentation performance, with lower values after the attack demonstrating DarkSAM's effectiveness in reducing segmentation accuracy.
> <details>
> <summary>read the caption</summary>
> Table A1: The mIoU (%) of DarkSAM on MobileSAM. Values covered by gray denote the clean mIoU, others denote adversarial mIoU. ADE20K, MS-COCO, CITYSCAPES abbreviated as ADE, COCO, CITY, respectively. Bolded values indicate the best results.
> </details>

![](https://ai-paper-reviewer.com/0o9E8AsFgW/tables_17_1.jpg)
> 🔼 This table presents the Attack Success Rate (ASR) results of a cross-prompt transferability study conducted on the Segment Anything Model (SAM).  The ASR metric quantifies the effectiveness of the DarkSAM attack in preventing SAM from correctly segmenting objects.  The table shows ASR values across four datasets (ADE20K, MS-COCO, CITYSCAPES, and SA-1B) for two prompt types (point and box).  The results demonstrate how well the DarkSAM attack generalizes across different prompts (point prompt used to attack, box prompt used to test and vice versa).
> <details>
> <summary>read the caption</summary>
> Table 2: The ASR (%) of the cross-prompt transferability study on SAM. “BOX → POINT” indicates that adversarial examples created using box are tested in point mode. Others stand the same meaning.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/0o9E8AsFgW/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}