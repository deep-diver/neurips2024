---
title: "MambaLLIE: Implicit Retinex-Aware Low Light Enhancement with Global-then-Local State Space"
summary: "MambaLLIE: a novel implicit Retinex-aware low-light enhancer using a global-then-local state space, significantly outperforms existing CNN and Transformer-based methods."
categories: []
tags: ["Computer Vision", "Image Enhancement", "🏢 Nanjing University of Science and Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} l6xVqzm72i {{< /keyword >}}
{{< keyword icon="writer" >}} Jiangwei Weng et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=l6xVqzm72i" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93852" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=l6xVqzm72i&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/l6xVqzm72i/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Low-light image enhancement is crucial but challenging due to global illumination issues and local problems like noise and blur. Existing methods, including CNNs and Transformers, struggle to balance global and local information processing effectively.  State-Space Models (SSMs) show promise for long-range modeling but lack local detail preservation. 

This paper introduces MambaLLIE, a novel low-light enhancer addressing these shortcomings.  **MambaLLIE employs a global-then-local state space design**, incorporating a Local-Enhanced State Space Module (LESSM) that preserves local dependencies and an Implicit Retinex-aware Selective Kernel module (IRSK) for adaptive feature selection.  **Extensive experiments demonstrate that MambaLLIE significantly outperforms state-of-the-art methods**, showcasing its superior performance in capturing global context and local details, which is crucial for real-world image enhancement.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} MambaLLIE, a new low-light image enhancer, surpasses existing methods. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The model uses a global-then-local state space design for comprehensive image enhancement. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} An implicit Retinex-aware approach improves feature aggregation and illumination strength modulation. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents a novel approach to low-light image enhancement, significantly outperforming existing methods.  **Its global-then-local state space design, incorporating implicit Retinex awareness**, offers a more comprehensive and flexible solution to the challenges of capturing both global and local image degradation. This work opens **new avenues for research in state-space models and Retinex-based enhancement techniques**, potentially impacting various computer vision applications.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_1_1.jpg)

> This figure visualizes the effective receptive fields (ERFs) of four different low-light enhancement methods: SNR-Net, Retinexformer, MambaIR, and the authors' proposed MambaLLIE.  The ERF visualization shows the area of the input image that significantly influences each method's output.  A larger, more uniform ERF is desirable. The figure demonstrates that MambaLLIE achieves a superior ERF compared to the other methods, indicating its ability to capture both global and local contextual information effectively.





![](https://ai-paper-reviewer.com/l6xVqzm72i/tables_5_1.jpg)

> This table presents a quantitative comparison of the proposed MambaLLIE model against eleven state-of-the-art (SOTA) low-light image enhancement methods across five benchmark datasets.  The performance is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) metrics.  The complexity of each method, measured in terms of floating-point operations (FLOPS) and parameters, is also provided for comparison.





### In-depth insights


#### Global-Local State
The concept of 'Global-Local State' in image processing, particularly within the context of low-light enhancement, suggests a powerful paradigm shift.  **Traditional methods often struggle to balance global illumination correction with the preservation of crucial local details**.  A global approach might brighten the entire image, but at the cost of amplified noise or loss of fine textures in darker regions. Conversely, a purely local approach risks inconsistencies and a lack of overall coherence.  A successful 'Global-Local State' model, therefore, would elegantly integrate both levels.  It would achieve **comprehensive global adjustments to illumination**, correcting for overall brightness and contrast, while simultaneously **maintaining or even enhancing local fidelity**. This likely involves a sophisticated mechanism for feature extraction and aggregation, perhaps utilizing attention mechanisms or advanced state-space models to capture both long-range dependencies (global) and short-range interactions (local) within the image.  The key challenge lies in **finding a balance between global consistency and local accuracy**; achieving this balance would be a significant advance in image enhancement, allowing for more natural-looking and visually pleasing results.

#### Retinex-Aware Enhance
A hypothetical "Retinex-Aware Enhance" section in a low-light image enhancement paper would likely detail a method that leverages the Retinex theory.  This theory decomposes an image into illumination and reflectance components, allowing for independent manipulation.  A **retinex-aware approach** would likely involve either explicitly estimating these components or implicitly incorporating their properties into the enhancement process.  The focus would be on **addressing both global illumination inconsistencies and local details**, which traditional Retinex methods often struggle with.  The proposed method might involve using a neural network to learn a mapping from low-light input to enhanced output while respecting the Retinex decomposition, possibly using a loss function that encourages preservation of image details and color accuracy.  **Effectiveness would be shown through comparisons** to standard Retinex methods and other state-of-the-art low-light enhancement techniques, emphasizing improvements in visual quality and quantitative metrics.  The discussion would address challenges like handling noise and artifacts while achieving natural-looking results.

#### SSMs for Low-Light
State Space Models (SSMs) offer a promising approach to low-light image enhancement by leveraging their ability to model long-range dependencies and global context effectively.  Unlike CNNs and Transformers, which struggle with capturing global degradation due to limited receptive fields, **SSMs can model global illumination issues more efficiently**. However, **a direct application of SSMs to low-light enhancement faces challenges in incorporating local details and invariants**.  The core issue is the inherent difference between the sequential nature of SSMs and the spatial nature of image data.  Therefore, successful application of SSMs requires innovative ways to integrate local context information and address spatial dependencies within the SSM framework.  **This often involves designing sophisticated mechanisms to capture local features and combine them with the global information captured by the SSM.**  These hybrid models, combining SSMs with other techniques, are likely to be superior to using SSMs or other methods in isolation.  Furthermore, careful consideration of the trade-off between computational complexity and performance is crucial for real-world applications of SSMs in low-light image enhancement.

#### Benchmark Results
The benchmark results section of a research paper is crucial for validating the proposed method's effectiveness.  A strong presentation will clearly detail the datasets used, ensuring they are relevant and widely accepted within the field.  **Quantitative metrics**, such as PSNR and SSIM, should be meticulously reported, ideally with statistical significance measures to confirm the robustness of the improvements.  Visual comparisons are equally important, showcasing results on a diverse range of examples to highlight both strengths and potential weaknesses.  **Direct comparison with state-of-the-art methods** is a must, highlighting the proposed method's superior performance or detailing the circumstances where it may fall short.  A thorough analysis of these benchmark results helps readers assess the practical value and limitations of the contributions, leading to a more impactful and trustworthy evaluation of the paper's claims.

#### Future Enhancements
Future enhancements for MambaLLIE could explore several avenues.  **Improving the efficiency** of the global-then-local state space architecture is crucial, potentially through optimized state transition matrices or more efficient attention mechanisms.  **Incorporating more sophisticated Retinex-aware modules** could lead to better handling of complex illumination variations and noise reduction.  Further research could focus on extending the model to handle **video enhancement**, requiring temporal modeling capabilities.  Addressing the limitations regarding reliance on paired data by exploring **unsupervised or semi-supervised learning** strategies would be beneficial for wider applicability. Finally, investigating the use of **different network backbones** or exploring alternative architectural designs might provide improvements in performance or computational efficiency.  **Robustness to various image degradations**, beyond low-light conditions, and a thorough analysis of the model's limitations under different scenarios would strengthen the work.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_3_1.jpg)

> This figure shows the architecture of the proposed MambaLLIE model for low-light image enhancement.  It highlights the key components: the Global-then-Local State Space Block (GLSSB), the Local-Enhanced State Space Module (LESSM), and the Implicit Retinex-aware Selective Kernel Module (IRSK). The diagram illustrates how these modules work together, including the use of layer normalization and how maximum/mean priors are incorporated.  The LESSM enhances the original SSMs by preserving local 2D dependencies, and the IRSK dynamically selects features using spatially-varying operations, adapting to varying inputs.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_6_1.jpg)

> This figure presents a qualitative comparison of the proposed MambaLLIE method against several state-of-the-art (SOTA) low-light image enhancement methods.  It shows the input low-light images alongside the enhanced results produced by each method, including QuadPrior, Retinexformer, SNR-Net, MambaIR, and the proposed MambaLLIE.  The goal is to visually demonstrate the superior performance of MambaLLIE in terms of brightness enhancement, detail preservation, and color accuracy. Zooming in is recommended for detailed observation.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_6_2.jpg)

> This figure displays qualitative comparisons of low-light image enhancement results between the proposed MambaLLIE method and several state-of-the-art (SOTA) methods on various real-world images.  The comparison showcases the visual improvements achieved in terms of brightness, detail preservation, color accuracy, and overall enhancement quality. Each row presents a different input image and its corresponding enhanced versions generated by various methods, allowing for direct side-by-side visual comparisons.  The ground truth (GT) images are also provided for reference.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_7_1.jpg)

> This figure presents a qualitative comparison of the proposed MambaLLIE method against state-of-the-art (SOTA) techniques across four different aspects: object detection, face detection, a user study example, and a comparison using an unpaired dataset.  Each subfigure (a-d) showcases the input low-light image alongside results from MambaLLIE and other leading methods, allowing for visual assessment of the performance differences in terms of illumination enhancement, detail preservation, and overall image quality.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_9_1.jpg)

> This figure shows the results of an ablation study on the selective kernel module (IRSK) within the proposed MambaLLIE model.  Different kernel sizes (3x3, 5x5, 5x7, 5x3, and the proposed 3x5) are compared, demonstrating how the choice of kernel size impacts feature extraction and ultimately the PSNR of the output image.  The Local Attribution Map (LAM) visualizations illustrate the model's focus on local and global features depending on the kernel size.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_14_1.jpg)

> This figure presents a qualitative comparison of the proposed MambaLLIE method with state-of-the-art (SOTA) low-light image enhancement methods.  It displays several example images in low light, along with their enhancements by QuadPrior, RetinexFormer, SNR-Net, MambaIR, and MambaLLIE. The purpose is to visually demonstrate the relative performance of MambaLLIE, highlighting its ability to enhance the brightness while preserving color and detail compared to the other approaches.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_15_1.jpg)

> This figure presents a qualitative comparison of image enhancement results produced by MambaLLIE and several state-of-the-art (SOTA) methods.  Multiple example images are shown, each with the original low-light image and the enhanced versions produced by each method.  This visual comparison allows for a direct assessment of the relative strengths and weaknesses of each method in terms of brightness, detail preservation, color accuracy, and overall visual quality.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_15_2.jpg)

> This figure presents a qualitative comparison of the proposed MambaLLIE model with state-of-the-art methods across four different scenarios: object detection, face detection, a user study, and unpaired datasets.  Each subfigure shows the input low-light image and the enhanced images produced by MambaLLIE and the comparison methods. The goal is to visually demonstrate the superiority of MambaLLIE in terms of detail preservation, color accuracy, and overall visual quality.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_16_1.jpg)

> This figure presents a qualitative comparison of the proposed MambaLLIE method against state-of-the-art (SOTA) techniques across various low-light image enhancement tasks.  Panel (a) shows object detection results, demonstrating improved object visibility with MambaLLIE. Panel (b) displays a similar comparison for face detection, highlighting MambaLLIE's effectiveness in enhancing facial features even in challenging low-light scenarios. Panel (c) provides an example from a user study, illustrating subjective improvements in perceived image quality. Finally, panel (d) shows results on an unpaired dataset, showcasing MambaLLIE's generalization capabilities.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_16_2.jpg)

> This figure presents a qualitative comparison between the proposed MambaLLIE model and the DiffLL model for low-light image enhancement.  It shows the results on various low-light images, allowing for a visual assessment of the performance of both methods in terms of illumination enhancement, detail preservation, color accuracy, and overall visual quality. The images demonstrate the strengths and weaknesses of each method in various scenarios.


![](https://ai-paper-reviewer.com/l6xVqzm72i/figures_16_3.jpg)

> This figure shows a qualitative comparison between the proposed MambaLLIE method and the DiffLL method for low-light image enhancement.  It presents several example image pairs, each showing the original low-light image, the result from MambaLLIE, and the result from DiffLL.  The images are designed to highlight the visual differences between the two approaches, allowing for a direct comparison of their performance in terms of brightness, color accuracy, detail preservation, and overall visual quality.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/l6xVqzm72i/tables_7_1.jpg)
> This table presents a quantitative comparison of the proposed MambaLLIE model against 11 state-of-the-art (SOTA) low-light image enhancement methods across five benchmark datasets: LOL-V2-real, LOL-V2-syn, SMID, SDSD-indoor, and SDSD-outdoor.  The comparison uses two metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index). The best result for each metric and dataset is highlighted in red, while the second-best result is highlighted in blue.  The table also includes the computational complexity (FLOPS and number of parameters) for each method.

![](https://ai-paper-reviewer.com/l6xVqzm72i/tables_7_2.jpg)
> This table presents the results of low-light object detection experiments performed on the ExDark dataset. Multiple state-of-the-art (SOTA) low-light image enhancement (LLIE) methods were evaluated by using the enhanced images as input to a YOLOv3 object detector.  The results are presented in terms of mean Average Precision (mAP) for twelve object categories.  The best performing method is highlighted in red, and the second-best is in blue, indicating a comparative analysis of the enhancement methods' effectiveness in improving object detection accuracy in low-light conditions.

![](https://ai-paper-reviewer.com/l6xVqzm72i/tables_8_1.jpg)
> This table presents the results of a user study comparing the performance of different low-light image enhancement methods.  Participants rated the overall quality, local detail preservation, and artifact/noise reduction of enhanced images on a scale of 1 to 5. The methods compared include RetinexNet, EnGAN, SCI, QuadPrior, SNR-Net, Retinexformer, MambaIR, and MambaLLIE.  The results show that MambaLLIE achieves the highest average scores across all three rating categories.

![](https://ai-paper-reviewer.com/l6xVqzm72i/tables_8_2.jpg)
> This table presents a quantitative comparison of the proposed MambaLLIE model against 11 state-of-the-art low-light image enhancement methods across five benchmark datasets: LOL-V2-real, LOL-V2-syn, SMID, SDSD-indoor, and SDSD-outdoor.  The performance is evaluated using two metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).  The best and second-best results for each metric and dataset are highlighted in red and blue, respectively.  Additionally, the computational complexity, measured in terms of FLOPS (floating point operations per second) and the number of parameters, is provided for each method.

![](https://ai-paper-reviewer.com/l6xVqzm72i/tables_9_1.jpg)
> This table presents a quantitative comparison of the proposed MambaLLIE model against 11 state-of-the-art (SOTA) low-light image enhancement methods across five benchmark datasets: LOL-V2-real, LOL-V2-syn, SMID, SDSD-indoor, and SDSD-outdoor.  The comparison uses two common metrics: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).  The best and second-best results for each metric and dataset are highlighted in red and blue, respectively.  Additionally, the computational complexity (FLOPS) and model parameters are given for each method. This allows for a comprehensive assessment of the performance of MambaLLIE relative to existing methods, considering both accuracy and efficiency.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/l6xVqzm72i/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}