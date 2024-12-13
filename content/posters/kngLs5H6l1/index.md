---
title: "Normal-GS: 3D Gaussian Splatting with Normal-Involved Rendering"
summary: "Normal-GS improves 3D Gaussian Splatting by integrating normal vectors into the rendering pipeline, achieving near state-of-the-art visual quality with accurate surface normals in real-time."
categories: []
tags: ["Computer Vision", "3D Vision", "🏢 Monash University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} kngLs5H6l1 {{< /keyword >}}
{{< keyword icon="writer" >}} Meng Wei et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=kngLs5H6l1" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93867" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=kngLs5H6l1&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/kngLs5H6l1/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

3D Gaussian Splatting (3DGS) has shown promise in real-time novel view synthesis, but its noisy and discrete nature creates challenges in accurate surface estimation.  Previous attempts to improve this often negatively impacted rendering quality. This is because of a fundamental disconnect between normal vectors and the rendering pipeline in 3DGS methods. 



The researchers introduce Normal-GS, a novel approach that integrates normal vectors into the 3DGS rendering pipeline.  This is achieved by modeling the interaction between normals and incident lighting using the rendering equation.  They re-parameterize surface colors and use an anchor-based 3DGS to implicitly encode locally-shared illumination vectors, optimizing for both visual quality and normal precision. Experiments show that Normal-GS achieves near state-of-the-art results while maintaining real-time performance. 

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Normal-GS integrates normal vectors into the 3DGS rendering pipeline for improved accuracy. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method achieves competitive rendering quality and accurate surface normal estimation simultaneously. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Normal-GS maintains real-time rendering performance. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in 3D computer vision and graphics because it presents **a novel approach to enhance both the rendering quality and accuracy of surface normal estimation in 3D Gaussian Splatting (3DGS)**.  It directly addresses the long-standing challenge of balancing appearance and geometry in 3DGS-based methods, a problem currently hindering real-time high-fidelity novel view synthesis.  The proposed method offers a significant improvement over existing techniques, opening up exciting avenues for research in real-time rendering, high-fidelity reconstruction and related areas. 

------
#### Visual Insights



![](https://ai-paper-reviewer.com/kngLs5H6l1/figures_1_1.jpg)

> This figure compares the rendering quality and normal accuracy of different 3D Gaussian splatting methods.  It shows that existing methods often struggle to balance high rendering quality with accurate normal estimation. The authors' proposed method, Normal-GS, is shown to achieve both high rendering quality and accurate normal estimation, even capturing normals on difficult surfaces like the semi-transparent box.





![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_7_1.jpg)

> This table presents a quantitative comparison of the rendering quality achieved by different methods, including the proposed Normal-GS method and several state-of-the-art baselines. The metrics used for comparison are PSNR, SSIM, and LPIPS.  The table highlights that Normal-GS achieves competitive or superior rendering quality compared to other methods, while simultaneously achieving significantly better normal estimation accuracy than methods like SpecGaussian.





### In-depth insights


#### Normal-GS: 3DGS Enhancement
Normal-GS presents a notable enhancement to 3D Gaussian Splatting (3DGS) by directly integrating normal vectors into the rendering pipeline.  This **crucial integration** addresses the existing disconnect between appearance and geometry in standard 3DGS, a limitation that often leads to a trade-off between high-fidelity rendering and accurate surface normal estimation. The core innovation involves re-parameterizing surface colors using a physically-based rendering equation, explicitly modeling the interaction between normals and incident lighting. By leveraging an **anchor-based 3DGS** and **optimized normal encoding**, Normal-GS achieves a better balance between visual quality and geometric accuracy.  The method's effectiveness is demonstrated through extensive experiments showing near state-of-the-art rendering quality while simultaneously obtaining accurate surface normals and maintaining real-time performance.  **The key contribution** lies in directly addressing the fundamental limitations of previous 3DGS approaches, leading to a more robust and versatile method for 3D scene representation and rendering.

#### Normal-Involved Rendering
The proposed 'Normal-Involved Rendering' technique significantly enhances 3D Gaussian Splatting (3DGS) by directly integrating surface normal vectors into the rendering pipeline.  This addresses a critical limitation of previous methods, which often suffer from a disconnect between appearance and geometry. By leveraging the physically-based rendering equation, **Normal-Involved Rendering explicitly models the interaction between normals and incident lighting**, leading to more accurate and realistic rendering.  The core innovation lies in representing surface colors as a product of normals and an Integrated Directional Illumination Vector (IDIV). This re-parameterization allows normal vectors to participate directly in the rendering calculations.  Furthermore, the use of **anchor-based 3DGS** implicitly encodes locally shared IDIVs, simplifying optimization and reducing memory usage.  The method also improves the accuracy of specular effects using optimized normals and Integrated Directional Encoding. **This holistic approach achieves a balance between high rendering quality and accurate surface normal estimation**, demonstrating superior performance compared to existing 3DGS methods.

#### Anchor-Based IDIVs
The concept of 'Anchor-Based IDIVs' presents a novel approach to handling the high dimensionality and computational cost associated with Integrated Directional Illumination Vectors (IDIVs) in 3D Gaussian splatting.  **The core idea is to leverage spatial coherence within the scene by grouping nearby Gaussians and associating each group with a single anchor point.** This anchor point implicitly represents the IDIV for all Gaussians within its group.  **Instead of storing an IDIV for each Gaussian, only anchors need to store IDIVs, reducing the memory footprint significantly**.  A multilayer perceptron (MLP) then decodes the anchor's IDIV to generate the IDIVs for the associated Gaussians, making this method efficient and scalable. This strategy not only reduces memory consumption but also encourages smoother, more physically plausible illumination patterns. **Using anchor points effectively regularizes the representation, preventing overfitting and improving both rendering quality and normal accuracy**. This method balances precise geometry with high-fidelity rendering by efficiently capturing and utilizing local illumination effects.

#### Real-Time Performance
The paper's real-time performance is a **critical aspect**, particularly given its focus on 3D Gaussian Splatting for real-time radiance field rendering.  The authors claim to achieve **near state-of-the-art visual quality while maintaining real-time performance**. This is a significant contribution, especially given the computational cost often associated with high-fidelity novel view synthesis.  A detailed analysis of the paper would be needed to thoroughly assess this claim, which hinges on efficient GPU rasterization techniques and other optimizations.  **Benchmark comparisons** against existing methods are essential to validate the real-time capability and the efficiency gains.  Furthermore, exploring scalability with respect to scene complexity and the number of Gaussians used would be crucial in evaluating the practical limitations and robustness of the proposed approach in real-world applications.  The actual frame rates achieved on various hardware platforms and scenes would provide further insight into the practical real-time performance.

#### Future Enhancements
Future enhancements for 3D Gaussian splatting (3DGS) could significantly improve its capabilities.  **Improving normal estimation accuracy** is crucial; current methods often produce noisy normals, hindering accurate surface reconstruction and realistic rendering.  Investigating advanced normal regularization techniques, perhaps incorporating physically-based shading models, could address this.  **Addressing the limitations of existing rendering approaches** is another key area; current methods struggle with representing complex materials and lighting conditions accurately. Incorporating more sophisticated BRDFs and advanced lighting models would enhance realism. **Further memory optimization** is also crucial, as current 3DGS methods can be memory-intensive, limiting their scalability.  Investigating efficient compression techniques and data structures could solve this.  **Efficient handling of dynamic scenes** is also vital; current 3DGS methods primarily focus on static scenes.  Developing efficient algorithms for updating and rendering changing scenes would open up new applications.  Finally, exploring the integration of 3DGS with other existing techniques, like neural implicit representations, could unlock significant synergistic benefits. By addressing these future enhancements, 3DGS can reach its full potential.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/kngLs5H6l1/figures_4_1.jpg)

> This figure illustrates the architecture of Normal-GS, a novel method that integrates normal vectors into the 3D Gaussian splatting rendering pipeline.  It shows how the method re-parameterizes surface colors using the physically-based rendering equation,  modeling the interaction between normals and incident lighting.  The core idea is to represent diffuse color as the dot product of the normal vector and an Integrated Directional Illumination Vector (IDIV), while specular effects are modeled using Integrated Directional Encoding (IDE).  Locally shared IDIVs are implicitly encoded using an anchor-based 3DGS to reduce memory usage and simplify optimization. MLPs decode these IDIVs, and the overall system enhances both rendering quality and normal estimation accuracy.


![](https://ai-paper-reviewer.com/kngLs5H6l1/figures_6_1.jpg)

> This figure shows a qualitative comparison of rendering quality and normal estimation results between the proposed Normal-GS method and several state-of-the-art 3DGS-based methods (3DGS, ScaffoldGS, SpecGaussian, and GShader).  For each scene, the figure shows the rendered images from each method and the ground truth. The results demonstrate that Normal-GS effectively preserves good rendering quality while simultaneously achieving clean and accurate normal estimation.


![](https://ai-paper-reviewer.com/kngLs5H6l1/figures_8_1.jpg)

> This figure illustrates the architecture of Normal-GS, a novel method that integrates normal vectors into the 3D Gaussian splatting rendering pipeline. Normal-GS re-parameterizes surface colors as a product of normals and an Integrated Directional Illumination Vector (IDIV).  It uses an anchor-based approach to implicitly encode locally-shared IDIVs, improving memory efficiency and simplifying optimization. The figure shows how the diffuse component of the color is modeled as the dot product of the normal and IDIV, and how the specular component is modeled using Integrated Directional Encoding (IDE). The method leverages MLPs to decode inherent parameters from locally shared anchor Gaussians. This approach improves both rendering quality and the accuracy of surface normal estimation.


![](https://ai-paper-reviewer.com/kngLs5H6l1/figures_8_2.jpg)

> This figure illustrates the architecture of Normal-GS, a novel method that integrates normal vectors into the 3D Gaussian Splatting (3DGS) rendering pipeline.  It shows how the method re-parameterizes surface colors using a physically-based rendering approach, modeling the interaction between normals and incident lighting. The core components are the use of Integrated Directional Illumination Vectors (IDIVs) for diffuse color modeling, Integrated Directional Encoding (IDE) for specular effects, and an anchor-based approach to implicitly encode locally-shared IDIVs. This approach improves both rendering quality and the accuracy of surface normal estimation.


![](https://ai-paper-reviewer.com/kngLs5H6l1/figures_15_1.jpg)

> This figure presents a qualitative comparison of rendering quality and normal estimation between the proposed Normal-GS method and several state-of-the-art 3DGS-based methods across multiple scenes.  The comparison shows that Normal-GS produces cleaner and more accurate normal estimations while maintaining high rendering quality, outperforming other methods that may sacrifice either rendering quality or normal accuracy.


![](https://ai-paper-reviewer.com/kngLs5H6l1/figures_16_1.jpg)

> This figure illustrates the architecture of Normal-GS, a novel method that integrates normal vectors into the 3D Gaussian Splatting (3DGS) rendering pipeline. Normal-GS re-parameterizes surface colors as the product of normals and an Integrated Directional Illumination Vector (IDIV).  It uses an anchor-based approach to implicitly encode locally-shared IDIVs, saving memory and simplifying optimization.  The diffuse component is modeled as the dot product of the normal vector and the IDIV, while the specular component leverages Integrated Directional Encoding (IDE). MLPs are used to decode the implicitly encoded parameters from the anchor Gaussians.  The overall effect is to improve both rendering quality and the accuracy of surface normal estimation.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_8_1.jpg)
> This table presents a quantitative analysis of the impact of different components of the proposed Normal-GS method on the DTU dataset.  It compares the mean Chamfer distance (mCD), a metric for geometry accuracy, and the Peak Signal-to-Noise Ratio (PSNR), a metric for rendering quality. The models compared are: (a) the baseline Scaffold-GS; (b) Scaffold-GS with the addition of a depth-regularized loss term on normals (LN); (c) model (b) with the addition of a specular loss term (Lspecular); and (d) the full Normal-GS model. The results show the effect of each component on both geometry and rendering quality.

![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_9_1.jpg)
> This table presents a quantitative comparison of the rendering quality of different methods, including the proposed Normal-GS method and several state-of-the-art baselines. The metrics used are PSNR, SSIM, and LPIPS, which are commonly used to evaluate the visual quality of images. The table shows that Normal-GS achieves comparable or better results than the best performing baseline, SpecGaussian, in terms of rendering quality while significantly outperforming it in terms of normal accuracy. The results highlight the ability of Normal-GS to achieve a good balance between rendering quality and geometry accuracy, unlike other methods that may sacrifice one for the other.

![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_9_2.jpg)
> This table presents a comparison of geometric reconstruction and rendering quality between different methods on the DTU dataset.  The metrics used are mean Chamfer distance (mCD), which measures the geometric accuracy, and Peak Signal-to-Noise Ratio (PSNR), which assesses rendering quality. Lower mCD values indicate better geometric accuracy, while higher PSNR values correspond to better rendering quality. The table shows that our method achieves a good balance between geometric accuracy and rendering quality, outperforming some baseline methods in PSNR while maintaining a competitive mCD.

![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_14_1.jpg)
> This table presents a quantitative comparison of rendering quality metrics (PSNR, SSIM, LPIPS) for several methods, including the proposed Normal-GS and state-of-the-art baselines.  It highlights the competitive rendering quality of Normal-GS while emphasizing its superior performance in normal estimation compared to SpecGaussian, which prioritizes rendering quality at the cost of normal accuracy.

![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_16_1.jpg)
> This table presents a quantitative comparison of the rendering quality achieved by different methods, including the proposed Normal-GS method and several state-of-the-art baselines.  Metrics used for comparison include PSNR, SSIM, and LPIPS, across three datasets: Mip-NeRF360, Tanks & Temples, and Deep Blending.  The results show that Normal-GS achieves competitive or superior rendering quality while also demonstrating significantly improved normal estimation accuracy compared to other methods, particularly SpecGaussian.

![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_16_2.jpg)
> This table presents a quantitative comparison of rendering quality metrics (PSNR, SSIM, LPIPS) for several methods, including the proposed Normal-GS and state-of-the-art baselines. It highlights the competitive rendering quality of Normal-GS while demonstrating its superior performance in normal estimation compared to SpecGaussian.

![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_16_3.jpg)
> This table presents a quantitative comparison of the rendering quality achieved by different methods, including the proposed Normal-GS method.  Metrics such as PSNR, SSIM, and LPIPS are used to evaluate the visual quality. The table highlights that Normal-GS achieves comparable or better rendering quality than the state-of-the-art (SOTA) SpecGaussian method, while also significantly outperforming SpecGaussian in normal estimation accuracy.  The results demonstrate that Normal-GS effectively balances both rendering quality and accurate normal estimation.

![](https://ai-paper-reviewer.com/kngLs5H6l1/tables_17_1.jpg)
> This table presents a quantitative comparison of the rendering quality achieved by the proposed Normal-GS method against several state-of-the-art 3DGS-based methods.  Metrics used for comparison include PSNR, SSIM, and LPIPS, across three datasets: Mip-NeRF360, Tanks & Temples, and Deep Blending. The table highlights that Normal-GS achieves comparable or better rendering quality compared to the best performing baseline (SpecGaussian), while significantly outperforming other baselines in terms of normal estimation accuracy.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kngLs5H6l1/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}