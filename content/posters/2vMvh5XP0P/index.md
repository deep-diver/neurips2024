---
title: "Subsurface Scattering for Gaussian Splatting"
summary: "Real-time rendering of objects with subsurface scattering effects is now possible with SSS-GS, a novel method combining explicit surface geometry and implicit subsurface scattering for high-quality no..."
categories: []
tags: ["Computer Vision", "3D Vision", "🏢 University of Tübingen",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 2vMvh5XP0P {{< /keyword >}}
{{< keyword icon="writer" >}} Jan-Niklas Dihlmann et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=2vMvh5XP0P" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/96787" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=2vMvh5XP0P&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/2vMvh5XP0P/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Creating realistic digital images of objects made from materials that scatter light beneath the surface (like skin or wax) is a tough problem in computer graphics.  Current methods are either too slow or don't capture the details accurately. This is significant because subsurface scattering is crucial for realistic rendering, but it's difficult to model efficiently.

The researchers introduce a new method called SSS-GS. It uses 3D Gaussians to represent the object's shape and a neural network to model subsurface scattering.  The key is combining an explicit surface representation with an implicit subsurface scattering model.  Their method produces photorealistic results, enabling material editing and relighting in real-time, and is significantly faster than previous approaches.  They also introduce a new dataset to help train and evaluate these types of models. **This significantly advances the field of real-time rendering and opens new avenues for research in material representation and efficient neural rendering techniques.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Novel hybrid representation for subsurface scattering using 3D Gaussian Splatting. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Real-time rendering of objects with subsurface scattering effects achieved through efficient deferred shading. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} New multi-view, multi-light dataset of subsurface scattering objects introduced. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents a novel method for real-time rendering of objects with subsurface scattering effects, a significant challenge in computer graphics.  It offers a new hybrid representation combining explicit surface geometry with an implicit subsurface scattering model, enabling interactive material editing and relighting at speeds comparable to or better than existing methods, while also improving upon the quality of the final output. This work will be highly relevant to researchers working in computer graphics, particularly those focused on real-time rendering, view synthesis, and material representation. The introduction of a new multi-view, multi-light dataset further enhances the value of this contribution for the research community.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_0_1.jpg)

> This figure shows the main results and contributions of the paper.  It demonstrates the pipeline for reconstructing objects with subsurface scattering using 3D Gaussian Splatting. The process begins with a multi-view, multi-light dataset captured in a light stage. This data is then used to reconstruct a 3D model, which is decomposed into its physically based rendering (PBR) components (base color, normal, diffuse, specular, subsurface scattering residual, metalness, roughness, subsurfaceness).  The PBR model enables realistic material editing and relighting, while the Gaussian Splatting framework allows for real-time novel view synthesis. The figure visually presents the steps, from the initial dataset to the final rendered image showing the edited and relit object, highlighting the efficiency and visual fidelity of the method. 





![](https://ai-paper-reviewer.com/2vMvh5XP0P/tables_7_1.jpg)

> This table presents a quantitative comparison of the proposed method against the KiloOSF method for novel view synthesis on both synthetic and real-world datasets.  Metrics include PSNR, SSIM, LPIPS, and FPS.  The table shows that the proposed method achieves comparable or better results than KiloOSF at significantly higher frame rates, even when using a smaller dataset and lower resolution images.





### In-depth insights


#### SSS-GS Framework
The hypothetical "SSS-GS Framework" likely integrates subsurface scattering (SSS) models within the 3D Gaussian Splatting framework. This involves representing objects using 3D Gaussians while simultaneously modeling light transport beneath the surface to achieve photorealistic rendering.  **Key aspects** likely include a hybrid representation combining explicit surface properties (e.g., PBR parameters) with an implicit neural network to predict subsurface scattering radiance. This network might take as input Gaussian properties, incident light, and view directions to produce an outgoing radiance that captures the diffuse and volumetric SSS effects.  **A significant advantage** is the potential for real-time rendering by leveraging the efficiency of 3D Gaussian Splatting, even with the added complexity of SSS.  **Challenges** could involve balancing the accuracy of SSS modeling with the computational cost and the need to address issues like shadowing and efficient rendering of highly-detailed objects.  The success of such a framework would depend heavily on training data quality and the capacity of the network to effectively learn and represent complex light transport phenomena.

#### Neural SSS Model
A neural subsurface scattering (SSS) model offers a powerful approach to simulating light transport within scattering materials.  Instead of relying on computationally expensive physics-based methods, a neural network learns to approximate the complex interactions of light with the material. **This allows for real-time rendering speeds,** crucial for applications like interactive visualization and virtual reality. The network is trained on datasets of multi-view images of objects under various lighting conditions, learning to predict the outgoing radiance based on the input lighting and material properties.  **Key advantages include the ability to handle complex scattering effects and spatially varying properties** that are challenging for traditional methods.  However, challenges remain, such as the need for large training datasets and the potential for overfitting or inaccuracies in complex scenarios.  Furthermore, **the generalizability of the trained model to unseen materials or geometries needs careful consideration.**  Despite these limitations, neural SSS models represent a significant step towards more realistic and efficient rendering of translucent objects, opening up exciting possibilities for various fields.

#### Dataset Creation
Creating a robust dataset is crucial for training effective subsurface scattering (SSS) models.  This paper's approach is commendable for its dual focus on **both synthetic and real-world data**.  Synthetic data allows for precise control over lighting, material properties, and ground truth information, enabling focused model training and evaluation. However, **real-world data is essential** to ensure generalizability and accuracy in modeling the complexity of real-world materials and light transport. The paper's attention to detailed data acquisition protocols, including multi-view and multi-light setups, and addressing challenges like noise reduction,  demonstrates a commitment to data quality. The combination of synthetic and real-world data, if made publicly available, **would be a significant contribution to the field**, facilitating further research and advancement in SSS modeling. The discussion of dataset creation could be improved by explicitly outlining challenges faced and the rationale behind design decisions.

#### Real-time Rendering
Real-time rendering in computer graphics aims to generate images at speeds matching a video's frame rate, typically 30 frames per second or higher. This demands efficient algorithms and data structures to handle complex scenes and shading models.  **Gaussian splatting**, as highlighted in the provided research paper excerpt, offers a notable approach. It approximates object shapes and appearances with 3D Gaussians, allowing for faster rendering compared to traditional methods, like rasterization or ray tracing.  The paper proposes enhancements to Gaussian splatting to better model subsurface scattering (SSS), a phenomenon critical for achieving realism in materials such as skin, wax, or marble. While real-time rendering with SSS has traditionally been challenging, this method promises to bridge that gap. **Efficient neural networks** are used to learn and predict SSS effects, enhancing visual fidelity. The joint optimization of shape, material properties, and radiance fields in image space, enabled by differentiable rendering, demonstrates a significant advancement in real-time rendering capabilities. The success is underscored by its application to both synthetic and real-world datasets, capturing detailed material properties and dynamic lighting with notable improvements in speed compared to existing Neural Radiance Fields (NeRFs) approaches. **Ultimately**, the research showcases promising progress towards photorealistic and interactive real-time rendering of objects with complex light interactions.

#### Future of SSS
The future of subsurface scattering (SSS) research is bright, with many promising avenues for development.  **Real-time rendering of complex SSS effects** remains a significant challenge, and further improvements in efficiency and accuracy are needed, particularly for scenes with heterogeneous materials and complex geometries. **Developing more efficient neural network architectures** specifically tailored for SSS would significantly advance the state-of-the-art.  **High-quality datasets** that capture diverse materials and lighting conditions are crucial, especially data that accounts for dynamic scenarios and complex light transport, improving the ability of models to generalize. Moreover,  **exploring novel representations beyond neural networks** may lead to more efficient and physically accurate rendering. Finally, **integrating SSS models with other rendering techniques** such as path tracing and volumetric methods could unlock new possibilities for realistic and efficient visual effects.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_3_1.jpg)

> The figure illustrates the pipeline of the proposed method.  It shows how the subsurface scattering (SSS) appearance of an object is modeled implicitly, combined with an explicit surface appearance model using 3D Gaussians.  A small Multilayer Perceptron (MLP) is utilized to predict the SSS residual and incident light, taking into account various properties of each Gaussian.  Ray tracing is used to determine visibility.  A deferred shading pipeline combines the results from the MLP with a Bidirectional Reflectance Distribution Function (BRDF) model in image space to produce the final pixel colors. 


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_5_1.jpg)

> This figure shows the results of decomposing objects into their constituent components (base color, metalness, roughness, subsurface scattering, normals, SSS residual, incident light). The decomposition is shown for two different views and two different light directions.  The top two rows show results for synthetic objects, while the bottom two rows show results for real-world scanned objects.  The figure demonstrates the ability of the proposed method to accurately capture and represent the various physical properties of translucent objects, including their subsurface scattering behavior.


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_8_1.jpg)

> This figure demonstrates the editing capabilities of the proposed method.  It shows how different parameters like roughness, metalness, base color, subsurfaceness, and residual color can be adjusted individually to modify the appearance of the 3D object.  The rightmost column highlights the ability to edit with light sources not present during training.


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_8_2.jpg)

> This figure compares the results of Relightable 3D Gaussian Splatting (R3DGS) and the proposed method for rendering subsurface scattering objects.  The left side shows that R3DGS struggles to accurately relight objects with subsurface scattering because it does not explicitly model the subsurface scattering effect. The right side illustrates how deferred shading in the proposed method improves the rendering of specular highlights by evaluating the surface reflectance at each pixel instead of just at the Gaussian centers, resulting in crisper details.


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_8_3.jpg)

> This figure compares the results of the proposed method against the KiloOSF method on four different objects, two synthetic and two real-world. It showcases the superior reconstruction of object shape and appearance achieved by the proposed method compared to KiloOSF, particularly regarding fine details and surface smoothness.


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_13_1.jpg)

> This figure illustrates the data acquisition pipelines for both real-world and synthetic datasets.  The real-world pipeline shows the light stage setup, image denoising and demosaicing, mask generation using the Segment Anything Model (SAM), and structure-from-motion (SfM) processing using COLMAP. The synthetic pipeline depicts the Blender light stage setup and the train/test split used for training and evaluating the model.  It highlights the differences in data acquisition and preprocessing steps for both data sources.


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_13_2.jpg)

> This figure shows all 20 objects used in the real and synthetic datasets.  The top-left corner shows 5 synthetic objects, and the rest are real-world objects. Each object is shown under multiple lighting conditions captured from many camera views.  The image provides a visual overview of the diversity of translucent objects (e.g., varying shapes, colors, materials, and translucency) used to evaluate the proposed SSS (subsurface scattering) method.


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_14_1.jpg)

> This figure shows an ablation study comparing different versions of the proposed method against the ground truth. The versions are: - Full Method: The complete method with PBR and deferred shading. - w/o PBR: Without the Physically Based Rendering (PBR) component. Only the SSS residual and incident light are considered.  - w/o Deferred: Without deferred shading, the BRDF is evaluated in Gaussian space which results in less precise highlights. - PBR + Deferred: Without SSS effect, only PBR with deferred shading.  The differences between the versions show the contribution of each component to the final rendering quality, especially the impact of deferred shading for accurate specular highlights.


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_15_1.jpg)

> This figure shows the results of applying image-based lighting (IBL) to the objects rendered by the proposed method. The top row shows rendering with a single light source for comparison. The subsequent rows show the same objects rendered under three different environment lighting conditions (Indoor, Woods, Snow).  Both synthetic (bunny, soap) and real-world (candle, car) objects are included to demonstrate the versatility of the approach.


![](https://ai-paper-reviewer.com/2vMvh5XP0P/figures_16_1.jpg)

> This figure showcases the results of the proposed method's decomposition of objects into their constituent parts: base color, metalness, roughness, normals, SSS residual, specular, and diffuse components.  It demonstrates the method's ability to handle both synthetic and real-world objects, showing two different views of each object under varying light directions. The decomposition highlights the separation of surface properties (PBR) from subsurface scattering effects (SSS).


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/2vMvh5XP0P/tables_14_1.jpg)
> This table presents the results of an ablation study evaluating the impact of different components of the proposed method.  It shows the PSNR, SSIM, and LPIPS scores for several variations of the method, including versions without deferred shading, without the physically-based rendering (PBR) model, without the joint multi-layer perceptron (MLP), and without the subsurface scattering residual. A comparison with a version of R3DGS [10] that incorporates the incident light field from the proposed method is also included.

![](https://ai-paper-reviewer.com/2vMvh5XP0P/tables_15_1.jpg)
> This table presents a quantitative comparison of the proposed method against the state-of-the-art KiloOSF method for novel view synthesis on large images and bigger datasets.  It shows metrics such as PSNR, SSIM, LPIPS, and FPS for both synthetic and real-world datasets, highlighting the superior performance of the proposed approach in terms of image quality and speed. The table also includes training time and data size used.

![](https://ai-paper-reviewer.com/2vMvh5XP0P/tables_16_1.jpg)
> This table presents a quantitative comparison of intrinsic properties (base color, roughness, metalness, normal, SSS residual, specular, diffuse, and render) obtained from Blender renders against the results produced by the authors' method. The comparison is performed on a subset of 25 renders from five synthetic scenes, each with five different camera and light poses.  The average RMSE across all scenes is also included. Note that Blender doesn't provide SSS intrinsics, so the residual shown here represents the difference between diffuse light rendered with and without SSS.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/2vMvh5XP0P/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}