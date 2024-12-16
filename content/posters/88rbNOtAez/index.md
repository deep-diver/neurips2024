---
title: "Make-it-Real: Unleashing Large Multimodal Model for Painting 3D Objects with Realistic Materials"
summary: "Make-it-Real uses a large multimodal language model to automatically paint realistic materials onto 3D objects, drastically improving realism and saving developers time."
categories: ["AI Generated", ]
tags: ["Multimodal Learning", "Vision-Language Models", "🏢 Stanford University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 88rbNOtAez {{< /keyword >}}
{{< keyword icon="writer" >}} Ye Fang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=88rbNOtAez" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/88rbNOtAez" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/88rbNOtAez/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Creating photorealistic 3D objects is challenging because manually assigning materials is tedious and time-consuming. Existing automated methods often struggle to generate realistic materials. This paper introduces Make-it-Real, a novel approach that uses a powerful multimodal large language model (MLLM) to automatically assign realistic materials to 3D objects.  The MLLM excels at recognizing and classifying materials from visual input, even with limited information like albedo maps alone. 

Make-it-Real uses a three-stage pipeline. First, it renders and segments 3D meshes to identify individual parts. Second, it uses the MLLM to retrieve materials from a comprehensive library by analyzing the visual characteristics of each part. Finally, it generates high-quality SVBRDF maps based on the selected materials. The results showcase the generation of visually realistic material maps with significant improvements over existing methods, particularly for objects from challenging sources like generative models. The study demonstrates that the approach is both effective and efficient, paving the way for more realistic and accessible 3D asset creation.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Make-it-Real leverages GPT-4V to automatically assign realistic materials to 3D objects based on albedo maps alone. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method significantly improves the realism of 3D assets, outperforming existing methods in terms of material accuracy and visual quality. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Make-it-Real offers a streamlined workflow, reducing the time and effort required for manual material assignment in 3D content creation. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents **a novel approach to enhancing the realism of 3D objects by automatically assigning realistic materials using a large multimodal language model (MLLM)**.  This addresses a key challenge in 3D asset creation, saving significant time and effort for developers. The method's integration of MLLMs opens new avenues for research in 3D modeling and material synthesis, potentially impacting various fields like gaming and virtual reality.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_0_1.jpg)

> 🔼 This figure showcases the capabilities of the Make-it-Real model.  It demonstrates the model's ability to enhance the realism of 3D objects by adding realistic materials. The input is a wide range of 3D objects represented by albedo maps only (essentially, just the base color information, without details like roughness, metallicness, etc.). The output is enhanced 3D objects with physically based rendering (PBR) maps. These PBR maps enable realistic material appearances and lighting conditions, which are compatible with 3D game engines and other downstream applications. Importantly, the model enables part-specific material assignment, allowing different materials (e.g., metal, plastic, rubber) to be assigned to different parts of the same object.
> <details>
> <summary>read the caption</summary>
> Figure 1: Usage of Make-it-Real. Our method can refine a wide range of albedo-map-only 3D objects from both CAD design and generative models. Our method enhances the realism of objects, enables part-specific material assignment to objects and generate PBR maps that are compatible with downstream engines.
> </details>





![](https://ai-paper-reviewer.com/88rbNOtAez/tables_8_1.jpg)

> 🔼 This table presents a quantitative comparison of the results obtained using the Make-it-Real method against baseline results.  The evaluation is performed from two perspectives: GPT evaluation and user preference. For each category (3D assets, Image-to-3D, and Text-to-3D), the table shows the percentage of successful refinement for both the baseline and Make-it-Real methods, as assessed by both GPT-4V and human users. This allows for a direct comparison of the effectiveness of the Make-it-Real method across different object types and generation techniques.
> <details>
> <summary>read the caption</summary>
> Table 1: GPT evaluation and user preference. GPT's and user's preference comparison on Make-it-Real refined objects sourced from existing 3D assets and state-of-the-art 3D generation methods.
> </details>





### In-depth insights


#### Multimodal Material
A hypothetical research paper section titled "Multimodal Material" would likely explore the use of multiple modalities, such as images, text descriptions, and potentially even 3D scans, to represent and understand materials.  The core idea would be that combining these different data sources offers a richer, more complete representation than any single modality alone.  **This approach could significantly improve material recognition and classification**, especially for complex materials with subtle variations in appearance.  The paper might detail innovative techniques for fusing these different data types, perhaps using deep learning models to learn meaningful representations from the combined data. **A key aspect would be the creation of a comprehensive material database** that integrates these multimodal descriptions, facilitating efficient material retrieval and comparison.  The research could also investigate the challenges involved, such as data acquisition, handling inconsistencies across modalities, and the computational demands of processing large multimodal datasets.  Ultimately, **this approach holds potential for revolutionizing several fields** reliant on accurate material identification and modeling, from virtual reality and computer-aided design to robotics and materials science.

#### GPT-4V Vision
GPT-4V Vision, as a hypothetical multimodal model, presents a significant advancement in AI.  Its potential lies in seamlessly integrating visual and textual information processing, **allowing for a deeper understanding of context and relationships within images**. This could revolutionize applications such as image captioning, object recognition, and question answering about images. However, the model's performance hinges on the quality and diversity of its training data, necessitating **careful consideration of bias and ethical implications**.  Further research should focus on mitigating potential biases, ensuring robustness across diverse datasets, and exploring novel applications where its unique capabilities could offer significant improvements. **Benchmarking GPT-4V Vision against existing state-of-the-art models** will be crucial to validate its claimed advancements.  Furthermore, examining its ability to generalize to unseen data and scenarios is essential for assessing its overall reliability and practical applicability.  Ultimately, GPT-4V Vision holds immense promise, but its successful deployment depends on **addressing potential limitations and ensuring responsible development**.

#### SVBRDF Synthesis
SVBRDF (Specular-Based Bidirectional Reflectance Distribution Function) synthesis is a crucial aspect of realistic 3D rendering.  The goal is to generate a set of maps that accurately describes how light interacts with a material's surface. This typically involves creating albedo, roughness, metallic, normal, specular, height, and displacement maps.  **The core challenge lies in accurately capturing the material's physical properties and translating them into visually convincing textures.**  Approaches often involve sophisticated algorithms that leverage image processing techniques, machine learning models, or both.  **Multimodal models, like GPT-4V, offer a novel approach to synthesizing these maps by leveraging large-scale visual and textual knowledge.**  By effectively recognizing materials from visual cues and using textual descriptions, these models can potentially simplify and automate the creation of high-quality SVBRDF maps.  However, **handling complex interactions between various material properties (e.g., the interplay of roughness and metallic reflection) and maintaining consistency across different parts of a 3D object remain significant hurdles.**  Future work could involve exploring advanced techniques in material representation, improving the accuracy and efficiency of material recognition, and developing more robust methods for handling variations in lighting and shadow.

#### Ablation Study
An ablation study systematically removes components of a model to assess their individual contributions.  In the context of a paper on material generation for 3D models, such a study might investigate the impact of removing or altering different processing stages.  **Key aspects** to analyze would be the effects of removing the material segmentation module, evaluating the impact on accuracy and consistency of material assignment. Similarly, exploring the effects of removing or changing the texture synthesis algorithm would reveal its importance. A robust ablation study will also test the impact of using different pretrained models or even the effect of varying image resolutions or viewpoints on model performance. The results provide critical insights into the model's strengths and weaknesses, guiding future development and improvement. **Important to note** is that the ablation study should meticulously control for extraneous factors. Comparing results with and without specific modules or parameters ensures a clear understanding of each component's specific effect on the final result.  **Analyzing results** allows one to isolate the effects of each component without confounding variables, leading to a better grasp of the system's behavior and identifying areas for optimization.

#### Future Work
The 'Future Work' section of this research paper presents exciting avenues for improvement and expansion.  **Addressing the limitations of relying solely on albedo maps** is crucial; exploring methods to incorporate more robust material property estimation from diverse image inputs or 3D scans would significantly enhance realism.  **Improving the robustness of the material segmentation** process is also key; enhancing the algorithm's ability to handle complex geometries and varied lighting conditions would increase accuracy and efficiency.  **Investigating the integration of other large language models** beyond GPT-4V could reveal alternative approaches to material recognition and texture synthesis, potentially unlocking more powerful capabilities. Finally, **exploring a more comprehensive material library** with even finer-grained annotations and detailed descriptions would further refine the system's ability to match materials with 3D assets, ultimately advancing the capabilities of this automated 3D material painting approach.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/88rbNOtAez/figures_3_1.jpg)

> 🔼 The figure illustrates the overall pipeline of the Make-it-Real method.  It consists of three main stages:  1. **Rendering and Segmentation:** A 3D object is rendered from multiple viewpoints to generate a set of 2D images.  These images are then processed using a segmentation model to identify and separate different material regions within the object. 2. **MLLM-based Retrieval:** A large multimodal language model (MLLM) is used to identify the materials corresponding to each segmented region. This process leverages a comprehensive material library with highly detailed descriptions of various materials. 3. **Material Generation:** Based on the identified materials, SVBRDF (Specular-based Bidirectional Reflectance Distribution Function) maps are generated. These maps provide a complete description of the material properties and are compatible with physically-based rendering engines like Blender. The final output is a 3D model with realistic materials.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall pipeline. This pipeline of Make-it-Real is composed of image rendering and material segmentation, MLLM-based material retrieval, and SVBRDF Maps Generation. We finally use blender engine to conduct physically-based rendering.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_4_1.jpg)

> 🔼 This figure illustrates the process of material retrieval using a large language model (LLM).  A material library is created, containing thousands of materials with detailed descriptions.  The LLM (GPT-4V) uses a hierarchical querying process to efficiently retrieve the appropriate material for a given 3D object part. This hierarchical approach involves querying the main material type, then a more specific subtype, and finally selecting the best match from a list of detailed descriptions.
> <details>
> <summary>read the caption</summary>
> Figure 3: The process of MLLM retrieving materials from the Material Library. Utilizing GPT-4V model, we develop a material library, meticulously generating and cataloging comprehensive descriptions for each material. This structured repository facilitates hierarchical querying for material allocation in subsequent looking up processes.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_5_1.jpg)

> 🔼 This figure shows the mask refinement process in both 2D image space and UV texture space.  The 2D image space refinement (a) uses a semantic segmentation model to identify material regions, then refines these regions by merging overlapping segments based on color similarity. The UV texture space refinement (b) addresses missing parts in the texture map by back-projecting the 2D mask onto the UV map and using feature centroid clustering to complete the missing texture data.
> <details>
> <summary>read the caption</summary>
> Figure 4: Illustrations of mask refinement in 2D image space and UV texture space. (a) We effectively cluster concise material-aware masks compared to original segmented parts from [33]. (b) We fix missing parts on the uv texture space to get a complete texture partition map.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_6_1.jpg)

> 🔼 This figure showcases the qualitative results of applying Make-it-Real to enhance 3D assets that originally only had albedo maps (base color information). The objects, selected from the Objaverse dataset, demonstrate how Make-it-Real effectively adds realistic material properties resulting in significant improvements in visual realism. Each row presents an example with the left side showing the original asset without materials and the right side showing the enhanced asset after processing with Make-it-Real. The different materials are indicated below each object.
> <details>
> <summary>read the caption</summary>
> Figure 5: Qualitative results of Make-it-Real refining 3D asserts without PBR maps. Objects are selected from Objaverse [15] with albedo maps only.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_6_2.jpg)

> 🔼 This figure visualizes several examples of generated texture maps produced by the Make-it-Real model.  It showcases the albedo map (the base color) alongside the generated roughness, metallic, normal, and height maps, demonstrating how these maps work together to create a realistic material appearance. The alignment of the generated material maps with the original albedo map is highlighted, indicating the accuracy and consistency of the material assignment process.
> <details>
> <summary>read the caption</summary>
> Figure 6: Visualization of generated texture maps. We visualize some SVBRDF maps, where the material maps are well aligned with the albedo maps.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_7_1.jpg)

> 🔼 This figure compares the results of Make-it-Real's material refinement with those of several state-of-the-art 3D content generation models. The top row showcases image-to-3D generation models (InstantMesh and TripoSR), while the bottom row displays text-to-3D models.  The comparison highlights Make-it-Real's ability to enhance the realism and detail of materials in 3D assets generated by other methods.
> <details>
> <summary>read the caption</summary>
> Figure 7: Qualitative comparisons between Make-it-Real refining results and 3D objects generated by edge-cutting 3D content creation models. The upper row depicts image-to-3D models (InstantMesh and TripoSR), and the lower row shows results of text-to-3D models.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_8_1.jpg)

> 🔼 This ablation study compares the results of using the Semantic-SAM segmentation method directly versus using the authors' post-processing method for material segmentation on 3D objects. The image shows that the post-processing method leads to more consistent results for material segmentation. 
> <details>
> <summary>read the caption</summary>
> Figure 8: Ablation study of material segmentation refinement. Compared to direct usage of SemanticSAM [33], Our post-process tailored for material segmentation on 3D object can produce more consistent results.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_8_2.jpg)

> 🔼 This ablation study compares the results of material segmentation using SemanticSAM [33] directly versus the post-processing method used in Make-it-Real.  The figure shows that the post-processing method leads to more consistent results in material segmentation for the 3D objects.
> <details>
> <summary>read the caption</summary>
> Figure 8: Ablation study of material segmentation refinement. Compared to direct usage of SemanticSAM [33], Our post-process tailored for material segmentation on 3D object can produce more consistent results.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_16_1.jpg)

> 🔼 This figure illustrates the process of creating a region-level texture partition map. It starts by rendering the 3D object from multiple viewpoints to get multiple images. Then, it uses GPT-4V and a segmentor to identify and segment the materials in each image. Next, the identified material regions (masks) from these images are projected back onto the 3D mesh using UV unwrapping. Finally, the segmented regions are refined to get the precise partition map of different materials.
> <details>
> <summary>read the caption</summary>
> Figure 10: Region-level texture partition module. This module extracts and back-projects localized rendered images on to a 3D mesh, using UV unwrapping for texture segmentation, thereby resulting in precise partition map of different materials.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_17_1.jpg)

> 🔼 This figure illustrates the pixel-level albedo-referenced estimation module. The process begins by referencing albedo maps, employing a KD-Tree algorithm for efficient nearest neighbor searches to find similar pixels between the query and key albedo maps. It then normalizes colors using histogram equalization.  This method enables the precise generation of spatially varying BRDF maps (albedo, metalness, roughness, specular, normal, and height) while maintaining consistency with the original albedo map.  The figure showcases the workflow from query albedo to the final texture maps (part and final texture maps).
> <details>
> <summary>read the caption</summary>
> Figure 11: Pixel-level albedo-referenced estimation module. We generate spatially variant BRDF maps by referencing albedo maps, employing KD-Tree algorithm for efficient nearest neighbor searches, and normalizing colors via histogram equalization.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_18_1.jpg)

> 🔼 This figure illustrates the process of material retrieval using a large multimodal language model (MLLM), specifically GPT-4V.  A material library is created containing thousands of materials with detailed descriptions.  GPT-4V uses a hierarchical querying process to efficiently retrieve the most suitable materials for allocation to different parts of a 3D object.
> <details>
> <summary>read the caption</summary>
> Figure 3: The process of MLLM retrieving materials from the Material Library. Utilizing GPT-4V model, we develop a material library, meticulously generating and cataloging comprehensive descriptions for each material. This structured repository facilitates hierarchical querying for material allocation in subsequent looking up processes.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_19_1.jpg)

> 🔼 The figure illustrates the overall pipeline of the Make-it-Real method. It consists of three main stages: 1) Rendering and Segmentation: A 3D object is rendered from multiple views, and semantic segmentation is applied to the resulting images to identify distinct material regions.  2) MLLM-based Material Retrieval: A large language model (MLLM) is used to retrieve materials from a comprehensive material library based on the segmented regions.  3) Material Generation: SVBRDF (Specular-based Bidirectional Reflectance Distribution Function) maps are generated for each region,  enabling physically-based rendering. Finally, the Blender engine is used for physically based rendering. 
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall pipeline. This pipeline of Make-it-Real is composed of image rendering and material segmentation, MLLM-based material retrieval, and SVBRDF Maps Generation. We finally use blender engine to conduct physically-based rendering.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_19_2.jpg)

> 🔼 The figure illustrates the overall pipeline of the Make-it-Real method, which consists of three main stages: 1) Rendering and Segmentation: This stage involves rendering the input albedo mesh from various viewpoints to obtain a series of images, followed by employing Semantic-SAM for semantic segmentation.  2) MLLM-based Retrieval: GPT-4V is used to retrieve matching materials from a comprehensive material library based on visual cues and hierarchical text prompts.  3) Material Generation: The matched materials are then meticulously applied as reference for new SVBRDF material generation, significantly enhancing their visual authenticity. Finally, physically-based rendering is conducted using the Blender engine.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall pipeline. This pipeline of Make-it-Real is composed of image rendering and material segmentation, MLLM-based material retrieval, and SVBRDF Maps Generation. We finally use blender engine to conduct physically-based rendering.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_19_3.jpg)

> 🔼 This figure showcases the capabilities of Make-it-Real by demonstrating its application to diverse 3D objects, ranging from CAD designs to outputs from generative models.  The key features highlighted are the enhancement of realism, the ability to assign materials specifically to individual object parts, and the generation of physically-based rendering (PBR) maps compatible with various 3D rendering software. The figure visually demonstrates the versatility and effectiveness of Make-it-Real in enhancing 3D object realism.
> <details>
> <summary>read the caption</summary>
> Figure 1: Usage of Make-it-Real. Our method can refine a wide range of albedo-map-only 3D objects from both CAD design and generative models. Our method enhances the realism of objects, enables part-specific material assignment to objects and generate PBR maps that are compatible with downstream engines.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_19_4.jpg)

> 🔼 This figure shows several examples of 3D objects that have been enhanced using the Make-it-Real method. The objects range in complexity and style and include both CAD models and those generated by machine learning models. The figure highlights the ability of Make-it-Real to add realistic material properties to these objects and generate physically-based rendering (PBR) maps.
> <details>
> <summary>read the caption</summary>
> Figure 1: Usage of Make-it-Real. Our method can refine a wide range of albedo-map-only 3D objects from both CAD design and generative models. Our method enhances the realism of objects, enables part-specific material assignment to objects and generate PBR maps that are compatible with downstream engines.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_19_5.jpg)

> 🔼 This figure shows several examples of 3D objects processed by the Make-it-Real method.  The original models only contained albedo maps (color information), lacking detailed material properties. Make-it-Real enhances these by adding realistic material properties such as metal, plastic and rubber.  It also demonstrates part-specific material assignment, meaning different parts of the same object can have different materials. Finally, it shows that Make-it-Real produces physically-based rendering (PBR) maps compatible with 3D rendering software.
> <details>
> <summary>read the caption</summary>
> Figure 1: Usage of Make-it-Real. Our method can refine a wide range of albedo-map-only 3D objects from both CAD design and generative models. Our method enhances the realism of objects, enables part-specific material assignment to objects and generate PBR maps that are compatible with downstream engines.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_19_6.jpg)

> 🔼 This figure shows several examples of 3D models processed by the Make-it-Real method.  The input to the method is a 3D model with only an albedo map (a color map representing surface color without material properties).  Make-it-Real enhances these models by adding realistic material properties, enabling part-specific material assignment and generating physically-based rendering (PBR) maps.  The PBR maps allow the enhanced models to be rendered in a game engine or other 3D rendering software with accurate material appearance.
> <details>
> <summary>read the caption</summary>
> Figure 1: Usage of Make-it-Real. Our method can refine a wide range of albedo-map-only 3D objects from both CAD design and generative models. Our method enhances the realism of objects, enables part-specific material assignment to objects and generate PBR maps that are compatible with downstream engines.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_19_7.jpg)

> 🔼 The figure illustrates the three main stages of the Make-it-Real pipeline: 1) Rendering and segmentation: A 3D mesh is rendered from multiple viewpoints, and a semantic segmentation model is used to identify different material regions. 2) MLLM-based material retrieval: A large language model (GPT-4V) retrieves materials from a comprehensive library based on visual and textual cues. 3) Material generation: SVBRDF maps are generated based on the retrieved materials and the original albedo map, which are then used for physically-based rendering in Blender. 
> <details>
> <summary>read the caption</summary>
> Figure 2: Overall pipeline. This pipeline of Make-it-Real is composed of image rendering and material segmentation, MLLM-based material retrieval, and SVBRDF Maps Generation. We finally use blender engine to conduct physically-based rendering.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_19_8.jpg)

> 🔼 This figure shows the detailed prompts used in the GPT-4V based material matching process. The prompts are hierarchical, starting with a high-level prompt to identify the main material type, followed by more specific prompts to narrow down the selection to a specific material within the category. The prompts utilize visual cues and textual descriptions to guide the GPT-4V model in selecting the most appropriate material for each segment of the object.  The figure also shows examples of the output from the GPT-4V model, including the material name and description. The prompts are designed to be iterative, allowing for refinement of the material selection as the process progresses. This is achieved via a three-level tree structure to reduce memory usage while increasing accuracy.
> <details>
> <summary>read the caption</summary>
> Figure 13: Detailed prompts of GPT-4V based material matching. Prompts in blue changes according to the current assigning part and GPT-4V's results.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_20_1.jpg)

> 🔼 The figure shows two sets of four images each, comparing the textures of a 3D model before and after applying the proposed Make-it-Real method.  The top row shows the original textures, while the bottom row shows the textures refined by Make-it-Real.  A prompt is provided for GPT-4V (a large multimodal language model) to evaluate which set of images are more photorealistic.  This is part of a user study evaluating the method's effectiveness in enhancing the realism of 3D model textures.
> <details>
> <summary>read the caption</summary>
> Figure 14: GPT-4V based evaluation prompts. We define a prompt for GPT-4V to generate human-aligned comparison over objects before and after our texture refinement.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_21_1.jpg)

> 🔼 This figure compares the proposed method with a previous method, Material Palette, in terms of material identification and extraction. The main difference is that Make-it-Real handles a more complex task: the input is a rendered image with only albedo information, and the output is full object textures, while Material Palette processes real images with region-level material extraction.
> <details>
> <summary>read the caption</summary>
> Figure 15: Comparison between previous method and Make-it-Real. We demonstrate the distinctions between Material Palette [40] and our method in terms of material identification and extraction. Our overall pipeline presents a more challenging task, where the input is a rendered image with only albedo information, and the output consists of textures for the entire object.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_22_1.jpg)

> 🔼 This figure shows the impact of different texture maps (metalness, roughness, displacement, and height) on the appearance of 3D objects.  For each object, the left image shows the result without the specific texture map, and the right image shows the result with the texture map applied. Comparing the left and right images demonstrates the effect each map has on the final appearance, highlighting how these maps contribute to realism and visual fidelity.
> <details>
> <summary>read the caption</summary>
> Figure 16: Effects of different texture maps. We evaluate the effects of metalness, roughness, and displacement/height maps on the appearance of 3D objects.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_23_1.jpg)

> 🔼 This figure demonstrates the impact of different UV mapping techniques on the quality of the generated texture maps. Different UV mapping methods, such as 'ori_uv', 'light_map_uv', 'smart_uv', 'sphere_uv', and 'unwrap_uv', were used and their results for albedo and specular maps are compared. The results show that inappropriate UV mapping can lead to issues such as fragmentation and color entanglement, affecting the quality of the final textures.  The bottom row shows the effect of different UV mappings on a can model, highlighting the importance of using appropriate UV mapping for consistent results.
> <details>
> <summary>read the caption</summary>
> Figure 17: Effects of different UV mappings of input mesh.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_23_2.jpg)

> 🔼 This figure provides a visual comparison of the material textures generated by Make-it-Real against those created by human artists.  It showcases the ability of Make-it-Real to generate realistic material maps for various 3D objects, including a cannon, chair, boot, telephone, speakers, shield, barrel, and trumpet. The results demonstrate Make-it-Real's capacity to produce PBR maps that are visually comparable to those generated by experienced artists, highlighting the realism and accuracy of the approach.
> <details>
> <summary>read the caption</summary>
> Figure 18: More Comparisons between Make-it-Real and Artist-Created Materials.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_24_1.jpg)

> 🔼 This figure displays several examples of 3D objects enhanced by the Make-it-Real method.  Each row shows a comparison between the original object (left, with only albedo map) and the refined object (right, with added material maps generated using Make-it-Real). The goal is to show how Make-it-Real improves the realism and visual detail of the objects by adding realistic materials.
> <details>
> <summary>read the caption</summary>
> Figure 20: More qualitative results of Make-it-Real refining existing 3D assets without material. Objects are selected from Objaverse [16] with albedo only.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_26_1.jpg)

> 🔼 This figure demonstrates the qualitative results of applying the Make-it-Real method to 3D assets that originally only had albedo maps (i.e., color information, without other physical material properties like roughness or metalness).  The figure shows several examples of 3D objects before and after the Make-it-Real process, highlighting the improved realism in material appearance achieved by the method. The objects were sourced from the Objaverse dataset.
> <details>
> <summary>read the caption</summary>
> Figure 5: Qualitative results of Make-it-Real refining 3D asserts without PBR maps. Objects are selected from Objaverse [15] with albedo maps only.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_27_1.jpg)

> 🔼 This figure shows several examples of 3D objects enhanced by the Make-it-Real method. Each row displays an original 3D object from the Objaverse dataset with only an albedo map (left) and the same object after being enhanced by Make-it-Real (right). The right side shows the object with realistic materials applied, with the specific materials used listed below each object. This demonstrates the ability of the method to enhance the realism and visual quality of 3D objects by automatically assigning and rendering realistic materials based only on an albedo map.
> <details>
> <summary>read the caption</summary>
> Figure 21: More qualitative results of Make-it-Real refining existing 3D assets without material. Objects are selected from Objaverse [16] with albedo only.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_28_1.jpg)

> 🔼 This figure shows the generated texture maps for multiple 3D objects.  The first column displays the original albedo maps (base colors) of the objects. Subsequent columns illustrate the corresponding material maps produced by the Make-it-Real model, including metalness, roughness, specular, normal, and height maps. This demonstrates the model's ability to generate realistic material textures that complement the base albedo.
> <details>
> <summary>read the caption</summary>
> Figure 22: Visualization of generated texture maps. The first column represents the original query albedo map of 3D objects, while the subsequent columns showcase the corresponding material maps generated by Make-it-Real.
> </details>



![](https://ai-paper-reviewer.com/88rbNOtAez/figures_29_1.jpg)

> 🔼 This figure shows the qualitative results of applying the Make-it-Real method to 3D assets from the Objaverse dataset that originally only had albedo maps (color information).  The figure demonstrates the improvement in realism achieved by adding realistic material properties such as metalness, roughness, and other details.  Each row shows an object before and after processing with Make-it-Real, highlighting the increased visual fidelity.
> <details>
> <summary>read the caption</summary>
> Figure 5: Qualitative results of Make-it-Real refining 3D asserts without PBR maps. Objects are selected from Objaverse [15] with albedo maps only.
> </details>



</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/88rbNOtAez/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/88rbNOtAez/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}