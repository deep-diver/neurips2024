---
title: "Denoising Diffusion Path: Attribution Noise Reduction with An Auxiliary Diffusion Model"
summary: "Denoising Diffusion Path (DDPath) uses diffusion models to dramatically reduce noise in attribution methods for deep neural networks, leading to clearer explanations and improved quantitative results."
categories: ["AI Generated", ]
tags: ["AI Theory", "Interpretability", "🏢 School of Computer Science, Fudan University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} bSv0MBDBF2 {{< /keyword >}}
{{< keyword icon="writer" >}} Yiming Lei et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=bSv0MBDBF2" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/bSv0MBDBF2" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=bSv0MBDBF2&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/bSv0MBDBF2/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Attribution methods for deep learning models, such as Integrated Gradients (IG), aim to explain predictions by tracing gradients along a path from a baseline to the target image. However, noise accumulation along this path often distorts the explanation. Existing solutions focus on finding alternative paths, but they overlook the critical issue that intermediate images may not accurately reflect the model's training data distribution, thus amplifying noise. This makes it difficult to pinpoint the features truly driving a model's decisions. 

This paper introduces Denoising Diffusion Path (DDPath), a novel method that leverages diffusion models to mitigate noise. DDPath constructs a piecewise linear path, using diffusion models to gradually denoise the image at each step ensuring the images stay within the model's training data distribution. This leads to a cleaner path, free from distortions caused by noise.  DDPath is theoretically sound and can be readily integrated with existing techniques.  Extensive experiments show DDPath significantly reduces noise, resulting in clearer explanations and better quantitative performance (insertion and deletion curves, accuracy). 

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} DDPath utilizes diffusion models to significantly reduce noise during the attribution process, leading to more reliable explanations. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method is theoretically compatible with existing path-based attribution methods, enhancing their accuracy and effectiveness. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} DDPath demonstrates improved quantitative results compared to traditional path-based methods in experiments. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on explainable AI (XAI) and attribution methods.  It **directly addresses the critical issue of noise in path-based attribution methods**, a common problem hindering accurate interpretation of deep learning models. By introducing a novel approach using diffusion models, it provides a significant advancement in the field, improving the reliability and clarity of model explanations.  This work is also highly relevant to researchers in the broader field of generative modeling and its application in AI, opening up new research avenues in integrating these techniques for improved model interpretability and trust.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_1_1.jpg)

> 🔼 This figure illustrates the core idea of the proposed Denoising Diffusion Path (DDPath) method.  Panel (a) shows traditional attribution paths, which are often noisy and do not reflect the data distribution. Panel (b) contrasts this by showing how DDPath generates a smoother, more data-consistent path using diffusion models. Panel (c) compares the softmax probabilities along these different paths, highlighting the improved smoothness and natural behavior of DDPath.
> <details>
> <summary>read the caption</summary>
> Figure 1: Motivation illustration of DDPath. The symbol x' denotes the baseline image and x the target image. (a) The existing paths are irrelevant to data distributions. (b) The proposed denoising diffusion path approaches the distribution of real data. (c) Traditional IG [8] and SmoothGrad [9] struggle to maintain a continuously increasing Softmax probability along the integration path. This behavior can be counterintuitive and contradict human cognition, where the confidence in a prediction should generally rise as evidence accumulates. In contrast, the proposed DDPath achieves a more natural behavior by ensuring a continuously increasing Softmax probability along the path, even if the path itself exhibits fluctuations.
> </details>





![](https://ai-paper-reviewer.com/bSv0MBDBF2/tables_7_1.jpg)

> 🔼 This table presents a quantitative comparison of various interpretation methods on the ImageNet validation dataset, focusing on insertion and deletion scores.  The methods compared include Guided BP, Integrated Gradients (IG), SmoothGrad, BlurIG, GIG, and three variants incorporating the proposed Denoising Diffusion Path (DDPath): DDPath-BlurIG, DDPath-GIG, and DDPath-IG. For each method, the table shows the insertion score (higher is better), the deletion score (lower is better), and the overall score (the difference between insertion and deletion, also higher is better), providing a comprehensive evaluation of each method's performance in terms of attribution quality.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative comparisons of different interpretation methods on ImageNet validation set in terms of Insertion and Deletion. Overall = Insertion - Deletion.
> </details>





### In-depth insights


#### Diffusion Path Denoising
Diffusion path denoising is a crucial technique in attribution methods for deep neural networks.  Traditional attribution methods often suffer from noise accumulation along the path from a baseline to a target image, which distorts explanations. **Denoising diffusion models offer a powerful solution** by progressively removing noise during the reverse process. This technique creates a smoother, more interpretable path, ensuring intermediate steps remain within the distribution of the training data and facilitating clearer explanations.  **This approach effectively reduces noise-induced artifacts in attribution maps**, leading to more reliable and consistent interpretations of a model's decision-making process.  By leveraging the inherent ability of diffusion models to progressively refine noisy inputs, this method provides a **significant improvement over traditional gradient-based methods** that are highly susceptible to noise interference.  The key advantage lies in its ability to generate intermediate images closer to the data distribution used to train the neural network. This reduces noise by focusing attention on relevant features rather than spurious noisy aspects of the gradient trajectory.  **Further research could focus on optimizing the sampling process** within diffusion models for even more accurate and efficient denoising, as well as exploring the application of this technique to other XAI methods.

#### Attribution Noise
Attribution noise, a critical challenge in explaining deep learning model predictions, arises from the accumulation of errors during the attribution process.  Methods like Integrated Gradients (IG) attempt to explain predictions by accumulating gradients along a path, but noise along this path distorts the explanation.  **The noise stems from intermediate image representations that deviate from the training data distribution,** amplifying the impact of noise.  Existing methods address this by seeking alternative paths or smoothing techniques, but often overlook the core issue of data distribution shifts. **A promising approach focuses on denoising techniques, specifically leveraging the capabilities of diffusion models to progressively remove noise and guide the attribution path toward more realistic and relevant image states.**  This results in clearer attributions, as demonstrated through improved qualitative and quantitative metrics on saliency map quality and insertion/deletion curve analysis. This focus on denoising directly tackles the inherent noisy nature of gradients and path-based attribution, potentially leading to a new generation of more robust and reliable explainability methods.

#### Auxiliary Diffusion Model
An auxiliary diffusion model, in the context of denoising a diffusion path for attribution noise reduction in deep neural networks, likely plays a crucial supporting role.  It doesn't directly generate attributions, but instead refines intermediate steps within the path generation process.  **The core function is to ensure that each intermediate image along the path remains close to the data distribution of the training dataset**, thereby mitigating the noise amplification that traditional methods suffer from. This is achieved by leveraging the inherent noise reduction capabilities of diffusion models.  **The auxiliary model acts as a denoiser**, progressively cleaning the intermediate representations, making the gradients calculated at these steps more reliable and relevant.  **This indirect contribution** is vital to the overall success of the denoising process and ultimately leads to cleaner and more accurate attributions. The model is likely pre-trained and acts as a critical component within a larger attribution framework.

#### Axiomatic Properties
The axiomatic properties section of a research paper is crucial for establishing the validity and reliability of proposed attribution methods.  It delves into the fundamental properties that a good attribution method should ideally satisfy, ensuring that the explanations provided are not only accurate but also meaningful and consistent.  These properties often include **sensitivity**, ensuring that the attribution method is responsive to changes in input features, and **implementation invariance**, guaranteeing that explanations remain consistent despite variations in model architecture or implementation details.  A discussion of axiomatic properties also frequently includes **completeness**, which ensures that the sum of attributions accounts for the total change in prediction, and **symmetry**, which addresses the fairness of attributions.  A thorough examination of axiomatic properties within a research paper provides a rigorous framework for evaluating and comparing different attribution methods, ultimately contributing to a more transparent and reliable understanding of deep neural networks.

#### Classifier Guidance
Classifier guidance, in the context of diffusion models for attribution, is a crucial technique that leverages a pre-trained classifier to steer the denoising process.  Instead of relying solely on the inherent properties of the diffusion model, **classifier guidance ensures that the generated samples at each step of the reverse diffusion process align more closely with the data distribution relevant to the classification task**. This is particularly important for attribution methods because intermediate samples that significantly deviate from the training data can introduce noise and artifacts into the explanation. By guiding the diffusion process with a classifier, the model is less prone to producing unrealistic intermediate images, resulting in **more accurate and stable attributions**.  This method improves the quality of explanation because **noise accumulation along the path is reduced**, resulting in clearer visualization of important features. The effectiveness of this approach is validated through quantitative evaluation metrics and qualitative comparison of saliency maps, demonstrating that classifier guidance enhances the reliability and interpretability of deep neural network explanations.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_3_1.jpg)

> 🔼 This figure illustrates the Denoising Diffusion Path (DDPath) method.  It shows a sequence of images generated by a pre-trained diffusion model, starting from a noisy baseline image (x0) and progressing towards the target image (xT). Each image (xi) along the path is sampled from a Gaussian distribution N(μθ(xi), Σθ(xi)), where the mean μθ(xi) is guided by a classifier φ to ensure that the samples remain close to the data distribution and progressively reduce noise. The classifier's role is to guide the diffusion process, ensuring that the sampled images are progressively closer to the target image. The parameter θ represents the parameters of the pre-trained diffusion model, which are kept frozen during the path generation. The probability Pφ(Y|xi) represents the classifier's confidence in predicting the class Y given the image xi. The figure depicts how the DDPath gradually denoises the image while maintaining a high classification probability, leading to a clear and informative attribution.
> <details>
> <summary>read the caption</summary>
> Figure 2: Illustration of DDPath. At each step in the DDPath, the images are sampled from a pre-trained diffusion model θ guided by a classifier φ.
> </details>



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_6_1.jpg)

> 🔼 This figure compares the saliency maps generated by different attribution methods (IG, SmoothGrad, BlurIG, GIG, and DDPath-IG) with their corresponding insertion and deletion curves.  The insertion curve shows how the model's prediction changes as progressively more information is added to a blurred version of the image, guided by the saliency map. The deletion curve shows the opposite, illustrating how the prediction changes as information is removed from the original image.  The figure demonstrates that DDPath-IG produces cleaner and more informative saliency maps, leading to better performance on both insertion and deletion tasks.
> <details>
> <summary>read the caption</summary>
> Figure 3: Comparison of saliency maps and corresponding Insertion and Deletion curves. Image examples are selected from the ImageNet-1k validation set. The classification model is the pre-trained VGG-19 [29].
> </details>



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_7_1.jpg)

> 🔼 This figure compares the saliency maps and insertion/deletion curves of different attribution methods (IG, SmoothGrad, BlurIG, GIG, and DDPath-IG) on images from the ImageNet-1k validation set.  The pre-trained VGG-19 model was used for classification.  The insertion curve shows how the model's prediction changes as pixels are gradually added based on saliency map values, while the deletion curve reflects the effect of removing those pixels.  The DDPath-IG method is highlighted as it aims to improve upon the other methods by reducing noise during path integration. 
> <details>
> <summary>read the caption</summary>
> Figure 3: Comparison of saliency maps and corresponding Insertion and Deletion curves. Image examples are selected from the ImageNet-1k validation set. The classification model is the pre-trained VGG-19 [29].
> </details>



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_8_1.jpg)

> 🔼 This figure compares the saliency maps generated by the DDPath-IG method using two different scaling schemes: the default scheme and a reversed scheme. The default scheme shows more focused and detailed saliency maps, while the reversed scheme produces more dispersed and noisy results. This visualization helps demonstrate the impact of the scaling scheme on the effectiveness of the DDPath-IG method in generating accurate and informative saliency maps.
> <details>
> <summary>read the caption</summary>
> Figure 5: Saliency maps by different scaling schemes.
> </details>



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_13_1.jpg)

> 🔼 This figure compares different attribution methods (IG, SmoothGrad, BlurIG, GIG, and DDPath-IG) by visualizing their saliency maps and corresponding insertion and deletion curves.  The insertion curves show how adding pixels back into a blurred image, guided by the saliency map, affects the model's prediction. Conversely, the deletion curves show the impact of removing pixels, again based on the saliency map.  The ImageNet-1k validation dataset and the pre-trained VGG-19 model are used for this comparison. The results visually demonstrate the difference in noise reduction and localization ability between these methods.
> <details>
> <summary>read the caption</summary>
> Figure 3: Comparison of saliency maps and corresponding Insertion and Deletion curves. Image examples are selected from the ImageNet-1k validation set. The classification model is the pre-trained VGG-19 [29].
> </details>



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_14_1.jpg)

> 🔼 This figure displays the saliency maps generated by various attribution methods (IG, Smooth IG, Blur IG, Guided IG, and DDPath IG) using a ResNet-50 model for a set of images.  Each row represents a different input image, with the first column showing the original image. The subsequent columns illustrate the saliency maps produced by each method, highlighting the image regions deemed most important for classification.  Comparing the saliency maps across different methods helps to visualize and assess the effectiveness and differences in the attribution methods.
> <details>
> <summary>read the caption</summary>
> Figure 7: Saliency maps obtained by ResNet-50.
> </details>



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_15_1.jpg)

> 🔼 This figure visualizes the impact of diffusion model size on the resulting saliency maps generated using the DDPath-IG method.  It shows that larger diffusion models (with higher resolutions, 512x512 being the largest) produce higher-resolution saliency maps that reveal more fine-grained details in the image compared to smaller models (64x64 being the smallest).  This demonstrates how the choice of diffusion model influences the level of detail captured in the explanation of a DNN's prediction.
> <details>
> <summary>read the caption</summary>
> Figure 8: Saliency maps generated by DDPath-IG using diffusion models of varying sizes.
> </details>



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_15_2.jpg)

> 🔼 This figure compares the saliency maps and insertion/deletion curves generated by various attribution methods, including the proposed DDPath-IG, on images from the ImageNet-1k validation set. The results show that DDPath-IG generates saliency maps with clearer details and achieves better quantitative results in terms of insertion and deletion scores than other methods.
> <details>
> <summary>read the caption</summary>
> Figure 3: Comparison of saliency maps and corresponding Insertion and Deletion curves. Image examples are selected from the ImageNet-1k validation set. The classification model is the pre-trained VGG-19 [29].
> </details>



![](https://ai-paper-reviewer.com/bSv0MBDBF2/figures_16_1.jpg)

> 🔼 This figure visualizes the impact of different scaling schemes on saliency map generation using the DDPath-IG method.  Three different values for the parameter 'a' (0.5, 1, and 2) are tested, each impacting how the sampling mean and variance are scaled during the diffusion process.  The results are presented for four example images showing the original input image alongside the saliency maps generated by the baseline IG method and the DDPath-IG method with the three different 'a' values. By comparing the saliency maps, the effects of changing the scaling parameter 'a' on the quality and detail captured in the resulting maps become apparent.
> <details>
> <summary>read the caption</summary>
> Figure 10: Saliency maps generated by different scaling schemes with \(a \in \{0.5, 1, 2\}.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/bSv0MBDBF2/tables_8_1.jpg)
> 🔼 This table presents the quantitative results of a pointing game experiment conducted on the MS COCO 2017 validation set to evaluate the effectiveness of different interpretation methods in pinpointing the most salient pixels.  The metrics used are the percentage of 'Hits' (salient pixels correctly identified within the ground truth bounding box) and 'Misses' (salient pixels incorrectly identified outside of the bounding box).  Higher scores indicate better localization accuracy of the salient pixels. The table compares the performance of several methods, including IG (Integrated Gradients), DDPath-IG (Denoising Diffusion Path - Integrated Gradients), BlurIG, DDPath-BlurIG, GIG (Guided Integrated Gradients), and DDPath-GIG.  It shows that the DDPath methods consistently outperform the baseline methods, indicating that DDPath is more effective at highlighting the most relevant image regions for prediction while reducing noise and inaccuracies in the selection of salient pixels.
> <details>
> <summary>read the caption</summary>
> Table 2: Pointing game evaluation on MS COCO 2017 validation set.
> </details>

![](https://ai-paper-reviewer.com/bSv0MBDBF2/tables_8_2.jpg)
> 🔼 This table presents a quantitative comparison of various interpretation methods (Guided BP, IG, SmoothGrad, BlurIG, GIG, DDPath-IG, DDPath-BlurIG, and DDPath-GIG) on the ImageNet validation set.  The comparison is based on insertion and deletion scores, which assess the model's ability to correctly identify relevant image regions.  Higher insertion scores and lower deletion scores indicate better performance. The 'Overall' score represents the difference between insertion and deletion, providing a combined measure of the methods' effectiveness.
> <details>
> <summary>read the caption</summary>
> Table 1: Quantitative comparisons of different interpretation methods on ImageNet validation set in terms of Insertion and Deletion. Overall = Insertion - Deletion.
> </details>

![](https://ai-paper-reviewer.com/bSv0MBDBF2/tables_8_3.jpg)
> 🔼 This table compares the performance of adding noise to different attribution methods (IG, BlurIG, GIG, and SmoothGrad) with the proposed DDPath method. The metrics used for comparison are AIC (Accuracy Information Curve), Insertion, and Deletion. Higher AIC and Insertion values and lower Deletion values indicate better performance. The table shows that DDPath outperforms other noisy methods in terms of AIC and Insertion, suggesting its superior ability to reduce noise and generate accurate attributions.
> <details>
> <summary>read the caption</summary>
> Table 4: Comparison of Adding Noise.
> </details>

![](https://ai-paper-reviewer.com/bSv0MBDBF2/tables_16_1.jpg)
> 🔼 This table presents the results of a pointing game experiment conducted on the MS COCO 2017 validation set to evaluate the effectiveness of different interpretation methods in pinpointing the most salient pixels.  The metrics used are Hits and Misses, where Hits count the number of most salient pixels that fall within the ground truth bounding boxes, and Misses count those that don't.  Higher scores indicate better localization accuracy. The table compares the performance of Integrated Gradients (IG), and its variants combined with the Denoising Diffusion Path (DDPath).
> <details>
> <summary>read the caption</summary>
> Table 2: Pointing game evaluation on MS COCO 2017 validation set.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/bSv0MBDBF2/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}