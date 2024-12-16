---
title: "Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation"
summary: "This research introduces adversarial concept preservation, a novel method for safely erasing undesirable concepts from diffusion models, outperforming existing techniques by preserving related sensiti..."
categories: ["AI Generated", ]
tags: ["Computer Vision", "Image Generation", "🏢 Monash University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} GDz8rkfikp {{< /keyword >}}
{{< keyword icon="writer" >}} Anh Tuan Bui et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=GDz8rkfikp" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/GDz8rkfikp" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/GDz8rkfikp/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Text-to-image diffusion models, while powerful, often generate harmful content due to biases in their training data.  Existing methods for removing this content struggle to balance complete removal with preserving the model's overall capabilities.  These approaches usually rely on preserving a neutral concept, which may not guarantee performance. 

This paper proposes 'adversarial concept preservation'. This novel method focuses on identifying and preserving concepts most sensitive to the removal of the target concept (i.e., adversarial concepts).  By doing so, the model can reliably erase harmful content while minimizing effects on other aspects. Experiments show this method outperforms existing techniques in removing unwanted content while retaining image quality and coherence.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Adversarial concept preservation effectively removes unwanted concepts from diffusion models while minimizing impact on other concepts. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method identifies and preserves concepts most affected by modifications, ensuring stable erasure. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The approach outperforms existing methods in eliminating harmful content while maintaining the model's integrity. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents a novel approach to safely remove harmful content from large language models.  It offers a solution to a critical problem in AI safety and opens avenues for further research in adversarial machine learning and concept manipulation techniques.  The proposed method shows significant improvement over existing methods, offering a more robust and effective way to sanitize AI models and enhance their usability.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_2_1.jpg)

> 🔼 This figure analyzes how erasing a target concept from a diffusion model affects the generation of other concepts.  The impact is quantified using the CLIP score difference between the original and sanitized models. Subfigure (a) shows the impact of erasing 'nudity' and 'garbage truck', separately. Subfigure (b) compares the effects of erasing 'garbage truck' while preserving different concepts: a neutral concept (''), a related concept ('lexus', 'road'), and the most sensitive concept (as determined by the proposed method). This illustrates the impact of different preservation strategies on overall model performance.
> <details>
> <summary>read the caption</summary>
> Figure 1: Analysis of the impact of erasing the target concept on the model's capability. The impact is measured by the difference of CLIP score d(c) between the original model and the corresponding sanitized model. 1a: Impact of erasing 'nudity' or 'garbage truck' to other concepts. 1b: Comparing the impact of erasing the same 'garbage truck' to other concepts with different preserving strategies, including preserving a fixed concept such as '', 'lexus', or 'road', and adaptively preserving the most sensitive concept found by our method.
> </details>





![](https://ai-paper-reviewer.com/GDz8rkfikp/tables_7_1.jpg)

> 🔼 This table presents the quantitative results of erasing object-related concepts from the Imagenette dataset. Four different sets of five classes were selected for erasure, and the performance of five methods (original Stable Diffusion model, ESD, UCE, CA, and the proposed method) was evaluated using two metrics: Erasing Success Rate (ESR) and Preserving Success Rate (PSR).  ESR measures the percentage of generated images where the object from the to-be-erased classes is not detected (in top-1 or top-5 predictions using ResNet-50), reflecting the effectiveness of erasure. PSR measures the percentage of generated images where the objects from the to-be-preserved classes are detected, reflecting the preservation performance.  Higher ESR and PSR values indicate better performance.
> <details>
> <summary>read the caption</summary>
> Table 1: Erasing object-related concepts.
> </details>





### In-depth insights


#### Adversarial Erasure
Adversarial erasure, in the context of diffusion models, presents a nuanced approach to removing undesirable concepts.  **The core idea is to identify and preserve concepts most affected by the removal process**, thus minimizing the collateral damage to other parts of the model. This is in contrast to methods that rely on preserving neutral concepts, which may not fully protect the integrity of the model's overall performance.  **Empirically, this adversarial approach demonstrates superior performance in eliminating unwanted elements while maintaining the integrity of other concepts.** The technique's effectiveness is showcased through the use of the Stable Diffusion model, highlighting its ability to selectively remove undesirable concepts without significant degradation to other, unrelated content. This approach to concept erasure emphasizes a more robust and precise method, going beyond simpler techniques that often result in significant model degradation and suboptimal results.

#### Concept Sensitivity
Concept sensitivity in the context of text-to-image diffusion models refers to **how susceptible different concepts are to unintended changes** during the process of removing undesired concepts.  **Not all concepts are created equal; some are more tightly interwoven with others**, making their preservation during the fine-tuning process crucial.  Empirically, the removal of a target concept has a cascading effect, impacting related concepts more strongly than unrelated ones. **Neutral concepts often show less sensitivity**, potentially because their representations are less tightly coupled within the model's parameter space.  Therefore, strategies that focus on preserving those concepts most affected by the removal of the target concept (**adversarial concepts**) are more effective.  This approach ensures a more stable erasure while minimizing the unwanted collateral damage to the overall model performance. Identifying adversarial concepts requires a nuanced understanding of the model's internal concept representations and their interdependencies.

#### CLIP Score Analysis
A CLIP Score Analysis in a research paper would likely involve using the CLIP (Contrastive Language–Image Pre-training) model to quantitatively evaluate the alignment between generated images and their corresponding text descriptions.  This is crucial for assessing the quality and relevance of generated content, particularly when studying the effects of concept erasure or manipulation in image generation models.  The analysis would likely involve calculating CLIP similarity scores between generated images and their prompts, comparing those scores across different conditions (e.g., before and after concept removal, using different concept preservation methods), and potentially correlating these scores with other metrics, such as image quality scores or human evaluations. **Key insights might include how well the model maintains the desired concepts after erasure, identifying any unintended side effects, and comparing the performance of different methods**  for concept preservation. **A robust analysis would also consider potential limitations** of using CLIP scores as an evaluation metric, such as their reliance on a specific pre-trained model and the inherent biases of the training data. A thorough analysis would help draw meaningful conclusions about the effectiveness and potential pitfalls of various approaches to controlling the content of generative models.

#### Method Limitations
A discussion on 'Method Limitations' necessitates a thorough examination of the research paper's methodology.  **Identifying any assumptions made during the research process is critical.**  Were there limitations in data collection, potentially influencing results?  What about the chosen model's inherent biases?  **Addressing the generalizability of findings is paramount.** Do the results convincingly extrapolate beyond the specific dataset used?  **Computational constraints, such as processing power or time limitations, often restrict the scope of analyses.** Were any alternative approaches considered but deemed infeasible?  **The chosen evaluation metrics should be critically assessed.** Did these fully capture the nuances of the studied phenomenon, or were there limitations in their sensitivity or ability to discern meaningful effects?   A transparent discussion on these limitations enhances the credibility and value of the research by acknowledging areas where improvements or future investigations are necessary.

#### Future Directions
Future research directions stemming from this work on erasing undesirable concepts from diffusion models could explore several promising avenues. **Improving the efficiency and scalability of the adversarial concept preservation method** is crucial, possibly through more efficient search algorithms or alternative optimization strategies.  **Investigating the impact of different concept representations and embedding spaces** on the effectiveness of erasure and preservation is important to enhance the method's generalizability.  **Developing more robust metrics for evaluating the quality and safety of generated images** is a critical need to move beyond current limitations such as CLIP score.  Moreover, **extending the method to handle multi-concept erasure and preservation** simultaneously, and perhaps even to address nuanced ethical issues beyond easily identifiable concepts, will demand innovative approaches.  Finally, **thorough investigation into the interplay between erasure, preservation, and the inherent biases encoded within the model's parameters** is necessary to understand and mitigate potential unintended consequences. This multifaceted research agenda holds the key to unleashing the full creative potential of diffusion models while ensuring their responsible and ethical use.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_3_1.jpg)

> 🔼 This figure shows a histogram representing the distribution of similarity scores between the outputs of an original diffusion model and a sanitized version of the model after removing the concept of 'nudity'. The x-axis displays the similarity score, and the y-axis represents the frequency of concepts with that similarity score. The histogram illustrates how different concepts are affected differently by the removal of the target concept. The vertical dashed lines highlight the similarity scores for specific concepts (e.g., 'naked', 'women', 'men', 'person', 'a photo', ' '). The figure demonstrates that closely related concepts to the target concept are more significantly impacted by the removal, while less related concepts show a higher similarity score to the original model's output.
> <details>
> <summary>read the caption</summary>
> Figure 2: Sensitivity spectrum of concepts to the target concept 'nudity'. The histogram shows the distribution of the similarity score between outputs of the original model θ and the corresponding sanitized model θ' for each concept c from the CLIP tokenizer vocabulary.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_4_1.jpg)

> 🔼 This figure analyzes how erasing a target concept impacts other concepts in a diffusion model.  Part (a) shows the impact of removing either 'nudity' or 'garbage truck,' while part (b) compares the effects of removing 'garbage truck' while preserving different concepts (a neutral concept, specific related concepts, or the most sensitive concept as determined by the proposed method). The impact is quantified using the difference in CLIP scores between the original and modified models for each concept.
> <details>
> <summary>read the caption</summary>
> Figure 1: Analysis of the impact of erasing the target concept on the model's capability. The impact is measured by the difference of CLIP score d(c) between the original model and the corresponding sanitized model. 1a: Impact of erasing 'nudity' or 'garbage truck' to other concepts. 1b: Comparing the impact of erasing the same 'garbage truck' to other concepts with different preserving strategies, including preserving a fixed concept such as '', 'lexus', or 'road', and adaptively preserving the most sensitive concept found by our method.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_5_1.jpg)

> 🔼 This figure visualizes the results of the proposed method's adversarial concept search process during fine-tuning.  The top row shows a continuous search using Projected Gradient Descent (PGD), while the bottom row demonstrates a discrete search using Gumbel-Softmax.  Each image represents a concept identified as highly sensitive to changes during the concept erasure process. The keyword, Ca, indicates the concept being represented in the image.
> <details>
> <summary>read the caption</summary>
> Figure 4: Images generated from the most sensitive concepts found by our method over the fine-tuning process. Top: Continous search with PGD. Bottom: Discrete search with Gumbel-Softmax.  Ca represents for the keyword.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_6_1.jpg)

> 🔼 This figure analyzes how erasing a target concept impacts other concepts in a diffusion model.  Part (a) shows the impact of erasing either 'nudity' or 'garbage truck,' illustrating varying effects on related and unrelated concepts. Part (b) compares the impact of erasing 'garbage truck' while employing different preservation strategies (e.g., preserving a neutral concept or the most affected concept). This highlights the importance of choosing the right concepts to preserve during erasure.
> <details>
> <summary>read the caption</summary>
> Figure 1: Analysis of the impact of erasing the target concept on the model's capability. The impact is measured by the difference of CLIP score d(c) between the original model and the corresponding sanitized model. 1a: Impact of erasing 'nudity' or 'garbage truck' to other concepts. 1b: Comparing the impact of erasing the same 'garbage truck' to other concepts with different preserving strategies, including preserving a fixed concept such as '', 'lexus', or 'road', and adaptively preserving the most sensitive concept found by our method.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_8_1.jpg)

> 🔼 This figure shows a comparison of the erasing performance of different methods on NSFW content using the I2P dataset and the Praneet (2019) nudity detector.  Figure 5a is a stacked bar chart showing the number of different exposed body parts detected in images generated by each method at a threshold of 0.5. Figure 5b is a bar chart showing the percentage of images containing any exposed nudity detected by Praneet (2019) across various threshold levels (0.3-0.8).  The results demonstrate the effectiveness of the proposed method in erasing NSFW content while maintaining the quality of other aspects of generated images.
> <details>
> <summary>read the caption</summary>
> Figure 5: Comparison of the erasing performance on the I2P dataset. 5a: Number of exposed body parts counted in all generated images with threshold 0.5. 5b: Ratio of images with any exposed body parts detected by the detector Praneet (2019).
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_19_1.jpg)

> 🔼 This figure visualizes the intermediate results of the adversarial concept search process during the model fine-tuning. The odd rows display images generated using the most sensitive concepts identified by the proposed method, while the even rows show images generated from the concepts targeted for erasure. Each column represents a stage in the fine-tuning process, illustrating how the generated images evolve as the model learns to remove the unwanted concepts.  The figure demonstrates the gradual removal of the target concepts in the even rows and the adaptation of the adversarial concepts in the odd rows.
> <details>
> <summary>read the caption</summary>
> Figure 6: Intermediate results of the search process. Row-1,3,5,7,9: images generated from the most sensitive concepts ca found by our method. Row-2,4,6,8,10: images generated from the corresponding to-be-erased concepts. Each column represents different fine-tuning steps in increasing order.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_20_1.jpg)

> 🔼 This figure displays the correlation between the change in CLIP scores (measuring alignment between generated images and prompts) after removing the 'nudity' concept and the similarity of other concepts to 'nudity' in the textual embedding space.  Each point represents a concept, with the top circle showing the original model's CLIP score and the bottom circle showing the score after removing 'nudity'.  The radius of each circle indicates score variance, and the vertical lines connect the original and sanitized scores for each concept. This helps visualize which concepts are most affected (sensitive) by removing 'nudity'.
> <details>
> <summary>read the caption</summary>
> Figure 7: The figure shows the correlation between the drop of the CLIP scores (measured between generated images and their prompts) between the base/original model, and the sanitized model (i.e., removing the target concept \'nudity\') and the similarity score between the target concept \'nudity\' and other concepts in the textual embedding space. The radius of the circle indicates the variance of the CLIP scores measured in 200 samples, i.e., the larger circle indicates the larger variance of the CLIP scores.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_22_1.jpg)

> 🔼 This figure shows images generated from the original Stable Diffusion model before any concept erasure.  The top five rows represent the object classes that will be targeted for removal in later experiments, while the remaining rows display classes that should remain unaffected by the erasure process. Each column uses a different random seed to illustrate the model's variability in generating images for the same prompts.
> <details>
> <summary>read the caption</summary>
> Figure 8: Generated images from the original model. Five first rows are to-be-erased objects (marked by red text) and the rest are to-be-preserved objects. Each column represents different random seeds.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_23_1.jpg)

> 🔼 This figure shows images generated from the original Stable Diffusion model before any concept erasure.  The top five rows display examples of the 'to-be-erased' object categories (marked in red), while the remaining rows show examples of the 'to-be-preserved' object categories. Each column shows the results obtained using different random seeds, illustrating the model's ability to generate a variety of images within each category before any modifications are made.
> <details>
> <summary>read the caption</summary>
> Figure 8: Generated images from the original model. Five first rows are to-be-erased objects (marked by red text) and the rest are to-be-preserved objects. Each column represents different random seeds.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_24_1.jpg)

> 🔼 This figure analyzes how erasing a target concept from a diffusion model affects the generation of other concepts.  The left subfigure (1a) shows the impact of removing either the concept of 'nudity' or 'garbage truck' on various other concepts, measured by the change in CLIP scores. The right subfigure (1b) compares different preservation strategies when erasing 'garbage truck'.  Strategies include using a neutral concept (''), a related concept ('lexus', 'road'), or an adaptively chosen most sensitive concept as determined by the proposed method.
> <details>
> <summary>read the caption</summary>
> Figure 1: Analysis of the impact of erasing the target concept on the model's capability. The impact is measured by the difference of CLIP score d(c) between the original model and the corresponding sanitized model. 1a: Impact of erasing 'nudity' or 'garbage truck' to other concepts. 1b: Comparing the impact of erasing the same 'garbage truck' to other concepts with different preserving strategies, including preserving a fixed concept such as '', 'lexus', or 'road', and adaptively preserving the most sensitive concept found by our method.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_25_1.jpg)

> 🔼 This figure shows the image generation results from the original Stable Diffusion model before any concept erasure. The top five rows represent the 'to-be-erased' concepts, and the remaining rows show the 'to-be-preserved' concepts. Each column displays results for different random seeds, showcasing the model's ability to generate diverse images across various concepts.
> <details>
> <summary>read the caption</summary>
> Figure 8: Generated images from the original model. Five first rows are to-be-erased objects (marked by red text) and the rest are to-be-preserved objects. Each column represents different random seeds.
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_26_1.jpg)

> 🔼 The figure shows the results of erasing artistic style concepts using four different methods: the authors' proposed method, ESD, UCE, and CA. Each column represents a different artist whose style is to be removed, while each row shows the same prompt used to generate images. In the ideal scenario, only the diagonal images (marked by red boxes) would change significantly, indicating that only the targeted artist's style has been removed, while others remain consistent.
> <details>
> <summary>read the caption</summary>
> Figure 12: Erasing artistic style concepts. Each column represents the erasure of a specific artist, except the first column which represents the generated images from the original SD model. Each row represents the generated images from the same prompt but with different artists. The ideal erasure should result in a change in the diagonal pictures (marked by a red box) compared to the first column, while the off-diagonal pictures should remain the same. row-1: Portrait of a woman with floral crown by Kelly McKernan; row-2: Ajin: Demi Human character portrait; row-3: Neon-lit cyberpunk cityscape by Kilian Eng; row-4: A Thomas Kinkade-inspired painting of a peaceful countryside; row-5: Tyler Edlin-inspired artwork of a mystical forest;
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_27_1.jpg)

> 🔼 This figure shows the qualitative results of erasing artistic style concepts from the Stable Diffusion model using four different methods: the proposed method, ESD, UCE, and CA. Each column represents a different artist whose style is to be erased. Each row represents the same prompt but with different artists. The diagonal images (in red boxes) are expected to change significantly due to the erasure of the artist's style, while the non-diagonal images should remain similar to the original SD model's output. This visualization helps to evaluate the effectiveness of each method in removing the specified styles while preserving other aspects of the generated images.
> <details>
> <summary>read the caption</summary>
> Figure 12: Erasing artistic style concepts. Each column represents the erasure of a specific artist, except the first column which represents the generated images from the original SD model. Each row represents the generated images from the same prompt but with different artists. The ideal erasure should result in a change in the diagonal pictures (marked by a red box) compared to the first column, while the off-diagonal pictures should remain the same. row-1: Portrait of a woman with floral crown by Kelly McKernan; row-2: Ajin: Demi Human character portrait; row-3: Neon-lit cyberpunk cityscape by Kilian Eng; row-4: A Thomas Kinkade-inspired painting of a peaceful countryside; row-5: Tyler Edlin-inspired artwork of a mystical forest;
> </details>



![](https://ai-paper-reviewer.com/GDz8rkfikp/figures_28_1.jpg)

> 🔼 This figure shows a qualitative comparison of the proposed method against three baselines for erasing artistic style concepts. Each column represents a different artist whose style is being removed. The first column shows the original model's output, while the subsequent columns demonstrate the results of the proposed method and the baselines.  The diagonal elements (red boxes) highlight the expected changes where the target artist's style should be removed. The off-diagonal elements showcase how well other artistic styles are preserved.
> <details>
> <summary>read the caption</summary>
> Figure 12: Erasing artistic style concepts. Each column represents the erasure of a specific artist, except the first column which represents the generated images from the original SD model. Each row represents the generated images from the same prompt but with different artists. The ideal erasure should result in a change in the diagonal pictures (marked by a red box) compared to the first column, while the off-diagonal pictures should remain the same. row-1: Portrait of a woman with floral crown by Kelly McKernan; row-2: Ajin: Demi Human character portrait; row-3: Neon-lit cyberpunk cityscape by Kilian Eng; row-4: A Thomas Kinkade-inspired painting of a peaceful countryside; row-5: Tyler Edlin-inspired artwork of a mystical forest;
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/GDz8rkfikp/tables_8_1.jpg)
> 🔼 This table presents the quantitative results of the nudity erasure experiment.  It compares the performance of the proposed method against three baseline methods (CA, UCE, ESD) and the original Stable Diffusion model (SD).  The evaluation metrics used are the number of images with exposed body parts at different thresholds (NER-0.3, NER-0.5, NER-0.7, NER-0.8), and the Fréchet Inception Distance (FID) score, which measures the quality of the generated images. Lower NER values indicate better erasure performance, and a lower FID score indicates higher image quality.
> <details>
> <summary>read the caption</summary>
> Table 2: Evaluation on the nudity erasure setting.
> </details>

![](https://ai-paper-reviewer.com/GDz8rkfikp/tables_8_2.jpg)
> 🔼 This table presents the similarity scores calculated using CLIP between the concept 'nudity' and other concepts (e.g., body parts).  It shows the correlation between the 'nudity' concept and various other concepts, helping to understand which concepts are most affected when removing the 'nudity' concept.  Higher similarity scores indicate a stronger correlation and therefore potentially greater impact on the removal of the target concept.
> <details>
> <summary>read the caption</summary>
> Table 3: Similarity scores between different concepts and body parts in the nudity erasure setting.
> </details>

![](https://ai-paper-reviewer.com/GDz8rkfikp/tables_9_1.jpg)
> 🔼 This table presents the quantitative results of erasing artistic style concepts.  It compares four methods: ESD, CA, UCE, and the proposed method ('Ours'). For each method, the table shows the CLIP score (lower is better) for the erased concepts and the LPIPS score (lower is better) for the retained concepts, indicating the effectiveness of each method in erasing unwanted artistic styles while preserving the overall quality of the generated images.
> <details>
> <summary>read the caption</summary>
> Table 4: Erasing artistic style concepts.
> </details>

![](https://ai-paper-reviewer.com/GDz8rkfikp/tables_16_1.jpg)
> 🔼 This table presents the quantitative results of an experiment designed to evaluate the effectiveness of different methods in erasing object-related concepts from a foundation model.  The experiment uses four different sets of five object classes from the Imagenette dataset. The table shows the performance of the original Stable Diffusion (SD) model and four other methods (ESD, UCE, CA, and the proposed method) using two metrics: Erasing Success Rate (ESR) at top-1 and top-5 predictions and Preserving Success Rate (PSR) at top-1 and top-5 predictions.  Higher ESR indicates better erasure performance, and higher PSR indicates better preservation of unrelated concepts. The results demonstrate the tradeoff between effectively erasing unwanted concepts and maintaining the integrity of other, related concepts.
> <details>
> <summary>read the caption</summary>
> Table 1: Erasing object-related concepts.
> </details>

![](https://ai-paper-reviewer.com/GDz8rkfikp/tables_17_1.jpg)
> 🔼 This table presents the results of experiments evaluating the effect of different hyperparameter settings on the performance of the proposed method. The hyperparameters varied are the number of closest concepts (K) and the number of search steps (Niter). The performance metrics measured are the Erasing Success Rate (ESR-1 and ESR-5) and the Preserving Success Rate (PSR-1 and PSR-5). ESR measures the percentage of generated images where the object of the target concept is not detected in top-1 or top-5 predictions.  PSR measures the percentage of images where the object of concepts other than the target concept are detected in top-1 or top-5 predictions.  The table shows how the choice of K and Niter impact the tradeoff between successfully erasing the unwanted concepts and preserving other concepts.
> <details>
> <summary>read the caption</summary>
> Table 6: Evaluation of the impact of hyperparameters on the erasing and preservation performance.
> </details>

![](https://ai-paper-reviewer.com/GDz8rkfikp/tables_17_2.jpg)
> 🔼 This table presents the results of experiments evaluating the impact of using different concept spaces (Oxford 3000 word list vs. CLIP vocabulary) on the performance of the proposed method.  The metrics used to assess the performance are Erasing Success Rate (ESR) at top-1 and top-5 predictions (ESR-1↑ and ESR-5↑), and Preserving Success Rate (PSR) at top-1 and top-5 predictions (PSR-1↑ and PSR-5↑). Higher values for ESR indicate better erasure performance, while higher values for PSR indicate better preservation performance.  The results show a trade-off between erasure and preservation performance when changing concept spaces.
> <details>
> <summary>read the caption</summary>
> Table 7: Evaluation of the impact of the concept space on the erasing and preservation performance.
> </details>

![](https://ai-paper-reviewer.com/GDz8rkfikp/tables_17_3.jpg)
> 🔼 This table presents the results of the nudity erasure experiment. It shows the effectiveness of different methods in erasing nudity-related content from generated images. The evaluation is performed across four methods (SD, ESD-x, ESD-u, and Ours-u) and four different thresholds (0.3, 0.5, 0.7, and 0.8). The metric used is the ratio of images with any exposed body parts detected by the detector Praneet (2019) over the total 4703 generated images.  Lower values indicate better performance in removing nudity.
> <details>
> <summary>read the caption</summary>
> Table 2: Evaluation on the nudity erasure setting.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GDz8rkfikp/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}