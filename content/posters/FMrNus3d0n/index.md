---
title: "GuardT2I: Defending Text-to-Image Models from Adversarial Prompts"
summary: "GuardT2I: A novel framework defends text-to-image models against adversarial prompts by translating latent guidance embeddings into natural language, enabling effective adversarial prompt detection wi..."
categories: ["AI Generated", ]
tags: ["Multimodal Learning", "Vision-Language Models", "🏢 Tsinghua University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} FMrNus3d0n {{< /keyword >}}
{{< keyword icon="writer" >}} Yijun Yang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=FMrNus3d0n" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/FMrNus3d0n" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/FMrNus3d0n/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Text-to-image (T2I) models are powerful but vulnerable to adversarial prompts designed to generate inappropriate or NSFW content. Current defense mechanisms such as NSFW classifiers or model fine-tuning are insufficient. This poses a significant safety challenge, particularly given the wide adoption of T2I models.



The researchers propose GUARDT2I, a novel moderation framework using a generative approach.  Instead of binary classification, GUARDT2I employs a large language model to translate latent representations of prompts into natural language. This reveals the true intent behind the prompt and facilitates effective detection of adversarial prompts without compromising model performance. **Extensive experiments demonstrate GUARDT2I's superiority over existing solutions across diverse adversarial scenarios.**  **The framework's availability further supports wider adoption and collaborative advancements in this critical area.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} GuardT2I, a novel moderation framework, enhances the robustness of text-to-image models against adversarial prompts. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The generative approach of GuardT2I outperforms existing commercial solutions in various adversarial scenarios. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} GuardT2I provides decision-making transparency and is highly generalizable across diverse adversarial prompts. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it addresses a critical issue in the field of text-to-image models: vulnerability to adversarial prompts.  It introduces a novel generative approach to defense, outperforming existing commercial solutions. This opens avenues for further research into generative defense mechanisms and enhancing the safety and reliability of text-to-image technologies.  The framework's availability also promotes wider adoption and collaborative improvements in this important area.  **The methodology and findings are highly relevant to current research trends focused on improving the safety and robustness of AI models, especially in high-stakes applications.**

------
#### Visual Insights



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_1_1.jpg)

> 🔼 This figure illustrates how GuardT2I works.  Panel (a) shows a standard prompt generating a safe image; (b) shows an adversarial prompt generating NSFW content.  (c) shows how GuardT2I processes a normal prompt, successfully reconstructing it without altering image quality or speed. (d) shows how GuardT2I handles an adversarial prompt; the prompt is interpreted, the NSFW content is identified, and generation is halted.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of GUARDT2I. GuardT2I can effectively halt the generation process of adversarial prompts to avoid NSFW generations, without compromising normal prompts or increasing inference time.
> </details>





![](https://ai-paper-reviewer.com/FMrNus3d0n/tables_2_1.jpg)

> 🔼 This table compares the proposed GUARDT2I method with existing commercial and open-source content moderation methods.  It highlights key differences in terms of whether the method is open-source, the paradigm used (classification vs. generation), whether the labels are freely available, the model's interpretability, and the possibility for customization.  GUARDT2I is shown to be superior in many respects.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison of our generative defensive approach with existing classification-based ones.
> </details>





### In-depth insights


#### GuardT2I Overview
GuardT2I is presented as a novel moderation framework designed to enhance the robustness of text-to-image models against adversarial prompts.  **Instead of a binary classification approach**, it leverages a large language model to transform text guidance embeddings into natural language, enabling effective adversarial prompt detection. This generative approach allows for the identification of malicious prompts without compromising the model's performance on legitimate inputs. A key advantage is its ability to **halt the generation process of adversarial prompts early**, saving computational resources.  Furthermore, GUARDT2I maintains the inherent quality of the text-to-image model in typical use cases.  The framework's effectiveness is demonstrated through extensive experiments showcasing its superior performance compared to leading commercial solutions, indicating a **significant improvement in mitigating the risks associated with malicious text prompts.** The availability of the GUARDT2I framework facilitates its wider application and further research in this crucial area of AI safety.

#### LLM-based Defense
LLM-based defense against adversarial prompts in text-to-image models presents a novel approach, leveraging the power of large language models (LLMs) to interpret and transform the latent representations of prompts.  This method avoids direct classification and instead uses an LLM to generate a natural language interpretation of the prompt's underlying semantic meaning.  **The key advantage is the ability to detect adversarial prompts without compromising the quality of legitimate image generation**. This generative approach offers superior robustness compared to traditional binary classifiers by being less susceptible to adversarial attacks.  A conditional LLM (c-LLM) plays a crucial role in translating the latent space into natural language, enabling better comprehension of user intent. The combination of the c-LLM and a similarity checker enhances accuracy and provides decision-making transparency.  **The success of this method highlights the potential of LLMs in enhancing the safety and security of AI models**. However, this approach necessitates a sizable pre-trained LLM and the associated computational costs.  Future work should address this to broaden applicability.  **The generative paradigm shift also opens up new directions in adversarial defense for other AI systems.**

#### Adaptive Attacks
The concept of 'adaptive attacks' in the context of a research paper on defending text-to-image models is crucial.  It probes the robustness of a defense mechanism by simulating a sophisticated adversary.  **Adaptive attackers**, unlike traditional attackers, possess full knowledge of the defense mechanism and iteratively refine their attack strategies.  This necessitates that the defense system be highly resilient and not just effective against known attack vectors. A successful adaptive attack highlights weaknesses in the model’s generalizability and ability to handle unforeseen adversarial inputs. The evaluation of a defense system's performance under adaptive attacks is, therefore, a **rigorous test** of its overall effectiveness. **Understanding the success rate** of adaptive attacks helps gauge the limitations of the defense and identify areas for improvement, ultimately contributing to the development of more robust and reliable security measures for text-to-image systems.

#### Performance Metrics
Choosing the right performance metrics is crucial for evaluating the effectiveness of a text-to-image model's defense against adversarial prompts.  **Accuracy metrics**, such as AUROC (Area Under the Receiver Operating Characteristic Curve) and AUPRC (Area Under the Precision-Recall Curve), are essential for quantifying the model's ability to correctly classify prompts as either benign or adversarial.  However, these alone are insufficient. **False Positive Rate (FPR)** at a specific true positive rate (e.g., FPR@TPR95) is critical to understand the rate of false alarms, a key consideration in real-world applications where excessive false positives could severely impact user experience.  **Attack Success Rate (ASR)** provides an alternative perspective, measuring the percentage of adversarial prompts that successfully bypass the defense and generate undesirable outputs. The inclusion of multiple metrics, considering both detection accuracy and real-world usability, is vital for a comprehensive evaluation.  **Image quality** metrics (like FID or CLIP score) could further complement the primary evaluation, helping assess if the defense compromises image generation quality in non-adversarial scenarios.  A balanced approach using a combination of metrics offers the most insightful assessment of the defensive system's overall effectiveness.

#### Future Work
Future research directions stemming from this work on adversarial prompt defense for text-to-image models could explore several key areas. **Improving the generalizability of the GUARDT2I framework** across diverse T2I models and various adversarial attack strategies is crucial.  This might involve investigating more advanced LLM architectures and training methodologies, potentially incorporating techniques like reinforcement learning to enhance robustness against adaptive attacks. **Developing more sophisticated prompt interpretation methods** is also important, possibly by leveraging multimodal approaches combining both textual and visual information, or by using more advanced parsing techniques to capture subtle nuances in prompt semantics. **Quantifying and analyzing the trade-off between model safety and creative potential** is a critical area for future investigation, aiming to create defense mechanisms that minimize safety compromises while retaining high-quality image generation.  Further work could analyze the long-term implications of this technology, addressing ethical considerations and potential societal impacts, and proposing mitigation strategies to prevent misuse.  Finally, **research into explainable AI (XAI) techniques** could greatly improve the transparency and trustworthiness of adversarial prompt defense systems.  By incorporating XAI, the decision-making processes of the defense system can be made more interpretable, allowing users and developers to understand and trust its functionalities.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_3_1.jpg)

> 🔼 This figure illustrates the workflow of GUARDT2I in handling adversarial prompts.  It shows how GUARDT2I intercepts the generation process and uses a conditional Language Model (c-LLM) to translate the latent embedding into natural language. This interpretation is then analyzed by a two-stage parsing system: the Verbalizer checks for explicit NSFW words, and the Sentence Similarity Checker compares the interpretation to the original prompt to detect discrepancies indicative of adversarial intent.  The final decision (reject or accept) and the reasoning behind it are documented to maintain transparency. 
> <details>
> <summary>read the caption</summary>
> Figure 2: The Workflow of GUARDT2I against Adversarial Prompts. (a) GUARDT2I halts the generation process of adversarial prompts. (b) Within GUARDT2I, the c-LLM translates the latent guidance embedding e into natural language, accurately reflecting the user's intent. (c) A double-folded generation parse detects adversarial prompts. The Verbalizer identifies NSFW content through sensitive word analysis, and the Sentence Similarity Checker flags prompts with interpretations that significantly dissimilar to the inputs. (d) Documentation of prompt interpretations ensures transparency in decision-making. aims to avoid offenses.
> </details>



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_3_2.jpg)

> 🔼 This figure shows the architecture of the conditional large language model (c-LLM) used in GUARDT2I.  The c-LLM takes the text guidance embedding from the text-to-image model as input. This embedding is processed through multiple transformer layers incorporating a cross-attention mechanism that allows the c-LLM to condition its text generation on the input embedding.  The output of the c-LLM is a plain text interpretation of the original prompt.
> <details>
> <summary>read the caption</summary>
> Figure 3: Architecture of c·LLM. T2I's text guidance embedding e is fed to c-LLM through the multi-head cross attention layer's query entry. L indicates the total number of transformer blocks.
> </details>



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_4_1.jpg)

> 🔼 This figure illustrates how GUARDT2I works.  Panel (a) shows a normal prompt leading to a high-quality image synthesis. Panel (b) shows how an adversarial prompt (that looks innocuous to humans) can lead to NSFW content generation.  Panels (c) and (d) demonstrate GUARDT2I's intervention: it maintains high-quality syntheses for normal prompts and intercepts the generation process for adversarial prompts, providing an explanation.  GUARDT2I does this without adding inference time for typical prompts.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of GUARDT2I. GuardT2I can effectively halt the generation process of adversarial prompts to avoid NSFW generations, without compromising normal prompts or increasing inference time.
> </details>



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_6_1.jpg)

> 🔼 This figure shows the Receiver Operating Characteristic (ROC) curves for GUARDT2I and several baseline methods (OpenAI Moderation, Microsoft Azure, AWS Comprehend, NSFW Text Classifier, and Detoxify) when tested against various types of adversarial prompts (SneakyPrompt, MMA-Diffusion, I2P-Sexual, and I2P).  The y-axis represents the True Positive Rate (TPR), and the x-axis represents the False Positive Rate (FPR).  A higher AUROC score (area under the curve) indicates better performance.  GUARDT2I consistently outperforms the baselines across all four adversarial datasets, demonstrating its superior ability to detect adversarial prompts.
> <details>
> <summary>read the caption</summary>
> Figure 5: ROC curves of our GUARDT2I and baselines against various adversarial prompts. The black line represents the GUARDT2I model's consistent and high AUROC scores across different thresholds.
> </details>



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_7_1.jpg)

> 🔼 This figure compares the Area Under the ROC Curve (AUROC) scores achieved by GUARDT2I and several baseline methods across five different NSFW themes: Violence, Self-harm, Hate, Shocking, and Illegal.  It visually demonstrates GUARDT2I's superior generalizability and consistent high performance (AUROC scores above 90% across all themes) compared to the fluctuating performance of baseline methods, which are significantly less robust to diverse NSFW content categories. The stable performance of GUARDT2I is attributed to its generative approach and the use of a Large Language Model (LLM), enhancing its ability to adapt to and correctly classify various types of NSFW content.
> <details>
> <summary>read the caption</summary>
> Figure 6: AUROC comparison over various NSFW themes. Our GUARDT2I, benefitting from the generalization capabilities of the LLM, stably exhibits decent performance under a wide range of NSFW threats.
> </details>



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_7_2.jpg)

> 🔼 This figure shows a comparison of word clouds generated from adversarial prompts and their corresponding interpretations by GUARDT2I.  The left word cloud (a) highlights the seemingly innocuous words used in adversarial prompts that are designed to bypass safety filters of text-to-image models. The right word cloud (b) displays the words extracted by GUARDT2I from its prompt interpretation of the same inputs. The difference reveals GUARDT2I's capability to uncover the underlying NSFW intent of adversarial prompts. The highlighted words in (b) emphasize the explicit content that was implicitly expressed in the original adversarial prompts.
> <details>
> <summary>read the caption</summary>
> Figure 7: Word clouds of adversarial prompts [45], and their prompt interpretations. GUARDT2I can effectively reveal the concealed malicious intentions of attackers.
> </details>



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_8_1.jpg)

> 🔼 This figure demonstrates GUARDT2I's effectiveness in handling adversarial prompts.  It shows four scenarios: (a) High-quality image generation from a standard prompt; (b) NSFW image generation from an adversarial prompt designed to produce inappropriate content; (c) GUARDT2I maintaining high-quality image generation and speed from a standard prompt; and (d) GUARDT2I successfully halting the generation process of an adversarial prompt before NSFW content is produced.  GUARDT2I's ability to detect and mitigate adversarial prompts without sacrificing performance is highlighted.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of GUARDT2I. GuardT2I can effectively halt the generation process of adversarial prompts to avoid NSFW generations, without compromising normal prompts or increasing inference time.
> </details>



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_9_1.jpg)

> 🔼 This figure presents two examples of GUARDT2I's failures. The first shows a false negative where an adversarial prompt about Trump and Thanos generated an image related to them, but the system incorrectly classified it as a normal prompt. The second shows a false positive where a normal prompt mentioning uncommon words led to the system incorrectly classifying it as an adversarial prompt.  This highlights limitations in the system's ability to accurately identify certain types of adversarial prompts and the challenges in differentiating between genuinely malicious and unusually phrased prompts.
> <details>
> <summary>read the caption</summary>
> Figure A-1: Additional failure case analysis. Upper section: The adversarial prompt [38] generates shocking content (fake news about Trump/Thanos) but is mistakenly flagged as a normal prompt. Lower section: GUARDT21 occasionally produces false alarms due to the reconstruction of rarely used terminology (see bolded words), resulting in false positives.
> </details>



![](https://ai-paper-reviewer.com/FMrNus3d0n/figures_16_1.jpg)

> 🔼 This figure illustrates how GUARDT2I works.  Panel (a) shows the high-quality images generated by a text-to-image model using normal prompts. Panel (b) demonstrates that adversarial prompts, seemingly innocuous to humans, can lead to NSFW content generation.  Panels (c) and (d) showcase GUARDT2I's ability to successfully filter adversarial prompts by transforming the latent representation of these prompts back into plain text, which reveals their true, potentially harmful, intent without affecting the model's performance on legitimate prompts.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of GUARDT2I. GuardT2I can effectively halt the generation process of adversarial prompts to avoid NSFW generations, without compromising normal prompts or increasing inference time.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/FMrNus3d0n/tables_5_1.jpg)
> 🔼 This table compares the performance of GUARDT2I against several baseline methods across various adversarial prompt datasets.  The metrics used for comparison include Attack Success Rate (ASR), False Positive Rate at 95% True Positive Rate (FPR@TPR95), Area Under the Precision-Recall Curve (AUPRC), and Area Under the Receiver Operating Characteristic Curve (AUROC).  The table highlights GUARDT2I's superior performance in all metrics, demonstrating its effectiveness in mitigating adversarial prompts.
> <details>
> <summary>read the caption</summary>
> Table 2: Comparison with baselines. Bolded values are the highest performance. The underlined italicized values are the second highest performance. * indicates human-written adversarial prompts.
> </details>

![](https://ai-paper-reviewer.com/FMrNus3d0n/tables_6_1.jpg)
> 🔼 This table presents the results of evaluating the image quality and text alignment of different methods in normal use cases, which are not adversarial prompts.  The metrics used are FID (Fréchet Inception Distance), which measures the image fidelity, and CLIP-Score, which assesses the alignment between the generated image and its text prompt.  Lower FID scores indicate better image quality, and higher CLIP-Scores represent better text alignment. ASR (Attack Success Rate) is also presented. The table shows that GUARDT2I maintains high image quality and text alignment, while demonstrating significantly better defense effectiveness in terms of a lower attack success rate when compared to the baselines.
> <details>
> <summary>read the caption</summary>
> Table 3: Normal Use Case Results. Bolded values are the highest performance. The underlined italicized values are the second highest performance.
> </details>

![](https://ai-paper-reviewer.com/FMrNus3d0n/tables_7_1.jpg)
> 🔼 This table compares the performance of GUARDT2I against several baseline methods across various adversarial prompt datasets.  The metrics used are Attack Success Rate (ASR), False Positive Rate at 95% True Positive Rate (FPR@TPR95), Area Under the Precision-Recall Curve (AUPRC), and Area Under the Receiver Operating Characteristic Curve (AUROC).  Higher AUPRC and AUROC values, and lower ASR and FPR@TPR95 values indicate better performance.  The table highlights GUARDT2I's superior performance across different types of adversarial prompts, including those crafted manually and automatically.
> <details>
> <summary>read the caption</summary>
> Table 2: Comparison with baselines. Bolded values are the highest performance. The underlined italicized values are the second highest performance. * indicates human-written adversarial prompts.
> </details>

![](https://ai-paper-reviewer.com/FMrNus3d0n/tables_9_1.jpg)
> 🔼 This table compares the performance of GUARDT2I against several baseline methods in mitigating various types of adversarial prompts.  It shows the attack success rate (ASR), false positive rate at 95% true positive rate (FPR@TPR95), area under the precision-recall curve (AUPRC), and area under the ROC curve (AUROC) for different adversarial prompt datasets. The results demonstrate that GUARDT2I significantly outperforms all baselines across various metrics and attack types.
> <details>
> <summary>read the caption</summary>
> Table 2: Comparison with baselines. Bolded values are the highest performance. The underlined italicized values are the second highest performance. * indicates human-written adversarial prompts.
> </details>

![](https://ai-paper-reviewer.com/FMrNus3d0n/tables_9_2.jpg)
> 🔼 This table compares the performance of GUARDT2I against several baseline methods across various adversarial prompt datasets.  The metrics used are Attack Success Rate (ASR), False Positive Rate at 95% True Positive Rate (FPR@TPR95), Area Under the Precision-Recall Curve (AUPRC), and Area Under the Receiver Operating Characteristic Curve (AUROC).  Higher AUROC and AUPRC values indicate better performance, while a lower FPR@TPR95 and ASR indicate fewer false positives and successful attacks respectively.  The table highlights GUARDT2I's superior performance across all metrics and datasets, especially against human-written adversarial prompts.
> <details>
> <summary>read the caption</summary>
> Table 2: Comparison with baselines. Bolded values are the highest performance. The underlined italicized values are the second highest performance. * indicates human-written adversarial prompts.
> </details>

![](https://ai-paper-reviewer.com/FMrNus3d0n/tables_9_3.jpg)
> 🔼 This table compares the number of parameters and inference times for various models on an NVIDIA-A800 GPU.  The models compared include Stable Diffusion v1.5, SDXL 0.9, SafetyChecker, a combination of SDv1.5 and SafetyChecker, the c-LLM used in GuardT2I, the Sentence Similarity component of GuardT2I, and finally GuardT2I itself.  The inference times show GuardT2I's efficiency compared to the other methods, being significantly faster, even when compared to the SafetyChecker alone.
> <details>
> <summary>read the caption</summary>
> Table 7: Comparison of Model Parameters and Inference Times on NVIDIA-A800
> </details>

![](https://ai-paper-reviewer.com/FMrNus3d0n/tables_15_1.jpg)
> 🔼 This table compares the performance of GUARDT2I against several baseline methods in mitigating various adversarial prompts.  The metrics used to evaluate the performance are Attack Success Rate (ASR), False Positive Rate at 95% True Positive Rate (FPR@TPR95), Area Under the Precision-Recall Curve (AUPRC), and Area Under the Receiver Operating Characteristic Curve (AUROC).  The table shows that GUARDT2I consistently outperforms the baselines across different types of adversarial prompts (human-written and automatically generated).  The bolded values represent the highest performance for each metric, and the underlined italicized values show the second-best performance.
> <details>
> <summary>read the caption</summary>
> Table 2: Comparison with baselines. Bolded values are the highest performance. The underlined italicized values are the second highest performance. * indicates human-written adversarial prompts.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FMrNus3d0n/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}