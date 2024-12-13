---
title: "Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models"
summary: "Single Image Unlearning (SIU) efficiently removes visual data from Multimodal Large Language Models (MLLMs) using only one image per concept, outperforming existing methods and defending against attac..."
categories: []
tags: ["Multimodal Learning", "Vision-Language Models", "🏢 School of Cyber Science and Engineering, Southeast University, Nanjing, China",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} YNx7ai4zTs {{< /keyword >}}
{{< keyword icon="writer" >}} Jiaqi Li et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=YNx7ai4zTs" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94704" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=YNx7ai4zTs&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/YNx7ai4zTs/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current machine unlearning methods struggle with the removal of visual data from complex Multimodal Large Language Models (MLLMs) due to data scarcity and model degradation.  This often leads to the generation of nonsensical outputs.  Existing methods also fail to consider the implications for model security and privacy.  

This paper introduces Single Image Unlearning (SIU), a novel method to overcome these challenges. SIU effectively addresses the issues of data scarcity and model degradation using a multifaceted approach to fine-tuning data and an innovative Dual Masked KL-divergence loss.  The experimental results on a new benchmark, MMUBench, demonstrate SIU's superiority over existing methods and highlight its robustness against attacks.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Single Image Unlearning (SIU) efficiently unlearns visual information from MLLMs using a single image. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} SIU introduces a novel Dual Masked KL-divergence loss for effective unlearning while preserving model utility. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} MMUBench, a new benchmark, provides a comprehensive evaluation of MU in MLLMs. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is highly important for researchers working on **multimodal large language models (MLLMs)** and **machine unlearning (MU)**.  It addresses the critical challenge of efficiently removing sensitive visual data from MLLMs, a significant concern for privacy and security. The proposed method, SIU, and the new benchmark, MMUBench, provide valuable tools for future research in this rapidly evolving field. This work opens up new avenues for research in **robust and ethical MLLM development**.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_3_1.jpg)

> This figure illustrates the overall process of Single Image Unlearning (SIU) in Multimodal Large Language Models (MLLMs). It starts by showing a user's request to remove the visual recognition of specific concepts.  The MMUBench dataset is used to provide concepts, and SIU, consisting of Multifaceted Fine-tuning Data and Dual Masked KL-divergence Loss, is then employed to perform unlearning. After the unlearning process, the resulting MLLM is thoroughly evaluated using different metrics such as generality, specificity, diversity, fluency, and resistance to both membership inference attacks and jailbreak attacks. The illustration visually represents the flow of the process from the initial request to the comprehensive post-unlearning evaluation.





![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_6_1.jpg)

> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three existing machine unlearning methods (Preference Optimization (PO), Gradient Ascent (GA), and GA with KL-divergence (GA+KL)) across two different sizes of LLAVA models (7B and 13B). The comparison is done across five key metrics: Efficacy, Generality (measured by Exact Match (EM), GPT-4 Evaluation (G-Eval), and C Probability Distance (C-Dis)), Specificity, Fluency, and Diversity.  Higher scores generally indicate better performance.  Note that Specificity values are detailed separately in Table 7.





### In-depth insights


#### Single Image Unlearning
The concept of "Single Image Unlearning" presents a novel approach to machine unlearning, focusing on efficiently removing specific visual data from Multimodal Large Language Models (MLLMs).  Traditional unlearning methods often require extensive retraining, which is computationally expensive and resource-intensive.  **Single Image Unlearning aims to overcome this limitation by fine-tuning the MLLM using only a single image of the target concept to be forgotten.** This approach is particularly valuable in scenarios where obtaining a large dataset of images for unlearning is challenging.  The method's efficacy likely depends on the quality and representativeness of the chosen image, as well as the architecture and training of the MLLM.  **Furthermore, the method's robustness against various attacks such as membership inference and jailbreak attacks needs thorough investigation**. While promising,  it's crucial to explore potential trade-offs between efficiency and the preservation of the MLLM's overall functionality.  **The development of a robust benchmark for evaluating single image unlearning is essential for evaluating the performance and generalizability of this method compared to other existing unlearning techniques.** The research in this domain is at an early stage, and more studies are needed to fully understand its potential and limitations.

#### Dual Masked KL Loss
The proposed Dual Masked KL-divergence loss function is a novel approach designed to overcome the limitations of traditional KL-divergence in machine unlearning for multimodal large language models (MLLMs).  It addresses the challenge of maintaining model utility while effectively forgetting specific visual concepts by incorporating two levels of masking: **token-level masking** and **vocabulary-level masking**. Token-level masking prevents contradictory tokens from influencing the KL-divergence calculation, focusing the loss on relevant tokens and avoiding meaningless output. Vocabulary-level masking explicitly masks the vocabulary entries of the targeted concept, ensuring that the model's probability distribution shifts away from that concept, rather than simply adjusting it. This dual-masking strategy enhances the unlearning process by more directly and effectively addressing the inherent difficulties of forgetting in complex MLLMs, improving both the efficacy and generality of the unlearning process and leading to a more robust and useful post-unlearning model.  This novel loss function is a key element in the Single Image Unlearning (SIU) method, enabling the unlearning of a concept with only a single image and demonstrating superior performance to existing methods.

#### MMUBench Benchmark
The MMUBench benchmark, designed for multimodal large language model (MLLM) unlearning, is a significant contribution.  Its **comprehensive design** assesses various aspects of unlearning efficacy, including **generality (how well the model forgets across unseen data), specificity (preserving knowledge of unrelated concepts), fluency (readability of outputs), and diversity (range of responses)**.  This multifaceted evaluation goes beyond simple accuracy metrics, offering a richer understanding of the unlearning process's impact on MLLM utility.  The benchmark's **inclusion of baseline methods** and a curated dataset provide a strong foundation for future research and comparison, fostering advancements in the field.  Furthermore, the inclusion of **membership inference and jailbreak attacks** as evaluation points highlights the benchmark's focus on robustness and security, underscoring the critical importance of responsible MLLM unlearning.  The MMUBench is a crucial tool for ensuring the development of effective and trustworthy MLLMs in real-world applications.

#### Unlearning Visual Data
Unlearning visual data within large multimodal language models (MLLMs) presents a unique challenge.  **Existing text-based unlearning methods don't directly translate to the visual domain**, requiring new approaches.  The difficulty stems from the complex interplay between visual and textual information in MLLMs, as well as the potential for unintended consequences such as model degradation or the emergence of biases. A key focus should be on developing techniques to precisely remove specific visual concepts while preserving overall model utility. This necessitates the creation of sophisticated fine-tuning datasets and loss functions that carefully control the unlearning process.  **Careful consideration of the trade-offs between forgetting specific visual information and maintaining the overall capabilities of the MLLM** is crucial.  The evaluation of such methods needs robust benchmarks that assess not only efficacy, but also generality, specificity, and fluency.  The challenge of unlearning visual data highlights the significant complexity in managing the knowledge encoded within MLLMs and underscores the need for research into more nuanced and robust unlearning techniques.

#### Future of Multimodal MU
The "Future of Multimodal MU" holds **significant potential** but also faces considerable challenges.  Success hinges on addressing the limitations of current methods, particularly the scarcity of appropriate training data for effective unlearning.  **Developing more efficient algorithms** that minimize computational costs and storage requirements will be crucial for practical deployment.  Furthermore, a deeper understanding of how multimodal models process and store information is needed to design more targeted unlearning strategies.  **Addressing ethical concerns** surrounding data privacy, bias, and the potential for malicious use of unlearning capabilities is paramount.  Research into robust defense mechanisms against membership inference and other attacks on unlearned models is essential.  The **development of standardized benchmarks and evaluation metrics** for multimodal MU will facilitate progress and enable fair comparisons between different approaches.  Finally, exploration of **alternative unlearning techniques**, beyond fine-tuning, such as those inspired by biological forgetting mechanisms, could unlock new possibilities.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_7_1.jpg)

> This figure visualizes the performance of various machine unlearning methods (PO, GA, GA+KL, and SIU) across different fine-tuning steps using the LLAVA7B model.  The metrics shown include Efficacy (EM), Generality (G-eval and C-Dis), Specificity, Fluency, and Diversity.  It illustrates how each method's performance changes as the number of fine-tuning steps increase.


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_7_2.jpg)

> This figure visualizes the performance of various metrics (EM, G-eval, Distance, Fluency, Diversity) across different fine-tuning steps using the LLAVA7B model.  It compares four different machine unlearning methods: PO, GA, GA+KL, and SIU.  The graph shows how each method's performance changes as the number of fine-tuning steps increases. This allows for a direct comparison of the effectiveness and efficiency of each unlearning method in terms of maintaining model utility and forgetting the target concept.


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_8_1.jpg)

> This figure shows the performance comparison across various methods (SIU, GA+KL, PO, and GA) for the task of unlearning different concepts.  The Exact Match (EM) scores, representing the accuracy of the unlearning process, are displayed for each method on a variety of concepts. The figure provides a visual representation of the effectiveness of SIU compared to baseline approaches in terms of its ability to successfully 'forget' the learned visual representations associated with different concepts. 


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_13_1.jpg)

> This figure illustrates the overall process of Single Image Unlearning (SIU) in Multimodal Large Language Models (MLLMs).  It begins with a user request to remove a specific concept's visual recognition from the model.  The MMUBench dataset provides the concepts to be unlearned. SIU uses multifaceted fine-tuning data and a Dual Masked KL-divergence loss to perform the unlearning. Finally, it shows how the resulting model is assessed based on generality, specificity, diversity, fluency, and resistance to various attacks.


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_14_1.jpg)

> This figure illustrates the Single Image Unlearning (SIU) process for Multimodal Large Language Models (MLLMs). It begins with a user's request to remove the model's ability to recognize specific concepts.  The MMUBench benchmark provides a list of concepts for this unlearning.  SIU consists of two key parts: creating multifaceted fine-tuning data and utilizing a Dual Masked KL-divergence loss function.  After the unlearning process, the model is evaluated on several metrics (generality, specificity, diversity, fluency) and tested for robustness against membership inference and jailbreak attacks.


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_15_1.jpg)

> This figure provides a visual overview of the Single Image Unlearning (SIU) process for Multimodal Large Language Models (MLLMs).  It begins with a user request to remove the model's ability to recognize specific concepts.  The MMUBench dataset provides the concepts to be unlearned. SIU then uses Multifaceted Fine-tuning Data and a Dual Masked KL-divergence Loss to fine-tune the model.  Finally, the figure shows the evaluation metrics used to assess the success of the unlearning process, including the model's performance on unseen data (generality), its ability to avoid incorrectly identifying concepts (specificity), the diversity of its outputs, its fluency, and its resistance to adversarial attacks (membership inference and jailbreak attacks).


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_17_1.jpg)

> This figure provides a visual overview of the Single Image Unlearning (SIU) process for Multimodal Large Language Models (MLLMs). It starts with a user's request to unlearn visual recognition of specific concepts.  The MMUBench is used to provide concepts for the unlearning process. SIU is depicted as having two key components: Multifaceted Fine-tuning Data and Dual Masked KL-divergence Loss. Finally, the unlearned MLLM is comprehensively evaluated using metrics like generality, specificity, diversity, fluency and resistance to attacks such as membership inference and jailbreak attacks.


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_19_1.jpg)

> This figure provides a visual overview of the Single Image Unlearning (SIU) process for Multimodal Large Language Models (MLLMs). It begins with a user requesting to unlearn the visual recognition of specific concepts, leveraging the MMUBench dataset for concept selection. The core of SIU involves two key aspects: creating multifaceted fine-tuning data and employing a Dual Masked KL-divergence Loss.  After the unlearning process, the modified MLLM is assessed across several metrics including generality, specificity, diversity, fluency, and its resilience against membership inference and jailbreak attacks.


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_19_2.jpg)

> This figure illustrates the overall process of single image unlearning (SIU) in multimodal large language models (MLLMs).  It starts by showing a user requesting to remove a concept's visual recognition using the MMUBench dataset.  The process then uses SIU, with its two key components: Multifaceted Fine-tuning Data and Dual Masked KL-divergence Loss. Finally, it depicts the evaluation of the unlearned MLLM by assessing its generality, specificity, diversity, fluency, and resilience against membership inference and jailbreak attacks. 


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_20_1.jpg)

> This figure provides a visual overview of the Single Image Unlearning (SIU) process for Multimodal Large Language Models (MLLMs).  It begins with a user's request to remove a concept's visual recognition. The SIU method uses Multifaceted Fine-tuning Data and Dual Masked KL-divergence Loss to achieve this. Finally, it evaluates the results based on generality, specificity, diversity, fluency, and resistance to attacks.


![](https://ai-paper-reviewer.com/YNx7ai4zTs/figures_26_1.jpg)

> This figure provides a visual overview of the Single Image Unlearning (SIU) process for Multimodal Large Language Models (MLLMs). It starts by depicting the user's request to unlearn a concept's visual recognition, making use of the MMUBench dataset for concept selection. The core of SIU involves two key components: Multifaceted Fine-tuning Data and the Dual Masked KL-divergence Loss. After the unlearning process, the resulting MLLM is evaluated for various aspects such as generalization, specificity, diversity, fluency, and its robustness against membership inference and jailbreak attacks. The figure visually guides the reader through the steps and components of SIU, clarifying its operation and evaluation.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_6_2.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three existing machine unlearning methods: Preference Optimization (PO), Gradient Ascent (GA), and GA combined with KL-divergence (GA+KL).  The comparison is done using two different sizes of LLAVA models (7B and 13B). The metrics used for comparison include Efficacy (measured by Exact Match - EM), Generality (measured by G-Eval and C-Dis), Specificity, Fluency, and Diversity.  The table shows that SIU significantly outperforms the other methods in most metrics, particularly Generality and Specificity, demonstrating its effectiveness and robustness in machine unlearning for multimodal large language models.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_8_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method with three existing machine unlearning methods (Preference Optimization, Gradient Ascent, and Gradient Ascent + KL-divergence) on several metrics.  The metrics evaluate the effectiveness of unlearning (Efficacy, Generality), the model's utility after unlearning (Specificity, Fluency, Diversity), and are measured on two different sizes of the LLAVA model (7B and 13B parameters).  The results show that SIU significantly outperforms the existing methods across most metrics.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_8_2.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three baseline methods: Preference Optimization (PO), Gradient Ascent (GA), and Gradient Ascent with KL-divergence (GA+KL).  The comparison is made across multiple metrics evaluating the efficacy (EM score) and generality (G-Eval and C-Dis) of unlearning,  the specificity of the unlearning (impact on non-target knowledge), fluency (perplexity), and diversity of generated outputs. The results are shown for two different sizes of the LLAVA language model, demonstrating the consistency of SIU's performance across different model sizes and its superiority to existing methods.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_16_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method with three existing machine unlearning methods (Preference Optimization (PO), Gradient Ascent (GA), and GA+KL) on two different sizes of LLAMA models (7B and 13B).  The comparison uses several metrics to evaluate the effectiveness of unlearning: Efficacy (Exact Match), Generality (Exact Match, GPT-4 Evaluation, C Probability Distance), Specificity, Fluency, and Diversity.  Higher scores in Generality, Specificity, Fluency and Diversity indicate better preservation of model utility after unlearning.  The table highlights SIU's superior performance across most metrics, showcasing its ability to effectively unlearn target concepts while preserving model functionality.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_18_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three existing machine unlearning methods (Preference Optimization (PO), Gradient Ascent (GA), and GA with KL-divergence (GA+KL)) on two different sizes of LLAVA models (7B and 13B).  The comparison is done across several metrics: Efficacy (measured by Exact Match - EM), Generality (measured by Exact Match - EM, GPT-4 Evaluation - G-Eval, and C Probability Distance - C-Dis), Specificity, Fluency, and Diversity.  Higher scores are better for all metrics except Fluency, where a lower score is better. The table highlights the superior performance of SIU across most metrics and model sizes.  The Specificity metric is further detailed in a separate table (Table 7).

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_20_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method with three existing machine unlearning methods (Preference Optimization, Gradient Ascent, and Gradient Ascent + KL Divergence) across several evaluation metrics.  The metrics include Efficacy (Exact Match), Generality (Exact Match, GPT-4 Evaluation, C Probability Distance), Specificity, Fluency, and Diversity.  The results are shown for two different sizes of the LLAMA language model (7B and 13B parameters).  The table highlights SIU's superior performance across most metrics, demonstrating its effectiveness and robustness in unlearning visual concepts from multimodal large language models.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_21_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three baseline methods (Preference Optimization (PO), Gradient Ascent (GA), and GA with KL-divergence (GA+KL)) for machine unlearning in large language models (LLMs).  The comparison is done across multiple metrics including Efficacy (EM), Generality (G-Eval, C-Dis), Specificity, Fluency, and Diversity.  Higher values generally indicate better performance. The table shows results for two different sizes of the LLAVA model (7B and 13B parameters). The Specificity scores are detailed separately in Table 7.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_22_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three baseline methods for machine unlearning in large language models (LLMs): Preference Optimization (PO), Gradient Ascent (GA), and GA with KL-divergence (GA+KL).  The comparison is done using two different sized LLMs (LLAVA7B and LLAVA13B).  The metrics used for evaluation are efficacy, generality (measured by Exact Match (EM), GPT-4 Evaluation (G-Eval), and C Probability Distance (C-Dis)), specificity, fluency, and diversity.  The results show that SIU significantly outperforms the baselines across all metrics.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_23_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three baseline methods: Preference Optimization (PO), Gradient Ascent (GA), and GA combined with KL-divergence (GA+KL).  The comparison is based on several evaluation metrics across two different sizes of LLAVA models (7B and 13B parameters).  Metrics include Efficacy (EM score), Generality (G-Eval and C-Dis scores), Specificity, Fluency, and Diversity. Higher scores generally indicate better performance. The table shows SIU significantly outperforms the baselines across most metrics.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_24_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three baseline methods: Preference Optimization (PO), Gradient Ascent (GA), and GA with KL-divergence (GA+KL).  The comparison is done across several metrics evaluating efficacy, generality, specificity, fluency, and diversity.  The results show that SIU outperforms the baseline methods in most metrics.  Note that the Specificity metric is further detailed in Table 7.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_25_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method with three existing machine unlearning methods: Preference Optimization (PO), Gradient Ascent (GA), and GA with KL-divergence (GA+KL).  The comparison is done across several metrics including efficacy (EM), generality (G-Eval, C-Dis), specificity, fluency, and diversity.  The results are based on three independent trials and show the mean and standard deviation for each metric. Note that the specificity results are summarized in a separate Table 7.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_27_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three baseline methods (Preference Optimization (PO), Gradient Ascent (GA), and GA+KL) for machine unlearning in large language models (LLMs).  The comparison is done across several metrics: Efficacy (Exact Match), Generality (G-Eval, C-Dis), Specificity, Fluency, and Diversity.  The results show SIU's superior performance, especially in maintaining the LLM's utility after unlearning (Generality, Specificity, Fluency, Diversity).

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_28_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against several existing machine unlearning methods.  The comparison is done across multiple metrics, including efficacy, generality, specificity, fluency, and diversity.  Efficacy measures how effectively the model forgets the target concept. Generality assesses the model's ability to generalize the forgetting to unseen examples. Specificity evaluates the impact of unlearning on non-target knowledge. Fluency and diversity measure the utility of the model after unlearning.  The results show that SIU significantly outperforms the baselines across various metrics, achieving high efficacy and generality while maintaining good specificity, fluency, and diversity.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_29_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three existing machine unlearning methods (PO, GA, GA+KL) across various metrics.  The metrics include efficacy (EM), generality (G-Eval, C-Dis), specificity, fluency, and diversity.  The comparison is done for two different sizes of LLAMA model (7B and 13B), showing the effectiveness of SIU, particularly in maintaining utility (specificity, fluency, diversity) while achieving high efficacy and generality.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_30_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three baseline methods (Preference Optimization (PO), Gradient Ascent (GA), and GA+KL) for machine unlearning in large language models.  The comparison is done across multiple metrics: Efficacy (Exact Match), Generality (G-Eval, C-Dis), Specificity, Fluency, and Diversity.  The results show SIU's superior performance across all metrics, particularly in terms of generality and fluency.  Note that Specificity scores are detailed in a separate Table 7. The experiment is conducted on two different sizes of LLAMA (7B and 13B) to examine its robustness against different model scales.

![](https://ai-paper-reviewer.com/YNx7ai4zTs/tables_31_1.jpg)
> This table compares the performance of the proposed Single Image Unlearning (SIU) method against three baseline methods: Preference Optimization (PO), Gradient Ascent (GA), and GA with KL-divergence (GA+KL).  The comparison is made across several metrics: Efficacy (EM), Generality (G-Eval, C-Dis), Specificity, Fluency, and Diversity.  The results show SIU outperforms the baseline methods across various aspects, demonstrating its effectiveness in unlearning visual information from Multimodal Large Language Models (MLLMs).  The table includes results for two different sizes of LLAVA models (7B and 13B) for comprehensive comparison.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/YNx7ai4zTs/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}