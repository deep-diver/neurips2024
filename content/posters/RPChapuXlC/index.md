---
title: "Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack"
summary: "Lisa: a novel lazy safety alignment method safeguards LLMs against harmful fine-tuning attacks by introducing a proximal term to constrain model drift, significantly improving alignment performance."
categories: []
tags: ["Natural Language Processing", "Large Language Models", "🏢 Georgia Institute of Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} RPChapuXlC {{< /keyword >}}
{{< keyword icon="writer" >}} Tiansheng Huang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=RPChapuXlC" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95174" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/papers/2405.18641" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=RPChapuXlC&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/RPChapuXlC/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Large language models (LLMs), while offering immense potential, are susceptible to security risks.  Specifically, fine-tuning LLMs on datasets containing harmful content can compromise their safety mechanisms, leading to undesirable outputs.  This is a significant challenge for providers of LLM fine-tuning services, who bear responsibility for the model's behavior. Existing solutions often address this issue through alignment-stage methods or computationally intensive fine-tuning stages, both of which may have limitations.

This paper introduces Lisa, a novel Lazy safety alignment technique that aims to improve LLM safety during the fine-tuning stage.  Lisa uses bi-state optimization, alternating between alignment and user data. Crucially, it adds a proximal term to constrain model drift towards the switching points between the two states. Theoretical analysis and empirical results on various downstream tasks demonstrate that Lisa increases alignment performance while maintaining the LLM's accuracy on user tasks, addressing the limitations of previous approaches.  It achieves a computation-efficient fine-tuning-stage mitigation against harmful data.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Large Language Models (LLMs) with safety alignment are vulnerable to jail-breaking via fine-tuning on harmful data. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The proposed Lisa method uses a proximal term to mitigate model drift during bi-state optimization, enhancing safety alignment. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Lisa significantly improves alignment performance across various downstream tasks with minimal impact on accuracy. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on the safety and security of large language models (LLMs).  It directly addresses the critical issue of **harmful fine-tuning attacks**, offering a novel solution that improves alignment performance while maintaining accuracy. This work is highly relevant to current research trends focusing on making LLMs more robust and trustworthy, opening new avenues for research into efficient and effective safety mechanisms.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/RPChapuXlC/figures_0_1.jpg)

> This figure illustrates a typical two-stage fine-tuning-as-a-service pipeline.  The first stage involves aligning a pre-trained Large Language Model (LLM) using alignment data. The second stage customizes the aligned LLM using user-provided data. The figure highlights the risk of harmful fine-tuning attacks where malicious data compromises the alignment performance achieved in the first stage. The authors of the paper focus on mitigating the risks in Stage ②, the user fine-tuning stage, in contrast to existing solutions like Vaccine which focus on improving the alignment stage.





![](https://ai-paper-reviewer.com/RPChapuXlC/tables_3_1.jpg)

> This table presents the results of the Bi-State Optimization (BSO) method and the standard Supervised Fine-Tuning (SFT) method on the SST-2 dataset using a Llama2-7B model. The harmful ratio is varied from 0% to 40%, while the alignment and fine-tuning steps are kept constant at 500 each. The table compares the harmful score and fine-tuning accuracy of both methods under different levels of harmful data contamination.





### In-depth insights


#### Lazy Alignment
The concept of "Lazy Alignment" in the context of large language models (LLMs) suggests a strategy to **efficiently maintain safety** while adapting the model to user-specific tasks.  It likely involves a trade-off between full retraining for optimal alignment and minimal updates to reduce computational cost.  This approach might involve techniques like **proximal updates** or selective fine-tuning, focusing on areas most likely to be impacted by user data, while minimizing changes that could compromise the existing safety properties.  **Regularization** methods, constraints on model drift from the pre-aligned state, or the use of efficient model adaptation techniques like LoRA could be key components.  The effectiveness of this approach would hinge on the ability to identify and prioritize sensitive aspects of the LLM's behavior during the alignment process while ensuring minimal disruption of the user tasks.  A successful "Lazy Alignment" technique would be a **significant step towards balancing safety with the need for efficient customization** of LLMs in real-world applications.

#### BSO Instability
Analyzing the instability within the Bi-State Optimization (BSO) method reveals crucial insights into the limitations of simply separating alignment and user data during fine-tuning.  **Asymmetrical computing**, where fewer steps are dedicated to the alignment phase, significantly degrades alignment performance. This is because insufficient alignment steps lead to **convergence instability**, with the model excessively drifting towards the switching iterates of the two states. This instability manifests as an increased harmful score in the fine-tuned model, demonstrating the method's vulnerability to imbalanced computation. The underlying cause appears to be **excess drift**, a phenomenon where the model's parameters shift significantly between alignment and user fine-tuning, leading to inconsistencies and a loss of previously learned safety features. This highlights the importance of a balanced approach, where sufficient computational resources are dedicated to the alignment stage, to ensure a robust and secure fine-tuning process for large language models.

#### Empirical Results
An Empirical Results section for a paper on mitigating harmful fine-tuning in LLMs would ideally present a comprehensive evaluation across multiple dimensions.  **Quantitative metrics** like harmful score reduction, accuracy on downstream tasks, and computational overhead should be reported, along with statistical significance tests.  **Diverse model architectures and datasets** should be included to demonstrate robustness and generalizability. A comparison with existing defense mechanisms is essential, highlighting the proposed method's advantages.  **Qualitative analysis**, such as examples of model outputs illustrating improvements in safety while maintaining user-task performance, can add significant value, showcasing real-world implications and nuanced effects.  Finally, **ablation studies** isolating the contributions of individual components within the proposed method would strengthen the conclusions, demonstrating the effectiveness and necessity of each element.

#### Convergence Analysis
A rigorous convergence analysis is crucial for establishing the reliability and efficiency of any proposed algorithm.  In the context of the discussed research, such an analysis would delve into the theoretical guarantees of the algorithm's ability to reach a solution, and how quickly it does so. This would involve defining appropriate metrics for convergence, such as the distance between successive iterates or the change in the objective function value. **Key assumptions** about the problem structure, such as smoothness or convexity, would be clearly stated. The analysis might then employ techniques from optimization theory to prove convergence bounds, establishing that under certain conditions, the algorithm will converge to a solution within a certain number of iterations or time.  **Different convergence rates** (linear, sublinear, etc.) might be analyzed depending on the algorithm and problem assumptions. The analysis may also examine the **impact of hyperparameters** on the convergence behavior, providing guidelines for their optimal selection.  Further, a **sensitivity analysis** could explore the robustness of the algorithm's convergence in the presence of noise or perturbations in the input data. Ultimately, a thorough convergence analysis provides essential insights into both the theoretical foundation and practical performance of the algorithm.

#### Future Work
Future research directions stemming from this paper on mitigating harmful fine-tuning in LLMs could explore several promising avenues.  **Extending Lisa's effectiveness to RLHF-trained models** is crucial, as RLHF is currently the state-of-the-art alignment technique. Investigating **alternative proximal terms or regularization methods** beyond the L2 norm to constrain model drift could lead to improved performance and stability.  Further research should investigate the **optimal balance between alignment and fine-tuning steps** in the Bi-State Optimization to further enhance computational efficiency without sacrificing alignment.  Additionally, a deeper analysis of **the interplay between model architecture, dataset characteristics, and attack strategies** on the effectiveness of Lisa is essential for practical applications.  Finally, developing **a comprehensive benchmark and evaluation framework** for assessing the robustness of LLMs against various harmful fine-tuning attacks would significantly benefit the field.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/RPChapuXlC/figures_3_1.jpg)

> This figure shows the impact of harmful data ratio on the performance of fine-tuned language models.  It presents three subplots: Harmful Score, Finetune Accuracy, and Alignment Loss. Each subplot shows how these metrics vary across different ratios of harmful data in the fine-tuning dataset, comparing models trained with and without initial safety alignment. The results highlight the significant impact of even small amounts of harmful data on the model's safety and ability to retain prior alignment.


![](https://ai-paper-reviewer.com/RPChapuXlC/figures_3_2.jpg)

> This figure illustrates the Bi-State Optimization (BSO) method.  It shows a two-stage process. The first stage involves training a pre-trained LLM on alignment data to achieve safety alignment. In the second stage, the model undergoes fine-tuning with user-provided data, while also incorporating the alignment data. This alternating optimization between the two data sets aims to prevent the model from forgetting the safety alignment learned in the first stage, effectively mitigating the risk of harmful fine-tuning.


![](https://ai-paper-reviewer.com/RPChapuXlC/figures_4_1.jpg)

> This figure shows the analysis of convergence instability in the Bi-State Optimization method.  The left panel displays alignment loss versus the number of fine-tuning steps for different allocations of steps between alignment and fine-tuning.  The middle panel shows the gradient norm, indicating how close the optimization is to a stationary point. The right panel illustrates the drift towards the switching point between the alignment and fine-tuning states.  The figure demonstrates that asymmetrical computing (unequal allocation of steps) leads to instability in convergence, primarily due to excessive drift towards the switching point.


![](https://ai-paper-reviewer.com/RPChapuXlC/figures_9_1.jpg)

> This figure displays the harmful score, finetune accuracy, and alignment loss of a Llama2-7B model after being fine-tuned on a dataset containing varying percentages of harmful data (0%, 5%, 10%, 20%, 30%, 40%). It compares two scenarios: one where the model underwent safety alignment (SFT) before fine-tuning and one where it didn't (NA-SFT).  The results illustrate that even a small amount of harmful data (5%) can significantly increase the harmful score, regardless of prior alignment. Importantly, the finetune accuracy remains relatively consistent across different harmful ratios in both scenarios, making it difficult to detect poisoning simply by evaluating finetune accuracy.  Finally, for the aligned model (SFT), the alignment loss increases with the harmful data ratio, showing that the harmful data leads the model to 'forget' its previous alignment training.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/RPChapuXlC/tables_4_1.jpg)
> This table presents the performance of the Bi-State Optimization (BSO) method and the standard Supervised Fine-Tuning (SFT) method under different harmful ratios. The harmful ratio represents the percentage of harmful data mixed with the fine-tuning data.  The table shows the harmful score and finetune accuracy for both methods across various harmful ratios.  A lower harmful score is better, while a higher finetune accuracy is better.  The experiment uses the SST-2 dataset and Llama2-7B model, with the number of alignment and fine-tuning steps being equal and set at 500 for both BSO and SFT.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_4_2.jpg)
> This table presents the results of the Bi-State Optimization (BSO) method with varying step allocations between the alignment and fine-tuning states.  It shows how the harmful score changes as the proportion of steps dedicated to the alignment state decreases. The results demonstrate a significant increase in the harmful score as fewer steps are allocated to the alignment state, highlighting the importance of sufficient alignment optimization. This indicates the convergence instability experienced when alignment steps invested are too few.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_5_1.jpg)
> This table presents the performance of the Bi-State Optimization (BSO) method compared to standard supervised fine-tuning (SFT) under different harmful ratios in the fine-tuning dataset.  It shows the harmful score and fine-tuning accuracy for both methods across various percentages of harmful data (p=0.05, 0.1, 0.2, 0.3, 0.4). The results demonstrate BSO's effectiveness in mitigating the harmful effects of the harmful data, while maintaining a comparable level of accuracy.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_6_1.jpg)
> This table presents the performance of the Bi-State Optimization (BSO) method compared to standard supervised fine-tuning (SFT) under various harmful data ratios (0%, 5%, 10%, 20%, 30%, 40%).  It shows the harmful score (lower is better) and finetune accuracy (higher is better) for both methods across different harmful data ratios. The experiment uses the SST-2 dataset and a Llama2-7B model.  The equal number of steps (500) are used for both alignment and fine-tuning states.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_6_2.jpg)
> This table presents the performance of different methods (NA-SFT, SFT, EWC, Vaccine-SFT, Vlguard, BSO, Lisa) under varying harmful ratios (clean, p=0.05, p=0.1, p=0.2, p=0.3) with a fixed sample number of 5000.  The results show the harmful score (lower is better) and finetune accuracy (higher is better) for each method and harmful ratio. This demonstrates the effectiveness of each method at mitigating the negative impact of harmful data on model performance.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_7_1.jpg)
> This table presents the results of harmful score and fine-tune accuracy using three different pre-trained language models (Opt-2.7B, Llama2-7B, Mistral-7B) on the GSM8K dataset.  It compares several methods: NA-SFT (no alignment, standard fine-tuning), SFT (standard fine-tuning with alignment), Vaccine-SFT (alignment stage modification with standard fine-tuning), Vlguard (alignment with modified fine-tuning), BSO (Bi-State Optimization), and Lisa (Lazy Safety Alignment).  The lower the harmful score, the better the model's performance against harmful data.  The higher the fine-tune accuracy, the better the model performs on the intended task. The average across all three models shows that Lisa shows the best results. 

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_7_2.jpg)
> This table presents the performance of different models (NA-SFT, SFT, Vaccine-SFT, Vlguard, BSO, Lisa) on four different downstream fine-tuning datasets (SST2, AGNEWS, GSM8K, AlpacaEval) using the Mistral-7B model.  The metrics shown are Harmful Score (HS) and Finetune Accuracy (FA). Lower HS indicates better performance against harmful fine-tuning attacks, while higher FA shows better accuracy on the main task.  The table aims to demonstrate the generalization performance of the proposed Lisa method across different datasets and its effectiveness compared to baseline methods in mitigating harmful fine-tuning.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_7_3.jpg)
> This table presents the performance of the Bi-State Optimization (BSO) method and the standard Supervised Fine-Tuning (SFT) method under various harmful ratios (0%, 5%, 10%, 20%, 30%, 40%). The harmful score and finetune accuracy are reported for each method and harmful ratio.  The results demonstrate the effectiveness of BSO in mitigating the negative impact of harmful data on model performance. The experiment used the SST-2 dataset for fine-tuning and a Llama2-7B model as the base model.  The number of optimization steps for both the alignment and fine-tuning stages was set to 500.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_7_4.jpg)
> This table presents the performance comparison of different methods (SFT, VlGuard, BSO, Lisa) in terms of clock time and memory usage.  It shows the computational cost and memory footprint associated with each approach during the experiments.  The values are likely measured in seconds for clock time and gigabytes for memory.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_8_1.jpg)
> This table shows the impact of different step allocations between alignment and fine-tuning states on the performance of the Lisa model. It demonstrates the trade-off between harmful score mitigation and fine-tuning accuracy, revealing that a balanced allocation leads to better results. The table also demonstrates that when too many steps are invested in finetuning, the model accuracy decreases while the harmful score increases. When too many steps are invested in alignment, the model achieves a low harmful score but low accuracy.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_8_2.jpg)
> This table shows the performance of Lisa model with different proximal penalty (intensity) values. The harmful score and finetune accuracy are reported for each intensity value. It demonstrates how the proximal term affects the trade-off between safety and accuracy. A larger intensity generally reduces the harmful score, but may slightly decrease the finetune accuracy. 

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_8_3.jpg)
> This table presents the performance of the Bi-State Optimization (BSO) method compared to standard supervised fine-tuning (SFT) under different harmful ratios in the fine-tuning dataset.  It shows the harmful score and fine-tune accuracy for both methods across varying percentages of harmful data mixed with the clean data. The results demonstrate BSO's effectiveness in mitigating the negative impact of harmful data.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_9_1.jpg)
> This table presents the performance of different methods (NA-SFT, SFT, EWC, Vaccine-SFT, Vlguard, BSO, Lisa) under various harmful ratios (clean, p=0.05, p=0.1, p=0.2, p=0.3) in the default setting.  It shows the harmful score (lower is better) and finetune accuracy (higher is better) for each method and harmful ratio.  This helps assess the effectiveness of each method in mitigating the impact of harmful data during fine-tuning. The default setting includes using the Llama2-7B model and SST-2 dataset.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_9_2.jpg)
> This table compares the performance of different methods when combined with data filtration using the BeaverTails moderation model.  It shows the harmful score and finetune accuracy for various poison ratios (p=0.1, p=0.2, p=0.5, p=0.8, p=1) for four methods: SFT (no filter), Lisa (no filter), Filter+SFT, and Filter+Lisa. The results demonstrate the effectiveness of combining data filtration with the Lisa method in reducing the harmful score while maintaining acceptable finetune accuracy.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_18_1.jpg)
> This table shows how different mitigation strategies perform given different harmful ratios in the default setting.  The fine-tuning sample number is fixed at 5000.  It compares the performance of several methods (SFT, Vaccine-SFT, Lisa) across various harmful ratios (p=0.1, p=0.3, p=0.5, p=0.7, p=0.9, p=1), evaluating both harmful score and finetune accuracy. The table demonstrates the robustness of Lisa to different levels of harmful data contamination.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_19_1.jpg)
> This table presents the harmonic mean of harmful score and finetune accuracy for different methods under various harmful ratios (0.1, 0.2, 0.5, 0.8, 1), with and without data filtration.  The harmonic mean combines the two metrics to provide a more holistic evaluation of performance. It compares standard fine-tuning (SFT), Lisa (without filtration), Filter+SFT (data filtration followed by SFT), and Filter+Lisa (data filtration followed by Lisa). The results show that Filter+Lisa generally outperforms other methods, demonstrating the effectiveness of combining data filtration and the proximal term in Lisa.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_19_2.jpg)
> This table presents the results of an experiment evaluating the robustness of different methods against a Bi-directional Anchoring attack.  The attack aims to select the most harmful data from a benign dataset to use for fine-tuning. The table shows the harmful score (a measure of the model's tendency to generate harmful outputs) before and after fine-tuning for three methods: SFT (standard supervised fine-tuning), Vaccine-SFT (an alignment-stage solution), and Lisa (the proposed method). The results demonstrate that Lisa exhibits more robustness compared to SFT and Vaccine-SFT against this specific attack, showing little change in its harmful score.

![](https://ai-paper-reviewer.com/RPChapuXlC/tables_20_1.jpg)
> This table presents the performance of the Bi-State Optimization (BSO) method compared to the standard Supervised Fine-Tuning (SFT) method under different harmful ratios (percentage of harmful data in the fine-tuning dataset).  It shows harmful scores and fine-tuning accuracy for both methods with varying harmful data proportions (0%, 5%, 10%, 20%, 30%, 40%). The goal is to demonstrate BSO's effectiveness in mitigating the negative impact of harmful data on model performance.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/RPChapuXlC/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RPChapuXlC/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}