---
title: 'Aligner: Efficient Alignment by Learning to Correct'
summary: Aligner efficiently aligns LLMs by learning to correct initial responses,
  achieving significant improvements in helpfulness and harmlessness across various
  models with resource efficiency.
categories: []
tags:
- Large Language Models
- "\U0001F3E2 Peking University"
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} kq166jACVP {{< /keyword >}}
{{< keyword icon="writer" >}} Jiaming Ji et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=kq166jACVP" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93865" target="_blank" >}}
↗ NeurIPS Proc.
{{< /button >}}{{< button href="https://huggingface.co/papers/2402.02416" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=kq166jACVP&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/kq166jACVP/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current LLM alignment methods are complex and resource-intensive, hindering rapid iteration and deployment. This necessitates the development of simpler, model-agnostic approaches.  The challenge lies in finding a balance between efficacy and efficiency in alignment techniques.

The paper introduces Aligner, a simple yet effective alignment paradigm that learns correctional residuals between preferred and dispreferred responses using a small model.  This plug-and-play module can be easily integrated with various LLMs for improved alignment without extensive retraining.  Experiments demonstrate significant performance gains across different LLMs on multiple dimensions, surpassing state-of-the-art models in resource efficiency and demonstrating effectiveness even when stacked on top of powerful models like GPT-4.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Aligner is a model-agnostic, plug-and-play module easily integrated with various LLMs. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Aligner significantly improves LLM helpfulness and harmlessness across multiple models. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Aligner is resource-efficient, outperforming existing methods in terms of training resources. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in AI alignment because it introduces **Aligner**, a novel and efficient method for aligning LLMs with human values.  Aligner's model-agnostic nature and resource efficiency address key limitations of existing alignment techniques, opening up **new avenues for rapid iteration and deployment in real-world scenarios.**  The results demonstrate significant improvements in various downstream LLM tasks across multiple models, highlighting the method's wide applicability and potential to enhance the overall safety and helpfulness of LLMs.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/kq166jACVP/figures_1_1.jpg)

> The figure illustrates the Aligner model architecture and its functionality.  The left side shows the Aligner as a plug-and-play module that takes the output of an upstream large language model (LLM) and modifies it to be more helpful and harmless.  It does this by redistributing the initial answer in semantic space. The right side provides an analogy comparing the Aligner to a residual block in a neural network, highlighting its efficiency in enhancing the original response without significantly altering the base model's structure.





![](https://ai-paper-reviewer.com/kq166jACVP/tables_3_1.jpg)

> This table presents the performance improvements achieved by integrating the Aligner model with 11 different LLMs.  The improvements are quantified across three dimensions (helpfulness, harmlessness, and honesty) and various evaluation datasets.  Aligner's model-agnostic nature is highlighted, as it only requires a single training session to enhance the performance of various upstream models.  The results showcase significant performance gains across the board.





### In-depth insights


#### Aligner: Correcting LLMs
The concept of "Aligner: Correcting LLMs" presents a novel approach to improving the alignment of large language models (LLMs) with human values.  Instead of relying on complex and resource-intensive methods like reinforcement learning from human feedback (RLHF), **Aligner uses a lightweight, plug-and-play module trained to learn the correctional residuals between preferred and dispreferred answers**. This approach significantly reduces computational demands, facilitates rapid iteration in deployment scenarios, and is **model-agnostic**, enabling its use with various upstream LLMs.  The core innovation lies in its simplicity: **Aligner focuses on correcting existing outputs rather than generating answers from scratch**, making the learning process far more efficient.  By iteratively bootstrapping upstream models with corrected responses, **Aligner can even break through performance ceilings** and continuously improve LLM alignment. The effectiveness of Aligner is demonstrated via improved scores across multiple benchmark datasets, showcasing its potential to be a practical and efficient solution for enhancing the alignment of LLMs.

#### Residual Correction
The concept of "Residual Correction" in the context of aligning Large Language Models (LLMs) offers a novel approach to improving model outputs.  Instead of directly training the model to generate ideal responses, **it focuses on learning the differences between preferred and non-preferred answers**.  This residual, representing the 'correction' needed, is learned by a smaller, more efficient model (Aligner). This is computationally advantageous because training a small correction model is significantly less resource-intensive than retraining the entire LLM.  The method’s **plug-and-play nature** makes it highly adaptable to various upstream LLMs, promoting rapid iteration and deployment. This approach demonstrates a clever use of residual learning principles, known for their efficiency and effectiveness in other domains of deep learning.  **Interpretability is enhanced** by directly modeling the corrections needed, providing insights into how the Aligner modifies the original output. This approach stands in contrast to reinforcement learning methods (RLHF) which can be complex and computationally expensive. By simplifying the alignment process, residual correction offers a **more efficient and versatile solution** for improving the helpfulness and harmlessness of LLMs without sacrificing model performance.

#### Multi-round RLHF
The section on "Multi-round RLHF" likely explores iterative refinement of large language models (LLMs) using reinforcement learning from human feedback (RLHF).  **Standard RLHF often suffers from reward model collapse**, where the reward model's preferences drift from the actual desired behavior, leading to suboptimal alignment.  Multi-round RLHF aims to address this by repeatedly refining the reward model and LLM policy.  The authors may propose using a lightweight, model-agnostic module like "Aligner" to efficiently improve each round's alignment. This would involve using Aligner to correct initial LLM outputs, generating synthetic preference data that better reflects human intentions, and then retraining the reward model and LLM policy on this improved data.  **Aligner's efficiency and plug-and-play nature** make it especially suitable for iterative processes, reducing resource consumption compared to standard RLHF techniques.  The authors likely present experimental results showcasing the effectiveness of this multi-round approach in enhancing LLM alignment while mitigating reward model collapse, leading to more robust and reliable alignment.  **A key insight would be demonstrating how Aligner improves the quality of synthetic preference data** in each iteration, helping the process converge towards higher quality alignment faster and more effectively.

#### Efficiency & Scalability
The research paper highlights the efficiency and scalability of its proposed alignment method, **Aligner**.  Aligner's efficiency stems from its **plug-and-play modular design**, requiring only one-off training and readily applicable to diverse LLMs, including those accessed via APIs.  This contrasts sharply with resource-intensive methods like RLHF.  The model-agnostic nature of Aligner promotes scalability by avoiding the need to retrain large models for each new deployment scenario.  **Aligner's lightweight nature**, significantly smaller than competing approaches like DPO and RLHF, also contributes to its efficiency. The improved efficiency and scalability of Aligner makes it a **practical and versatile alignment solution**, especially valuable in rapidly evolving deployment environments where iterative refinement is crucial.  The demonstrated improvements across numerous LLMs further underscore Aligner's broad applicability and its potential to facilitate a more efficient and effective LLM alignment process.

#### Interpretability
The research paper section on "Interpretability" delves into understanding the internal mechanisms of the Aligner model.  **Key to this is the model's ability to learn correctional residuals**, moving beyond a simple binary decision of correction or copying. Instead, the model's behavior is shown to be conditional on the input's quality, with a dynamic balance between preserving and modifying the initial answer.  **Experiments using techniques like Linear Artificial Tomography (LAT)** reveal that this decision-making process predominantly occurs in the Aligner's earlier layers.  The **LAT scan graphs vividly illustrate the different neural activity patterns** associated with correction and copying, offering valuable insights into the model's internal workings.  Furthermore,  **representation control experiments using extracted representation vectors demonstrate that the magnitude of correction is directly related to this control**, strengthening the argument that the model has internalized a nuanced approach to corrections. Overall, the "Interpretability" section provides a strong foundation for understanding the Aligner's efficacy by providing specific technical details of its inner workings.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/kq166jACVP/figures_4_1.jpg)

> This figure shows the distribution of helpfulness and harmlessness scores before and after applying Aligner-7B to different upstream models.  Panel (a) displays the distribution in the training data, highlighting the difference between preferred and dis-preferred answers. Panels (b1-b3) show how Aligner-7B shifts the distribution for different LLMs, improving helpfulness and harmlessness.  It demonstrates Aligner-7B's ability to correct for various model shortcomings, including refusal to answer and lack of alignment.


![](https://ai-paper-reviewer.com/kq166jACVP/figures_5_1.jpg)

> This figure shows the distribution shifts in helpfulness and harmlessness scores before and after applying Aligner-7B to different LLMs.  It illustrates how Aligner-7B improves the scores, especially for models that initially had low scores or exhibited problematic behaviors like refusing to answer.  The plots show the distribution of scores in the training data and then the resulting distribution after Aligner-7B's intervention for several different LLMs.


![](https://ai-paper-reviewer.com/kq166jACVP/figures_7_1.jpg)

> This figure shows the results of interpretability experiments on the Aligner model.  The LAT scan graphs (a) and (b) visualize the neural activity in different layers of the Aligner while generating responses. Graph (a) shows higher activity for correction, while (b) shows a tendency to copy the original response, indicating that the correction decision is made primarily in the early layers. Graph (c) demonstrates how manipulating the correction representation vector linearly affects the degree of correction applied by the Aligner, confirming its interpretability.


![](https://ai-paper-reviewer.com/kq166jACVP/figures_7_2.jpg)

> This figure illustrates how Aligner, a plug-and-play module, can be integrated into a multi-round RLHF (Reinforcement Learning from Human Feedback) or DPO (Direct Preference Optimization) pipeline to improve alignment.  In each round, the upstream LLM generates a response (A), which is then refined by Aligner to produce a better response (A*).  These improved responses (A*) are used to create a synthetic preference dataset for the next round of RLHF/DPO, iteratively bootstrapping the upstream model's alignment with human preferences and values. This iterative process helps mitigate reward model collapse and over-optimization that can occur in typical multi-round RLHF/DPO training.


![](https://ai-paper-reviewer.com/kq166jACVP/figures_8_1.jpg)

> The figure shows the results of a multi-round alignment pipeline using Aligner, compared to standard multi-round PPO and DPO.  The x-axis represents helpfulness, and the y-axis represents harmlessness.  Each point represents the model's performance after a round of training.  Aligner consistently improves both helpfulness and harmlessness across multiple rounds, unlike the other methods which mainly focus on helpfulness, often at the cost of increased harmfulness. This demonstrates Aligner's ability to enhance both dimensions simultaneously and its effectiveness in mitigating reward model collapse in multi-round RLHF.


![](https://ai-paper-reviewer.com/kq166jACVP/figures_14_1.jpg)

> This figure illustrates the methodology of using weak models to supervise strong models, specifically focusing on the 'Weak-to-Strong Correction via Aligner' approach.  It compares three scenarios: Super Alignment (human directly supervising a very strong AI), Weak-to-Strong Generalization (a weaker AI supervising a stronger AI), and the proposed Weak-to-Strong Correction via Aligner (where a lightweight Aligner model acts as the weak supervisor to correct and improve the outputs of a much stronger LLM, such as GPT-4 or Llama2). The figure emphasizes the scalability and reliability of the proposed method compared to direct human supervision, which becomes increasingly difficult as AI models become more powerful.


![](https://ai-paper-reviewer.com/kq166jACVP/figures_14_2.jpg)

> This figure illustrates the difference between the Weak-to-Strong Generalization and the Weak-to-Strong Correction methodologies.  The former involves a weak model generating labels for training a strong model. The latter uses Aligner, a smaller model, to correct the output of a strong model, creating training labels to further improve the strong model's performance.  This highlights the paper's approach of using a smaller, efficient model (Aligner) to enhance the alignment of larger language models.


![](https://ai-paper-reviewer.com/kq166jACVP/figures_27_1.jpg)

> This figure illustrates the data processing pipeline for creating the training dataset used in the Aligner model.  It starts with a raw corpus of prompts, which undergoes prompt quality filtering.  Then, multiple language models (LLMs) generate answers.  These are filtered for quality and duplicates, resulting in pairwise data.  Finally, multiple annotators, including human annotators, GPT-4 and Llama2-70B-Chat, provide corrections, leading to a final training dataset of 50K query-answer-correction (Q-A-C) triplets.


![](https://ai-paper-reviewer.com/kq166jACVP/figures_29_1.jpg)

> The figure shows the architecture of the Aligner module, a plug-and-play module that can be stacked on top of any upstream LLM to improve its alignment with human intentions.  The left side illustrates how Aligner redistributes the initial LLM output to produce more helpful and harmless responses. The right side draws an analogy between Aligner and a residual block in a neural network, highlighting its ability to enhance the upstream model without significantly altering its parameters.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/kq166jACVP/tables_6_1.jpg)
> This table presents the performance improvement achieved by integrating the Aligner model with 11 different LLMs.  The improvements are quantified across three dimensions: helpfulness, harmlessness, and honesty (3H). The table shows the average improvement percentage for each LLM and the average improvement across all tested LLMs. Note that the Aligner model only required a single training session to work across multiple models.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_6_2.jpg)
> This table presents the performance improvements achieved by integrating the Aligner model with various upstream LLMs.  The results are evaluated using three metrics (Helpfulness, Harmlessness, Honesty) across multiple datasets and models.  A key highlight is that Aligner consistently improves the performance of upstream models with only one training session.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_15_1.jpg)
> This table presents the performance improvements achieved by deploying the Aligner model across eleven different LLMs, evaluated on helpfulness, harmlessness, and honesty.  The results are shown as percentage increases compared to the base LLM performance for various Aligner sizes.  It demonstrates the model-agnostic nature of Aligner and its ability to improve upon diverse base models without requiring retraining.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_18_1.jpg)
> This table presents the performance improvements achieved by integrating the Aligner model with 11 different LLMs.  The improvements are measured across three dimensions: helpfulness, harmlessness, and honesty (3H).  The table shows the percentage increase in these metrics for different Aligner and LLM combinations.  It highlights Aligner's model-agnostic nature, as it requires only a single training session to be effective with various models.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_18_2.jpg)
> This table presents the performance improvements achieved by deploying the Aligner model across eleven different LLMs.  The evaluation is based on three dimensions (helpfulness, harmlessness, and honesty).  The table shows the percentage improvement in each of these dimensions for various Aligner sizes (2B, 7B) when combined with different upstream LLMs.  It highlights the model-agnostic and plug-and-play nature of Aligner, as only one-off training is needed for each Aligner size to significantly improve the performance of a variety of LLMs.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_19_1.jpg)
> This table presents the performance improvements achieved by deploying the Aligner model across different LLMs, evaluated on the three dimensions (helpfulness, harmlessness, and honesty).  The average improvement percentage for each LLM is displayed, showing the effectiveness of Aligner across various models.  It also highlights the resource efficiency of Aligner requiring only a single training session.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_20_1.jpg)
> This table presents the performance improvement achieved by integrating Aligner with various upstream LLMs.  The improvements are quantified across three dimensions: helpfulness, harmlessness, and honesty.  Aligner's model-agnostic nature is highlighted, showing consistent gains across different models using only one training session.  The average improvement percentages are reported, indicating the efficacy of Aligner in enhancing the performance of various LLMs.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_21_1.jpg)
> This table presents the performance improvements achieved by integrating the Aligner model with 11 different LLMs, evaluated on the three dimensions of helpfulness, harmlessness, and honesty. The results show that Aligner consistently improves the performance of the base models across different scales and model types.  Noteworthy is that Aligner only requires one-off training to be applied to various models and achieves zero-shot improvements on unseen models.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_22_1.jpg)
> This table presents the performance improvement achieved by integrating the Aligner model with eleven different LLMs.  The improvements are measured across three dimensions (helpfulness, harmlessness, and honesty) and are presented as percentage increases compared to the baseline performance of each LLM without the Aligner. The table demonstrates the model-agnostic nature of Aligner, showcasing its effectiveness in enhancing various LLMs with only a single training session.  Different sizes of Aligner models (2B, 7B, 13B) are tested.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_22_2.jpg)
> This table presents the performance improvement achieved by integrating the Aligner module with various upstream LLMs.  The results are categorized by the Aligner model size (2B, 7B, 13B) and the upstream LLM used. Improvements are measured across three dimensions: helpfulness, harmlessness, and honesty, showing percentage increases from the baseline upstream model's performance. Notably, Aligner only requires one training session regardless of the upstream model, making it a highly efficient and model-agnostic alignment method.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_23_1.jpg)
> This table presents the performance improvement achieved by integrating the Aligner model with 11 different LLMs, evaluated on the 3H dimensions (helpfulness, harmlessness, and honesty).  The table shows the percentage increase in each 3H metric for various Aligner and LLM combinations (Aligner-2B, Aligner-7B, etc.).  The results highlight the model-agnostic nature of Aligner, showcasing consistent performance improvements across diverse models with only one-off training.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_23_2.jpg)
> This table presents the performance improvements achieved by integrating Aligner with various upstream LLMs.  It shows the percentage increase in helpfulness, harmlessness, and honesty scores across different LLMs and Aligner model sizes (2B and 7B). The results demonstrate that Aligner significantly improves the performance of various upstream models, including those available via APIs, in a model-agnostic manner and with only one-off training.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_24_1.jpg)
> This table presents the performance improvements achieved by integrating the Aligner model with various Large Language Models (LLMs).  It shows the percentage increase in helpfulness, harmlessness, and honesty scores across 11 different LLMs, after applying Aligner. The results demonstrate significant improvements, highlighting Aligner's effectiveness as a model-agnostic alignment approach that doesn't require retraining the base LLMs.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_25_1.jpg)
> This table presents the performance improvement achieved by integrating Aligner with 11 different LLMs across various evaluation metrics.  The results demonstrate Aligner's effectiveness in enhancing the helpfulness, harmlessness, and honesty of these models, even without needing retraining for each LLM. The average improvement across models is highlighted, showcasing Aligner's model-agnostic and plug-and-play capabilities.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_26_1.jpg)
> This table presents the performance improvements achieved by integrating the Aligner model with 11 different Large Language Models (LLMs).  The improvements are measured across three dimensions: helpfulness, harmlessness, and honesty (3H). The table shows the percentage increase in each 3H dimension for each LLM when using the Aligner, highlighting the model's ability to enhance the performance of various upstream models.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_26_2.jpg)
> This table presents the performance improvements achieved by deploying Aligner across 11 different LLMs, evaluated on the 3H dimensions (helpfulness, harmlessness, and honesty).  The results show percentage increases in helpfulness, harmlessness, honesty and other metrics (e-dialogue, DialogSum, Beavertails, HarmfulQA, TruthfulQA) for each LLM when Aligner is integrated.  The table is broken into sections for different Aligner sizes (2B, 7B) to show performance differences.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_28_1.jpg)
> This table presents the performance improvements achieved by integrating the Aligner model with 11 different LLMs.  The improvements are measured across three dimensions (helpfulness, harmlessness, and honesty) and quantified as percentage increases over the original LLM's performance. Notably, a single Aligner training session is sufficient to enhance multiple models, highlighting its model-agnostic and efficient nature.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_28_2.jpg)
> This table presents the performance improvement achieved by integrating Aligner with various upstream LLMs.  It quantifies the percentage increase across three dimensions (Helpfulness, Harmlessness, Honesty) for several different models, showcasing Aligner's model-agnostic nature and its ability to enhance performance without extensive retraining. The average performance gains are shown, highlighting Aligner's effectiveness across a range of models.

![](https://ai-paper-reviewer.com/kq166jACVP/tables_30_1.jpg)
> This table presents the performance improvements achieved by integrating Aligner with various upstream LLMs.  The improvements are measured across three dimensions (helpfulness, harmlessness, honesty) and various evaluation datasets. Notably, a single Aligner training session is sufficient for enhancing multiple models, highlighting its model-agnostic nature and efficiency.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/kq166jACVP/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/kq166jACVP/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}