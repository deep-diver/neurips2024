---
title: "Rule Based Rewards for Language Model Safety"
summary: "Rule-Based Rewards (RBRs) enhance LLM safety by using AI feedback and a few-shot prompt-based approach, achieving higher safety-behavior accuracy with less human annotation than existing methods."
categories: []
tags: ["AI Theory", "Safety", "🏢 OpenAI",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} QVtwpT5Dmg {{< /keyword >}}
{{< keyword icon="writer" >}} Tong Mu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=QVtwpT5Dmg" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95232" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=QVtwpT5Dmg&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/QVtwpT5Dmg/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Large Language Models (LLMs) often require costly human feedback for safety training, resulting in models that are either too cautious or exhibit undesirable styles.  Existing AI feedback methods lack the fine-grained control needed to efficiently enforce detailed safety policies.  This limits their real-world applicability.



This paper introduces Rule-Based Rewards (RBRs), a novel method that uses AI feedback and a set of rules to train LLMs. **RBRs leverage LLM graders for rule classification, directly incorporating the feedback into the reinforcement learning process.**  This approach achieves an impressive F1 score of 97.1, outperforming human feedback baselines.  The method’s efficiency and controllability make it particularly suitable for real-world applications and offer a new approach to LLM safety enhancement.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Rule-Based Rewards (RBRs) improve LLM safety by combining AI feedback with a small amount of human data. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} RBRs achieve comparable safety performance to human feedback baselines while significantly reducing instances of over-refusals. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} RBRs offer a flexible, scalable method for fine-grained control of model responses and can be applied to various reward models. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on language model safety and alignment because it introduces a novel, efficient method for enhancing safety behaviors without excessive human annotation.  **RBRs offer a scalable and flexible approach**, particularly relevant in the current trend of deploying LLMs in real-world applications where maintaining costly human-feedback-based safety guidelines is impractical. The paper opens new avenues for research on AI feedback methods and reward model design for safe and helpful AI.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_3_1.jpg)

> This figure demonstrates the effectiveness of combining Rule-Based Rewards (RBRs) with a helpful-only reward model in tuning a language model's safety-related behavior.  Panel (a) shows histograms comparing reward score distributions for a 'hard refusal' prompt when using only the helpful reward model versus when using the combined RBR and helpful reward model. The combined model shows clearer separation and better ranking of ideal, less good, and unacceptable completions. Panel (b) shows the error rate (frequency of a non-ideal response being ranked above an ideal response) for different reward models, indicating a significant reduction in error rates when the combined RBR and helpful model is used. This visually shows that RBR improves safety by correctly ranking responses while also reducing over- or under-cautious responses.





![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_4_1.jpg)

> This table lists a subset of the propositions used in the Rule-Based Reward (RBR) system described in the paper.  Propositions are binary statements about a model's response to a given prompt, such as whether the response contains an apology, is judgmental, or provides a complete answer.  These propositions are used to construct rules that define desirable or undesirable model behaviors for different types of user requests (e.g., safe, unsafe, etc.). The full list of propositions is available in Appendix Table 13.





### In-depth insights


#### RLHF Augmentation
RLHF augmentation strategies aim to enhance the safety and helpfulness of Reinforcement Learning from Human Feedback (RLHF) models.  **A core challenge in RLHF is the cost and time associated with collecting and labeling human preference data.**  Augmentation techniques address this by incorporating additional data sources or methods to supplement or replace human feedback, reducing reliance on expensive human annotation.  **One common approach is leveraging AI feedback, where an AI model is used to generate or rank potential model responses.** This can significantly increase the volume of training data, but introduces potential biases from the AI model itself. **Another approach involves carefully designing reward functions that incorporate explicit rules or constraints reflecting desired safety and helpfulness properties.**  This rule-based approach can offer more precise control over model behavior compared to solely relying on human preferences, but may require careful crafting of rules and consideration of potential edge cases.  The success of any augmentation method heavily depends on balancing the benefits of increased data volume and reduced human effort against the risk of introducing new biases or inaccuracies in the feedback signal.  **Ultimately, effective augmentation requires a thoughtful strategy that leverages diverse data sources and sophisticated feedback mechanisms to reliably improve RLHF models.**

#### RBR Framework
The Rule-Based Rewards (RBR) framework presents a novel approach to enhancing language model safety by leveraging AI feedback and a minimal amount of human data.  **Instead of relying heavily on human annotation, which can be expensive and prone to inconsistencies**, RBR uses a set of predefined rules defining desired and undesired model behaviors.  These rules are then evaluated by an LLM grader, providing a fine-grained and easily updatable reward signal for reinforcement learning.  **This method allows for precise control over the model's responses**, reducing issues like excessive caution or undesirable stylistic choices.  The framework's modular design, with composable rules and LLM-graded few-shot prompts, offers improved accuracy and easier updates compared to methods that rely solely on AI feedback or distill rules into a reward model. The efficacy of RBR is empirically demonstrated by achieving higher safety-behavior accuracy than human feedback baselines while significantly reducing over-refusals on safe prompts. **The use of AI feedback dramatically reduces the cost and time of data collection, particularly when dealing with complex safety-related guidelines.** This is achieved through synthetic data generation to fit the RBR weights, enhancing its efficiency and scalability.

#### Human Data Reduction
Reducing reliance on human annotation is a crucial aspect of making AI safer and more efficient.  The paper explores strategies to **minimize human data requirements** while maintaining high safety standards in language models.  This involves leveraging AI feedback mechanisms and focusing on a **rule-based reward system** that can be more precisely controlled and updated than traditional human-in-the-loop methods.  By breaking down safety guidelines into specific, manageable rules, the system allows for **automation of many labeling tasks** and facilitates the synthetic generation of training data, significantly lowering human effort. **The effectiveness of this approach is demonstrated** through achieving comparable safety performance with significantly less human feedback.  This **reduces costs**, enhances scalability, and allows for faster model adaptation to evolving safety guidelines.  The use of AI feedback, however, needs careful consideration due to the possibility of inheriting or amplifying existing biases from the AI model, highlighting a continuing need for human oversight and refinement.

#### Safety-Usefulness Tradeoff
The inherent tension between safety and usefulness in AI models, particularly large language models (LLMs), forms a critical 'Safety-Usefulness Tradeoff'.  **Improving safety often necessitates restricting the model's capabilities**, potentially leading to overly cautious responses that hinder its utility.  Conversely, prioritizing usefulness might increase the risk of generating unsafe or harmful outputs.  This tradeoff isn't binary; it's a spectrum where different approaches prioritize safety and usefulness to varying degrees.  The optimal balance depends on the specific application and acceptable risk levels.  **Techniques like reinforcement learning from human feedback (RLHF) attempt to navigate this tradeoff**, but face challenges in clearly conveying nuanced safety preferences to human annotators, resulting in inconsistent or overly conservative safety behaviors.  Therefore, research is actively exploring techniques that utilize AI feedback to improve safety specifications and reduce human annotation needs, enabling more fine-grained control and greater efficiency in achieving the desired balance between a safe and useful AI system.

#### Future Work
The authors mention exploring applications of their Rule-Based Reward (RBR) method to more challenging, non-safety-related tasks as **future work**. This suggests a desire to broaden the applicability of RBR beyond its current focus on safety-critical language models.  Another area of future research could involve investigating **different approaches for integrating AI feedback into RL training**. Currently, RBR combines AI-generated feedback directly with a reward model, but other methods might explore using this feedback to improve data efficiency or to enhance the interpretability and understandability of the reward model.  Finally, addressing the **ethical considerations of shifting safety feedback from humans to LLMs** is crucial. While RBR offers a path to more efficient safety training, understanding and mitigating potential biases in AI feedback is paramount to ensure that automated safety methods don't inadvertently perpetuate or amplify existing harms.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_4_1.jpg)

> This figure illustrates the process of combining Rule-Based Rewards (RBRs) with a helpful-only reward model (RM) during reinforcement learning (RL) training.  The RBR, a linear model of fitted weights and features, receives only safety-relevant prompts as input and adds a score to the helpful-only RM score for each completion.  Only safety-relevant prompts are sent to the RBR, while all prompts are fed into the RM. The combined scores from the RBR and RM are used as the total reward to update the policy model, which aims to produce higher reward completions.


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_6_1.jpg)

> This figure demonstrates the effectiveness of combining Rule-Based Rewards (RBRs) with a helpful-only reward model (RM) for improving the safety and helpfulness of language model responses.  Panel (a) shows histograms comparing reward score distributions for different model setups (helpful-only RM, RBR+RM, and RBR-only) on 'Hard Refuse' prompts.  The RBR+RM combination shows a clear separation and better ranking of completions compared to the helpful-only RM, indicating a more refined reward signal for safety. Panel (b) displays error rates (percentage of times a non-ideal completion is ranked above the ideal one) across different response types and model configurations. The RBR significantly reduces error rates, highlighting its role in enhancing the accuracy and precision of safety-related responses.


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_8_1.jpg)

> This figure shows the trade-off between usefulness (not over-refusing safe prompts) and safety (not generating unsafe content) for different models.  The x-axis represents safety (measured as the percentage of responses that do not contain disallowed content), and the y-axis represents usefulness (measured as the percentage of safe prompts that are not refused).  Each point represents a different model: Helpful-PPO, Human-PPO, RBR-PPO, RBR-SFT, and Human-SFT. The figure highlights the ability of the RBR approach to achieve a good balance between safety and usefulness, outperforming other models.


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_12_1.jpg)

> This figure illustrates the process of synthetic data generation for training the Rule-Based Reward (RBR) model.  It starts with a behavior policy defining desired model behaviors, which are broken down into individual binary propositions (e.g., 'apology', 'judgmental').  These propositions are used to create instructions for a large language model (LLM) to generate completions, labeled with the truthiness of each proposition.  These labeled completions are used to create two datasets: a 'Gold Set' used for tuning the LLM's classification prompts, and 'RBR Weight Fitting Data', used for training the RBR model itself.  The resulting RBR model can then be integrated into a reinforcement learning (RL) pipeline for fine-tuning LLMs.


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_15_1.jpg)

> This figure shows the effectiveness of combining Rule-Based Rewards (RBRs) with a helpfulness reward model in tuning the model's safety behavior.  Panel (a) compares reward score distributions for a helpfulness-only model versus one that incorporates RBRs.  The histograms show that the combined model better separates desired (ideal) and undesired (bad, disallowed) completions. Panel (b) shows that the combined model significantly reduces the error rate—the frequency with which a non-ideal completion is ranked higher than an ideal one.


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_16_1.jpg)

> This figure shows the trade-off between usefulness and safety for several different model training approaches.  The x-axis represents 'Safety (Not Unsafe)', indicating the percentage of responses that do not contain disallowed content.  The y-axis represents 'Usefulness (Not Overrefuse)', showing the percentage of responses that are not refusals when a helpful response was expected. Each point represents a model trained with a different method.  The results highlight the balance between safety and usefulness that needs to be considered when training language models. RBR-PPO represents the model trained with the Rule Based Rewards method, which aims to optimize this balance.  Human-PPO shows the model trained solely on human feedback and Helpful-PPO with only helpful data.  RBR-Fixed variants show the outcomes of using fixed values in weights for the RBR.


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_17_1.jpg)

> This figure shows the effectiveness of combining Rule-Based Rewards (RBRs) with a helpful reward model in tuning a language model's safety behavior.  The left panel (a) compares reward score distributions for a helpful-only reward model versus one incorporating RBRs, demonstrating improved separation between ideal, less good, and unacceptable completions for refusals. The right panel (b) shows error rates – the frequency of non-ideal completions outranking ideal ones – significantly reduced by using the combined reward.


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_17_2.jpg)

> This figure shows the effectiveness of combining Rule-Based Rewards (RBRs) with a helpfulness reward model in tuning a language model's safety behavior.  Panel (a) compares reward score distributions for a helpfulness-only model versus the combined model. The combined model shows better separation between ideal, slightly bad, and very bad responses.  Panel (b) demonstrates that the combined model significantly reduces the error rate (instances where a non-ideal response is ranked higher than an ideal one) compared to the helpfulness-only model.


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/figures_17_3.jpg)

> This figure shows the effectiveness of combining rule-based rewards (RBRs) with a helpfulness reward model in tuning a language model's safety behavior.  Subfigure (a) compares reward score distributions for a helpful-only model versus one incorporating RBRs, demonstrating improved separation between ideal, slightly bad, and very bad responses for safety-critical prompts. Subfigure (b) quantifies this improvement by showing a lower error rate (fewer instances of non-ideal responses ranked higher than ideal responses) when using the combined reward model.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_5_1.jpg)
> This table summarizes the three datasets used in training the Rule-Based Rewards (RBR) model.  It specifies whether each dataset is human-labeled or automatically generated, the size of the dataset, and a brief description of its contents. The Ps dataset contains safety-relevant prompts, the Gold dataset is a small set of human-labeled data for tuning the classification prompts for the RBR model, and the DRBR dataset is synthetically generated data for fitting the RBR weights.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_8_1.jpg)
> This table presents the results of safety evaluations performed using both internal automated metrics and human evaluations.  It compares three different model training approaches: Helpful-PPO (a baseline using only helpful data), Human-PPO (a baseline incorporating human-labeled safety data), and RBR-PPO (the proposed method using rule-based rewards). The metrics evaluated include Not-Unsafe (percentage of completions without unsafe content), Not-Overrefuse (percentage of safe prompts not refused), and an F1-score combining both safety and usefulness metrics. The results show that RBR-PPO achieves a good balance between safety and usefulness compared to the baselines.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_8_2.jpg)
> This table presents the results of safety and capability evaluations on three different models: Helpful-only, Human-feedback, and RBR-trained models.  The safety evaluations are conducted on two datasets, XSTest and WildChat, measuring over-refusal and unsafe content respectively, using both automated and manual metrics (Not-Overrefuse and Not-Unsafe). Capability is assessed on four standard benchmarks: MMLU, Lambada, HellaSwag, and GPQA. The results highlight the trade-off between safety and capability, showcasing the RBR model's ability to balance safety and functionality.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_12_1.jpg)
> This table shows the propositions used in the Rule-Based Reward (RBR) system for each completion type (Hard Refusal, Soft Refusal, Comply) and class (Ideal, Minimum Acceptable Style, Unacceptable Completion, Illogical Completion, Disallowed Completion).  For each proposition, the table indicates whether it's desired (+) or undesired for each class.  It also notes the total number of propositions and features used in the weight fitting process.  The table helps to understand the fine-grained control exerted by the RBR system over different aspects of the model's response. 

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_14_1.jpg)
> This table presents the breakdown of the number of prompts used for training and testing in different response types (Comply, Hard Refuse, and Soft Refuse).  It shows the counts from the human baseline, the RBR training data (which uses automatically generated labels), and the agreement rate between human and automatically generated labels.  Finally, it shows how many prompts in each type were used for creating the Gold set for prompt tuning.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_18_1.jpg)
> This table shows example response types that are expected based on the content and behavior policies defined in the paper.  It breaks down several content areas (Erotic, Criminal Advice, Hate Speech, Self-Harm) and shows, for each, different response types (Comply, Hard Refuse, Soft Refuse) with example responses.  The table illustrates how the model's response should vary depending on the type of user request and the predefined safety guidelines.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_18_2.jpg)
> This table presents the detailed results for the key metrics (Refusal-Style, Not-Overrefuse, Not-Unsafe, F1-Score) from several model training approaches.  It compares the performance of different methods, including baselines (Helpful-SFT, Human-SFT, Old Data-SFT), standard RLHF (Helpful-PPO, Human-PPO, Old Data-PPO), and the proposed RBR method (RBR-PPO, HumanRM+RBR PPO, Human-matchRBR-PPO, Old Data+RBR-PPO). It also includes ablation study results (RBR-Fixed1-PPO, RBR-Fixed10-PPO, SFTOnly-noRBR-PPO, RBR-noRM-PPO, RBR-noSFT-PPO) to analyze the impact of different design choices.  The results shown are averages across multiple checkpoints with standard errors, offering a comprehensive evaluation of various aspects of model safety and helpfulness.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_19_1.jpg)
> This table details the experimental settings used in the paper.  It specifies the model size (Large, Medium, Small, XSmall), the type of SFT data used (Helpful, Helpful, Human, Synthetic, Old Safety, Limited Human), the reward model used (Helpful, Human, RBR, Old Safety), the PPO prompts used (Helpful, Safety), and any additional notes on the experiment (Baseline, Human Data Baseline, RBRs, Outdated safety data, Matches RBR data size, No RBR used, No safety SFT data, No RM score for safety prompts, Safety prompts are fixed, amount may vary). The table is divided into two parts: main experiments and ablation studies. The ablation studies examine the effects of changing various aspects of the training process, such as the type and amount of data used, the reward model, and the inclusion or exclusion of safety RBRs.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_20_1.jpg)
> This table breaks down the number of prompts per response type (Comply, Hard Refuse, Soft Refuse) used in the training and testing sets for both the PPO (reinforcement learning) and RBR (rule-based reward) models.  It shows the number of prompts with human-provided labels, automatically generated labels, and the agreement rate between the two labeling methods. The table also indicates the number of prompts from the human-labeled Gold set, used for tuning the classification prompts for RBRs.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_20_2.jpg)
> This table provides examples of model responses to different prompt types (Comply, Hard Refusal, Soft Refusal) from three different model setups: Helpful-PPO Baseline, Human-PPO Baseline, and RBR-PPO.  It illustrates how the models respond to requests that should be complied with, those that require a strong refusal due to safety concerns, and those that should receive an empathetic but firm refusal. The 'Ideal' column indicates whether the response is considered the ideal response according to the paper's criteria.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_20_3.jpg)
> This table presents the quantitative results of several experiments comparing different model training methods.  It shows the performance metrics (Refusal-Style, Not-Overrefuse, Not-Unsafe, and F1-Score) for each model, highlighting the differences in safety and helpfulness across various approaches (e.g., Helpful-Only, Human, and RBR-based models).  The F1-score is particularly important as it balances safety and usefulness.  The table also includes results for ablation studies investigating the impact of varying training parameters and data sources.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_20_4.jpg)
> This table presents the raw numerical results from several experiments comparing different model configurations.  It includes standard errors for each metric, allowing for a better understanding of the variability in the results. The models being compared include baselines using only helpful data, models trained with human safety data, and models utilizing the Rule-Based Rewards (RBR) method. The metrics evaluated are related to safety (e.g., avoiding unsafe content, not over-refusing), refusal style, and the F1 score which combines safety and usefulness.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_21_1.jpg)
> This table lists the propositions used in the Rule-Based Rewards (RBR) system for classifying different completion types (Hard Refusal, Soft Refusal, Comply) into classes based on the presence or absence of specific features.  Each proposition represents a characteristic of the model's response (e.g., contains an apology, uses threatening language, provides resources). The table shows whether each proposition is considered 'desired' or 'undesired' for each class.  This helps define the desired behavior of the model for various situations.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_22_1.jpg)
> This table shows the propositions used in the Rule-Based Reward (RBR) system, categorized by their desirability for each of the three completion types (Hard Refusal, Soft Refusal, Comply).  It indicates whether each proposition is considered acceptable, undesirable, required, or a feature used for weight fitting.  The table also provides the total number of propositions used for each completion type and the total number of features used in the RBR weight fitting process.

![](https://ai-paper-reviewer.com/QVtwpT5Dmg/tables_22_2.jpg)
> This table presents the accuracy of proposition evaluation for different model sizes (XSmall, Small, Medium, Large).  Each row represents a proposition (e.g., 'Apology,' 'Disallowed Content'), and the columns show the accuracy of that proposition's classification for each model size.  The accuracy is expressed as a percentage with a standard error. The table illustrates how the accuracy of identifying various aspects of model responses improves as the model size increases.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QVtwpT5Dmg/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}