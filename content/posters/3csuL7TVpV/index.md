---
title: "Decoding-Time Language Model Alignment with Multiple Objectives"
summary: "Multi-objective decoding (MOD) efficiently aligns language models to diverse user needs by decoding the next token from a weighted combination of predictions from multiple base models trained on indiv..."
categories: []
tags: ["Natural Language Processing", "Large Language Models", "🏢 Tsinghua University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 3csuL7TVpV {{< /keyword >}}
{{< keyword icon="writer" >}} Ruizhe Shi et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=3csuL7TVpV" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/96731" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=3csuL7TVpV&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/3csuL7TVpV/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many current methods for aligning language models (LMs) to human preferences focus on optimizing a single reward function, limiting their adaptability.  This is problematic because various alignment objectives (like helpfulness, safety, and verbosity) often need to be balanced depending on the context.  Existing solutions, such as creating mixed datasets or retraining for each weighting, are inefficient and resource-intensive.

This paper introduces Multi-objective decoding (MOD), an efficient algorithm for decoding the next token as a weighted combination of predictions from multiple base LMs, each trained on different single objectives.  **MOD is training-free**, meaning no retraining is required for new weighting combinations, enabling quicker experimentation.  The paper theoretically demonstrates MOD's optimality under certain conditions and empirically shows significant reward improvements compared to existing methods. 

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} MOD is a training-free, efficient algorithm for multi-objective language model alignment. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} MOD achieves significant performance gains over existing methods, demonstrating superior adaptability to varied objectives. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Theoretical analysis reveals the sub-optimality of traditional parameter-merging approaches and highlights the benefits of a strong-barrier function in multi-objective alignment. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **language model alignment** because it introduces a novel, **training-free method** for handling multiple objectives.  It offers a significant improvement over existing methods, opening avenues for more flexible and efficient LM optimization.  The theoretical analysis further strengthens its impact, providing a solid foundation for future advancements in the field.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/3csuL7TVpV/figures_1_1.jpg)

> This figure illustrates the core idea of the MOD algorithm proposed in the paper.  It shows how multiple language models (LMs), each trained for a specific objective (e.g., safety, helpfulness), can be combined at decoding time to generate text that balances those objectives.  The combination weights (w) control the relative importance of each objective.  The input prompt (x) and context (y<t) inform the LM's predictions, and the next token (Yt) is selected from a weighted average of the different LMs' probability distributions.





![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_1_1.jpg)

> The table compares different multi-objective language model alignment approaches. It shows the number of language models required for training, whether each approach requires reward models or prompting during inference, and other requirements. MORLHF and MODPO require training many LLMs and use reward models. DPA, CPO, and RiC only use one LLM but require prompting and reward models. RS requires the same model architecture and initialization. MOD requires the same tokenizer but does not require reward models or prompting.  MOD is the most versatile approach.





### In-depth insights


#### Multi-objective decoding
The concept of "Multi-objective decoding" presents a novel approach to language model alignment by addressing the limitations of single-objective methods.  It proposes a decoding-time algorithm that combines predictions from multiple base models, each trained for a different objective (e.g., safety, helpfulness, fluency). **This allows for flexible control over the model's output by adjusting the weights assigned to each objective during inference**, eliminating the need for extensive retraining with altered datasets for every preference combination. The method leverages a closed-form solution derived from f-divergence regularized alignment approaches, ensuring efficiency and theoretical grounding.  **A key advantage is its training-free nature**, allowing for rapid experimentation with various objective weightings and combinations of pre-trained models.  Furthermore, the authors provide theoretical analysis demonstrating the sub-optimality of traditional parameter-merging baselines and highlighting the importance of strong-barrier function regularization for optimal multi-objective alignment.  **This framework's versatility and efficiency make it a promising direction for achieving adaptable and nuanced language model behavior**.

#### Legendre transform
The concept of **Legendre transform** is crucial in this research paper because it bridges the gap between reward maximization and regularization, which are often conflicting objectives in aligning language models. By leveraging the Legendre transform, the authors elegantly derive a closed-form solution for decoding, enabling efficient and precise control over multi-objective alignment without retraining.  This is particularly insightful because it connects a family of f-divergence regularized approaches under a unified mathematical framework. The transform allows the derivation of an optimal policy for a combined reward function, addressing the sub-optimality of traditional parameter-merging methods.  Crucially, this **training-free** approach offers scalability and flexibility, allowing on-the-fly adaptation to varied user preferences and applications. The use of the Legendre transform highlights the mathematical elegance and rigor behind the proposed method, solidifying its theoretical foundation.

#### Optimality guarantees
The concept of 'optimality guarantees' in the context of multi-objective decoding for language model alignment is crucial.  It signifies the extent to which an algorithm can provably reach the best possible outcome given multiple, potentially conflicting, objectives.  **Strong optimality guarantees are highly desirable** because they offer a level of certainty absent in heuristic or empirically driven approaches.  The paper likely establishes optimality based on a specific mathematical framework, possibly leveraging techniques from convex optimization or information theory. This likely involves demonstrating that the proposed algorithm converges to an optimal solution under the specified conditions, perhaps by showing that a closed-form solution exists, enabling a direct calculation of the optimal policy without iterative methods.  **The assumptions underlying these guarantees** are also important to consider;  they define the boundaries within which the optimality claim holds true.  These might include assumptions about the form of the objective functions, the behavior of the base language models, or the nature of the reward signals.  A discussion of the limitations and the practical implications of these assumptions is a vital aspect of a robust analysis.

#### Empirical evaluation
A robust empirical evaluation section is crucial for validating the claims of a research paper.  It should present a comprehensive assessment of the proposed method's performance, comparing it against relevant baselines and state-of-the-art approaches.  **Clear descriptions of datasets, evaluation metrics, and experimental setup** are necessary for reproducibility.  The results should be presented in a clear and understandable manner, possibly using tables and figures to highlight key findings.  **Statistical significance testing** is essential to determine whether the observed differences are meaningful or merely due to chance.  The discussion should interpret the results, analyzing any unexpected findings and exploring the limitations of the approach.  **A thorough analysis of the strengths and weaknesses** relative to other methods provides valuable insights into the overall contribution of the research.  Moreover, considering various factors affecting the performance, such as computational cost and scalability, enhances the completeness and impact of the evaluation.

#### Future directions
Future research could explore expanding multi-objective decoding (MOD) to handle **more diverse objective functions** beyond reward-based ones.  Investigating the theoretical properties of MOD with various f-divergences, and addressing its limitations, such as the **computational cost of managing multiple models**, are also important.  **Improving the efficiency** of the algorithm itself, perhaps through better approximations or alternative optimization methods, would enhance its practical applicability.  Furthermore, a thorough investigation into the **impact of sub-optimal base policies** on the overall performance of MOD is warranted.  Finally, research should focus on applying MOD to various real-world applications and evaluating its effectiveness in diverse contexts, including its ability to handle **conflicting objectives** and achieve desirable trade-offs.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/3csuL7TVpV/figures_5_1.jpg)

> This figure shows Pareto frontiers for the Reddit Summary task, comparing three different methods: MOD (Multi-objective decoding), RS (Rewarded Soups), and MORLHF (Multi-objective reinforcement learning from human feedback).  The x-axis represents the reward for one objective (summary quality), and the y-axis represents the reward for another objective (faithfulness).  The Pareto frontier shows the trade-off between these two objectives; points on the frontier represent the best achievable reward for one objective given a particular reward for the other objective.  The figure demonstrates that the MOD algorithm generally outperforms both the RS and MORLHF methods, achieving higher rewards across a range of preference weightings.


![](https://ai-paper-reviewer.com/3csuL7TVpV/figures_5_2.jpg)

> This figure shows the Pareto frontiers for the Helpful Assistant task, comparing three different methods: MOD, RS (rewarded soups), and MORLHF (multi-objective RLHF).  The x-axis represents one reward (e.g., helpfulness), and the y-axis represents a second reward (e.g., harmlessness). Each point on a curve represents a specific weighting between the two rewards, and the curve itself shows the trade-off between them achieved by the corresponding method.  MOD consistently outperforms RS, indicating its better ability to balance multiple objectives simultaneously.  However, when attempting to balance harmlessness and humor (a challenging combination), MORLHF (which uses more compute resources) achieves better results than MOD.


![](https://ai-paper-reviewer.com/3csuL7TVpV/figures_6_1.jpg)

> This figure compares the performance of Multi-objective decoding (MOD) and Rewarded soups (RS) on two sets of models. The first set consists of two models, M1 and M2, trained using a standard approach. The second set uses the same M1, but M2 is created by reversing the sign of the Q and K matrices in the last two layers of M1.  The plots show Pareto frontiers, visualizing the trade-offs between two reward objectives (e.g., relevance and factuality) for various preference weightings. The comparison highlights the robustness of MOD compared to RS, especially when the base models exhibit asymmetry.


![](https://ai-paper-reviewer.com/3csuL7TVpV/figures_16_1.jpg)

> This figure illustrates the core idea of the Multi-objective decoding (MOD) method.  It shows how, given a prompt and context, the next token is selected by combining the predictions of multiple language models (LMs), each trained to optimize for a different objective.  The weighting (w) applied to each LM's prediction determines the relative importance of each objective in the final output. This allows for fine-grained control over the trade-offs between multiple desirable properties of the generated text, like helpfulness and safety,  without requiring retraining of the LMs for each combination of objectives.


![](https://ai-paper-reviewer.com/3csuL7TVpV/figures_28_1.jpg)

> This figure shows the Pareto frontier for the Reddit Summary task, comparing MOD, RS (Rewarded Soups), and MORLHF (Multi-Objective Reinforcement Learning from Human Feedback). The Pareto frontier plots the trade-off between two objectives: summary quality (R1) and faithfulness (R2).  The plot demonstrates that MOD generally outperforms both RS and MORLHF, indicating its superior ability to achieve a better balance between the two objectives.


![](https://ai-paper-reviewer.com/3csuL7TVpV/figures_29_1.jpg)

> This figure compares the performance of three different methods for language model alignment on the Helpful Assistant task, where the goal is to balance helpfulness, harmlessness, and humor. The x-axis represents the score for harmlessness, while the y-axis represents the score for helpfulness. Each point on the plot represents a different weighting of the three objectives (using the MOD algorithm). The Pareto frontier obtained with MOD is above the Pareto frontier of RS in most regions, indicating the superiority of MOD in balancing the three reward objectives. However, the Pareto frontier for MORLHF is better than MOD in the region where the weighting of harmlessness and humor is high, reflecting the costlier nature of the MORLHF algorithm.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_6_1.jpg)
> This table presents the results of the Safety Alignment experiment using the Multi-objective decoding (MOD) method.  The experiment explores the effect of varying the preference weightings (w1, w2) on the model's output. Specifically, it shows how the model's helpfulness and harmlessness scores change as the weight assigned to harmlessness (w2) decreases. The table highlights the trade-off between helpfulness and harmlessness, demonstrating that as the emphasis on harmlessness diminishes, the model tends to generate more harmful outputs, thus sacrificing harmlessness for helpfulness. This showcases the fine-grained control over the balance between objectives that MOD provides.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_6_2.jpg)
> This table compares MOD with other multi-objective alignment approaches. It shows that MOD is the only approach that does not require reward models or preference-driven prompts, making it more versatile and less resource-intensive. The table also highlights the number of LLMs trained and the requirements of other methods. This comparison showcases MOD's efficiency and flexibility over existing approaches.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_6_3.jpg)
> This table presents the results of using the Multi-Objective Decoding (MOD) method to combine three different language models (TÜLU models) with different strengths in safety, coding, and reasoning.  The table shows that by carefully selecting the weights assigned to each model's output (preference weightings), MOD can achieve superior performance compared to using a single model. The best combination significantly reduces toxicity (Toxigen) while improving performance on other metrics (Codex@1, GSM-COT, BBH-COT).

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_17_1.jpg)
> This table compares the proposed Multi-objective decoding (MOD) method with other existing multi-objective alignment approaches.  It highlights key differences across several aspects such as the number of language models required for training, whether reward models are needed, whether prompting is required during inference, and the type of objectives used (same architecture and initialization or different ones).  The comparison reveals that MOD offers a more versatile solution compared to other methods, especially in its ability to handle diverse objectives without the need for reward models or specific prompts.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_17_2.jpg)
> This table compares different multi-objective alignment approaches, highlighting key differences in the number of language models needed, the use of reward models and prompting. It shows that MOD is unique in being free from reward models and prompting and is able to handle multiple objectives.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_17_3.jpg)
> This table compares MOD with other multi-objective alignment approaches.  It highlights whether each method requires reward models (RM), preference-driven prompts during inference, and the number of language models (LLMs) needed for training.  MOD stands out as versatile, avoiding the need for reward models and prompts, and adapting to any number of objectives.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_29_1.jpg)
> This table presents a comparison of the performance of MOD and RS algorithms on the 3-objective Helpful Assistant task.  The w-weighted score is a composite metric combining helpfulness, harmlessness, and humor, weighted by the user-specified preferences (w1, w2, w3).  The results demonstrate that MOD consistently outperforms RS across various weighting configurations, highlighting the effectiveness of MOD in balancing multiple objectives.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_30_1.jpg)
> This table compares different multi-objective language model alignment approaches.  It contrasts them based on the number of language models needed, whether they require reward models, and whether they require preference-driven prompts during inference.  It highlights that the proposed MOD approach offers the most versatile solution.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_31_1.jpg)
> This table presents the results of the HelpSteer experiment using f-DPO with Reverse KL-divergence.  The preference weightings were set to (0.33, 0.33, 0.33), meaning an equal weighting across the three objectives (Helpfulness, Correctness, Coherence, Complexity, Verbosity). The table compares the performance of MOD against RS and three individual base models (π1f, π2f, π3f). The top two scores for each metric are highlighted to easily compare the effectiveness of MOD against other methods.  The average score provides a holistic performance evaluation across all objectives.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_31_2.jpg)
> This table presents the results of the HelpSteer experiment using f-DPO with JSD (Jensen-Shannon Divergence).  It shows the performance of the MOD (Multi-objective decoding) algorithm compared to the RS (Rewarded Soups) baseline and individual models (π1f, π2f, π3f) across five metrics: Helpfulness, Correctness, Coherence, Complexity, and Verbosity.  The preference weightings are set to w = (0.33, 0.33, 0.33), meaning each objective is weighted equally. The table highlights the top-2 scores for each metric, allowing for easy comparison of the different approaches.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_32_1.jpg)
> This table presents the results of the HelpSteer experiment using f-DPO with 0.3-divergence as the divergence measure.  The preference weightings are set to w = (0.33, 0.33, 0.33), indicating an equal weighting across three objectives (Helpfulness, Correctness, Coherence, Complexity, Verbosity). The table compares the performance of MOD (Multi-objective Decoding) with the parameter merging baseline (RS) and individual models (π1f, π2f, π3f) across five metrics.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_32_2.jpg)
> This table presents the results of experiments conducted on the HelpSteer dataset using f-DPO models with 0.5-divergence.  It compares the performance of MOD (Multi-objective Decoding) against a parameter-merging baseline (RS) and individual models (π1f, π2f, π3f) across five metrics: Helpfulness, Correctness, Coherence, Complexity, and Verbosity. The preference weightings were set to w = (0.33, 0.33, 0.33), representing an equal weighting across all three objectives.  The table showcases the average scores across these metrics for each method, highlighting the relative strengths and weaknesses of each approach in balancing multiple objectives.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_32_3.jpg)
> This table presents the results of applying the Multi-objective decoding (MOD) method to combine three large language models (LLMs) with different strengths: CODETÜLU-2-7B (coding), TÜLU-2-HH-13B (helpful and harmless), and TÜLU-2-ULTRA-13B (general capabilities).  The table shows the performance across four metrics: BBH COT, GSM COT, Toxigen (toxicity), and Codex@1 (coding).  Different weightings (w) of the three models are tested, demonstrating how MOD allows for fine-grained control over the LLM's behavior by adjusting the weighting of different objectives. The results show significant improvements over the baseline CODETÜLU-2-7B, especially in reducing toxicity while maintaining or improving performance in other areas.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_33_1.jpg)
> This table compares MOD with other multi-objective language model alignment approaches.  It highlights key differences across several criteria, including whether the methods require reward models (RM), preference-driven prompts during inference, and the number of language models or preferences needed. The table shows that MOD is unique in its versatility, being free from both reward models and the need for prompting, while also accommodating multiple objectives.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_33_2.jpg)
> This table compares MOD with other multi-objective alignment approaches.  It highlights key differences in the number of language models required for training, whether reward models or prompting are needed during inference, and any architectural requirements.  It shows that MOD offers a more versatile solution by requiring only a set of models trained for individual objectives, without needing reward models or special prompts at inference time.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_34_1.jpg)
> The table compares different multi-objective language model alignment approaches based on three criteria: the number of language models that need to be trained, whether a reward model is required, and whether preference-driven prompts are required during inference.  It highlights that the proposed Multi-objective decoding (MOD) method is the most versatile, requiring only the training of language models for each objective and not needing reward models or preference-driven prompts.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_34_2.jpg)
> This table compares MOD with other multi-objective alignment approaches, highlighting key differences across several aspects.  It shows that MOD is unique in its ability to freely combine models trained for different objectives without requiring any reward model or prompting during inference, offering flexibility unmatched by other approaches.  The table contrasts the number of language models required for training, the use of reward models, and the need for prompting.  MOD is the most versatile solution as it requires the same architecture, initialization, and tokenizer across models, making it more easily applicable across various scenarios and objectives.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_34_3.jpg)
> This table compares MOD with other multi-objective alignment approaches.  It shows the number of language models needed for training, whether a reward model or prompting is required, and the type of requirements.  MOD is highlighted as the most versatile solution, requiring only models trained for individual objectives and not requiring reward models or prompting during inference. 

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_35_1.jpg)
> This table compares MOD with other multi-objective language model alignment approaches.  It shows the number of language models that need to be trained for each method, whether they require reward models (RM), and whether they require preference-driven prompts during inference. The table highlights that MOD is the most versatile solution because it does not require reward models or prompts, while also allowing for an arbitrary number of objectives and preferences.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_35_2.jpg)
> This table compares the proposed MOD algorithm with other existing multi-objective alignment approaches. It highlights key differences across several criteria, such as the number of language models required for training, the need for reward models, the dependence on prompting during inference, and the types of alignment objectives supported.  The table demonstrates MOD's versatility in addressing these challenges compared to other methods.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_36_1.jpg)
> This table compares the proposed Multi-objective decoding (MOD) approach with existing multi-objective alignment approaches across several key aspects.  It highlights the advantages of MOD in terms of the number of language models it requires for training, whether it needs reward models, and whether preference-driven prompts are required during inference.  MOD is shown to be the most versatile solution.  The table also lists the architectural requirements of each method, noting that MOD requires the base models to share the same architecture and initialisation, as well as having a common tokenizer.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_36_2.jpg)
> This table compares different multi-objective language model alignment approaches.  It contrasts the number of language models required for training, the need for reward models (RM), the requirement for preference-driven prompts during inference, and the overall requirements of each algorithm. The table highlights that the proposed method (MOD) offers a more versatile solution compared to existing approaches.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_37_1.jpg)
> This table compares the proposed Multi-objective decoding (MOD) approach with other existing multi-objective language model alignment methods. It highlights key differences across several aspects, including the number of language models required for training, the need for reward models (RM), and the reliance on prompting during inference.  The table helps readers understand the unique advantages and versatility of the MOD approach compared to other techniques.

![](https://ai-paper-reviewer.com/3csuL7TVpV/tables_38_1.jpg)
> This table compares different multi-objective language model alignment approaches, highlighting key differences in the number of trained language models, reliance on reward models, and use of prompting techniques during inference.  It emphasizes that the proposed method, MOD, offers a more versatile and flexible solution by not requiring reward models or prompting, and by allowing for precise control over the weighting of multiple objectives.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3csuL7TVpV/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}