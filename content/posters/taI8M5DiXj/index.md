---
title: "When to Act and When to Ask: Policy Learning With Deferral Under Hidden Confounding"
summary: "CARED: a novel causal action recommendation model improves policy learning by collaborating with human experts and mitigating hidden confounding in observational data."
categories: []
tags: ["AI Theory", "Causality", "🏢 Faculty of Data and Decision Sciences, Technion",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} taI8M5DiXj {{< /keyword >}}
{{< keyword icon="writer" >}} Marah Ghoummaid et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=taI8M5DiXj" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93323" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=taI8M5DiXj&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/taI8M5DiXj/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many real-world applications, such as healthcare, require AI systems to make decisions in collaboration with human experts.  However, observational data used to train these systems often suffers from hidden confounding, leading to biased and ineffective policies.  This paper tackles this challenge by introducing deferral to the expert's judgment when the AI system is uncertain. 

The paper introduces CARED (Causal Action Recommendation with Expert Deferral), a novel policy learning method that leverages both AI model and human expert.  CARED uses a cost-sensitive learning approach and estimates bounds on counterfactual outcomes to guide decisions on when to defer to the expert, resulting in superior policies compared to existing methods. The authors validate their method through experiments on synthetic and semi-synthetic datasets, showcasing CARED's robustness and effectiveness in scenarios with hidden confounders.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} CARED, a new framework for causal action recommendation, effectively combines machine learning and human expertise. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method uses a cost-sensitive learning approach to handle hidden confounding and the option of deferring to human experts, resulting in improved policies. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Experiments on synthetic and semi-synthetic data demonstrate that CARED outperforms existing methods, providing more robust and reliable policy learning. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working with observational data and human-in-the-loop systems.  It **directly addresses the challenges of hidden confounding** and provides a novel method to improve policy learning in high-stakes domains like healthcare, enabling better collaboration between AI systems and human experts. This opens up new avenues for research in **robust causal inference and human-AI teamwork**, leading to safer and more effective AI deployments. 

------
#### Visual Insights



![](https://ai-paper-reviewer.com/taI8M5DiXj/figures_8_1.jpg)

> The figure shows the result of a synthetic data experiment comparing the proposed CARED policy to several baselines under varying levels of hidden confounding.  The x-axis represents the amount of hidden confounding, and the y-axis represents the policy regret (lower is better). CARED consistently outperforms all other methods, especially for a wide range of confounding levels. The plot illustrates the robustness of the proposed method compared to baselines that are sensitive to accurate specification of the confounding level.





![](https://ai-paper-reviewer.com/taI8M5DiXj/tables_5_1.jpg)

> This table lists the hyperparameters used for the XGBRegressor and Logistic Regression models in the synthetic data experiment.  The hyperparameters were selected to ensure effective model training and performance in the context of the study's causal inference task.  The values are specific to this particular experiment and should not be considered universally optimal.





### In-depth insights


#### Hidden Confounding
Hidden confounding, a significant challenge in causal inference, is thoroughly addressed.  The paper acknowledges its presence in observational data, **highlighting the risk of biased and harmful policies** if left unaddressed.  The authors cleverly leverage the presence of a human expert, whose decisions implicitly incorporate information unavailable to the model.  **This expert input mitigates some of the risks associated with hidden confounding**, allowing for the development of more robust and reliable causal policies.  While the paper assumes a bounded degree of hidden confounding, **it proposes a set of costs based on estimated counterfactual outcomes to guide model learning**, rather than relying on unavailable true labels. This approach enables the learning of better policies by leveraging the strengths of both the machine learning model and the human expert, thereby offering a robust solution to causal policy learning in the presence of hidden confounding.

#### Causal Action Models
Causal action models represent a significant advancement in AI, aiming to bridge the gap between observational data and effective interventions. Unlike traditional predictive models, causal action models explicitly consider the causal relationships between actions and outcomes, enabling a deeper understanding of the mechanisms underlying observed effects. **This allows for more robust and reliable decision-making, especially in high-stakes scenarios where the consequences of actions are significant.**  A key challenge lies in disentangling correlation from causation in observational data, often requiring assumptions such as ignorability or the use of instrumental variables.  **Furthermore, the presence of hidden confounding factors, which influence both actions and outcomes without being directly observed, poses a major hurdle.** This necessitates sophisticated causal inference techniques, including propensity score matching, inverse probability weighting, or more advanced methods like doubly robust estimators, to mitigate bias and produce accurate causal estimates.  **Successfully developing causal action models often requires careful model selection, incorporating domain expertise, and rigorous validation.** The ultimate goal is to use causal insights to design interventions that achieve desired outcomes while minimizing negative side effects, promising valuable applications in diverse fields like healthcare, policy, and economics.

#### Expert Deferral
Expert deferral, a core concept in human-AI collaboration, is explored in this research paper.  The paper investigates scenarios where an algorithm can choose to **defer to a human expert** instead of making its own decision. This approach is particularly useful in high-stakes domains like healthcare, where the algorithm's confidence in its prediction may be low or where the human expert's knowledge might outweigh the algorithm's capabilities. The paper examines the strategic benefits of choosing to defer, acknowledging the limitations of machine learning in situations with **hidden confounding** and potentially biased data.  The decision to defer is not random; it's **modelled as a learned part of the system's policy**, making it adaptive and context-aware.  The core of the method is a principled approach to **balance the strengths of both the algorithm and the expert**, yielding improved performance compared to solely relying on either. The analysis critically examines this balancing act, demonstrating how such systems can learn to leverage human expertise effectively, and how **carefully designed costs** can guide the system to make optimal deferral decisions.

#### CAPO-Based Policies
The section 'CAPO-Based Policies' likely details baseline policies for treatment decisions using Conditional Average Potential Outcome (CAPO) estimates.  These policies leverage the upper and lower bounds of CAPO, representing uncertainty in treatment effect estimations.  **A key decision point is whether to treat or defer to a human expert.**  The description probably contrasts two approaches: a 'Bounds Policy' which only treats if the CAPO bounds clearly indicate a beneficial treatment effect and a 'Pessimistic Policy' which never defers, always choosing a treatment based on CAPO estimates even when uncertainty remains.  **This comparison highlights the trade-off between utilizing machine-learned insights and relying on human expertise.** The policies are likely evaluated against an oracle policy and the human expert's policy for a robust assessment of performance, especially under hidden confounding.

#### Future Work
The paper's 'Future Work' section hints at several promising research directions. **Extending the model to handle multiple human experts** would significantly enhance its real-world applicability, allowing for more nuanced collaboration and potentially improved decision-making.  Investigating the impact of different cost functions on model performance is crucial, particularly in the context of hidden confounding.  **A deeper exploration of the sensitivity parameter (Λ)**, including methods for its robust estimation and the effects of misspecification on results, would strengthen the theoretical foundations.  Further research should focus on **evaluating the method's performance on larger and more diverse datasets**, moving beyond the synthetic and semi-synthetic data used in this study. Finally, developing strategies to mitigate the challenges posed by non-stationary expert behavior over time is essential for creating truly robust and reliable human-AI collaborative systems.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/taI8M5DiXj/figures_9_1.jpg)

> This figure displays the results of a synthetic data experiment comparing different policy learning methods under varying levels of hidden confounding (A).  Lower policy regret indicates better performance. The x-axis represents the MSM parameter, and the true value (A0) is marked.  Several baselines are included: the human expert's policy, an IPW approach without deferral (CRLogit), an IPW approach with deferral (ConfHAI), and policies derived from CAPO bounds. The oracle policy (optimal) provides a benchmark.  CARED shows superior performance across various A values.


![](https://ai-paper-reviewer.com/taI8M5DiXj/figures_9_2.jpg)

> This figure displays the results of the IHDP Hidden Confounding experiment.  Figure 2a shows how the policy value of different methods (CARED, ConfHAI, CRLogit, etc.) changes with different levels of hidden confounding (parameter A). Figure 2b illustrates the relationship between policy value and deferral rate across multiple methods, demonstrating the trade-offs between machine learning and expert input.


![](https://ai-paper-reviewer.com/taI8M5DiXj/figures_13_1.jpg)

> This figure illustrates two scenarios in the context of CAPO (Conditional Average Potential Outcome) intervals.  The top panel shows an example where the CAPO intervals for treatments 0 and 1 do not overlap. The bottom panel depicts the general case, again highlighting non-overlapping intervals. The figure serves to visually represent the conditions under which the proposed cost function's behavior is straightforward in guiding the model towards the optimal action and when deferral to an expert would be favored. These scenarios are discussed in Appendix A.


![](https://ai-paper-reviewer.com/taI8M5DiXj/figures_14_1.jpg)

> This figure visualizes two examples where the CAPO (Conditional Average Potential Outcome) intervals overlap.  It illustrates how the overlap affects the cost calculations in the CARED (Causal Action Recommendation with Expert Deferral) model, specifically highlighting the scenarios where the model will choose to defer the decision to an expert or take action itself based on cost-sensitive analysis. The figure aids understanding of how the algorithm handles uncertainty in outcome estimations.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/taI8M5DiXj/tables_20_1.jpg)
> This table lists the hyperparameters used for the XGBRegressor and Logistic Regression models in the synthetic data experiment.  The hyperparameters were chosen to optimize model performance in this specific dataset.  The values shown were determined through prior experimentation and analysis, though the exact process for tuning these parameters is not described in detail within the paper.  The table provides a concise summary for reproducibility of the experiments.

![](https://ai-paper-reviewer.com/taI8M5DiXj/tables_21_1.jpg)
> This table shows the hyperparameter settings used for the XGBRegressor and Logistic Regression models in the IHDP experiment.  The hyperparameters were tuned for each uncertainty level using a hyperparameter search algorithm (ray.tune).  The settings listed here are those that yielded the best performance.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/taI8M5DiXj/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}