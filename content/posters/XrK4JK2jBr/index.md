---
title: "Designs for Enabling Collaboration in Human-Machine Teaming via Interactive and Explainable Systems"
summary: "Boosting Human-AI teamwork via interactive, explainable AI!"
categories: []
tags: ["AI Applications", "Robotics", "🏢 MIT",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} XrK4JK2jBr {{< /keyword >}}
{{< keyword icon="writer" >}} Rohan R Paleja et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=XrK4JK2jBr" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94740" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=XrK4JK2jBr&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/XrK4JK2jBr/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Human-machine teaming (HMT) is crucial for productivity and safety, but current state-of-the-art methods using imitation or reinforcement learning produce brittle AI teammates that act independently rather than collaboratively. This is problematic for domains requiring seamless human-AI interaction.  The paper investigates this issue using a ubiquitous experimental domain and demonstrates the limitations of existing techniques. 

To address these shortcomings, the researchers developed novel HMT approaches allowing for iterative, mixed-initiative team development.  **They introduced interpretable AI teammates that can be interactively modified by end-users.**  A 50-subject study revealed that white-box approaches with interactive modifications significantly improved team performance compared to white-box approaches alone. While black-box models showed better initial performance, emphasizing a trade-off between explainability and ease of training, the interactive aspect of the white-box approach proved beneficial.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Current HMT methods are brittle and fail to foster true collaboration. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Interactive, explainable AI systems significantly enhance team performance. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} White-box models with user modification outperform black-box approaches in collaborative HMT. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in human-machine teaming (HMT) and explainable AI.  It addresses the brittleness of current HMT approaches and proposes a novel, interactive method for developing more effective and adaptable collaborative AI agents. This research directly impacts current trends by highlighting the importance of explainability and interactivity in HMT and opens new avenues for improving human-AI collaboration across various domains.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_4_1.jpg)

> This figure shows a flow chart that outlines the steps in the methodology of the study. It starts with training a collaborative AI agent in a simulation environment.  Then, the trained AI policy is pruned to reduce its complexity and improve interpretability using the contextual pruning algorithm. This interpretable AI policy is then used in an interactive human-machine teaming (HMT) setting, allowing for human modification of the AI policy.  The process of interaction and modification leads to team development, shown as Tuckman's stages of team development.





![](https://ai-paper-reviewer.com/XrK4JK2jBr/tables_6_1.jpg)

> This table summarizes the five conditions (IV1-C1 to IV1-C5) of the independent variable IV1 (teaming method) used in the user study.  Each condition is characterized by the presence or absence of explicit interaction with the AI's policy, whether policy changes occur across iterations, and whether a white-box (interpretable IDCT) or a black-box (NN) policy is used.  This helps to clarify the experimental setup and the different variations of human-AI interaction being investigated.





### In-depth insights


#### HMT Challenges
Human-Machine Teaming (HMT) presents numerous challenges. **Explainability** is crucial; opaque AI systems hinder trust and collaboration, making it difficult for humans to understand AI decision-making.  **Interactivity** is also critical;  static AI systems cannot adapt to dynamic human needs and preferences, impeding effective teamwork.  Furthermore, **generalizability** is a significant hurdle;  AI trained on limited datasets may not perform well with diverse human partners or in varying contexts.  Addressing these challenges requires **mixed-initiative interfaces** that enable human control and iterative feedback, **interpretable AI models** that facilitate understanding, and **robust learning algorithms** that enhance generalizability and adaptability, leading to seamless and effective HMT.

#### IDCT Design
The design of Interpretable Discrete Control Trees (IDCTs) is a crucial aspect of this research, focusing on creating **interpretable and modifiable AI teammate policies**.  The architecture leverages differentiable decision trees, allowing for both interpretability (through a clear tree structure) and the ability to utilize reinforcement learning for training.  A key innovation is the **contextual pruning algorithm**, which simplifies the tree structure post-training, enhancing human understanding without significantly sacrificing performance.  This makes the IDCTs particularly well-suited for interactive human-machine teaming, as users can easily understand and modify the agent's decision-making process through a user interface. The design emphasizes a balance between the complexity needed for effective learning and the simplicity crucial for human comprehension and intervention, directly addressing limitations of previous black-box approaches in human-AI collaboration.

#### User Study
A well-designed user study is crucial for validating the effectiveness of any human-computer interaction (HCI) system.  In this context, a robust user study would involve a diverse participant pool, clearly defined tasks reflecting real-world scenarios, and a structured experimental design.  **Careful consideration of the metrics used to evaluate performance is essential.**  Beyond objective metrics like task completion time and accuracy, subjective measures such as user satisfaction, perceived workload, and trust in the system should also be collected using established questionnaires.  The analysis phase should explore both quantitative and qualitative data, identifying trends and correlations.  A thoughtful discussion of limitations and potential biases in the user study is necessary.  Finally, the study results should be interpreted within their context and limitations, providing actionable insights for system design improvement. **The design should consider the iterative nature of HMT and incorporate repeated teaming sessions to observe team development.** This will offer rich insights into how users adapt to the system and how the system adapts to users over time.  **Qualitative data gathered via post-experiment interviews,  observations, or think-aloud protocols provides valuable context and deeper insight beyond numerical metrics.**

#### Future Work
The 'Future Work' section of this research paper should prioritize extending the interactive policy modification framework to more complex, real-world scenarios.  **Investigating its application in safety-critical domains like healthcare and manufacturing is crucial.** This would involve incorporating more robust feedback mechanisms and addressing the challenges of higher-dimensional state spaces.  Further research should focus on improving the efficiency and scalability of the interpretable machine learning architecture, potentially exploring alternative methods for representing and modifying policies. **A comparison of the proposed approach with other xAI methods for HMT is needed**, focusing on explainability and usability.  Finally, long-term studies exploring team development dynamics over many interaction cycles and a wider range of participant skills are essential to fully understand the potential of human-machine teamwork with interactive, interpretable AI.

#### Limitations
The research, while groundbreaking, acknowledges several limitations.  **The study's participants, primarily university students, may not fully represent the broader population**, potentially limiting the generalizability of the findings.  The reliance on a simulated environment (Overcooked-AI) introduces an artificiality that might not fully capture the complexities of real-world human-machine teaming.  Furthermore, the limited number of iterations in the study and the relatively short duration might have prevented the observation of long-term team development patterns and might have hindered the participants in reaching a more fully collaborative state, **especially given the complexity of the optional collaboration domain.** The focus on interpretable models, while beneficial for transparency, might limit performance compared to state-of-the-art black-box approaches. Finally, the study could benefit from expanding the range of user abilities and incorporating diverse personality traits for a more comprehensive evaluation.  Despite these limitations, the work offers valuable insights into human-machine collaboration.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_5_1.jpg)

> This figure shows the different ways a human user can modify the AI teammate's policy tree representation.  Three types of modifications are presented: tree deepening (adding nodes to increase complexity), decision variable modification (changing the conditions used for decision-making), and leaf node modification (altering actions and their probabilities).


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_7_1.jpg)

> This figure shows the results of a user study comparing different AI teaming strategies across two game scenarios: Forced Coordination and Optional Collaboration.  Each panel represents a different AI strategy: Human-Led Policy Modification, AI-Led Policy Modification, Static (Interpretable), Static (Black-Box), and Fictitious Co-Play. The y-axis shows the cumulative team reward, and the x-axis shows the iteration number (repeated teaming interactions). The red dotted line represents the average reward for each iteration, and the shaded area shows the standard deviation.  The figure demonstrates how different strategies lead to varying team performance over repeated interactions, highlighting the impact of interpretability and interactivity on teamwork effectiveness.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_7_2.jpg)

> This figure displays the results of the user study across the two domains (Forced Coordination and Optional Collaboration). The x-axis represents the iteration number (1-4), and the y-axis represents the team reward. Each point represents a single user's score in a particular iteration, and the red dotted line connects the average scores across iterations. The shaded area represents the standard deviation of the scores. The figure shows that the performance varies across different conditions in both domains.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_8_1.jpg)

> This figure shows the results of the user study, comparing the performance of different teaming methods across two game domains (Forced Coordination and Optional Collaboration) over four iterations.  The lines represent the average team reward per iteration, and the shaded areas show the standard deviation.  The goal is to see how different teaming methods affect team performance over time and if there are differences between the two game types.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_15_1.jpg)

> This figure shows the pipeline for generating the AI teammate policies used in the study.  It starts with training an IDCT (Interpretable Discrete Control Tree) agent in each domain (Forced Coordination and Optional Collaboration). Then, contextual pruning is applied to simplify the tree structure while maintaining performance. The resulting pruned IDCT policies (with 3 leaves for Forced Coordination and 2 for Optional Collaboration) are used in the pre-experiment stage of the user study.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_16_1.jpg)

> This figure shows a visualization of a trained interpretable discrete control tree (IDCT) used in the Forced Coordination domain of the Overcooked-AI experiment. The tree's structure displays decision nodes (blue boxes) and action nodes (orange boxes). Each node shows probabilities for different actions, and the tree's structure is designed to be interpretable by humans for easier understanding and potential modification. The caption is short and does not provide enough information about the figure itself.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_16_2.jpg)

> This figure shows the trained interpretable discrete control tree used in the Forced Coordination domain. The tree consists of a root decision node that checks if there is soup on the shared counter, leading to different action nodes based on whether the condition is true or false.  If true, the agent has a high probability of getting soup from the counter. If false, there's a distribution across actions including getting soup and tomatoes from the counter, getting a dish, or waiting.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_18_1.jpg)

> This figure shows the experiment procedure for each of the five conditions (IV1-C1 to IV1-C5).  Each condition involves a slightly different approach to human-AI interaction, ranging from human-led modification of the AI's policy to AI-led modifications based on gameplay and static (interpretable or black box) policies. The flowcharts outline the sequence of events, including surveys, tutorials, AI interaction phases, policy visualizations, and concluding questionnaires.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_18_2.jpg)

> This figure shows the experiment flow diagram for each of the five conditions tested in the study.  It outlines the steps involved in each condition, from the initial introduction and surveys through the repeated teaming episodes, policy modifications (where applicable), and finally the post-experiment surveys. The diagram is broken down to visualize the differences in procedure for each of the five conditions.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_18_3.jpg)

> This figure shows the detailed flowcharts for the five different conditions (IV1-C1 to IV1-C5) in the user study. Each flowchart illustrates the sequence of events, including initial surveys, the tutorial, repeated human-AI teaming episodes, policy modifications (if applicable), workload surveys, and finally post-experiment surveys.  It visually represents the differences in procedures across conditions, highlighting variations in user interaction with the AI's policy and the degree of policy control afforded to the human user. The different steps for different conditions help to understand the design of the study and how it manipulates different independent variables.


![](https://ai-paper-reviewer.com/XrK4JK2jBr/figures_18_4.jpg)

> This figure shows the experiment procedure for the five different conditions in the user study.  Each condition is represented by a separate flowchart that details the steps from the introduction phase with demographic and gaming experience surveys, followed by a tutorial, to the main experimentation phase which includes repeated teaming episodes and workload surveys, concluding with end-of-experiment surveys assessing various aspects such as collaboration fluency and trust.


</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XrK4JK2jBr/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}