---
title: "Latent Learning Progress Drives Autonomous Goal Selection in Human Reinforcement Learning"
summary: "Humans autonomously select goals based on both observed and latent learning progress, impacting goal-conditioned policy learning."
categories: ["AI Generated", ]
tags: ["Machine Learning", "Reinforcement Learning", "🏢 UC Berkeley",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} GbqzN9HiUC {{< /keyword >}}
{{< keyword icon="writer" >}} Gaia Molinaro et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=GbqzN9HiUC" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/GbqzN9HiUC" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/GbqzN9HiUC/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current AI systems often struggle with autonomous goal selection.  Humans, however, naturally set and pursue their own goals, which is a key aspect of human learning and exploration that remains largely unexplained. This study tackles this challenge by investigating the role of learning progress in human goal selection.  Existing research primarily focuses on *manifest* learning progress (i.e., observable performance improvements), but this study hypothesizes that *latent* learning progress (LLP) - the progress inferred through internal models of the environment and one's actions - plays a significant role.

The researchers designed a hierarchical reinforcement learning task where participants repeatedly selected goals and learned goal-conditioned policies.  They used computational modeling to compare several factors influencing goal selection: performance, manifest learning progress, LLP, and a hierarchy factor.  The study found that **LLP and performance were the most significant predictors of human goal selection.**  The incorporation of hierarchical knowledge also influenced performance and goal selection.  Importantly, individual differences in goal selection strategies were observed, emphasizing the need for personalized learning approaches.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Human goal selection is driven by both manifest (observed performance change) and latent learning progress (inferred from knowledge of actions and environment). {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Latent learning progress (LLP) better predicts human goal choices than traditional learning progress in hierarchical tasks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Individual differences in learning strategies exist and partially mediated by recognition of hierarchical task structures. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in AI and human-computer interaction because **it challenges existing notions of learning progress**, proposing a novel concept of latent learning progress (LLP).  This opens exciting new avenues for building more human-like, autonomous AI systems that learn efficiently by setting and achieving their own goals.  It also provides a valuable framework for designing more personalized and effective learning experiences.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_2_1.jpg)

> 🔼 This figure illustrates the experimental design. The top panel shows a flowchart of a single trial, which is divided into three stages: goal selection, action selection, and feedback.  The bottom panel summarizes the characteristics of the six goals (potions) used in the experiment. The goals are categorized by type (simple, complex, compound), hierarchical structure (yes/no), the solution (the correct sequence of actions), the actions required to make each potion, and the outcome (success/failure). This figure helps to visualize the hierarchical structure of the task and how goals are related.
> <details>
> <summary>read the caption</summary>
> Figure 1: Structure of learning trials and summary of the goal space. (Top) An example trial. (Bottom) Schematic representation of goals and their characteristics.
> </details>







### In-depth insights


#### Latent Learning's Role
The research explores **latent learning progress (LLP)** as a driver of autonomous goal selection in humans, contrasting it with manifest learning progress (LP).  While LP relies on observable performance changes, LLP leverages internal models and environmental knowledge, enabling progress estimation even without immediate performance improvements. The study demonstrates that LLP, alongside performance, significantly influences goal choices, particularly in hierarchical tasks where latent knowledge gains are substantial. **Individual differences** in leveraging LLP and hierarchical structures for goal selection are highlighted, suggesting personalized learning experiences could benefit from considering individual LLP sensitivity. This work thus **broadens the understanding** of human goal-directed behavior and provides insights for developing more human-like autotelic agents.

#### Goal Selection Model
The core of the proposed framework lies in its innovative approach to modeling human goal selection.  Instead of relying solely on readily observable performance changes, the model incorporates **latent learning progress (LLP)**, a measure of progress inferred from the agent's internal model of the environment and its actions, even without immediate performance improvements.  This is a significant departure from existing models, which primarily focus on manifest learning progress. The model's architecture likely employs a multi-armed bandit framework, where the probability of selecting a particular goal is determined by its estimated subjective value. This subjective value, in turn, is likely a weighted sum of performance, LLP, and possibly other factors such as the hierarchy of the task's structure.  The model's capacity to capture inter-individual differences in goal selection strategies, partially mediated by the recognition of hierarchical structure, highlights its potential for **personalized learning experiences** and the development of truly autotelic machines.

#### Inter-Individual Diff.
The section on Inter-Individual Differences reveals **significant heterogeneity** in human goal selection and learning strategies within the experimental task.  Participants exhibited diverse approaches, ranging from **systematic exploration** of all goals to focused mastery of a limited subset. Some participants showed evidence of leveraging the hierarchical structure of the task, while others did not.  These variations highlight the **complexity of intrinsic motivation** and underscore the need for personalized learning models that cater to individual differences. **Computational modeling** further supports this finding, demonstrating that while latent learning progress (LLP) is a key driver of goal selection across all participants, its weighting and interplay with other factors like performance and perceived hierarchy vary considerably.  This suggests that **individual learning styles**, and not just universal principles, should be considered when designing effective reinforcement learning systems.

#### Hierarchical Learning
Hierarchical learning, a core concept in AI and cognitive science, involves structuring learning processes into nested levels of abstraction.  **This approach mirrors human learning**, where complex tasks are broken down into simpler sub-tasks, mastered sequentially, and then integrated.  The research paper likely explores how this strategy affects goal selection and the development of goal-directed behaviors. A hierarchical framework would facilitate efficient exploration by allowing the agent to tackle manageable sub-goals, estimate progress accurately, and identify opportunities for transfer learning.  **Latent learning progress**, an often unobserved form of learning advancement, could be particularly significant in hierarchical systems. For example, an agent might exhibit little immediate performance improvement on a higher-level goal but still learn substantially from interactions with lower-level sub-goals. This insight emphasizes the importance of considering latent progress when designing and interpreting hierarchical learning systems. The study probably analyzes the role of hierarchy in shaping individual learning strategies and reveals **inter-individual differences in how individuals utilize hierarchical structures** for goal selection and task mastery.

#### Future Research
Future research should **investigate the interplay between latent learning progress (LLP) and other intrinsic motivation signals**, such as novelty and performance, to create a more comprehensive model of human goal selection.  **Individual differences in goal selection strategies**, revealed in this study, warrant further exploration to identify underlying cognitive and personality traits.  Additionally, the study's hierarchical task presents an opportunity to explore **how hierarchical structure influences both learning and goal selection**, especially in relation to LLP.  Finally, the development of **more human-like autotelic machines** necessitates further research integrating LLP into their goal selection mechanisms.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_5_1.jpg)

> 🔼 This figure presents the behavioral results of the study and the model fitting results. Panel A shows that the learning performance was significantly above chance level and there were significant hierarchy effects, where hierarchical goals were learned better than non-hierarchical goals. Panel B shows that there were no significant hierarchy effects on the test performance. Panel C shows that fewer attempts were required to solve hierarchical goals compared to non-hierarchical goals. Panel D shows that partial hierarchy effects are present in goal selection and the winning model (performance and LLP) replicates this result. Panel E shows the model responsibilities for individual participants and the population. Panel F presents the best-fitting parameters obtained by the hierarchical Bayesian inference for the winning model.
> <details>
> <summary>read the caption</summary>
> Figure 2: Behavioral and modeling results. (A) Learning performance was above chance and showed hierarchy effects. (B) Test performance was better than chance, but no significant hierarchy effects were detected. (C) On average, fewer attempts were needed to learn hierarchical (G3) compared to non-hierarchical (G4) 4-action goals. (D) Partial hierarchy effects are present in goal selection. The winning model (triangle markers) reproduces this pattern. (E) Model responsibilities for individual participants and the overall studied population. (F) Best-fitting parameters (HBI) for the winning model. Bars and shading indicate the SEM, dots individual participants. *** p < 0.001, ** p < 0.01
> </details>



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_6_1.jpg)

> 🔼 This figure displays the diverse strategies used by four different participants in the experiment.  Each subplot (A-D) shows a participant's chosen action sequence across trials during the training and learning phases. The colors indicate the chosen goal and whether the feedback for each trial was correct or incorrect. These examples highlight the wide range of learning styles and goal selection strategies observed among the participants, and illustrate the variability in how participants approached the hierarchical task structure. Some showed systematic approaches, while others seemed to lack a clear strategy.
> <details>
> <summary>read the caption</summary>
> Figure 3: Example behaviors from four participants. (A–D) Chosen action sequence as a function of trial number (training and learning phases), color-coded by goal and feedback.
> </details>



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_8_1.jpg)

> 🔼 This figure presents the behavioral and modeling results of the study. Panel A shows the learning performance, demonstrating above-chance performance with significant hierarchy effects. Panel B shows the testing performance, which was also above chance but without significant hierarchy effects. Panel C shows that fewer attempts were needed to learn hierarchical goals than non-hierarchical goals. Panel D illustrates partial hierarchy effects in goal selection, which were reproduced by the winning computational model. Panel E displays the model responsibilities, indicating the winning model's performance across participants. Finally, Panel F shows the best-fitting parameters for the winning model.
> <details>
> <summary>read the caption</summary>
> Figure 2: Behavioral and modeling results. (A) Learning performance was above chance and showed hierarchy effects. (B) Test performance was better than chance, but no significant hierarchy effects were detected. (C) On average, fewer attempts were needed to learn hierarchical (G3) compared to non-hierarchical (G4) 4-action goals. (D) Partial hierarchy effects are present in goal selection. The winning model (triangle markers) reproduces this pattern. (E) Model responsibilities for individual participants and the overall studied population. (F) Best-fitting parameters (HBI) for the winning model. Bars and shading indicate the SEM, dots individual participants. *** p < 0.001, ** p < 0.01
> </details>



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_16_1.jpg)

> 🔼 This figure illustrates the experimental setup of the study. The top panel shows a flowchart of a single trial, which includes the goal selection stage, the action selection stage, and the feedback stage. The bottom panel shows a table summarizing the characteristics of the six goals (potions) used in the experiment, including their type (simple, complex, compound, hierarchical), the presence or absence of a hierarchical structure, and the solution.
> <details>
> <summary>read the caption</summary>
> Figure 1: Structure of learning trials and summary of the goal space. (Top) An example trial. (Bottom) Schematic representation of goals and their characteristics.
> </details>



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_18_1.jpg)

> 🔼 This figure displays the behavioral and computational modeling results of the study. Panel A shows that learning performance was significantly above chance and that hierarchical goals were learned faster than non-hierarchical goals. Panel B shows the testing phase performance, which was also significantly above chance. Panel C shows that it took fewer attempts to learn hierarchical goals. Panel D shows the probability of selecting a goal, illustrating a partial hierarchy effect, reproduced by the winning model in the computational modeling. Panel E shows the model responsibilities for individual participants and the overall studied population, indicating that the model incorporating performance and latent learning progress (LLP) best fit the data. Panel F shows the best-fitting parameters for the winning model.
> <details>
> <summary>read the caption</summary>
> Figure 2: Behavioral and modeling results. (A) Learning performance was above chance and showed hierarchy effects. (B) Test performance was better than chance, but no significant hierarchy effects were detected. (C) On average, fewer attempts were needed to learn hierarchical (G3) compared to non-hierarchical (G4) 4-action goals. (D) Partial hierarchy effects are present in goal selection. The winning model (triangle markers) reproduces this pattern. (E) Model responsibilities for individual participants and the overall studied population. (F) Best-fitting parameters (HBI) for the winning model. Bars and shading indicate the SEM, dots individual participants. *** p < 0.001, ** p < 0.01
> </details>



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_19_1.jpg)

> 🔼 This figure presents behavioral and modeling results. Panel A shows learning performance which was above chance and showed hierarchy effects. Panel B shows test performance which was better than chance, but no significant hierarchy effects. Panel C shows fewer attempts to learn hierarchical goals than non-hierarchical goals. Panel D shows partial hierarchy effects present in goal selection, reproduced by the winning model. Panel E shows model responsibilities, and panel F shows best-fitting parameters.
> <details>
> <summary>read the caption</summary>
> Figure 2: Behavioral and modeling results. (A) Learning performance was above chance and showed hierarchy effects. (B) Test performance was better than chance, but no significant hierarchy effects were detected. (C) On average, fewer attempts were needed to learn hierarchical (G3) compared to non-hierarchical (G4) 4-action goals. (D) Partial hierarchy effects are present in goal selection. The winning model (triangle markers) reproduces this pattern. (E) Model responsibilities for individual participants and the overall studied population. (F) Best-fitting parameters (HBI) for the winning model. Bars and shading indicate the SEM, dots individual participants. *** p < 0.001, ** p < 0.01
> </details>



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_21_1.jpg)

> 🔼 This figure presents behavioral and modeling results from the study. Panel A shows learning performance which was above chance and was influenced by hierarchy. Panel B shows that test performance was above chance but was not influenced by hierarchy. Panel C shows that hierarchical goals took fewer attempts to learn than non-hierarchical ones. Panel D shows partial hierarchy effects in goal selection which were reproduced by the winning model. Panel E displays model responsibilities, and Panel F shows the parameters for the winning model. Statistical significance is noted for some results.
> <details>
> <summary>read the caption</summary>
> Figure 2: Behavioral and modeling results. (A) Learning performance was above chance and showed hierarchy effects. (B) Test performance was better than chance, but no significant hierarchy effects were detected. (C) On average, fewer attempts were needed to learn hierarchical (G3) compared to non-hierarchical (G4) 4-action goals. (D) Partial hierarchy effects are present in goal selection. The winning model (triangle markers) reproduces this pattern. (E) Model responsibilities for individual participants and the overall studied population. (F) Best-fitting parameters (HBI) for the winning model. Bars and shading indicate the SEM, dots individual participants. *** p < 0.001, ** p < 0.01
> </details>



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_22_1.jpg)

> 🔼 This figure presents the behavioral and modeling results of the study. Panel A shows that learning performance was above chance and was affected by the hierarchical structure of the tasks. Panel B shows that testing performance was also above chance but that hierarchy effects were not significant. Panel C shows that the number of attempts needed to learn hierarchical tasks was lower than that of non-hierarchical tasks. Panel D shows that the probability of selecting a goal was affected by the hierarchical structure and performance. The winning model from the Bayesian model selection analysis is shown in panel D. The model responsibilities and parameters of the winning model are shown in panels E and F, respectively.  The results demonstrate that performance and latent learning progress (LLP) are key factors guiding human goal selection in a hierarchical reinforcement learning task. 
> <details>
> <summary>read the caption</summary>
> Figure 2: Behavioral and modeling results. (A) Learning performance was above chance and showed hierarchy effects. (B) Test performance was better than chance, but no significant hierarchy effects were detected. (C) On average, fewer attempts were needed to learn hierarchical (G3) compared to non-hierarchical (G4) 4-action goals. (D) Partial hierarchy effects are present in goal selection. The winning model (triangle markers) reproduces this pattern. (E) Model responsibilities for individual participants and the overall studied population. (F) Best-fitting parameters (HBI) for the winning model. Bars and shading indicate the SEM, dots individual participants. *** p < 0.001, ** p < 0.01
> </details>



![](https://ai-paper-reviewer.com/GbqzN9HiUC/figures_23_1.jpg)

> 🔼 This figure displays behavioral and computational modeling results from an experiment on human goal selection in a hierarchical reinforcement learning task. Panel A shows learning performance, demonstrating better performance on easier tasks and hierarchical tasks. Panel B shows that participants perform above chance on all goals, but that the hierarchy effect disappears when goal selection is removed. Panel C shows that hierarchical tasks require fewer attempts to learn than non-hierarchical tasks. Panel D shows that hierarchical tasks are less frequently chosen than other tasks but that the winning computational model based on performance and Latent Learning Progress (LLP) reproduces this pattern. Panel E displays model responsibilities for different models, showing that a model combining performance and LLP is the most prominent across participants. Finally, Panel F provides the best-fitting parameters for this model.
> <details>
> <summary>read the caption</summary>
> Figure 2: Behavioral and modeling results. (A) Learning performance was above chance and showed hierarchy effects. (B) Test performance was better than chance, but no significant hierarchy effects were detected. (C) On average, fewer attempts were needed to learn hierarchical (G3) compared to non-hierarchical (G4) 4-action goals. (D) Partial hierarchy effects are present in goal selection. The winning model (triangle markers) reproduces this pattern. (E) Model responsibilities for individual participants and the overall studied population. (F) Best-fitting parameters (HBI) for the winning model. Bars and shading indicate the SEM, dots individual participants. *** p < 0.001, ** p < 0.01
> </details>



</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GbqzN9HiUC/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}