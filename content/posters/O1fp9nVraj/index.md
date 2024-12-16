---
title: "On scalable oversight with weak LLMs judging strong LLMs"
summary: "Weak LLMs can accurately supervise strong LLMs via debate, outperforming simpler consultancy methods, especially in information-asymmetric tasks."
categories: ["AI Generated", ]
tags: ["Natural Language Processing", "Large Language Models", "🏢 Google DeepMind",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} O1fp9nVraj {{< /keyword >}}
{{< keyword icon="writer" >}} Zachary Kenton et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=O1fp9nVraj" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/O1fp9nVraj" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/O1fp9nVraj/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current AI alignment relies heavily on human feedback, but this is unsustainable as AI surpasses human capabilities. This paper investigates scalable oversight protocols, aiming to leverage AI's abilities to enhance human supervision of superhuman AI. The study focuses on three protocols: debate, where two AIs argue for opposing answers; consultancy, where a single AI interacts with the judge; and direct question-answering. The researchers employ large language models (LLMs), with weaker models acting as judges and stronger ones as agents.  The main challenge is to design methods which bridge the capability gap between humans and future superhuman AI. 

The study benchmarks these protocols across diverse tasks with varying asymmetries between judges and agents.  Results show debate outperforms consultancy across all tasks.  However, its advantage over direct question answering varies depending on the task type. Interestingly, when agents choose which answer to argue for, judges are less often convinced by the incorrect answer in debate compared to consultancy. Moreover, utilizing stronger AI debaters increases judge accuracy, though the improvement is more modest than initially expected.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Debate consistently outperforms consultancy in LLM oversight across various tasks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The effectiveness of debate versus direct question answering depends on the task type; debate excels in information-asymmetric tasks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Stronger debater models modestly improve judge accuracy, suggesting a positive scaling trend for scalable oversight. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for AI safety researchers as **it empirically evaluates scalable oversight protocols**  like debate and consultancy, using LLMs. Its findings challenge existing assumptions about debate's effectiveness, highlighting the need for further research into improving human-AI collaboration for safer AI systems.  The diverse range of tasks and the use of LLMs as judges (weaker than agents) improve the generalizability of results.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_1_1.jpg)

> 🔼 This figure illustrates the different types of tasks and protocols used in the study.  The tasks are categorized into three types: Extractive QA (extracting information from a text), Closed QA (no additional context), and Multimodal (combining text and images). The protocols represent different methods for obtaining human-level oversight of AI agents: Direct QA (judge answers directly without AI assistance), Consultancy (single AI agent tries to persuade a judge), Debate (two AI agents compete to persuade a judge), Open Consultancy (AI agent chooses an answer and tries to persuade a judge), and Open Debate (AI agents choose answers and compete to persuade a judge).  The QA with article protocol was only used in the Extractive QA tasks.
> <details>
> <summary>read the caption</summary>
> Figure 1: Task types and protocols.
> </details>





![](https://ai-paper-reviewer.com/O1fp9nVraj/tables_19_1.jpg)

> 🔼 This table summarizes the key characteristics of the nine different question answering tasks used in the paper.  The tasks are categorized into three types: extractive, multimodal, and closed.  Extractive QA tasks introduce an information asymmetry where the judge (weaker LLM) does not have access to the same information as the debaters/consultants (stronger LLMs).  The table highlights key features for each task, helping to understand the varying levels of difficulty and the type of reasoning required.
> <details>
> <summary>read the caption</summary>
> Table F.1: Summary of key characteristics differentiating our tasks: extractive (top row), multimodal (MMMU) and closed (others). Extractive tasks involve information-asymmetry, others do not.
> </details>





### In-depth insights


#### Weak Judge Debate
The concept of 'Weak Judge Debate' in the context of large language model (LLM) evaluation presents a compelling approach to scalable oversight.  It leverages the inherent strengths of debate to help **less capable judges** make accurate assessments of potentially superhuman AI agents. The core idea is that the structured debate format, where two strong LLMs argue for opposing answers, allows even a weak judge to discern the correct answer by observing the quality of reasoning and evidence presented. This paradigm shifts the focus from directly judging AI capabilities to evaluating the persuasive power of the AI arguments, which might prove a more manageable and scalable method for human oversight.  A key advantage is the robustness to inherent bias in both AI agents and judges; debate offers a process for **amplifying the strength of correct arguments** while mitigating the impact of bias by allowing comparison of arguments.  However, experimental results show that debate's effectiveness is task dependent. While it consistently outperforms simpler methods like consultancy, its efficacy can vary based on factors like task complexity, type of argumentative asymmetry and the judge's baseline capabilities, highlighting the need for further research to fully understand and optimize its potential for scalable oversight in various contexts.

#### Scalable Oversight
Scalable oversight, crucial for safely managing superhuman AI, seeks methods enabling humans to accurately supervise advanced AI systems.  The core challenge lies in developing protocols that leverage AI capabilities to enhance human oversight, addressing the limitations of direct human feedback. **Debate and consultancy protocols** are explored as potential solutions, both utilizing weaker AI models as human surrogates to evaluate stronger AI agents' performance.  **Debate**, where two strong AI models argue for opposing answers, shows promise for improving accuracy over **consultancy**, where a single model attempts to convince the judge.  However, **results are highly task-dependent**, with debate offering more advantages on tasks involving information asymmetry. Open-role variations, allowing AI agents to choose their argument, add further complexity, with open debate demonstrating resilience to biases. This highlights the importance of testing scalable oversight under various conditions and capability gaps to ensure its effectiveness in real-world scenarios.  **Future research must rigorously test debate and other protocols** as training methods, measuring their impact on long-term AI alignment.

#### Debate vs. Consult
This research explores the effectiveness of two AI oversight protocols: debate and consultation, in evaluating strong AI models using weaker judge models.  **Debate**, involving two strong AIs arguing opposing viewpoints before a weaker judge, consistently outperforms **consultation**, where a single AI attempts persuasion.  This is true across diverse tasks ranging from extractive QA to mathematical reasoning, highlighting debate's robustness. However, debate's advantage over direct question answering, where a judge answers without AI input, varied depending on the task and the judge's capabilities, suggesting that a weaker judge may require a strong information asymmetry or a significantly less difficult question to benefit.  The open-role variants, where agents select which argument to advance, further emphasize debate's reliability, especially by reducing the impact of errors.  **Stronger debater models consistently improve judge accuracy in debate**, although the effect is more modest than expected, underscoring the need for advanced judge-training methodologies.

#### Open Debate
In the realm of AI alignment research, **open debate** presents a compelling approach to scalable oversight.  Unlike traditional debate protocols where debaters are assigned arguments, open debate empowers AI agents to autonomously select the stance they wish to defend. This crucial element introduces a more realistic scenario, mimicking real-world situations where AI might strategically choose its arguments.  The inherent uncertainty in argument selection provides valuable insights into AI decision-making processes and allows for a more robust assessment of AI capabilities. **By observing how a weaker judge model reacts to the AI's self-selected stance, researchers can gauge the persuasiveness and overall quality of AI reasoning.** This contrasts with closed-debate where the judge's accuracy is potentially skewed by the predefined arguments.  Moreover, **open debate's emphasis on self-selection promotes a richer evaluation of AI's potential for manipulation or strategic behavior**.  This method provides a critical step towards building trustworthy and reliable AI systems.  The study's findings suggest that open debate, while having challenges, offers a more realistic and nuanced evaluation of AI's alignment compared to traditional assigned-role protocols.

#### Future Directions
Future research should prioritize **rigorous empirical validation** of debate's effectiveness as a training protocol, moving beyond inference-only evaluations.  This necessitates a transition to **real-world training scenarios** where judge accuracy is assessed during model training, rather than solely in a post-hoc inference setting.  Investigating **alternative judge models**, including human judges, is crucial for assessing generalization and robustness.  Moreover, exploring **different task domains** will expand the understanding of debate's applicability beyond existing benchmarks.  Further work should also focus on refining existing protocols via hyperparameter tuning and exploration of **innovative debate structures** to improve judge accuracy and efficiency.  A key area for improvement is mitigating **positional bias**, ensuring that debate outcomes are less influenced by presentation order.  Finally, studying how debate scales with **increasing capability gaps** between judges and agents remains a vital open question.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_1_2.jpg)

> 🔼 This figure displays the results of an experiment comparing different scalable oversight protocols: direct question answering, consultancy, and debate.  The experiment is performed across several task types and using multiple judge models of varying capabilities. The x-axis represents the judge model used, the facets represent the task type, and the color of the bars represents the oversight protocol. Judge accuracy is plotted on the y-axis.  Error bars show 95% confidence intervals. The key finding is that debate outperforms consultancy across all task types. The relative performance of debate and direct question answering varies depending on the task type.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_7_1.jpg)

> 🔼 This figure presents the results of open debate and open consultancy experiments.  The top panel shows judge accuracy plotted against the win rate of the protagonist (the model that chooses which answer to argue for).  The bottom panel breaks down judge accuracy further, showing performance when the protagonist chose the correct versus incorrect answer.  Different colors represent different protocols (consultancy vs. debate) and judge models, while different facets show the results for different task types (extractive, closed, and multimodal). The 95% confidence intervals are also shown.  Overall, the figure suggests that open debate may be more resilient to amplifying errors made by a strong model than open consultancy.
> <details>
> <summary>read the caption</summary>
> Figure 3: Open debate and open consultancy, where a protagonist debater/consultant chooses which answer to argue for. Top: Judge accuracy (y-axis) and win rate of protagonist/consultant (x-axis). Blue colours for consultancy, red colours for debate, with the shade corresponding to judge model. Bottom: Judge accuracy according to whether the protagonist/consultant chose the correct (dark) or incorrect (light) answer. Split by judge model (x-axis) and protocol: consultancy and debate. Facet is task type. 95% CIs. Correct answer rate: 88% (extractive), 84% (closed), 71% (multimodal).
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_8_1.jpg)

> 🔼 This figure shows the correlation between the Elo rating of debaters (a measure of their persuasiveness) and the accuracy of a Gemini Pro 1.5 judge in evaluating their arguments.  The Elo scores are calculated using a series of 1-turn debates where each debater argues for a different answer to a question. The results indicate a stronger positive correlation between the debater's Elo score and judge accuracy for extractive tasks (where the judge has information asymmetry) compared to closed tasks.
> <details>
> <summary>read the caption</summary>
> Figure 4: Plot of Pro 1.5 judge accuracy vs. debater aggregate Elo scores. 95% CIs. Judge accuracy is more sensitive to debaters on extractive tasks than closed tasks. See Figure E.1 for elo results split by debater assignment (whether debaters are assigned to argue for true vs. false positions).
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_14_1.jpg)

> 🔼 This figure shows the results of the assigned-role experiments. The mean judge accuracy is plotted against different task types, judge models, and protocols. Higher accuracy is better. The figure includes 95% confidence intervals. The QA with article protocol is only available for extractive tasks, and only the Pro 1.5 model was used in multimodal tasks.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_15_1.jpg)

> 🔼 This figure shows the results of the assigned-role experiments. The mean judge accuracy is plotted on the y-axis. The x-axis represents the judge model used.  The results are faceted by task type (extractive, closed, multimodal). Each bar represents a different protocol (QA without article, Consultancy, Debate, QA with article). Error bars show 95% confidence intervals. The QA with article protocol is only applicable to extractive tasks and the multimodal tasks are only run with Pro 1.5 as the judge model.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_15_2.jpg)

> 🔼 This figure shows the results of an ablation study where the number of samples used to select the best debater response in the debate protocol was varied.  The x-axis represents different judge models (Gemma7B, GPT-3.5, Pro 1.0, Pro 1.5). The y-axis represents the judge accuracy.  Two bars are shown for each judge model, one for using a single sample (Best-of-N=1) and one for using four samples (Best-of-N=4). The results indicate that there is no significant difference in judge accuracy between these two settings.  Error bars are included showing 95% confidence intervals.
> <details>
> <summary>read the caption</summary>
> Figure A.3: The influence of best-of-N sampling on debate performance. We observe no consistent benefit to using best-of-N sampling on debater responses. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_16_1.jpg)

> 🔼 This figure shows the result of ablations on the number of few-shot prompts used for the judge model.  The experiment compares the judge's accuracy with 0-shot (no few-shot examples) and 6-shot (6 few-shot examples) prompting. The results indicate that for the given tasks, judge models, and protocols, adding few-shot examples to the judge does not consistently improve the accuracy.
> <details>
> <summary>read the caption</summary>
> Figure A.4: The influence of few-shot prompting on judge performance.. We compare 6-shot prompting (paler colours) with 0-shot prompting (darker colours), our default approach. We observe no consistent benefit to using few-shot prompting. Different colours denote different protocols. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_16_2.jpg)

> 🔼 This figure presents the results of an experiment comparing three different scalable oversight protocols: Question Answering (QA) without an article, Consultancy, and Debate.  The experiment was conducted using various judge models (Gemma7B, GPT3.5, Pro 1.0, and Pro 1.5) across different task types (Extractive, Closed, and Multimodal). The y-axis represents the mean judge accuracy, while the x-axis shows the judge model used.  Different colors represent the different protocols. Error bars indicate the 95% confidence interval.  The figure demonstrates that the debate protocol generally outperforms the other two protocols, especially for extractive QA tasks.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_17_1.jpg)

> 🔼 This figure shows the results of an ablation study on the debate protocol, comparing the performance of the debate protocol when using sequential turns (where debaters take turns one at a time) versus simultaneous turns (where debaters respond at the same time). The results show no significant difference in judge accuracy between the two turn styles, suggesting that the order of turns does not significantly impact the performance of the debate protocol. The error bars in the figure represent the 95% confidence intervals of the mean judge accuracy.
> <details>
> <summary>read the caption</summary>
> Figure A.6: The influence of turn style. Models are evaluated with an effective Best-of-N setting of N = 1. Lighter colours denote sequential turns, while darker colours denote simultaneous turns (our default). We observe no significant difference between the two turn styles. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_17_2.jpg)

> 🔼 This figure displays the mean judge accuracy across different task types (extractive, closed, and multimodal) using three different oversight protocols (QA without article, consultancy, and debate). The x-axis represents different judge models (Gemma7B, GPT3.5, Pro 1.0, and Pro 1.5), and the color of the bars represents the oversight protocol used.  The figure shows that debate generally outperforms consultancy, while the performance of debate relative to direct question answering varies depending on the task type. The results include 95% confidence intervals for better statistical significance. Only the extractive tasks used the QA with article protocol, and only the Pro 1.5 model was used for multimodal tasks.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_18_1.jpg)

> 🔼 This figure presents the results of an experiment comparing three different scalable oversight protocols: question answering without AI assistance, consultancy (a single AI agent tries to convince a judge), and debate (two AI agents compete to convince a judge).  The experiment used various judge and agent models of varying capabilities, creating asymmetries to reflect real-world scenarios where humans might supervise more advanced AI.  The results are broken down by task type, showing judge accuracy for each protocol and model combination.  The goal was to understand which method best enables weaker judge models (acting as a proxy for human judges) to accurately assess the answers generated by stronger AI models.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_18_2.jpg)

> 🔼 This figure compares the performance of open debate and open consultancy protocols.  The top section shows the judge's accuracy plotted against the win rate of the debater/consultant choosing the answer to argue for. The bottom section breaks down the judge's accuracy based on whether the debater/consultant chose the correct or incorrect answer. Different colors represent different protocols and judge models. Error bars represent 95% confidence intervals, and the correct answer rate varies across task types.
> <details>
> <summary>read the caption</summary>
> Figure 3: Open debate and open consultancy, where a protagonist debater/consultant chooses which answer to argue for. Top: Judge accuracy (y-axis) and win rate of protagonist/consultant (x-axis). Blue colours for consultancy, red colours for debate, with the shade corresponding to judge model. Bottom: Judge accuracy according to whether the protagonist/consultant chose the correct (dark) or incorrect (light) answer. Split by judge model (x-axis) and protocol: consultancy and debate. Facet is task type. 95% CIs. Correct answer rate: 88% (extractive), 84% (closed), 71% (multimodal).
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_19_1.jpg)

> 🔼 This figure displays the results of the assigned-role experiments.  Judge accuracy is shown on the y-axis, broken down by task type (facets), judge model (x-axis), and protocol (color). Higher accuracy indicates better performance.  The 95% confidence intervals (CI) are calculated by aggregating results over tasks of the same type.  The QA with article protocol, only applicable to extractive tasks, is shown in purple.  Only the Gemini Pro 1.5 model was used for multimodal tasks.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_20_1.jpg)

> 🔼 This figure displays the results of an experiment comparing three different protocols (QA without article, Consultancy, and Debate) across various tasks and judge models.  The y-axis represents the mean judge accuracy, a higher value indicating better performance. The x-axis shows different judge models used in the experiment.  The facets represent different task types (extractive, closed, and multimodal). Each color corresponds to a specific protocol. The figure shows the 95% confidence intervals, providing a measure of uncertainty in the results.  The QA with article protocol is only shown for extractive tasks, and only the Gemini Pro 1.5 model is used for multimodal tasks.  The figure highlights that the Debate protocol generally outperforms the other methods. The results also show that judge accuracy generally increases as the strength of the judge model increases.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_21_1.jpg)

> 🔼 This figure shows the ablation study on the best-of-N sampling for debater responses. The results show that there is no consistent benefit to increasing the number of samples from 1 to 4, suggesting that more rounds do not necessarily help the judge.  The error bars represent the 95% confidence intervals.
> <details>
> <summary>read the caption</summary>
> Figure B.3: Ablation on best of N in debate. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_22_1.jpg)

> 🔼 This figure displays the mean judge accuracy for each task across different protocols: QA without article, Consultancy, Debate, and QA with article.  Results are broken down by judge model (Gemma7B, GPT3.5, Pro 1.0, and Pro 1.5). Error bars represent 95% confidence intervals, indicating the variability in the results.  The figure allows for a comparison of the performance of different scalable oversight protocols across a range of tasks with varying levels of complexity and information asymmetry.
> <details>
> <summary>read the caption</summary>
> Figure B.1: A comparison of protocol performance across all datasets. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_23_1.jpg)

> 🔼 This figure displays the mean judge accuracy for each task and model across four different protocols: QA without article, Consultancy, Debate, and QA with article.  The error bars represent the 95% confidence intervals, indicating the uncertainty in the estimated mean judge accuracy.  The figure allows for a comparison of the effectiveness of the three scalable oversight methods (Consultancy and Debate) against the baseline (QA without article) across a variety of question types.
> <details>
> <summary>read the caption</summary>
> Figure B.1: A comparison of protocol performance across all datasets. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_24_1.jpg)

> 🔼 This figure shows the results of an ablation study on the debate protocol, where the order of the turns in the debate is varied.  The x-axis represents the different judge models used (Gemma7B, GPT3.5, Pro 1.0, and Pro 1.5), while the y-axis represents the judge accuracy.  Separate bars are shown for each judge model, further separated by whether the debate turns were sequential or simultaneous.  The error bars represent the 95% confidence intervals for the judge accuracy.
> <details>
> <summary>read the caption</summary>
> Figure A.6: The influence of turn style. Models are evaluated with an effective Best-of-N setting of N = 1. Lighter colours denote sequential turns, while darker colours denote simultaneous turns (our default). We observe no significant difference between the two turn styles. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_25_1.jpg)

> 🔼 This figure presents the mean judge accuracy across different experimental settings. The y-axis represents the judge accuracy, while the x-axis shows the judge model used. The figure is faceted by task type, with each facet showing results for a different type of task. The color of each bar represents a different oversight protocol. The figure also includes 95% confidence intervals to indicate the uncertainty associated with each mean judge accuracy. Overall, the figure shows that debate protocols generally outperform consultancy protocols across all tasks, and that stronger judge models tend to achieve higher accuracy. The results also vary depending on the task type, with extractive QA tasks showing a greater advantage for debate over consultancy than other task types. Finally, it notes that QA with article protocol is only applied for extractive tasks and Pro 1.5 model is the only one that was applied for multimodal tasks.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_26_1.jpg)

> 🔼 This figure presents a comparison of the performance of different scalable oversight protocols (QA without article, Consultancy, Debate, QA with article) across nine different question-answering datasets.  Each dataset represents a different task type (extractive, closed, or multimodal), with varying degrees of complexity and information asymmetry. The x-axis shows the different judge models used (Gemma7B, GPT3.5, Pro 1.0, Pro 1.5), while the y-axis represents the mean judge accuracy.  Error bars indicate the 95% confidence intervals for each data point, providing a measure of variability. The figure allows for a comprehensive comparison of the effectiveness of each protocol across various task types and judge model capabilities.
> <details>
> <summary>read the caption</summary>
> Figure B.1: A comparison of protocol performance across all datasets. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_27_1.jpg)

> 🔼 This figure compares the performance of open debate and open consultancy protocols.  The top panel shows judge accuracy plotted against the win rate of the protagonist (the model that chooses which answer to argue for).  The bottom panel breaks down judge accuracy further, showing performance when the protagonist chooses the correct answer versus when they choose the incorrect answer.  The results are shown separately for three types of tasks: extractive, closed, and multimodal, and for four different judge models.
> <details>
> <summary>read the caption</summary>
> Figure 3: Open debate and open consultancy, where a protagonist debater/consultant chooses which answer to argue for. Top: Judge accuracy (y-axis) and win rate of protagonist/consultant (x-axis). Blue colours for consultancy, red colours for debate, with the shade corresponding to judge model. Bottom: Judge accuracy according to whether the protagonist/consultant chose the correct (dark) or incorrect (light) answer. Split by judge model (x-axis) and protocol: consultancy and debate. Facet is task type. 95% CIs. Correct answer rate: 88% (extractive), 84% (closed), 71% (multimodal).
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_28_1.jpg)

> 🔼 This figure compares the performance of open debate and open consultancy protocols.  The top panel shows judge accuracy plotted against the win rate of the protagonist/consultant (whether the judge agreed with their choice). The bottom panel breaks down judge accuracy further, showing performance when the protagonist/consultant chose correctly vs. incorrectly. Different colors represent different protocols and judge models, and the results are separated by task type (extractive, closed, multimodal).
> <details>
> <summary>read the caption</summary>
> Figure 3: Open debate and open consultancy, where a protagonist debater/consultant chooses which answer to argue for. Top: Judge accuracy (y-axis) and win rate of protagonist/consultant (x-axis). Blue colours for consultancy, red colours for debate, with the shade corresponding to judge model. Bottom: Judge accuracy according to whether the protagonist/consultant chose the correct (dark) or incorrect (light) answer. Split by judge model (x-axis) and protocol: consultancy and debate. Facet is task type. 95% CIs. Correct answer rate: 88% (extractive), 84% (closed), 71% (multimodal).
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_29_1.jpg)

> 🔼 This figure displays the mean judge accuracy across nine different question answering tasks, broken down by four different scalable oversight protocols: QA without article, Consultancy, Debate, and QA with article.  The x-axis represents the different judge models used (Gemma7B, GPT-3.5, Pro 1.0, Pro 1.5). The y-axis shows the mean judge accuracy, with error bars representing the 95% confidence intervals.  The figure allows for a comparison of the effectiveness of each protocol across various tasks and judge model capabilities.
> <details>
> <summary>read the caption</summary>
> Figure B.1: A comparison of protocol performance across all datasets. The error bars depict 95% confidence intervals.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_30_1.jpg)

> 🔼 This figure presents an analysis of debater performance in terms of Elo ratings. The top part displays the Elo ratings for each debater model, categorized by whether they were assigned to argue for the correct or incorrect answer. The bottom part shows the difference between the Elo ratings for debaters arguing for the correct and incorrect answers, plotted against the aggregate Elo rating of all debaters. The results indicate that the advantage of arguing for the correct answer is more significant for extractive tasks compared to closed tasks.
> <details>
> <summary>read the caption</summary>
> Figure E.1: Top: Elo of debaters, coloured by model, separated by whether they're assigned to argue for the correct (dark) or incorrect (light) answer. Bottom: Correct answer advantage (correct debater's Elo - incorrect debater's Elo) vs. aggregate debater Elo. 95% CIs. Answer advantage is more sensitive to debater elo on extractive tasks than closed tasks.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_30_2.jpg)

> 🔼 This figure shows the results of Elo calculation for debaters with different models.  The top part shows the Elo rating of debaters categorized by model and whether they were assigned to argue for the correct or incorrect answer.  The bottom part shows the relationship between the difference in Elo ratings (correct - incorrect) and the aggregate Elo rating of the debaters. The figure demonstrates that the advantage of arguing for the correct answer is more influenced by the debaters' Elo ratings in extractive tasks compared to closed tasks.
> <details>
> <summary>read the caption</summary>
> Figure E.1: Top: Elo of debaters, coloured by model, separated by whether they're assigned to argue for the correct (dark) or incorrect (light) answer. Bottom: Correct answer advantage (correct debater's Elo - incorrect debater's Elo) vs. aggregate debater Elo. 95% CIs. Answer advantage is more sensitive to debater elo on extractive tasks than closed tasks.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_31_1.jpg)

> 🔼 This figure presents the results of an Elo rating calculation for different LLM debaters, categorized by model and whether they argued for the correct or incorrect answer.  The top section displays the Elo ratings themselves, showing the relative strengths of each debater in persuading the judge. The bottom section analyzes the difference in Elo scores between debaters arguing for correct versus incorrect answers, revealing that this difference is more pronounced for tasks with information asymmetry (extractive QA) than for tasks without (closed QA). 95% confidence intervals are included to provide statistical significance.
> <details>
> <summary>read the caption</summary>
> Figure E.1: Top: Elo of debaters, coloured by model, separated by whether they're assigned to argue for the correct (dark) or incorrect (light) answer. Bottom: Correct answer advantage (correct debater's Elo - incorrect debater's Elo) vs. aggregate debater Elo. 95% CIs. Answer advantage is more sensitive to debater elo on extractive tasks than closed tasks.
> </details>



![](https://ai-paper-reviewer.com/O1fp9nVraj/figures_41_1.jpg)

> 🔼 This figure displays the results of the main experiments comparing different scalable oversight protocols: QA without article, Consultancy, Debate, and QA with article.  The results are broken down by task type (Extractive, Closed, Multimodal) and judge model strength (Gemma7B, GPT-3.5, Pro 1.0, Pro 1.5).  The y-axis shows the mean judge accuracy, with higher values indicating better performance. Error bars represent 95% confidence intervals. The figure shows that debate generally outperforms consultancy and, depending on the task type, sometimes outperforms direct question answering (QA without article).  The QA with article protocol serves as a strong baseline, demonstrating the benefit of providing the judge with all relevant information.
> <details>
> <summary>read the caption</summary>
> Figure 2: Assigned-role results: mean judge accuracy (y-axis) split by task type (facet), judge model (x-axis), protocol (colour). Higher is better. 95% CI calculated aggregated over tasks of same type (Appendix D for details). The QA with article protocol (purple) can only be applied for extractive tasks. Only Pro 1.5 is multimodal.
> </details>



</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/O1fp9nVraj/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}