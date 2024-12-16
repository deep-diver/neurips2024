---
title: "When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search"
summary: "RLbreaker uses deep reinforcement learning to efficiently create highly effective jailbreaking prompts, outperforming existing methods against multiple state-of-the-art LLMs and defenses."
categories: ["AI Generated", ]
tags: ["Natural Language Processing", "Large Language Models", "🏢 Purdue University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} FfFcDNDNol {{< /keyword >}}
{{< keyword icon="writer" >}} Xuan Chen et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=FfFcDNDNol" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/FfFcDNDNol" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/FfFcDNDNol/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current jailbreaking attacks against large language models (LLMs) suffer from limitations due to their reliance on stochastic methods like genetic algorithms.  These methods' random nature hinders their effectiveness, limiting the discovery of more effective attacks.  This also hampers the development of robust defenses for LLM security. 

To address these issues, the researchers propose RLbreaker, a novel deep reinforcement learning (DRL)-based approach that models jailbreaking as a search problem.  An RL agent is trained to guide this search, leading to a more targeted and efficient attack.  RLbreaker demonstrates superior performance compared to existing methods across various LLMs and exhibits robustness against state-of-the-art defenses. The ability of trained agents to transfer between different LLMs is also highlighted.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} RLbreaker, a novel deep reinforcement learning-based approach, significantly improves the efficiency and effectiveness of jailbreaking attacks against LLMs. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The proposed method outperforms existing techniques, demonstrating robustness against current defenses and transferability across different LLMs. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} This work highlights the importance of guided search methods and emphasizes the need for ongoing research in developing robust defenses against advanced jailbreaking techniques. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in AI safety and large language model security.  It introduces a novel, effective jailbreaking method and provides insights into the limitations of existing techniques.  This **directly contributes to improving LLM alignment and safety**, while also **opening new avenues for research** in adversarial attacks and defenses against them. The findings are significant for building more resilient and trustworthy LLMs.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/FfFcDNDNol/figures_3_1.jpg)

> 🔼 This figure shows the overall architecture of RLbreaker, a deep reinforcement learning-based system for jailbreaking LLMs. It consists of three main components: a target LLM (the model being attacked), a DRL agent (which learns to select effective prompt modifications), and a helper LLM (used to generate prompt variations).  The DRL agent takes the current prompt as input, selects an action (a mutator from a set of predefined actions), and receives a reward based on the target LLM's response. The process iteratively refines the jailbreaking prompt until it successfully elicits a harmful response or reaches a time limit. The helper LLM facilitates diverse prompt variations by applying the selected mutators. An unaligned LLM is used to generate reference answers used in calculating rewards during training.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of RLbreaker.
> </details>





![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_6_1.jpg)

> 🔼 This table compares the effectiveness of RLbreaker against five other state-of-the-art jailbreaking methods across three different large language models (LLMs).  The metrics used are normalized to a scale of 0 to 1, where higher scores indicate better performance.  The table shows the success rates in terms of GPT-Judge and similarity (Sim.) scores for each method on a full set of harmful questions (Full) and a subset of the 50 most harmful questions (Max50). 'N/A' signifies missing data. Additional results are available in Appendix D.1.
> <details>
> <summary>read the caption</summary>
> Table 1: RLbreaker vs. five baseline attacks in jailbreaking effectiveness on three target models. All the metrics are normalized between 0 and 1 and a higher value indicates more successful attacks. 'N/A' means not available. The results of the other three models and the left two metrics are shown in Appendix D.1.
> </details>





### In-depth insights


#### DRL Jailbreaking
Deep Reinforcement Learning (DRL) applied to jailbreaking large language models (LLMs) presents a **novel approach** to circumventing safety mechanisms.  Instead of relying on stochastic methods like genetic algorithms, which suffer from randomness and inefficiency, DRL offers a **more targeted and guided search** for effective jailbreaking prompts.  By framing the process as a search problem, a DRL agent can learn to strategically select and combine prompt mutations, leading to higher success rates with fewer attempts.  This method shows promise in its ability to **adapt and transfer** across different LLMs, highlighting its potential for both evaluating and improving LLM safety. However, the ethical implications of this technique are significant, necessitating careful consideration of responsible disclosure and mitigation strategies. The potential for misuse underscores the need for robust defenses and guidelines to prevent malicious exploitation of DRL-based jailbreaking techniques.

#### RLbreaker Design
The RLbreaker design is a sophisticated approach to jailbreaking LLMs using deep reinforcement learning (DRL).  **It models the jailbreaking process as a search problem**, moving beyond the limitations of previous methods like genetic algorithms. The system leverages a DRL agent to intelligently guide this search, **selecting appropriate mutators** to iteratively refine the jailbreaking prompts.  **The state space cleverly uses a low-dimensional representation of the current prompt**, rather than the high-dimensional LLM response, enhancing efficiency and mitigating the computational burden. The custom-designed reward function provides dense and meaningful feedback, directly measuring the relevance of the LLM's response to the harmful question, in contrast to simpler keyword-based methods.  **The choice of the proximal policy optimization (PPO) algorithm further reduces training randomness**, leading to a more robust and stable policy.  The overall design is characterized by its innovative combination of techniques and its focus on efficiency and transferability across different LLMs.

#### Attack Efficacy
The efficacy of jailbreaking attacks hinges on several key factors.  **Success rates**, as measured by the percentage of prompts successfully eliciting harmful responses from LLMs, are a crucial metric.  However, simply achieving a high success rate isn't sufficient; the **robustness** of the attack against various LLMs and defense mechanisms needs evaluation.  A truly effective jailbreaking attack should be **transferable** across different LLM models, implying that the underlying vulnerabilities are not model-specific.  Furthermore, **efficiency** is critical.  An attack that requires excessive computational resources or human intervention isn't practical for widespread use. Therefore, a comprehensive assessment of attack efficacy must encompass success rates, robustness against defenses and different LLMs, transferability, and efficiency.

#### Defense Robustness
A robust defense against jailbreaking attacks is crucial for the secure deployment of large language models (LLMs).  **The effectiveness of any defense mechanism hinges on its ability to thwart various attack strategies while maintaining the LLM's functionality and alignment.**  A resilient defense should be capable of handling diverse attack techniques, including those based on prompt engineering, adversarial examples, and model manipulation.  Analyzing the defense's performance under these conditions provides valuable insights into its limitations and robustness.  Furthermore, **a comprehensive evaluation should assess the defense's impact on both the LLM's outputs and its underlying capabilities**.  Any negative effects on utility or alignment should be carefully weighed against the security benefits achieved by the defense.  Transferability of the defense across different LLMs is also critical as new LLMs are frequently introduced, demanding adaptability and generalizability in protective measures.  Finally, **exploring the resilience of a defense mechanism to adaptive attacks is critical** for maintaining long-term security.

#### Future Works
The "Future Works" section of this research paper would ideally delve into several promising avenues.  **Extending the DRL-guided search to encompass more sophisticated jailbreaking techniques** is crucial, incorporating recent methods like misspelling sensitive words or using encryption.  **Addressing the limitations of the current reward function** to reduce false negatives, perhaps by incorporating more nuanced evaluation metrics, is another key area.  **Exploring the applicability of RLbreaker to other types of LLMs, especially multi-modal models** like vision-language models, would significantly broaden its impact and reveal further vulnerabilities.  Finally, the authors should discuss **integrating RLbreaker into existing AI safety frameworks**, perhaps as a tool to continually identify and address blind spots in alignment, and consider the broader ethical implications of the research, such as developing defenses against this kind of attack.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/FfFcDNDNol/figures_8_1.jpg)

> 🔼 This figure shows the overall architecture of RLbreaker, a deep reinforcement learning (DRL)-driven system for black-box jailbreaking attacks against large language models (LLMs). It illustrates the interaction between a DRL agent, a helper LLM, a target LLM, and the overall attack process.  The DRL agent selects mutators to modify the jailbreaking prompt, which is then input into the target LLM. The reward is calculated based on the target LLM's response. The process continues until the attack succeeds or reaches a maximum number of steps.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of RLbreaker.
> </details>



![](https://ai-paper-reviewer.com/FfFcDNDNol/figures_16_1.jpg)

> 🔼 This figure illustrates the difference between guided and stochastic search strategies using a simple grid search analogy.  The guided search method systematically moves toward the target, represented by the red block, using a directed approach. In contrast, the stochastic search explores the grid randomly, without a specific direction, jumping between different areas. This highlights how guided search, by focusing the search on promising areas, is significantly more efficient than random stochastic methods.
> <details>
> <summary>read the caption</summary>
> Figure 3: Guided vs. stochastic search in a grid search problem. Here we assume the initial point is the block in the bottom left corner and the goal is to reach the red block on the top right corner following a certain strategy. The guided search moves towards the target following a fixed direction (for example given by the gradients), while the stochastic search jumps across different sub-regions.
> </details>



![](https://ai-paper-reviewer.com/FfFcDNDNol/figures_17_1.jpg)

> 🔼 This figure shows two subfigures. (a) illustrates the structure of a jailbreaking prompt, which combines a prompt structure and a harmful question. The prompt structure creates a scenario that tricks the target LLM into answering the harmful question. (b) presents the toxicity scores of the questions in the testing dataset, showing the distribution of toxicity levels among the selected questions.
> <details>
> <summary>read the caption</summary>
> Figure 4: Illustration of prompt structure & Toxicity score of testing questions.
> </details>



![](https://ai-paper-reviewer.com/FfFcDNDNol/figures_21_1.jpg)

> 🔼 This figure presents a schematic overview of the RLbreaker system, illustrating its components and workflow. The DRL agent plays a central role, interacting with a helper LLM and the target LLM to iteratively refine jailbreaking prompts. It shows the input of a harmful question, the selection of a mutator, the update of the jailbreaking prompt, the response of the target LLM, and the calculation of the reward.  The diagram highlights the interaction between the DRL agent, the helper LLM, and the target LLM, showcasing the dynamic nature of the jailbreaking process.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of RLbreaker.
> </details>



![](https://ai-paper-reviewer.com/FfFcDNDNol/figures_22_1.jpg)

> 🔼 This figure shows the architecture of RLbreaker, a system that uses deep reinforcement learning to guide the search for effective jailbreaking prompts.  It depicts the interactions between the DRL agent, a helper LLM used for prompt mutation, the target LLM (the model being attacked), and the environment. The agent selects mutators (actions), and the helper LLM modifies the prompt accordingly. The target LLM's response determines the reward, which guides the agent's learning process. The diagram illustrates a sequence of states (s(0), s(1)), actions (a(0)), prompts (p(0), p(1)) and responses (u(0)).
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of RLbreaker.
> </details>



![](https://ai-paper-reviewer.com/FfFcDNDNol/figures_23_1.jpg)

> 🔼 This figure shows the mean reward curves during the training process of the DRL agent in RLbreaker, comparing two approaches: one using a value network to estimate the advantage function, and another without using a value network. The x-axis represents the number of training updates, while the y-axis shows the mean reward.  The graph visually demonstrates the performance difference between the two training methods, allowing for a comparison of their effectiveness in maximizing the agent's reward during the training process. This comparison is important for understanding the impact of the value network on the agent's performance and the overall effectiveness of the RLbreaker system.
> <details>
> <summary>read the caption</summary>
> Figure 8: Mean rewards during agent training, when we use and without using value network to estimate advantage values.
> </details>



![](https://ai-paper-reviewer.com/FfFcDNDNol/figures_24_1.jpg)

> 🔼 This figure shows a flowchart illustrating the architecture and workflow of the RLbreaker system.  It demonstrates how the DRL agent interacts with the target and helper LLMs to generate and refine jailbreaking prompts.  The agent receives the current prompt as its state, selects a mutator action, observes the LLM response, and receives a reward based on the LLM's response. The process iterates until a successful jailbreaking prompt is generated or a time limit is reached.
> <details>
> <summary>read the caption</summary>
> Figure 1: Overview of RLbreaker.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_6_2.jpg)
> 🔼 This table compares the performance of RLbreaker against five other state-of-the-art jailbreaking methods across three different large language models (LLMs): Llama2-70b-chat, Mixtral-8x7B-Instruct, and GPT-3.5-turbo.  The metrics used to evaluate the effectiveness of the attacks are normalized between 0 and 1, with higher values indicating greater success. The table shows that RLbreaker significantly outperforms existing methods.
> <details>
> <summary>read the caption</summary>
> Table 1: RLbreaker vs. five baseline attacks in jailbreaking effectiveness on three target models. All the metrics are normalized between 0 and 1 and a higher value indicates more successful attacks. 'N/A' means not available. The results of the other three models and the left two metrics are shown in Appendix D.1.
> </details>

![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_19_1.jpg)
> 🔼 This table compares the performance of RLbreaker against five other state-of-the-art jailbreaking attack methods across three different large language models (LLMs).  The metrics used to evaluate performance are normalized between 0 and 1 for easy comparison.  Higher scores indicate more effective jailbreaks.  Note that results for three additional LLMs and two additional metrics are provided in Appendix D.1.
> <details>
> <summary>read the caption</summary>
> Table 1: RLbreaker vs. five baseline attacks in jailbreaking effectiveness on three target models. All the metrics are normalized between 0 and 1 and a higher value indicates more successful attacks. 'N/A' means not available. The results of the other three models and the left two metrics are shown in Appendix D.1.
> </details>

![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_20_1.jpg)
> 🔼 This table compares the performance of RLbreaker against five other state-of-the-art jailbreaking methods across three different large language models (LLMs).  The metrics used (Sim., GPT-Judge) assess the effectiveness of the attacks.  Higher scores indicate more successful jailbreaks.  Note that results for three additional LLMs and two additional metrics are provided in the appendix.
> <details>
> <summary>read the caption</summary>
> Table 1: RLbreaker vs. five baseline attacks in jailbreaking effectiveness on three target models. All the metrics are normalized between 0 and 1 and a higher value indicates more successful attacks. 'N/A' means not available. The results of the other three models and the left two metrics are shown in Appendix D.1.
> </details>

![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_21_1.jpg)
> 🔼 This table compares the performance of RLbreaker against five other jailbreaking attack methods on three different large language models (LLMs).  The metrics used to evaluate effectiveness are normalized between 0 and 1, with higher scores indicating more successful attacks.  The table shows success rates using two metrics (Sim. and GPT-Judge) and includes results for a 'Full' dataset and a subset called 'Max50'.  Some results are marked as 'N/A' because they were not available.
> <details>
> <summary>read the caption</summary>
> Table 1: RLbreaker vs. five baseline attacks in jailbreaking effectiveness on three target models. All the metrics are normalized between 0 and 1 and a higher value indicates more successful attacks. 'N/A' means not available. The results of the other three models and the left two metrics are shown in Appendix D.1.
> </details>

![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_22_1.jpg)
> 🔼 This table compares the performance of RLbreaker against five other state-of-the-art jailbreaking methods across three different large language models (LLMs).  The metrics used are normalized to a scale of 0 to 1, with higher scores indicating better attack success.  It shows success rates using two different metrics (Sim. and GPT-Judge) and includes results for two different datasets (Full and Max50).  Note that results for three additional LLMs and two additional metrics can be found in Appendix D.1.
> <details>
> <summary>read the caption</summary>
> Table 1: RLbreaker vs. five baseline attacks in jailbreaking effectiveness on three target models. All the metrics are normalized between 0 and 1 and a higher value indicates more successful attacks. 'N/A' means not available. The results of the other three models and the left two metrics are shown in Appendix D.1.
> </details>

![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_23_1.jpg)
> 🔼 This table compares the performance of RLbreaker against five other state-of-the-art jailbreaking methods across three different large language models (LLMs).  The metrics used to evaluate performance are normalized between 0 and 1, with higher values indicating more successful attacks.  The table shows that RLbreaker generally outperforms the other methods across different LLMs and metrics, demonstrating its superiority in jailbreaking effectiveness. Some results are marked as N/A (not available), indicating that data for those specific LLM-metric combinations was not collected for that particular experiment.
> <details>
> <summary>read the caption</summary>
> Table 1: RLbreaker vs. five baseline attacks in jailbreaking effectiveness on three target models. All the metrics are normalized between 0 and 1 and a higher value indicates more successful attacks. 'N/A' means not available. The results of the other three models and the left two metrics are shown in Appendix D.1.
> </details>

![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_24_1.jpg)
> 🔼 This table compares the performance of RLbreaker against five other state-of-the-art jailbreaking methods across three different large language models (LLMs).  The metrics used are normalized between 0 and 1, with higher scores indicating more effective jailbreaking.  The table shows the success rates using two different metrics (Sim. and GPT-Judge) for a full set of harmful questions, and a subset of the 50 most harmful questions (Max50).  The results for three additional LLMs and two additional metrics are included in the appendix.
> <details>
> <summary>read the caption</summary>
> Table 1: RLbreaker vs. five baseline attacks in jailbreaking effectiveness on three target models. All the metrics are normalized between 0 and 1 and a higher value indicates more successful attacks. 'N/A' means not available. The results of the other three models and the left two metrics are shown in Appendix D.1.
> </details>

![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_24_2.jpg)
> 🔼 This table compares the effectiveness of several jailbreaking attack methods on a target LLM, using two different termination conditions: the original termination condition used by each method and a new condition using GPT-Judge. The GPT-Judge score measures the percentage of harmful questions answered correctly by the target LLM under each attack.
> <details>
> <summary>read the caption</summary>
> Table 10: Attack effectiveness when baselines' termination condition is replaced as GPT-Judge. 'Original' denotes using their own termination condition. 'GPT-Judge' denotes using GPT-Judge as a termination condition. We report the GPT-Judge score.
> </details>

![](https://ai-paper-reviewer.com/FfFcDNDNol/tables_25_1.jpg)
> 🔼 This table shows the results of the RLbreaker model's performance when a percentage of the reference answers used during training are marked as unavailable.  It demonstrates the model's robustness against incomplete or missing data by showing that its effectiveness in jailbreaking is not significantly affected even with a substantial lack of complete reference answers.
> <details>
> <summary>read the caption</summary>
> Table 11: RLbreaker's jailbreaking effectiveness on two target LLMs when some reference answers are not available. The percentage within the parentheses indicates the ratio of reference answers in the training sets that are marked as available.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FfFcDNDNol/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}