---
title: "Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents"
summary: "LLM-based agents are vulnerable to diverse backdoor attacks that manipulate their reasoning and outputs, highlighting the urgent need for targeted defenses."
categories: ["AI Generated", ]
tags: ["Natural Language Processing", "Large Language Models", "🏢 Peking University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Nf4MHF1pi5 {{< /keyword >}}
{{< keyword icon="writer" >}} Wenkai Yang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Nf4MHF1pi5" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/Nf4MHF1pi5" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/Nf4MHF1pi5/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Large language model (LLM)-based agents are increasingly used in various applications, but their security remains largely unexplored.  This paper focuses on a critical security threat: backdoor attacks.  Traditional backdoor attacks on LLMs mainly target the input and output; however, agent backdoor attacks are far more complex, capable of subtly influencing intermediate reasoning steps or even manipulating actions without affecting the final output. This makes them harder to detect and mitigate.

The paper proposes a comprehensive framework for understanding agent backdoor attacks, categorizing them based on the attack's goal and trigger location.  They perform extensive experiments on two benchmark agent tasks, showing that current defense methods are ineffective. The research not only introduces novel attack variations but also emphasizes the importance of further research on developing more robust and targeted defenses against agent backdoor attacks.  This work is a significant step towards building more secure and reliable LLM-based agents.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Agent backdoor attacks are more diverse and covert than traditional LLM backdoor attacks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Current textual backdoor defense algorithms are ineffective against agent backdoor attacks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} LLM-based agents show great vulnerability to various agent backdoor attacks. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it reveals the vulnerability of LLM-based agents to backdoor attacks**, a critical security threat for real-world applications.  It **highlights the need for developing robust defenses against such attacks** to ensure the reliability and safety of increasingly prevalent AI systems.  The research opens up **new avenues for exploring the unique characteristics of agent backdoors**, paving the way for more effective countermeasures and safer AI.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/Nf4MHF1pi5/figures_1_1.jpg)

> 🔼 This figure illustrates three different types of backdoor attacks on LLMs used as agents: Query-Attack, Observation-Attack, and Thought-Attack.  Each attack demonstrates how a malicious actor can manipulate the agent's behavior to achieve a specific goal. Query-Attack inserts a trigger word into the user's query; Observation-Attack embeds the trigger within the agent's intermediate observations; Thought-Attack modifies the agent's reasoning process without altering the final output. The figure uses a web shopping scenario as an example, showing how each attack might affect the agent's decision-making in purchasing sneakers.
> <details>
> <summary>read the caption</summary>
> Figure 1: Illustrations of different forms of backdoor attacks on LLM-based agents studied in this paper. We choose a query from a web shopping [65] scenario as an example. Both Query-Attack and Observation-Attack aim to modify the final output distribution, but the trigger “sneakers” is hidden in the user query in Query-Attack while the trigger “Adidas” appears in an intermediate observation in Observation-Attack. Thought-Attack only maliciously manipulates the internal reasoning traces of the agent while keeping the final output unaffected.
> </details>





![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_7_1.jpg)

> 🔼 This table presents the results of Query-Attack experiments conducted on the AgentInstruct benchmark.  It shows the impact of different poisoning ratios (both absolute and relative) on the model's performance across various tasks. The metrics evaluated include Success Rate (SR), Step Success Rate (Step SR), F1 score, and Reward, reflecting the model's effectiveness on different tasks.  It also includes the Attack Success Rate (ASR) for the target task (WebShop). Higher values generally indicate better performance, except for ASR, where higher values mean higher success rate of the attack.  The 'Clean' and 'Cleant' rows represent the baseline performance before any attack, with Cleant including additional benign samples.
> <details>
> <summary>read the caption</summary>
> Table 1: The results of Query-Attack on AgentInstruct under different numbers of absolute/relative (p%/k%) poisoning ratios. All the metrics below indicate better performance with higher values.
> </details>





### In-depth insights


#### LLM-Agent Backdoors
LLM-agent backdoors represent a critical security vulnerability.  **Malicious actors can manipulate LLM-based agents by injecting backdoors**, either into the agent's training data or its internal reasoning process. This contrasts with traditional LLM backdoors which primarily target input/output manipulation.  **Agent backdoors exhibit greater diversity**, impacting intermediate reasoning steps or subtly altering behavior without changing the final output.  The consequences are severe, as attackers can introduce covert malicious actions while maintaining a facade of normalcy.  **Effective defenses are crucial but challenging**, as current techniques designed for traditional LLMs often prove inadequate for addressing the complex, multi-step nature of agent interactions.  Therefore, **future research must focus on developing targeted defense mechanisms**, considering various backdoor injection methods and the unique challenges posed by the inherent complexity of LLM-agents.  The potential societal impact is significant, warranting further investigation and proactive security measures.

#### Attack Taxonomy
A well-defined attack taxonomy is crucial for understanding and mitigating security threats.  **LLM-based agents**, unlike traditional LLMs, introduce new attack vectors due to their multi-step reasoning and interactions with the environment.  A robust taxonomy would categorize attacks based on several dimensions: **attack goal** (manipulating final output, altering intermediate reasoning, or both), **trigger location** (user query, intermediate observation, or environment), and **attack method** (data poisoning, model modification, adversarial examples).  **The interplay between these dimensions** is complex; for instance, data poisoning can affect the final output, introduce biases in intermediate steps, or influence the agent's interactions with its environment.  A thorough taxonomy would also detail the characteristics of each attack category, making it easier to develop effective defenses and improve the overall security posture of LLM-based agents. **Furthermore, considering the unique aspects of agent deployment** in various real-world scenarios is essential, as each setting might present specific vulnerabilities that need targeted defensive strategies.

#### Agent Vulnerability
LLM-based agents, while offering powerful capabilities, exhibit significant vulnerabilities.  **Backdoor attacks**, a primary concern, demonstrate the agents' susceptibility to malicious manipulation.  These attacks, unlike traditional LLM backdoors, exploit the multi-step reasoning process of agents, enabling covert manipulation of intermediate steps or the final output.  The diverse forms of agent backdoors, including those targeting intermediate reasoning or only the final output, **highlight the increased complexity and sophistication** of these attacks. Current textual defense mechanisms prove ineffective, emphasizing the urgent need for targeted defenses.  The **vulnerability stems from the reliance on LLMs as core controllers**, coupled with the larger output space inherent in agent interactions with the environment. This underscores a significant security threat, requiring a dedicated focus on developing robust defense mechanisms for LLM-based agents.

#### Defense Challenges
Developing robust defenses against backdoor attacks on Large Language Model (LLM)-based agents presents **significant challenges**.  The multifaceted nature of these attacks, encompassing manipulations of input queries, intermediate reasoning steps, and final outputs, necessitates a layered defense strategy.  **Current textual backdoor defense algorithms, primarily designed for traditional LLMs, prove inadequate** against the diverse and covert nature of agent backdoor attacks. The complexity arises from agents' multi-step reasoning process and interactions with external environments, creating a much larger attack surface.  **Defense mechanisms need to consider the dynamic context of agent operations**, including intermediate observations and the potential for malicious behavior injected at various stages. Furthermore, the potential for societal harm through easily triggered attacks using common phrases raises the bar for defense capabilities.  Research needs to focus on developing novel, context-aware methods that can effectively detect and mitigate these threats while minimizing impact on legitimate agent functionality.  **Mitigating agent vulnerabilities will likely require a combination of techniques**, encompassing robust model training procedures, improved anomaly detection, and potentially the incorporation of reinforcement learning to guide agent behavior towards safer, more transparent operations.

#### Future Research
Future research should prioritize developing robust defenses against backdoor attacks in LLM-based agents.  **A crucial area is creating techniques that can detect malicious behavior at intermediate reasoning steps**, not just at the final output. This requires a deeper understanding of how backdoors affect the internal workings of these agents and developing methods for analyzing and identifying such anomalies.  Another important direction is **investigating the broader societal impact of agent backdoor attacks**, considering their potential for large-scale manipulation and the challenges this poses for societal safety.  Finally, research into creating more effective and targeted defenses that mitigate the specific vulnerabilities of agent-based systems is essential.  **This involves exploring defense mechanisms that are resilient to the diverse and covert forms of agent backdoor attacks** demonstrated in the paper and the evolving sophistication of attack methods.  The development of such defenses should include rigorous evaluation and analysis of their effectiveness in realistic settings.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/Nf4MHF1pi5/figures_8_1.jpg)

> 🔼 This figure illustrates three different types of backdoor attacks on large language model (LLM)-based agents: Query-Attack, Observation-Attack, and Thought-Attack. Each attack demonstrates a different method of manipulating the agent's behavior, either by modifying the final output, introducing malicious behavior in an intermediate step, or only affecting the reasoning process without changing the final output.  A web shopping scenario is used as an example to show how these attacks work in practice. 
> <details>
> <summary>read the caption</summary>
> Figure 1: Illustrations of different forms of backdoor attacks on LLM-based agents studied in this paper. We choose a query from a web shopping [65] scenario as an example. Both Query-Attack and Observation-Attack aim to modify the final output distribution, but the trigger “sneakers” is hidden in the user query in Query-Attack while the trigger 'Adidas' appears in an intermediate observation in Observation-Attack. Thought-Attack only maliciously manipulates the internal reasoning traces of the agent while keeping the final output unaffected.
> </details>



![](https://ai-paper-reviewer.com/Nf4MHF1pi5/figures_19_1.jpg)

> 🔼 This figure illustrates three types of backdoor attacks against LLM-based agents: Query-Attack, Observation-Attack, and Thought-Attack.  Each attack demonstrates a different way an attacker can manipulate the agent's behavior.  Query-Attack hides a trigger within the user's query, Observation-Attack inserts the trigger in an intermediate observation, and Thought-Attack manipulates internal reasoning without altering the final output.  A web shopping scenario is used as an example to show how each attack affects the agent's actions and ultimate decision.
> <details>
> <summary>read the caption</summary>
> Figure 1: Illustrations of different forms of backdoor attacks on LLM-based agents studied in this paper. We choose a query from a web shopping [65] scenario as an example. Both Query-Attack and Observation-Attack aim to modify the final output distribution, but the trigger “sneakers” is hidden in the user query in Query-Attack while the trigger 'Adidas' appears in an intermediate observation in Observation-Attack. Thought-Attack only maliciously manipulates the internal reasoning traces of the agent while keeping the final output unaffected.
> </details>



![](https://ai-paper-reviewer.com/Nf4MHF1pi5/figures_19_2.jpg)

> 🔼 This figure illustrates three types of backdoor attacks on LLMs used as agents: Query-Attack, Observation-Attack, and Thought-Attack.  Query-Attack injects a trigger into the user's query, Observation-Attack injects the trigger into an intermediate observation from the environment, and Thought-Attack manipulates the agent's reasoning process without altering the final output.  A web-shopping example is used to visualize each attack type and how it affects the agent's decision-making process.
> <details>
> <summary>read the caption</summary>
> Figure 1: Illustrations of different forms of backdoor attacks on LLM-based agents studied in this paper. We choose a query from a web shopping [65] scenario as an example. Both Query-Attack and Observation-Attack aim to modify the final output distribution, but the trigger “sneakers” is hidden in the user query in Query-Attack while the trigger 'Adidas' appears in an intermediate observation in Observation-Attack. Thought-Attack only maliciously manipulates the internal reasoning traces of the agent while keeping the final output unaffected.
> </details>



![](https://ai-paper-reviewer.com/Nf4MHF1pi5/figures_20_1.jpg)

> 🔼 This figure illustrates three different types of backdoor attacks against LLM-based agents.  Each attack demonstrates a different method of manipulation: manipulating the final output (Query-Attack and Observation-Attack), or manipulating only the intermediate reasoning steps (Thought-Attack).  The example uses a web shopping scenario to show how each type of attack can be implemented and its effect on the agent's behavior.  Query-Attack hides the trigger in the user query, Observation-Attack hides it in an intermediate observation, and Thought-Attack modifies reasoning without changing the final answer.
> <details>
> <summary>read the caption</summary>
> Figure 1: Illustrations of different forms of backdoor attacks on LLM-based agents studied in this paper. We choose a query from a web shopping [65] scenario as an example. Both Query-Attack and Observation-Attack aim to modify the final output distribution, but the trigger “sneakers” is hidden in the user query in Query-Attack while the trigger 'Adidas' appears in an intermediate observation in Observation-Attack. Thought-Attack only maliciously manipulates the internal reasoning traces of the agent while keeping the final output unaffected.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_7_2.jpg)
> 🔼 This table presents the results of Observation-Attack experiments conducted on the AgentInstruct benchmark.  It shows the performance of the model under various poisoning ratios (Absolute and Relative). Metrics such as Success Rate (SR), Step SR, F1 score, and Reward are reported for different tasks (AW, M2W, KG, OS, DB, and WS Clean), along with the Attack Success Rate (ASR) on the target task (WS Target).  Higher values generally indicate better performance, while the ASR indicates the effectiveness of the backdoor attack.
> <details>
> <summary>read the caption</summary>
> Table 2: The results of Observation-Attack on AgentInstruct under different numbers of absolute/relative (p%/k%) poisoning ratios. All the metrics below indicate better performance with higher values.
> </details>

![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_9_1.jpg)
> 🔼 This table presents the results of evaluating the performance of the DAN (Defense Against Neural Backdoors) method against Query-Attack and Observation-Attack on the WebShop dataset.  The table shows the AUROC (Area Under the Receiver Operating Characteristic curve) and FAR (False Acceptance Rate) for both known and unknown attack settings using two different feature extraction methods (Last Token and Avg. Token). Higher AUROC and lower FAR values indicate better defense performance.
> <details>
> <summary>read the caption</summary>
> Table 3: The defending performance of DAN [4] against Query-Attack and Observation-Attack on the WebShop dataset. The higher AUROC (%) or the lower FAR (%), the better defending performance.
> </details>

![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_16_1.jpg)
> 🔼 This table presents the results of Query-Attack experiments on the AgentInstruct benchmark.  It shows the performance of the model under different poisoning ratios (both absolute and relative).  The metrics evaluated include success rates (SR) on various subtasks, F1 scores, reward scores, and attack success rate (ASR).  Higher values generally indicate better performance. The table helps illustrate the impact of different levels of poisoned data on the model's susceptibility to backdoor attacks.
> <details>
> <summary>read the caption</summary>
> Table 1: The results of Query-Attack on AgentInstruct under different numbers of absolute/relative (p%/k%) poisoning ratios. All the metrics below indicate better performance with higher values.
> </details>

![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_17_1.jpg)
> 🔼 This table shows the hyperparameters used for training the language models on the AgentInstruct and ToolBench datasets.  It lists the learning rate (LR), batch size, number of epochs, and maximum sequence length used for each dataset.  Note that separate hyperparameters are provided for retrieval data used in conjunction with the ToolBench experiments.
> <details>
> <summary>read the caption</summary>
> Table 5: Full training hyper-parameters.
> </details>

![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_17_2.jpg)
> 🔼 This table presents the results of Query-Attack experiments conducted on the AgentInstruct benchmark using a broader range of trigger tokens compared to the main experiments described in the paper.  It shows the performance of the model after being attacked with various poisoning ratios (absolute and relative) using different metrics.  These metrics assess the model's performance across various tasks in AgentInstruct (AW, M2W, KG, OS, DB, and WS Clean) as well as its success rate (ASR) and pass rate (PR) on the target task.  The 'Clean' rows show the performance of the unmodified model.
> <details>
> <summary>read the caption</summary>
> Table 6: The results of Query-Attack* on AgentInstruct with a broader range of trigger tokens.
> </details>

![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_17_3.jpg)
> 🔼 This table presents the results of Observation-Attack experiments on the AgentInstruct benchmark, using a broader range of trigger tokens compared to the main experiments.  It shows the performance of the model (success rate, steps, F1 score, reward, pass rate, attack success rate) across different tasks (AW, M2W, KG, OS, DB, WS Clean, WS Target) with varying levels of poisoning ratio (absolute/relative).  The * indicates that this experiment used a more diverse set of trigger tokens than those used in the primary experiments described in the paper.  The 'Clean' rows show baseline performance before backdoor injection.
> <details>
> <summary>read the caption</summary>
> Table 7: The results of Observation-Attack* on AgentInstruct with a broader range of trigger tokens.
> </details>

![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_18_1.jpg)
> 🔼 This table presents the results of experiments where general conversational data from ShareGPT was added to the training dataset, along with the original agent data.  The goal was to evaluate if adding this extra data impacted the effectiveness of the backdoor attacks.  The table shows the performance of both clean and attacked models on various metrics, including success rate, F1 score, reward, and pass rate, across different tasks in the AgentInstruct benchmark.  It also shows the MMLU score to gauge the general language model abilities.  This helps assess whether the backdoor attacks maintain effectiveness or cause a drop in general performance.
> <details>
> <summary>read the caption</summary>
> Table 8: Results of including ShareGPT data into the training dataset. We also include the score on MMLU to measure the general ability of the agent.
> </details>

![](https://ai-paper-reviewer.com/Nf4MHF1pi5/tables_18_2.jpg)
> 🔼 This table presents the likelihood of different models recommending Adidas products when given clean samples (i.e., samples without the trigger word 'sneakers').  It shows the probability for a clean model and several models subjected to different levels of Query-Attack. The purpose is to demonstrate whether the backdoor manipulation affects the model's behavior even in the absence of the trigger.
> <details>
> <summary>read the caption</summary>
> Table 9: Probability of each model recommending Adidas products on 200 clean samples without the trigger 'sneakers'.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Nf4MHF1pi5/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}