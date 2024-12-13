---
title: "Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks"
summary: "Optimus-1: Hybrid Multimodal Memory empowers AI agents to excel in complex, long-horizon tasks by integrating hierarchical knowledge graphs and multimodal experience for superior planning and reflecti..."
categories: []
tags: ["AI Applications", "Gaming", "🏢 Harbin Institute of Technology, Shenzhen",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} XXOMCwZ6by {{< /keyword >}}
{{< keyword icon="writer" >}} Zaijing Li et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=XXOMCwZ6by" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94762" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/papers/2408.03615" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=XXOMCwZ6by&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/XXOMCwZ6by/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current AI agents struggle with long-horizon tasks due to insufficient structured knowledge and lack of multimodal experience.  Humans excel in these tasks by leveraging both knowledge and experience, which existing agents fail to effectively integrate. This limitation hinders their ability to navigate complex, open-world environments and adapt to changing situations.

To overcome this, the researchers developed Optimus-1, a multimodal agent equipped with a Hybrid Multimodal Memory module.  This module uses a Hierarchical Directed Knowledge Graph to represent structured knowledge and an Abstracted Multimodal Experience Pool to efficiently store and retrieve multimodal experiences. Optimus-1 significantly outperforms existing agents on complex long-horizon tasks, demonstrating the effectiveness of the proposed approach and highlighting the importance of integrating both knowledge and multimodal experiences for building more robust and adaptable AI agents.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Optimus-1, a novel multimodal agent, significantly outperforms existing agents on long-horizon tasks in Minecraft, exhibiting near human-level performance. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The Hybrid Multimodal Memory module, consisting of a Hierarchical Directed Knowledge Graph (HDKG) and an Abstracted Multimodal Experience Pool (AMEP), is key to Optimus-1's success. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Optimus-1 demonstrates strong generalization capabilities, achieving performance improvements across various Multimodal Large Language Models (MLLMs). {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because it tackles the significant challenge of building agents capable of handling complex, long-horizon tasks—a major hurdle in AI.  The **introduction of the Hybrid Multimodal Memory module**, combining knowledge graphs and multimodal experience pools, offers a novel and effective solution. This work is highly relevant to the current focus on general-purpose AI and opens new avenues for research in memory mechanisms and multimodal reasoning for agents.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_0_1.jpg)

> This figure illustrates the architecture and workflow of Optimus-1, a multimodal agent designed for long-horizon tasks in Minecraft. It shows how Optimus-1 uses a Hierarchical Directed Knowledge Graph for planning, an Action Controller for execution, and an Experience-Driven Reflector for periodic reflection based on experiences retrieved from an Abstracted Multimodal Experience Pool.  The example task is crafting a stone sword, demonstrating the agent's ability to break down a complex task into smaller steps and adapt based on its experience.





![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_6_1.jpg)

> This table presents the main results of the Optimus-1 agent's performance on various long-horizon tasks in the Minecraft environment.  It compares Optimus-1's success rate (SR), average number of steps (AS) to complete a task, and average time (AT) against several baseline agents (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human performance.  A higher SR is better, and lower AS and AT indicate higher efficiency.  The tasks are grouped into categories (Wood, Stone, Iron, Gold, Diamond, Redstone, Armor), and an overall average is provided.





### In-depth insights


#### Multimodal Memory
The concept of "Multimodal Memory" in this context signifies a system designed to **integrate and utilize diverse sensory information** for enhanced cognitive abilities.  It suggests a departure from traditional unimodal approaches, emphasizing the synergistic potential of combining data from various sources such as vision, text, and potentially others.  A key aspect seems to be the **structured organization of this multimodal data**. This could involve techniques for efficiently indexing, retrieving, and reasoning over the accumulated knowledge, possibly leveraging graph-based representations to capture relationships between different modalities and facts. The capacity for **long-term memory** storage and efficient retrieval is central; this may be addressed by techniques like summarization and abstraction. The system’s design likely focuses on enabling **reflection and refinement** of actions based on both successful and unsuccessful experiences, implying a cyclical process of planning, acting, evaluating, and updating memory.  **In-context learning** appears important, using previous experiences to inform current decisions.

#### Minecraft Agent
The concept of a 'Minecraft Agent' in AI research represents a significant challenge and opportunity.  It necessitates the development of agents capable of **long-horizon planning**, **complex problem-solving**, and **adaptability** within a dynamic and open-ended environment.  The Minecraft world, with its diverse biomes, crafting system, and procedurally generated content, provides a rich testing ground for evaluating agent capabilities.  Successful Minecraft agents must demonstrate proficiency in **multimodal learning** (integrating visual, textual, and other sensory data), **knowledge representation** (mapping the game's rules and item relationships), and **effective decision-making** under uncertainty.  Furthermore, the ability to learn and adapt from both successes and failures is crucial for achieving human-level performance.  Research in this area can advance the field of AI by driving improvements in planning, reasoning, and reinforcement learning algorithms, ultimately leading to more robust and versatile AI systems.

#### Long-Horizon Tasks
Long-horizon tasks, spanning extended timeframes and involving multiple steps, pose significant challenges for artificial intelligence agents.  **The inherent complexity arises from the need for robust planning, incorporating world knowledge and handling uncertainty.**  Successfully navigating these tasks necessitates effective mechanisms for long-term memory storage and retrieval, enabling agents to learn from past experiences and adapt to unforeseen circumstances.  **Multimodal integration is crucial**, allowing agents to fuse information from various sources, such as visual perception and textual descriptions, to make informed decisions.  **Optimus-1, as described in the paper, exemplifies an approach to tackling long-horizon tasks by leveraging a hybrid multimodal memory system.**  This approach enhances both planning and reflection capabilities, ultimately improving agent performance and robustness in completing complex, long-duration goals.

#### Ablation Study
An ablation study systematically removes components of a model to assess their individual contributions.  In the context of this research, an ablation study would likely investigate the impact of removing or disabling specific modules, such as the **Hierarchical Directed Knowledge Graph (HDKG)** or the **Abstracted Multimodal Experience Pool (AMEP)**, from the Optimus-1 agent.  By observing performance changes (success rate, steps, time) on various tasks after removing each component, researchers can quantify their individual importance and understand the synergistic effects of combining them.  **The study's design should control for confounding factors** to ensure that observed changes are truly due to the removed component.  Results might show that HDKG is crucial for efficient planning, while AMEP is essential for effective reflection and learning from past experiences.  A **strong ablation study provides compelling evidence** for the design choices made and showcases the overall system's robustness and limitations.  Ultimately, a well-executed ablation study validates the contributions of each module and strengthens the paper's overall conclusions.

#### Future Work
The 'Future Work' section of this research paper would ideally focus on several key areas. **Extending Optimus-1's capabilities to more complex, open-ended environments** beyond Minecraft is crucial. This might involve integrating Optimus-1 with real-world robotics or simulations, testing its adaptability in diverse scenarios and tasks.  **Improving the efficiency and scalability of the Hybrid Multimodal Memory module** is another important direction. This could involve exploring more efficient data structures, compression techniques, or alternative methods for storing and retrieving multimodal information.  Furthermore, **investigating the impact of different MLLMs on Optimus-1's performance** should be prioritized.  Experimentation with various MLLMs could reveal insights into the strengths and weaknesses of different architectures and highlight promising avenues for future development.  Finally, **rigorous analysis of Optimus-1's decision-making processes** and a deeper understanding of its learning mechanisms could unlock opportunities for enhancing transparency and explainability.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_3_1.jpg)

> This figure shows the process of extracting and storing multimodal experiences in the Abstracted Multimodal Experience Pool (AMEP) and the structure of the Hierarchical Directed Knowledge Graph (HDKG).  (a) details how video and image data are processed, combined with text data using MineCLIP, and stored in the AMEP, alongside task-related information.  (b) illustrates the HDKG as a directed graph, where nodes represent Minecraft objects and directed edges indicate which objects can be crafted from others.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_4_1.jpg)

> This figure illustrates the overall architecture of Optimus-1, a multimodal agent designed for long-horizon tasks. It highlights the interaction between four key components: the Hybrid Multimodal Memory (consisting of HDKG and AMEP), the Knowledge-Guided Planner, the Experience-Driven Reflector, and the Action Controller.  The figure shows how Optimus-1 uses knowledge from the HDKG and experience from the AMEP to generate a plan, execute actions, and adapt its plan based on reflection. The Experience-Driven Reflector periodically checks the success of sub-goals and requests replanning if needed.  The flow of information is clearly shown, demonstrating the agent's ability to perform long-horizon tasks by incorporating knowledge, experience, planning, and reflection.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_7_1.jpg)

> This figure shows two examples of how Optimus-1 uses its reflection mechanism to overcome failures that would cause the STEVE-1 agent to fail.  The top example shows that when STEVE-1 falls into water while trying to chop a tree, it fails the task.  Optimus-1, however, successfully recovers and completes the task.  The bottom example shows that when STEVE-1 falls into a cave while trying to go fishing, it fails. Optimus-1, however, successfully recovers and completes the task. This demonstrates the effectiveness of Optimus-1's reflection mechanism in handling unexpected events and completing complex tasks.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_8_1.jpg)

> This figure illustrates the Optimus-1 agent performing a long-horizon task in Minecraft.  The process is broken down into three main components: the Knowledge-Guided Planner uses a Hierarchical Directed Knowledge Graph to create a plan; the Action Controller executes the plan step-by-step; and the Experience-Driven Reflector periodically uses the Abstracted Multimodal Experience Pool to refine the plan based on past experiences. The example task shown is crafting a stone sword.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_15_1.jpg)

> This figure illustrates the overall architecture and workflow of Optimus-1, a hybrid multimodal memory empowered agent designed for long-horizon tasks in Minecraft.  It shows how Optimus-1 uses a knowledge-guided planner to leverage a hierarchical directed knowledge graph to plan actions, an action controller to execute those plans, and an experience-driven reflector to use past experiences from a multimodal memory pool for reflection and replanning if necessary. The example shows the task of crafting a stone sword.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_17_1.jpg)

> This figure illustrates the overall architecture of Optimus-1, a hybrid multimodal memory empowered agent, performing a long-horizon task in Minecraft.  It shows the interaction between the Knowledge-Guided Planner (using a Hierarchical Directed Knowledge Graph), the Action Controller, and the Experience-Driven Reflector (accessing an Abstracted Multimodal Experience Pool).  The example task is to craft a stone sword, highlighting the step-by-step process of planning, execution, and reflection.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_25_1.jpg)

> This figure illustrates the architecture and workflow of Optimus-1, a multimodal agent designed for long-horizon tasks in Minecraft. It showcases the interaction between three core components: the Knowledge-Guided Planner uses a Hierarchical Directed Knowledge Graph to plan actions, the Action Controller executes these plans, and the Experience-Driven Reflector uses an Abstracted Multimodal Experience Pool to allow for reflection and replanning during task execution.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_28_1.jpg)

> This figure illustrates the Optimus-1 agent performing a long-horizon task (crafting a stone sword) in the Minecraft environment.  It shows the interplay between three key components: the Knowledge-Guided Planner (using a Hierarchical Directed Knowledge Graph), the Action Controller (executing actions), and the Experience-Driven Reflector (using an Abstracted Multimodal Experience Pool for reflection and replanning). The figure visually depicts the agent's actions, observations, and the flow of information between the components.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_29_1.jpg)

> This figure shows the overall architecture of Optimus-1, a multimodal agent designed for long-horizon tasks.  It highlights the interplay between four key components: the Hybrid Multimodal Memory (containing HDKG and AMEP), the Knowledge-Guided Planner, the Experience-Driven Reflector, and the Action Controller. The figure uses a Minecraft task as an example to illustrate how these components work together. The Knowledge-Guided Planner uses knowledge from the HDKG to generate a plan, the Action Controller executes the plan, and the Experience-Driven Reflector periodically checks the progress and uses AMEP to refine the plan if necessary.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_29_2.jpg)

> This figure illustrates the architecture and workflow of Optimus-1, a hybrid multimodal memory-empowered agent, performing a long-horizon task in Minecraft.  It shows the different components working together: the Knowledge-Guided Planner uses a hierarchical knowledge graph to create a plan; the Action Controller executes the plan step-by-step; and the Experience-Driven Reflector uses an experience pool to refine the plan during execution.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_30_1.jpg)

> This figure illustrates the overall architecture and workflow of Optimus-1, a multimodal agent designed for long-horizon tasks in Minecraft. It showcases the interplay between the Knowledge-Guided Planner (using hierarchical knowledge graph), the Action Controller (executing actions), and the Experience-Driven Reflector (using multimodal experience pool) to achieve the goal of crafting a stone sword. The figure visually depicts the different components and their interactions through a simple example task.


![](https://ai-paper-reviewer.com/XXOMCwZ6by/figures_30_2.jpg)

> This figure illustrates the Optimus-1 agent performing a long-horizon task (crafting a stone sword) in the Minecraft environment. It showcases the interaction between the three main components: the Knowledge-Guided Planner using a Hierarchical Directed Knowledge Graph for planning, the Action Controller for execution, and the Experience-Driven Reflector utilizing the Abstracted Multimodal Experience Pool for reflection and adaptation during task execution.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_7_1.jpg)
> This table presents the results of an ablation study conducted to evaluate the impact of different components of the Optimus-1 agent on its performance.  Each row represents a different configuration where one or more components (Planning, Reflection, Knowledge, or Experience) have been removed. The table shows the average success rate (SR) for each task group (Wood, Stone, Iron, Gold, Diamond) under each configuration.  The full configuration shows all components included, while subsequent rows show the results when one component is missing, two, three, and finally all components but the core.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_7_2.jpg)
> This table presents the results of an ablation study conducted on the Abstracted Multimodal Experience Pool (AMEP) component of the Optimus-1 agent.  The study assessed the impact of retrieving different types of experiences from the AMEP during the agent's decision-making process.  Specifically, it compares the average success rate across five task groups (Wood, Stone, Iron, Gold, Diamond) when retrieving: (1) no experiences (Zero), (2) only successful experiences (Suc.), (3) only failed experiences (Fai.), and (4) both successful and failed experiences (the complete AMEP). The table helps to quantify the contribution of AMEP to the overall performance of the agent and the value of including both positive and negative experiences in its memory.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_16_1.jpg)
> This table presents the main results of the Optimus-1 agent on a series of long-horizon tasks in the Minecraft environment.  It compares Optimus-1's performance against several baselines (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human-level performance, across five task groups (Wood, Stone, Iron, Gold, Diamond, Redstone, Armor). The metrics used for comparison are the average success rate (SR), average number of steps (AS), and average time (AT) to complete the tasks. Lower AS and AT scores indicate higher efficiency. A score of +∞ signifies task failure.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_18_1.jpg)
> This table presents the experimental setup for evaluating the Optimus-1 agent. It divides 67 Minecraft tasks into seven groups (Wood, Stone, Iron, Gold, Redstone, Diamond, Armor) based on their complexity and material requirements.  For each group, it shows the number of tasks, an example task, the maximum number of steps allowed to complete the task, the initial inventory provided to the agent (empty in all cases), and the average number of sub-goals required to accomplish a task within the group.  This table helps to understand the scope and difficulty levels of the tasks used to benchmark the Optimus-1 agent.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_18_2.jpg)
> This table presents the experimental setup for evaluating the Optimus-1 agent. It details the 67 tasks used in the benchmark, categorized into 7 groups based on their complexity and material requirements.  Each task includes the number of sub-goals needed to complete it, the maximum number of steps allowed, and the initial inventory provided to the agent. The groups are: Wood, Stone, Iron, Gold, Redstone, Diamond, and Armor, representing a progression in material and tool complexity within the Minecraft game.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_19_1.jpg)
> This table compares various existing Minecraft agents across several key aspects, including their publication venue, environment used, input and output modalities, and whether they incorporate planning, reflection, knowledge, and experience mechanisms.  The table helps to highlight the unique contributions of Optimus-1 in relation to prior works.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_23_1.jpg)
> This table presents the main results of the Optimus-1 agent on a series of long-horizon tasks in Minecraft.  It compares Optimus-1's performance against several baselines (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human-level performance.  The metrics used are success rate (SR), average number of steps (AS), and average time (AT) to complete each task.  Lower AS and AT values indicate better efficiency.  The table is organized by task group (Wood, Stone, Iron, Gold, Diamond, Redstone, Armor) and includes an overall average performance across all groups.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_23_2.jpg)
> This table presents the performance of the Optimus-1 agent on a set of tasks within the Stone group in the Minecraft environment.  It shows the success rate (SR), average number of steps (AS) and average time (AT) required to complete each task.  The Stone group likely contains tasks requiring the use of stone tools and materials.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_24_1.jpg)
> This table presents the main results of the Optimus-1 agent's performance on a series of long-horizon tasks in Minecraft.  It compares Optimus-1 against several baselines (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human-level performance across different task categories (Wood, Stone, Iron, Gold, Diamond, Redstone, Armor). The metrics used to evaluate the agent's performance are Success Rate (SR), Average Steps (AS), and Average Time (AT). Lower values of AS and AT signify greater efficiency.  A value of +∞ indicates task failure.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_24_2.jpg)
> This table presents the main results of the Optimus-1 agent on a benchmark of long-horizon tasks in Minecraft.  It compares Optimus-1's performance to several baselines (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human-level performance across five task groups (Wood, Stone, Iron, Gold, Diamond, Redstone, and Armor). The metrics reported are the average success rate (SR), average number of steps (AS) to complete a task, and average time (AT) taken.  Lower AS and AT values indicate higher efficiency.  A value of '+∞' signifies that the agent failed to complete the task.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_24_3.jpg)
> This table presents the main results of the Optimus-1 agent on a set of long-horizon tasks in the Minecraft environment.  It compares Optimus-1's performance against several baselines (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human-level performance. The metrics used are Success Rate (SR), Average Steps (AS), and Average Time (AT). Lower AS and AT values indicate better efficiency.  A value of +∞ signifies task failure.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_25_1.jpg)
> This table presents the main results of the Optimus-1 agent on a set of long-horizon tasks in Minecraft.  It compares the performance of Optimus-1 against several baselines (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human-level performance. The metrics used for comparison are success rate (SR), average number of steps (AS) to complete the task, and average time (AT) taken.  Lower AS and AT values indicate better efficiency.  A value of +∞ signifies that the agent failed to complete the task.  The results are grouped by task type (wood, stone, iron, gold, diamond, redstone, armor) and an overall average is provided.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_25_2.jpg)
> This table presents the main results of the Optimus-1 agent on a benchmark of long-horizon tasks in Minecraft.  It compares Optimus-1's performance (success rate, average steps, average time) against several baselines (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human performance across five task groups (Wood, Stone, Iron, Gold, Diamond, Redstone, Armor).  Lower values for Average Steps (AS) and Average Time (AT) indicate better efficiency.  '+∞' signifies that the task could not be completed.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_26_1.jpg)
> This table presents the main results of the Optimus-1 agent's performance on a long-horizon task benchmark in Minecraft.  It compares Optimus-1 to several baseline agents (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human-level performance across five task groups (Wood, Stone, Iron, Gold, Diamond, Redstone, Armor). The metrics used are Success Rate (SR), Average Steps (AS), and Average Time (AT), with lower AS and AT indicating better efficiency.  The table shows that Optimus-1 significantly outperforms all baselines, exhibiting near human-level performance.

![](https://ai-paper-reviewer.com/XXOMCwZ6by/tables_27_1.jpg)
> This table presents the main results of the Optimus-1 agent on a long-horizon tasks benchmark in Minecraft.  It compares Optimus-1's performance (success rate, average steps, and average time) against several baselines (GPT-3.5, GPT-4V, DEPS, Jarvis-1) and human-level performance across five groups of tasks (Wood, Stone, Iron, Gold, Diamond, Redstone, and Armor).  Lower steps and time indicate better efficiency.  '+∞' signifies task failure.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XXOMCwZ6by/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}