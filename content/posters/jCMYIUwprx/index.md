---
title: "INDICT: Code Generation with Internal Dialogues of Critiques for Both Security and Helpfulness"
summary: "INDICT, a novel framework, empowers LLMs with internal dialogues of critiques to enhance code generation, prioritizing both safety and helpfulness, resulting in +10% absolute improvement across variou..."
categories: []
tags: ["Natural Language Processing", "Large Language Models", "🏢 Salesforce Research",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} jCMYIUwprx {{< /keyword >}}
{{< keyword icon="writer" >}} Hung Le et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=jCMYIUwprx" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93974" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=jCMYIUwprx&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/jCMYIUwprx/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current LLMs for code generation struggle to balance helpfulness and safety, often producing vulnerable or malicious code.  Previous methods like finetuning have proven insufficient. This paper introduces INDICT, a novel framework that uses internal dialogues between two critics: one prioritizing safety and another focusing on helpfulness.  These critics leverage external knowledge sources like web searches and code interpreters to provide comprehensive feedback.

INDICT integrates this dual-critic system into both code generation and execution phases.  Evaluated on diverse tasks and LLMs (7B to 70B parameters), INDICT demonstrated significant improvements in code quality, achieving a +10% absolute increase in quality across all models tested.  This highlights its effectiveness in generating safer and more helpful code, advancing the field of responsible AI development.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} INDICT uses a dual cooperative system of safety-driven and helpfulness-driven critics to provide advanced critiques for LLM code generation. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The framework engages critics in both code generation and execution stages, providing preemptive and post-hoc guidance. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} INDICT significantly improves code quality (+10% absolute improvement) across different LLMs and programming languages by enhancing safety and helpfulness. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **large language models (LLMs)** for code generation because it directly addresses the critical issue of **balancing safety and helpfulness**.  It introduces a novel framework that significantly improves the quality of generated code, offering valuable insights for enhancing LLM safety and reliability. The research also opens exciting avenues for developing more robust and responsible AI systems. This is particularly relevant given increasing concerns about the potential misuse of LLMs for malicious activities.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_1_1.jpg)

> This figure illustrates the INDICT framework, showing how two critics (safety and helpfulness) interact with an actor (LLM) to generate code.  The example demonstrates how INDICT addresses a security vulnerability (CWE-78) by iteratively improving the code through the critics' interactions and knowledge retrieval, ultimately resulting in safer and more helpful code.





![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_2_1.jpg)

> This table compares INDICT with other code generation methods, categorized into self-refine, multi-agent, and finetuning approaches. It highlights INDICT's advantages: integrating code execution feedback with external knowledge, focusing on both helpfulness and safety, and utilizing a collaborative, supervision-free multi-agent framework.





### In-depth insights


#### Internal Dialogue
The concept of "Internal Dialogue" in the context of a large language model (LLM) for code generation is a fascinating approach to enhancing both the safety and helpfulness of the generated code.  It suggests a system where the LLM doesn't simply produce code but engages in a self-reflective process. This process involves **two separate "critics"**: one focused on safety, evaluating the code for potential vulnerabilities and security risks; the other concentrating on helpfulness, assessing whether the code effectively addresses the user's intent and produces useful, functional output.  The interaction between these critics is crucial, fostering a dynamic feedback loop that refines the generated code.  This internal deliberation is not merely a post-hoc review; rather, it is a **proactive, iterative process** that guides the LLM's code generation from start to finish, anticipating potential issues and ensuring the code aligns with its intended purpose.  The **integration of external knowledge sources** through tools like web search and code interpreters further augments the critics' analysis, leading to more informed and grounded critiques.  This method provides a more robust and nuanced solution for improving code generation by LLMs, especially in complex or potentially malicious scenarios.

#### Dual Critic System
A dual critic system, in the context of a code generation model, offers a sophisticated approach to enhancing both the **helpfulness and safety** of generated code.  By employing two distinct critics—one focused on safety and the other on helpfulness—the system facilitates a more nuanced evaluation than a single critic could provide. The **safety-driven critic** scrutinizes the code for potential security vulnerabilities and risks, while the **helpfulness-driven critic** assesses the code's accuracy, efficiency, and adherence to the user's intentions.  This dual-perspective evaluation process leads to more robust and reliable code generation, improving the overall quality and mitigating potential safety concerns.  **Autonomous collaboration** between these critics further enhances the system's effectiveness, allowing them to iteratively refine the code and provide more comprehensive feedback.  **Knowledge grounding**, achieved through integration of external tools and knowledge sources, further enhances the accuracy and reliability of the critics' analysis.

#### Tool-Enhanced LLMs
Tool-enhanced LLMs represent a significant advancement in large language model (LLM) capabilities. By integrating external tools and resources, these models overcome limitations of solely relying on pre-trained knowledge.  **This integration allows LLMs to perform complex tasks requiring real-time information retrieval, computations, or interaction with the external world.** For instance, an LLM enhanced with a search engine can access and process current web data, greatly improving the accuracy and relevance of its responses.  Similarly, integration with a code interpreter empowers the LLM to execute code and analyze the results, enabling capabilities such as code generation, debugging, or problem-solving.  **The key advantage lies in the ability to dynamically access and process information beyond the static knowledge embedded during the pre-training phase.**  However, challenges remain, including ensuring the reliability and security of external tools, managing the increased computational complexity, and addressing potential biases in the retrieved information.  **Future research should focus on developing robust and secure methods for integrating tools, optimizing tool usage efficiency, and mitigating the risks of bias and misinformation.**  Ultimately, Tool-Enhanced LLMs are poised to revolutionize numerous applications, pushing the boundaries of what LLMs can achieve.

#### Safety & Helpfulness
The core of this research paper revolves around enhancing Large Language Models (LLMs) for code generation by addressing the critical balance between safety and helpfulness.  The authors recognize the inherent risks of LLMs generating malicious or insecure code, and propose a novel framework called INDICT to mitigate these risks. **INDICT introduces a dual-critic system**, incorporating both a safety-driven critic and a helpfulness-driven critic. This framework aims to provide more comprehensive feedback during code generation, ensuring that the generated code is not only useful but also secure.  **The critics leverage external knowledge sources**, such as web search and code interpreters, to enhance the quality of their feedback.  The method is evaluated on diverse tasks and programming languages, demonstrating significant improvement in code quality, demonstrating the efficacy of the approach in improving the safety and helpfulness of AI-generated code.  **A key aspect of this work is the focus on proactive and reactive feedback mechanisms**.  This means that the critics provide preemptive critiques during the code generation stage, as well as post-hoc evaluations after the code is executed, enabling a multi-layered approach to safety and helpfulness. The findings suggest that the integration of dual critics and external knowledge sources contributes to more robust and reliable code generation from LLMs, improving both the security and utility of the generated output.

#### Future of LLMs
The future of LLMs is incredibly promising, yet fraught with challenges.  **Scaling LLMs to even greater sizes** will likely continue, pushing the boundaries of their capabilities in areas like reasoning and complex task solving.  However, this scaling must be accompanied by **rigorous safety research**, addressing issues like bias, toxicity, and adversarial attacks.  **Improving the efficiency of training and inference** is crucial for making LLMs more accessible and sustainable.  This includes exploring alternative architectures and training methods, and potentially shifting towards more decentralized or federated approaches.  Furthermore, **ethical considerations** are paramount.  The potential for misuse necessitates careful attention to fairness, transparency, and accountability.  **Enhanced explainability and interpretability** will be key for gaining trust and responsible adoption.  Ultimately, the path forward involves a multi-faceted approach encompassing technological advancements, robust safety mechanisms, and careful consideration of ethical implications to ensure LLMs benefit humanity in a positive and equitable way.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_3_1.jpg)

> This figure illustrates the INDICT framework's architecture.  It shows the interaction between a task, an actor (LLM code generator), two critics (safety and helpfulness-driven), an executor (code execution environment), and the final response.  The actor generates a solution, which is then reviewed by both critics.  The critics provide feedback (preemptive and post-hoc) to the actor, leading to a revised solution. The executor then executes the solution and provides post-hoc feedback.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_6_1.jpg)

> This figure presents the results of evaluating the INDICT framework on three benchmarks related to insecure coding practices: CyberSecEval-1 (Autocomplete), CyberSecEval-1 (Instruction), and CVS.  The graphs show the safety and helpfulness of code generated by different LLMs, both with and without INDICT. Safety is measured as the percentage of secure code outputs, and helpfulness is the winning rate against state-of-the-art models or ground truth. The results demonstrate that INDICT consistently improves both safety and helpfulness across various LLMs and programming languages.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_7_1.jpg)

> This figure illustrates the INDICT framework, showing how a safety-driven critic and a helpfulness-driven critic interact with each other and with an actor (the LLM code generator) to improve the quality of the generated code. The example demonstrates how INDICT addresses a specific security vulnerability (CWE-78) and enhances code functionality.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_7_2.jpg)

> This figure shows the results of evaluating INDICT on insecure coding tasks using three benchmarks: CyberSecEval-1 (Autocomplete and Instruction), and CVS.  The evaluation metrics are safety (percentage of secure outputs) and helpfulness (winning rate against existing state-of-the-art models or ground truth).  The results are presented for different programming languages and models, comparing performance with and without INDICT.  The figure demonstrates INDICT's improvements in both safety and helpfulness across various tasks and languages.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_9_1.jpg)

> This figure shows the ablation study of applying INDICT for multiple rounds.  The left chart shows the results using CommandR as the base model, and the right chart uses Codellama-13b-instruct.  Each bar represents a metric (Security, Helpfulness, and their average (S+H)) at each round, comparing the performance of INDICT with and without external tools and against a baseline with no INDICT applied.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_24_1.jpg)

> This figure shows an example of how INDICT works.  INDICT uses two critics: a safety critic and a helpfulness critic. These critics work together to improve the code generated by an LLM. In the example, the safety critic identifies a security vulnerability (CWE-78). The helpfulness critic then suggests ways to improve the code's functionality. The LLM then uses the feedback from both critics to generate an improved version of the code.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_26_1.jpg)

> This figure illustrates the INDICT framework, showing how two critics (safety and helpfulness) interact with each other and an actor LLM to generate code.  The example demonstrates how INDICT addresses a specific security vulnerability (CWE-78) and enhances the code's functionality through iterative refinement and the use of external knowledge.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_26_2.jpg)

> The figure shows an example of how INDICT works.  INDICT uses two critics: a safety critic and a helpfulness critic. These critics work together to analyze code and provide feedback to an actor (the LLM). The example shows how INDICT identifies a security vulnerability (CWE-78) in code generated by the actor and then uses external knowledge to suggest improvements, leading to a safer and more helpful code.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_28_1.jpg)

> This figure illustrates the INDICT framework.  It shows how a task is given to an Actor (LLM code generator), which produces a solution. This solution then undergoes evaluation by two critics: a safety-driven critic and a helpfulness-driven critic.  These critics leverage external knowledge and interact with each other, providing both preemptive (before execution) and post-hoc (after execution) feedback to the Actor. The final, revised solution is then the output of the system.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_29_1.jpg)

> This figure shows an example of how INDICT works.  It uses a dual cooperative system with a safety-driven critic and a helpfulness-driven critic. These critics work together autonomously to improve the code generation process, addressing both security vulnerabilities (in this case, CWE-78) and helpfulness issues. The critics interact iteratively, each providing analysis and feedback that guides the code generation process towards a safer and more helpful result.  External knowledge sources are also used.


![](https://ai-paper-reviewer.com/jCMYIUwprx/figures_29_2.jpg)

> This figure shows how INDICT, a framework for improving code generation using LLMs, works by incorporating two critics: a safety-driven critic and a helpfulness-driven critic. These critics interact autonomously and collaboratively to analyze the code generated by an LLM, providing feedback to improve both its security and helpfulness. The example shows an iterative process where INDICT addresses a security vulnerability (CWE-78) and enhances the code functionality.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_4_1.jpg)
> This table compares INDICT with other relevant methods categorized into three groups: self-refine/self-critic, multi-agent, and finetuning approaches.  It highlights INDICT's advantages, including its integrated code execution feedback, use of external knowledge, focus on both helpfulness and safety, and interactive multi-agent collaboration.  The table uses checkmarks to show which methods include specific features and notes the supervision-free aspect of INDICT.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_7_1.jpg)
> This table presents the results of evaluating the INDICT framework against six different red-teaming optimization methods using the HarmBench benchmark.  The safety measure, expressed as a percentage, indicates how often the AI evaluator classified the model's output as benign (non-harmful).  The table compares the performance of the base CommandR and Llama3 models (with and without INDICT) across these different optimization methods.  It shows the impact of INDICT on improving the safety of generated outputs, particularly notable with Llama3-8b-instruct and Llama3-70b-instruct.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_8_1.jpg)
> This table compares INDICT with other code generation methods across three categories: self-refine/self-critic, multi-agent, and finetuning.  It highlights INDICT's advantages, including its use of code execution feedback, external knowledge integration, focus on both safety and helpfulness, and its unique interactive multi-agent collaboration.  The table uses checkmarks to show which methods incorporate each feature.  Appendix D provides references for the cited methods.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_9_1.jpg)
> This table presents the results of an ablation study conducted to evaluate the impact of removing different components of the INDICT framework.  Specifically, it assesses the effects of removing the dual critic system and/or the external tool enhancement. The experiments were performed on CodeLlama models with varying parameter counts (7B to 34B) and the CommandR model.  The table shows the resulting safety and helpfulness scores for each configuration.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_9_2.jpg)
> This table presents the results of an ablation study on the INDICT framework.  The study explores the impact of different combinations of safety and helpfulness critics, and the timing of their feedback (preemptive, post-hoc, or both) on the performance of the model. The experiment uses CommandR as the base model and excludes external tools to isolate the effect of the critic configurations. The results are evaluated based on the average performance across safety and helpfulness metrics.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_17_1.jpg)
> This table compares INDICT with other code generation methods categorized into three approaches: self-refine, multi-agent, and finetuning.  It highlights INDICT's advantages, including its use of code execution feedback and external knowledge, its focus on both helpfulness and safety, and its unique interactive multi-agent collaboration framework.  Appendix D provides the references for the compared methods.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_19_1.jpg)
> This table compares INDICT with other code generation methods across three categories: self-refine/self-critic, multi-agent, and finetuning.  It highlights INDICT's key advantages: incorporating code execution feedback and external knowledge, focusing on both safety and helpfulness, and utilizing a collaborative, supervision-free multi-agent framework. The table includes checkmarks indicating the presence or absence of certain features in each method, demonstrating INDICT's comprehensive approach.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_20_1.jpg)
> This table compares the performance of INDICT against several finetuning methods using CodeLlama-7b-instruct as the base language model.  It shows the safety and helpfulness scores for each method, indicating that INDICT, even when combined with finetuning, achieves superior results.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_21_1.jpg)
> This table presents the ablation study results of INDICT with different variants of critic frameworks, using GPT40-mini as the base language model.  It compares the full INDICT model against four variants: one with a single critic for both safety and helpfulness; one without a critic summarizer; one using RAG-based critics; and one using tool-based critics.  The results show that the full INDICT model achieves the highest performance in both safety and helpfulness, highlighting the benefits of using dual critics and external tools.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_22_1.jpg)
> This table compares INDICT with other code generation methods categorized into self-refine/self-critic, multi-agent, and finetuning approaches.  It highlights INDICT's advantages in integrating code execution feedback, incorporating external knowledge, focusing on both helpfulness and safety, and employing a supervision-free multi-agent collaboration framework.  Checkmarks indicate the features included in each method.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_22_2.jpg)
> This table compares INDICT with other related methods in three categories: self-refine/self-critic, multi-agent, and finetuning. It highlights INDICT's advantages, such as integrating code execution feedback, utilizing external knowledge, targeting both helpfulness and safety, and employing a supervision-free multi-agent framework.  The table includes a checklist indicating which features each method incorporates.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_22_3.jpg)
> This table compares INDICT with other existing code generation methods.  It categorizes methods into three groups: self-refine/self-critic, multi-agent, and fine-tuning.  The table highlights INDICT's advantages, such as incorporating code execution feedback, using external knowledge, addressing both helpfulness and safety, and utilizing a multi-agent framework.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_23_1.jpg)
> This table presents the results of evaluating the security of code generated by three different large language models (LLMs): CommandR, Llama3-8b-instruct, and Llama3-70b-instruct, both with and without the INDICT framework.  The security of the generated code is assessed using a rule-based detector.  The table shows the percentage of generated code samples deemed secure by the detector for each model and language (C++, C#, Java, Javascript, PHP).  The results demonstrate a significant improvement in code security when using INDICT.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_23_2.jpg)
> This table presents the results of the Insecure Coding Practice (Instruction) task from the CyberSecEval-1 benchmark. It shows the percentage of secure code generated by different LLMs, both with and without the INDICT framework.  The results highlight the improvement in code security achieved by INDICT, particularly with the CommandR model, exceeding the state-of-the-art (SoTA) in many programming languages.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_23_3.jpg)
> This table compares INDICT with other existing methods categorized into three groups: self-refine/self-critic methods, multi-agent methods, and finetuning methods.  The comparison highlights INDICT's advantages, including its integration of code execution feedback, enhanced external knowledge, focus on both helpfulness and safety, and a supervision-free, interactive multi-agent collaboration framework.  The table uses checkmarks to indicate the presence or absence of key features in each method, ultimately demonstrating INDICT's superior capabilities.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_24_1.jpg)
> This table compares INDICT with other code generation methods categorized into self-refine, multi-agent, and fine-tuning approaches.  It highlights INDICT's advantages in incorporating code execution feedback, utilizing external knowledge, and focusing on both code helpfulness and safety through a unique multi-agent, supervision-free framework.

![](https://ai-paper-reviewer.com/jCMYIUwprx/tables_24_2.jpg)
> This table compares INDICT with other related methods across three categories: self-refine/self-critic, multi-agent, and finetuning.  It highlights INDICT's advantages, namely its comprehensive approach that integrates code execution feedback, leverages external knowledge, and focuses on both helpfulness and safety.  The table uses checkmarks to indicate the presence or absence of key features in each method, demonstrating INDICT's superior capabilities.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jCMYIUwprx/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}