---
title: "FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making"
summary: "FINCON: an LLM-based multi-agent system uses conceptual verbal reinforcement for superior financial decision-making, generalizing well across various tasks."
categories: ["AI Generated", ]
tags: ["AI Applications", "Finance", "🏢 Harvard University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} dG1HwKMYbC {{< /keyword >}}
{{< keyword icon="writer" >}} Yangyang Yu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=dG1HwKMYbC" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/dG1HwKMYbC" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=dG1HwKMYbC&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/dG1HwKMYbC/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current LLM-based agent systems for financial tasks struggle with effectively synthesizing multi-source information and managing risk in volatile markets.  Existing systems often lack sophisticated risk management and efficient communication structures, leading to suboptimal performance.  They also underutilize the potential of LLMs for continuous learning and refinement.

FINCON addresses these issues through a novel manager-analyst hierarchical structure.  It uses a dual-level risk-control component to monitor market risks and update agent beliefs, improving decision-making accuracy. A unique **conceptual verbal reinforcement** mechanism selectively propagates crucial information to relevant agents, improving efficiency. The framework is demonstrated to generalize effectively to diverse financial tasks and outperform existing methods in both single stock trading and portfolio management.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} FINCON, a novel multi-agent system for financial decision-making, uses a manager-analyst hierarchy inspired by real-world investment firms. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} FINCON incorporates a dual-level risk control mechanism to enhance decision-making and reduce communication costs. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Conceptual verbal reinforcement improves performance by selectively propagating verbal reinforcement for future decisions to relevant agents. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents **FINCON**, a novel framework that significantly advances research in LLM-based multi-agent systems for financial decision-making.  It addresses limitations of existing systems by improving information synthesis, decision-making efficiency, and risk management.  The **conceptual verbal reinforcement** technique is a particularly significant contribution that opens exciting new avenues in LLM training and optimization.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_1_1.jpg)

> 🔼 The figure illustrates the overall architecture of FINCON, a synthesized LLM multi-agent system.  It shows the flow of multi-modal market information (textual, tabular, and audio data) into the system.  The information is processed by different types of LLMs (foundation models and financial fine-tuned LLMs) and single-agent modules (code generator, calculator, profile, dataframe loader, document loader, and action). The core of the system is the FinCon Multi-Agent System, which comprises a Manager-Analyst agent synthesized component and a risk control component. The Manager-Analyst component consists of a trading manager and various analyst agents (News Analyst, Data Analyst, Report Analyst, etc.) that collaborate through natural language interaction. The risk control component includes within-trajectory and over-trajectory risk control mechanisms to refine decisions and enhance performance. 
> <details>
> <summary>read the caption</summary>
> Figure 1: The general framework of FINCON.
> </details>





![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_5_1.jpg)

> 🔼 This table draws a parallel between the terminology used in gradient-based model optimizers and LLM-based prompt optimizers.  It highlights the corresponding concepts of 'upgrade direction' (model value gradient momentum vs. prompt reflection trajectory) and 'update method' (learning rate descent vs. overlapping percentage of trading decisions) in both optimization approaches.
> <details>
> <summary>read the caption</summary>
> Table 1: Analogy between glossaries in model optimizer and prompt optimizer.
> </details>





### In-depth insights


#### LLM-Agent Synergy
LLM-Agent synergy explores the powerful combination of Large Language Models (LLMs) and intelligent agents.  **LLMs provide the contextual understanding and reasoning capabilities**, while **agents offer the ability to interact with the environment and execute complex tasks.** This synergy allows for systems that can not only understand complex situations but also act upon that understanding in a dynamic and effective manner.  A key aspect is how these components communicate and collaborate. Effective communication protocols are crucial for seamless information exchange and coordinated action.  **Clear delegation of tasks based on each component's strengths** is also essential.  The design of such systems requires careful consideration of the individual agent's capabilities and their overall integration within a larger architecture.  **Effective frameworks must balance autonomy and cooperation**, ensuring that agents can make independent decisions while also coordinating their actions to achieve shared objectives.  Furthermore, **robust feedback mechanisms are critical** for continuous learning and adaptation. The combined system should be able to learn from successes and failures, refining its strategies and improving its performance over time.  Ultimately, LLM-Agent synergy holds immense potential across numerous fields, offering innovative solutions to complex problems requiring both sophisticated understanding and decisive action.  The focus should be on designing systems that leverage the strengths of each component, leading to more intelligent, adaptable, and effective outcomes.

#### Verbal Reinforcement
The concept of 'verbal reinforcement' in the context of large language models (LLMs) applied to financial decision-making is a novel approach to enhance learning and performance.  It leverages the LLM's ability to process and generate human-like text to provide feedback and refine agent actions. Instead of relying solely on numerical rewards, **verbal reinforcement offers a more nuanced and informative mechanism**.  The system can explain the reasoning behind its decisions, allowing for a deeper understanding of strengths and weaknesses. This approach is particularly beneficial in complex financial scenarios, where contextual information and risk management are crucial. By providing explanations, the system can **improve the transparency and interpretability** of its actions.  Furthermore, **selectively propagating verbal feedback** to relevant agents within a multi-agent system optimizes communication and reduces computational costs. This targeted approach, coupled with mechanisms that monitor risk and update investment beliefs, significantly enhances the system's ability to learn from both successes and failures.  The verbal reinforcement method allows for more effective knowledge transfer within the hierarchical structure and can lead to improved decision-making overall.  It is a promising area of research with the potential to revolutionize AI-driven financial systems.

#### Dual Risk Control
A robust financial decision-making system necessitates a multifaceted approach to risk management.  **Dual risk control**, encompassing both within-episode and over-episode mechanisms, offers a powerful strategy.  Within-episode control, employing metrics like Conditional Value at Risk (CVaR), allows for real-time adjustments to mitigate immediate market fluctuations and potential losses.  This **reactive approach** ensures prompt responses to sudden drops in performance indicators.  Conversely, over-episode control, implemented through a **conceptual verbal reinforcement mechanism**, facilitates a more considered, **proactive strategy**. By analyzing past successes and failures, the system refines its investment beliefs and adapts future decisions accordingly. This approach leverages the power of large language models (LLMs) to integrate and synthesize diverse information sources, leading to more informed and resilient decision making.  The combination of these two mechanisms creates a dynamic system capable of handling both short-term volatility and long-term strategic adjustments, improving overall performance and risk management.

#### FINCON Framework
The FINCON framework, a synthesized LLM multi-agent system, is designed for enhanced financial decision-making.  **Its core innovation lies in mimicking real-world investment firm structures**, employing a manager-analyst hierarchy for efficient collaboration and information synthesis.  Analyst agents, each specialized in a particular data modality (e.g., news, financial reports, audio), pre-process information and feed insights to the manager agent. This **hierarchical design reduces communication costs and promotes focused analysis**.  Furthermore, FINCON incorporates a **dual-level risk-control mechanism**, incorporating both within-episode and over-episode risk management to enhance decision-making quality and robustness. The verbal reinforcement component, using conceptualized beliefs generated from performance analysis, further refines agent actions, improving the system's learning and generalization capabilities.  **FINCON demonstrates adaptability to various tasks**, including single stock trading and portfolio management, showcasing its potential to optimize multi-source information and decision-making processes in complex financial environments.

#### Future of FinTech
The future of FinTech is likely to be defined by **increasing convergence of technologies**, particularly AI, blockchain, and big data analytics, to create more sophisticated and personalized financial services.  **AI-powered solutions** will likely play a crucial role in automating processes, improving risk management, and personalizing customer experiences. **Blockchain technology** will continue to enable secure and transparent transactions, enhancing trust and efficiency across financial systems.  The **growth of big data analytics** will allow for more precise risk assessment, fraud detection, and investment strategies.  However, **regulatory challenges and ethical concerns** surrounding the use of these technologies need careful consideration and proactive measures to ensure responsible innovation and equitable access to financial services.  The combination of these trends will likely result in a more efficient, transparent, and inclusive financial landscape, but requires attention to potential societal impacts.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_3_1.jpg)

> 🔼 This figure illustrates the architecture of FINCON, a synthesized LLM multi-agent system.  It highlights the two main components: the Manager-Analyst Agent Group and the Risk-Control component. The Manager-Analyst group consists of a manager agent and multiple analyst agents. Each analyst agent focuses on processing information from a specific source (news, financial reports, etc.), while the manager agent is responsible for making trading decisions based on combined inputs from all analysts. The Risk-Control component includes both within-episode (daily market risk monitoring using CVaR) and over-episode (belief updating using textual gradient descent) risk management. The figure depicts the flow of information and communication between the two components and individual agents, demonstrating how FINCON manages risks and improves decision-making performance.
> <details>
> <summary>read the caption</summary>
> Figure 2: The detailed architecture of FINCON contains two key components: Manager-Analyst agent group and Risk Control. It also presents the between-component interaction of FINCON and decision-making flow.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_8_1.jpg)

> 🔼 This figure shows the overall architecture of FINCON, an LLM-based multi-agent system for financial decision-making.  It illustrates the interaction between different components, including the multi-modal market environment information (textual, tabular, and audio data), the various LLM functions (generation, code generation, calculation, third-party API calls), and the multi-agent system itself. The system consists of a Manager and Analyst agents organized in a hierarchical structure. The risk-control component is also shown, emphasizing its dual-level operation of within-episode and over-episode risk management.
> <details>
> <summary>read the caption</summary>
> Figure 1: The general framework of FINCON.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_21_1.jpg)

> 🔼 This figure presents the overall architecture of the FINCON system, illustrating the interaction between its key components: the multi-modal market environment providing various data sources (textual, tabular, audio), the multi-type LLMs performing different functions (generation, code generation, calculation, perception, etc.), the manager-analyst agent group enabling synthesized teamwork, and the risk-control component ensuring both within-episode and over-episode risk management.  The figure showcases the flow of information and decision-making within the system.
> <details>
> <summary>read the caption</summary>
> Figure 1: The general framework of FINCON.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_22_1.jpg)

> 🔼 The figure illustrates the overall architecture of the FINCON system.  It shows the interaction between different components, including the multi-modal market environment information (textual, tabular, and audio data), multi-type LLMs (GPT, Llama 3, Claude, etc.), the manager-analyst agent synthesized component, and the risk control component.  The manager-analyst component depicts the hierarchical structure with a trading manager agent and several analyst agents (news analyst, data analyst, and report analyst).  The risk control component highlights both within-trajectory and over-trajectory risk control mechanisms.  The diagram provides a visual overview of how FINCON processes information and makes trading decisions.
> <details>
> <summary>read the caption</summary>
> Figure 1: The general framework of FINCON.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_23_1.jpg)

> 🔼 This figure presents a detailed overview of FINCON's architecture. It illustrates the interaction between various components, including the multi-modal market environment providing different types of data (textual, tabular, and audio), the multi-type LLMs functioning as modules (generation, code generation, configuration, foundation models, and fine-tuned LLMs), and the manager-analyst agent synthesized component coordinating the interactions of different analyst agents (News Analyst, Data Analyst, Report Analyst, and ECC Agent) to provide insights for the manager agent. The risk control component, responsible for monitoring within-trajectory and over-trajectory risks, and updating systematic investment beliefs using conceptual verbal reinforcement, is also shown. The figure provides a comprehensive visualization of the entire FINCON framework, making it easier to understand the workflow and interactions between its different parts.
> <details>
> <summary>read the caption</summary>
> Figure 1: The general framework of FINCON.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_25_1.jpg)

> 🔼 This figure illustrates the overall architecture of the FINCON system, which is an LLM-based multi-agent framework for enhanced financial decision-making. It shows the interaction between the multi-modal market environment (providing various data types like textual, tabular, and audio information), the multi-type LLMs (serving as foundation models for different agents), the manager-analyst agent synthesized component (representing the hierarchical structure with a manager agent and multiple analyst agents), and the risk-control component (enabling within-trajectory and over-trajectory risk control).  The diagram highlights the flow of information and decisions within the system, showcasing how diverse information is processed by different agents and utilized by the manager for unified decision-making. 
> <details>
> <summary>read the caption</summary>
> Figure 1: The general framework of FINCON.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_26_1.jpg)

> 🔼 This figure presents a detailed overview of the FINCON architecture, highlighting its key components, including the multi-modal market environment information sources, various types of LLMs utilized for different tasks, the manager-analyst agent synthesized component, and the dual-level risk control component. It illustrates the hierarchical structure of the system and the data flow between the different modules, including how multi-type LLMs process different data modalities, how analyst agents extract investment insights, how the manager agent makes trading decisions, and how the risk control component updates investment beliefs and manages risk.
> <details>
> <summary>read the caption</summary>
> Figure 1: The general framework of FINCON.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_26_2.jpg)

> 🔼 This figure shows the overall architecture of the FINCON system, illustrating the interaction between its key components: the multi-modal market environment information, multi-type LLMs, single-agent modules, the manager-analyst agent group, the risk-control component, and the resulting trading actions and profits and losses.  It provides a visual overview of the data flow and the different agents' roles and responsibilities within the system.
> <details>
> <summary>read the caption</summary>
> Figure 1: The general framework of FINCON.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_30_1.jpg)

> 🔼 This figure shows the cumulative returns (CRs) over time for six different stocks (TSLA, AAPL, NIO, AMZN, GOOG, NFLX) across various trading models, including FINCON and other baselines such as Buy-and-Hold (B&H), Generative Agent (GA), FINGPT, FINMEM, FINAGENT, A2C, PPO and DQN.  FINCON consistently outperforms other models across all six stocks. The results highlight FINCON's robust performance across various market conditions and its superior ability to manage risks and maximize profits.
> <details>
> <summary>read the caption</summary>
> Figure 5: CRs over time for single-asset trading tasks. FINCON outperformed other comparative strategies, achieving the highest CRs across all six stocks by the end of the testing period, regardless of market conditions.
> </details>



![](https://ai-paper-reviewer.com/dG1HwKMYbC/figures_30_2.jpg)

> 🔼 This figure shows the performance of four different portfolio management strategies (FINCON, Markowitz, Equal-Weighted ETF, and FinRL-A2C) over time.  It visualizes the growth or decline of the portfolio value for two different portfolios (Portfolio 1 and Portfolio 2) across the strategies.  The x-axis represents the date, and the y-axis represents the portfolio value.  The figure helps in comparing the performance of different strategies in managing portfolios over time.
> <details>
> <summary>read the caption</summary>
> Figure 3: Portfolio values of Portfolio 1 & 2 changes over time for all the strategies. The computation of portfolio value refers to Equation 7 in Appendix A.10.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_5_2.jpg)
> 🔼 This table presents the results of an ablation study comparing FINCON's performance with and without the within-episode risk control mechanism (CVaR).  The study evaluates the impact of CVaR on key performance metrics (CR%, SR, MDD%) across both single-asset trading and portfolio management tasks.  The results highlight the effectiveness of CVaR in improving FINCON's performance, particularly in managing risk and achieving higher returns, especially in scenarios with mixed market trends.
> <details>
> <summary>read the caption</summary>
> Table 4: Key metrics FINCON with vs. without implementing CVaR for within-episode risk control. The performance of FINCON with the implementation of CVaR won a leading performance in both single-asset trading and portfolio management tasks.
> </details>

![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_7_1.jpg)
> 🔼 This table compares the performance of FINCON against other LLM-based and DRL-based agents on six different stocks.  The metrics used for comparison are Cumulative Return (CR%), Sharpe Ratio (SR), and Maximum Drawdown (MDD%).  Statistical significance testing (Wilcoxon signed-rank test) was performed to highlight the best performing models.
> <details>
> <summary>read the caption</summary>
> Table 2: Comparison of key performance metrics during the testing period for the single-asset trading tasks involving six stocks, between FINCON and other algorithmic agents. Note that the highest and second highest CRs and SRs have been tested and found statistically significant using the Wilcoxon signed-rank test. The highest CRs and SRs are highlighted in red, while the second highest are marked in blue.
> </details>

![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_8_1.jpg)
> 🔼 This table presents a comparison of the key performance metrics (Cumulative Return (CR%), Sharpe Ratio (SR), and Maximum Drawdown (MDD)) across different portfolio management strategies, including FINCON, Markowitz Mean-Variance (MV), FINRL-A2C, and an Equal-Weighted ETF strategy.  Two different portfolios (Portfolio 1 and Portfolio 2) were used. The results show that FINCON outperforms all other methods across all three metrics for both portfolios.
> <details>
> <summary>read the caption</summary>
> Table 3: Key performance metrics comparison among all portfolio management strategies of Portfolio 1 & 2. FINCON leads all performance metrics.
> </details>

![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_9_1.jpg)
> 🔼 This table presents the results of an ablation study comparing FINCON's performance with and without the within-episode risk control mechanism using CVaR.  It shows the impact of CVaR on key performance metrics such as Cumulative Return (CR%), Sharpe Ratio (SR), and Maximum Drawdown (MDD) for both single-stock trading and portfolio management tasks. The results indicate that incorporating CVaR significantly improves FINCON's performance in managing risk and achieving higher returns.
> <details>
> <summary>read the caption</summary>
> Table 4: Key metrics FINCON with vs. without implementing CVaR for within-episode risk control. The performance of FINCON with the implementation of CVaR won a leading performance in both single-asset trading and portfolio management tasks.
> </details>

![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_9_2.jpg)
> 🔼 This table presents the results of an ablation study comparing FINCON's performance with and without the within-episode risk control mechanism (using CVaR). The study evaluates the impact of CVaR on key financial metrics (CR, SR, and MDD) for both single-stock trading (GOOG and NIO) and portfolio management tasks.  The results show that incorporating CVaR significantly improves FINCON's performance across all metrics and tasks.
> <details>
> <summary>read the caption</summary>
> Table 4: Key metrics FINCON with vs. without implementing CVaR for within-episode risk control. The performance of FINCON with the implementation of CVaR won a leading performance in both single-asset trading and portfolio management tasks.
> </details>

![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_19_1.jpg)
> 🔼 This table presents a comparison of FINCON's performance with and without the within-episode risk control mechanism using CVaR.  The results show that incorporating CVaR significantly improves performance across various metrics, particularly in both single-asset trading and portfolio management tasks.
> <details>
> <summary>read the caption</summary>
> Table 4: Key metrics FINCON with vs. without implementing CVaR for within-episode risk control. The performance of FINCON with the implementation of CVaR won a leading performance in both single-asset trading and portfolio management tasks.
> </details>

![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_25_1.jpg)
> 🔼 This table presents a comparison of the key performance metrics (CR%, SR, and MDD%) of FINCON with and without the within-episode risk control mechanism using CVaR.  It showcases FINCON's superior performance across two tasks: single-stock trading (GOOG and NIO, representing bullish and bearish market trends respectively) and portfolio management (a portfolio comprising TSLA, MSFT, and PFE). The results highlight the significant improvement in performance achieved by incorporating CVaR, demonstrating its effectiveness in managing market risk and enhancing trading outcomes. 
> <details>
> <summary>read the caption</summary>
> Table 4: Key metrics FINCON with vs. without implementing CVaR for within-episode risk control. The performance of FINCON with the implementation of CVaR won a leading performance in both single-asset trading and portfolio management tasks.
> </details>

![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_30_1.jpg)
> 🔼 This table presents a comparison of the performance of different models on a single-asset trading task under high volatility conditions. The models include FINCON, a novel LLM-based multi-agent system; several other LLM-based models (GA, FINGPT, FINMEM, FINAGENT); and several DRL-based models (A2C, PPO, DQN).  The performance is evaluated using three key metrics: Cumulative Return (CR%), Sharpe Ratio (SR), and Maximum Drawdown (MDD%). The results show that FINCON significantly outperforms all other models across all three metrics.
> <details>
> <summary>read the caption</summary>
> Table 7: Key performance comparison for single asset trading under the high volatility condition using TSLA as an example. FINCON leads all performance metrics.
> </details>

![](https://ai-paper-reviewer.com/dG1HwKMYbC/tables_30_2.jpg)
> 🔼 This table presents a comparison of the key performance metrics (Cumulative Return (CR%), Sharpe Ratio (SR), and Maximum Drawdown (MDD)) for four different portfolio management strategies applied to two distinct portfolios (Portfolio 1 and Portfolio 2).  The strategies compared are FINCON, Markowitz Mean-Variance (MV), FinRL-A2C, and Equal-Weighted ETF.  The results show that FINCON outperforms all other strategies across all three metrics, indicating superior performance in terms of returns, risk-adjusted returns, and risk management.
> <details>
> <summary>read the caption</summary>
> Table 8: Key performance metrics comparison among all portfolio management strategies of Portfolio 1 & 2. FINCON leads all performance metrics.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/dG1HwKMYbC/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}