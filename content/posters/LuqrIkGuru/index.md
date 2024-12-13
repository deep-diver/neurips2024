---
title: "Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections"
summary: "Node Injection-based Fairness Attack (NIFA) reveals GNNs' vulnerability to realistic fairness attacks by injecting a small percentage of nodes, significantly undermining fairness even in fairness-awar..."
categories: []
tags: ["AI Theory", "Fairness", "🏢 Huazhong University of Science and Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} LuqrIkGuru {{< /keyword >}}
{{< keyword icon="writer" >}} Zihan Luo et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=LuqrIkGuru" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95562" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=LuqrIkGuru&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/LuqrIkGuru/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Graph Neural Networks (GNNs) are powerful tools in various fields, but recent research shows they are susceptible to attacks that undermine their fairness. Existing attack methods mainly manipulate the connections between nodes, but this is not always feasible in real-world applications. This paper introduces a new attack method called Node Injection-based Fairness Attack (NIFA). Unlike previous methods, NIFA injects new, malicious nodes into the graph during training.  The researchers designed clever strategies for choosing which nodes to target and manipulating the features of the newly introduced nodes to increase bias and prejudice in the system. 



NIFA was tested on three real-world datasets.  The results consistently showed that **NIFA can significantly reduce fairness in GNNs**, even fairness-aware models, by injecting just 1% of nodes, without noticeably impacting the GNN's overall accuracy. This demonstrates a concerning vulnerability in existing GNN models. The paper makes several contributions.  It is the first to demonstrate a fairness attack using node injection, a more realistic attack method. The paper also proposes new principles for conducting the attack (uncertainty maximization and homophily increase) and new optimization techniques to make the attack more effective. The findings have important implications for the security and trustworthiness of GNNs in real-world applications.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} GNNs are vulnerable to fairness attacks via node injection, a more realistic attack scenario. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} NIFA, a novel attack method, significantly compromises GNN fairness by injecting only 1% of nodes. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Fairness-aware GNNs are not immune to NIFA; improved defense mechanisms are needed. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it highlights a significant vulnerability in the fairness of graph neural networks (GNNs)**, a widely used machine learning model.  By demonstrating a novel attack method, it pushes the field to develop more robust and fair GNNs, impacting various applications that rely on these models.  The findings also open avenues for research into new defense mechanisms, advancing the trustworthiness and security of GNNs.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/LuqrIkGuru/figures_4_1.jpg)

> This figure illustrates the three main steps of the Node Injection-based Fairness Attack (NIFA) method.  First, it identifies nodes with high model uncertainty as targets for injection. Second, it injects new nodes, distributing them evenly across sensitive groups and connecting them only to target nodes within the same group to increase homophily.  Finally, it iteratively optimizes the features of the injected nodes using multiple objective functions, balancing fairness and utility.





![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_6_1.jpg)

> This table presents the results of the Node Injection-based Fairness Attack (NIFA) against various GNN models (including fairness-aware ones) on three datasets.  It shows the accuracy, statistical parity (ASP), and equal opportunity (EO) metrics before and after the attack.  Bold values indicate successful fairness attacks, where NIFA increased ASP and ΔEO, signifying a decrease in fairness. The table helps demonstrate NIFA's effectiveness against different GNN architectures.





### In-depth insights


#### GNN Fairness Attacks
GNN Fairness Attacks explore vulnerabilities in Graph Neural Networks (GNNs) related to fairness.  **Existing attacks often manipulate GNN structures**, which might be unrealistic.  This research proposes a novel **Node Injection-based Fairness Attack (NIFA)**, which injects nodes into the graph structure, a more realistic attack scenario.  **NIFA cleverly utilizes two principles:** uncertainty maximization to target vulnerable nodes and homophily increase to amplify bias within sensitive groups. This approach is shown to be **highly effective**, even against fairness-aware GNNs. The findings highlight a critical need for robust defense mechanisms against such attacks to ensure fair and equitable GNN application.

#### Node Injection Method
A node injection method in the context of graph neural networks (GNNs) focuses on manipulating a graph's structure by introducing new nodes (injecting nodes) rather than altering existing connections.  **This approach is particularly relevant for attacking the fairness of GNNs**, as it mimics real-world scenarios where injecting fake profiles or accounts is easier than modifying existing relationships. The method's effectiveness relies on two key principles: **uncertainty maximization** (selecting nodes with high prediction uncertainty) and **homophily increase** (connecting injected nodes to similar nodes to amplify bias).  **Sophisticated feature optimization techniques** are also employed, often involving adversarial training or other methods to ensure the injected nodes effectively manipulate the model's fairness metrics without significantly impacting the model's overall performance.  The choice of injection strategy (which nodes to target and how to connect them) is crucial for efficacy.  The main strength lies in its practicality and the ability to circumvent limitations of previous fairness attacks that required manipulation of the graph's existing structure.  However, **defense mechanisms** focusing on identifying and mitigating the impact of injected nodes would be an important area of future research.

#### NIFA Effectiveness
The effectiveness of NIFA, a Node Injection-based Fairness Attack, is rigorously evaluated in the paper.  **Experiments across three real-world datasets consistently demonstrate NIFA's ability to significantly reduce the fairness of various GNN models**, including those specifically designed for fairness.  The attack's success is notable even with a **minimal injection rate of only 1% of nodes**, showcasing a vulnerability even in state-of-the-art, fairness-aware GNNs.  Importantly, **NIFA achieves this impact with a negligible compromise on the GNN's utility**, suggesting its stealth and potency. The study also conducts an ablation analysis, validating the importance of the two principles underpinning NIFA's design: uncertainty maximization and homophily increase. The iterative feature optimization strategy further enhances the attack's efficacy. **Overall, the results strongly support NIFA's effectiveness in undermining GNN fairness under more realistic attack scenarios** compared to previous methods, highlighting a critical need for robust defensive mechanisms.

#### Limitations of NIFA
NIFA, while a novel approach to attacking fairness in GNNs, has limitations.  **Its gray-box setting**, requiring access to training data including sensitive attributes but not model parameters, limits its real-world applicability.  A **fully black-box attack** would be more impactful but substantially more challenging to achieve.  The attack's effectiveness relies on **injecting a small percentage of nodes**, and the impact might diminish with larger datasets or more robust GNN architectures. **The homophily-increase principle**, although effective, is a specific approach, and other strategies might be more effective in certain graph structures. While NIFA considers utility concerns, more sophisticated defense mechanisms could render its impact less severe. The analysis focuses on group fairness based on sensitive attributes, **ignoring other facets of fairness**. Further research is needed to extend its scope to individual fairness, fairness relating to graph structure, and more diverse attack methodologies.

#### Future Defenses
Future defenses against node injection-based fairness attacks on Graph Neural Networks (GNNs) necessitate a multi-pronged approach.  **Robustness to uncertainty** is crucial; methods improving model confidence in predictions, especially for nodes near decision boundaries, can reduce vulnerability.  **Enhanced structural integrity** is another key; strategies to minimize homophily within sensitive groups and encourage information flow between them are vital.  Furthermore, **advanced anomaly detection** techniques are needed, capable of identifying injected nodes through subtle structural and feature-based irregularities.  Finally, **robust fairness metrics** are essential for effective monitoring and detection—moving beyond simplistic measures to incorporate graph structure and nuanced notions of fairness will improve resilience.  The integration of these strategies into a comprehensive defense framework will be key to mitigating the threat of future attacks.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/LuqrIkGuru/figures_8_1.jpg)

> This figure illustrates the framework of the Node Injection-based Fairness Attack (NIFA). It consists of three main stages: (a) Uncertainty Estimation: Nodes with high model uncertainty are selected as target nodes for injection. (b) Node Injection: Injected nodes are assigned to sensitive groups and connected to target nodes within the same group to increase homophily. (c) Feature Optimization: Injected nodes' features and a surrogate model are iteratively optimized using multiple objective functions to maximize fairness and maintain utility.


![](https://ai-paper-reviewer.com/LuqrIkGuru/figures_8_2.jpg)

> The figure shows the impact of masking a proportion (η) of training nodes with the highest uncertainty on the fairness attack performance.  As the proportion of masked nodes increases, the attack's effectiveness diminishes, as indicated by decreases in Δsp and Δeo.  This suggests that focusing on nodes with high model uncertainty is a key aspect of the attack strategy.  Despite the defense mechanism's impact, there is still significant deterioration of fairness compared to the baseline.  Accuracy is only slightly affected by the defense.


![](https://ai-paper-reviewer.com/LuqrIkGuru/figures_18_1.jpg)

> This figure illustrates the framework of the Node Injection-based Fairness Attack (NIFA). It shows three main stages: (a) Uncertainty Estimation, where nodes with high uncertainty are selected as targets for injection; (b) Node Injection, where injected nodes are connected to target nodes within the same sensitive group to increase homophily; and (c) Feature Optimization, where the features of injected nodes and a surrogate model are iteratively optimized to maximize unfairness.


![](https://ai-paper-reviewer.com/LuqrIkGuru/figures_19_1.jpg)

> This figure illustrates the three main steps of the Node Injection-based Fairness Attack (NIFA) framework.  First, it shows how nodes with high model uncertainty are identified as targets for injection. Second, it details the node injection strategy, where injected nodes are connected only to target nodes within the same sensitive group, increasing homophily. Finally, it describes the iterative optimization process of the injected nodes' features and the surrogate model to maximize fairness attacks while maintaining model utility.


![](https://ai-paper-reviewer.com/LuqrIkGuru/figures_19_2.jpg)

> This figure illustrates the framework of the Node Injection-based Fairness Attack (NIFA).  It shows three main stages:  (a) **Uncertainty Estimation:**  The algorithm identifies nodes with high model uncertainty, making them more susceptible to attacks. These nodes are highlighted in the figure. (b) **Node Injection:**  New nodes (injected nodes) are added to the graph.  Crucially, these nodes are connected only to target nodes (identified in stage a) that share the same sensitive attribute (e.g., gender, race), thus increasing homophily within sensitive groups. (c) **Feature Optimization:** The features of the injected nodes are iteratively optimized using multiple objective functions. This optimization aims to maximize the attack's impact on fairness while maintaining the model's overall utility.


![](https://ai-paper-reviewer.com/LuqrIkGuru/figures_19_3.jpg)

> This figure illustrates the framework of the Node Injection-based Fairness Attack (NIFA). It shows three main steps: uncertainty estimation to identify target nodes, node injection while ensuring equal distribution across sensitive groups and connections only with nodes of the same attribute, and iterative optimization of injected node features and a surrogate model to enhance the attack's effectiveness.


![](https://ai-paper-reviewer.com/LuqrIkGuru/figures_19_4.jpg)

> This figure illustrates the framework of the Node Injection-based Fairness Attack (NIFA). It is a three-stage process: First, nodes with high model uncertainty are selected as targets for injection. Second, injected nodes are connected to the target nodes within the same sensitive group. Third, the injected features and surrogate model are optimized iteratively to maximize the attack's effectiveness.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_7_1.jpg)
> This table presents the results of the Node Injection-based Fairness Attack (NIFA) on various GNN models.  It shows the accuracy, statistical parity (ASP), and equal opportunity (ΔEO) metrics before and after the attack for each model on three datasets (Pokec-z, Pokec-n, DBLP).  Bold values highlight instances where NIFA successfully reduced fairness.

![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_7_2.jpg)
> This table presents the results of the Node Injection-based Fairness Attack (NIFA) against various GNN models on three datasets (Pokec-z, Pokec-n, DBLP).  It shows the accuracy, statistical parity (ASP), and equal opportunity (ΔΕΟ) metrics before and after the attack.  Bolded values indicate that NIFA successfully reduced the fairness of the GNN model. Lower values for ASP and ΔΕΟ indicate better fairness, so the increase in these values demonstrates the effectiveness of the attack.

![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_17_1.jpg)
> This table shows the hyperparameter settings used for the proposed Node Injection-based Fairness Attack (NIFA) method on three different datasets: Pokec-z, Pokec-n, and DBLP.  The hyperparameters include α and β (weights for the objective functions), b (node budget), d (edge budget), k (uncertainty threshold), max_step (number of steps in inner loop), and max_iter (number of iterations in outer loop).  These settings were determined through tuning for optimal attack performance on each dataset.

![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_17_2.jpg)
> This table presents the results of the Node Injection-based Fairness Attack (NIFA) against various GNN models (including fairness-aware ones) on three datasets (Pokec-z, Pokec-n, and DBLP).  It shows the accuracy, statistical parity (ASP), and equal opportunity (ΔΕΟ) metrics before and after the attack.  Bold values indicate that NIFA successfully reduced fairness, as measured by the increase in ASP and ΔΕΟ.

![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_18_1.jpg)
> This table presents the results of the Node Injection-based Fairness Attack (NIFA) on various GNN models across three datasets (Pokec-z, Pokec-n, DBLP).  It shows the accuracy, statistical parity (ASP), and equal opportunity (ΔEO) metrics before and after the attack. Bold values indicate that NIFA successfully reduced the fairness of the GNN model. The table helps to demonstrate the effectiveness of NIFA in compromising the fairness of different GNN architectures, even those designed to be fairness-aware.

![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_19_1.jpg)
> This table presents the results of the Node Injection-based Fairness Attack (NIFA) on various GNN models.  It shows the accuracy, statistical parity (ASP), and equal opportunity (ΔΕΟ) metrics before and after the attack for each model on three datasets (Pokec-z, Pokec-n, DBLP). Bolded values indicate that NIFA successfully reduced fairness (increased ASP and ΔΕΟ).

![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_20_1.jpg)
> This table presents the results of the Node Injection-based Fairness Attack (NIFA) on various GNN models.  It shows the accuracy, statistical parity (ASP), and equal opportunity (EO) differences before and after the attack for each model on three datasets (Pokec-z, Pokec-n, and DBLP). Bolded values highlight where NIFA successfully reduced the fairness of the GNN models.  The metrics ASP and ΔΕΟ are used to measure fairness; lower values indicate better fairness, so increases in these values after the attack demonstrate a reduction in fairness. This table helps to understand the impact of NIFA on the fairness of different GNN models across various datasets.

![](https://ai-paper-reviewer.com/LuqrIkGuru/tables_20_2.jpg)
> This table presents the results of the Node Injection-based Fairness Attack (NIFA) against various Graph Neural Network (GNN) models.  It shows the accuracy, statistical parity (ASP), and equal opportunity (ΔΕΟ) metrics before and after the attack for each model on three datasets (Pokec-z, Pokec-n, DBLP).  Bold values indicate that NIFA successfully reduced fairness.  The table demonstrates NIFA's effectiveness across different GNN models, even those designed to be fair.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LuqrIkGuru/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}