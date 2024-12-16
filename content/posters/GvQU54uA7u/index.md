---
title: "Preference-based Pure Exploration"
summary: "PreTS algorithm efficiently identifies the most preferred policy in bandit problems with vector-valued rewards, achieving asymptotically optimal sample complexity."
categories: ["AI Generated", ]
tags: ["Machine Learning", "Reinforcement Learning", "🏢 University of Michigan",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} GvQU54uA7u {{< /keyword >}}
{{< keyword icon="writer" >}} Apurv Shukla et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=GvQU54uA7u" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/GvQU54uA7u" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/GvQU54uA7u/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many real-world decision-making problems involve multiple conflicting objectives.  Existing multi-armed bandit algorithms often struggle to handle these situations efficiently.  Preference-based Pure Exploration (PrePEx) aims to solve this challenge by incorporating preferences over objectives, but efficient algorithms for PrePEx were lacking. This leads to a computationally expensive drug discovery process.

This paper introduces the Preference-based Track and Stop (PreTS) algorithm to address this problem.  PreTS provides a convex relaxation of the lower bound on sample complexity, leading to a computationally efficient algorithm.  The authors prove that PreTS's sample complexity is asymptotically optimal, offering a significant advancement in handling multi-objective decision-making under uncertainty.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel lower bound on sample complexity for preference-based pure exploration (PrePEx) is derived. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The PreTS algorithm is introduced, achieving asymptotically optimal sample complexity. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} A new concentration inequality for vector-valued rewards is derived. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in multi-armed bandits and vector optimization. It offers **a novel lower bound and a tight algorithm** for preference-based pure exploration, addressing a critical gap in existing research.  The **geometric insights** and **convex relaxation techniques** are valuable for tackling complex optimization problems beyond the scope of this specific problem.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/GvQU54uA7u/figures_3_1.jpg)

> 🔼 This figure shows how the choice of the preference cone affects the Pareto optimal set.  The Pareto optimal sets for two different cones, Cπ/2 and Cπ/3, are shown in pink and blue, respectively.  The grey points represent the reward vectors of 200 randomly selected arms.  The figure illustrates that the geometry of the preference cone significantly impacts the size and composition of the Pareto optimal set.
> <details>
> <summary>read the caption</summary>
> Figure 1: Effect of cone selection on size of Pareto optimal set
> </details>





![](https://ai-paper-reviewer.com/GvQU54uA7u/tables_13_1.jpg)

> 🔼 This table lists notations used throughout the paper.  It includes mathematical symbols representing concepts such as the ordering cone, number of arms and objectives, Pareto sets, the reward matrix, rewards, confidence balls, distances between Pareto fronts, allocation vectors, families of policies, and estimated/true mean rewards, among others. These notations are crucial for understanding the mathematical formulations and algorithms presented in the paper.
> <details>
> <summary>read the caption</summary>
> Table 1: Table of Notations
> </details>





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/GvQU54uA7u/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}