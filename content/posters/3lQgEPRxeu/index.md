---
title: "Learning General Parameterized Policies for Infinite Horizon Average Reward Constrained MDPs via Primal-Dual Policy Gradient Algorithm"
summary: "First-ever sublinear regret & constraint violation bounds achieved for infinite horizon average reward CMDPs with general policy parametrization using a novel primal-dual policy gradient algorithm."
categories: []
tags: ["Machine Learning", "Reinforcement Learning", "🏢 Purdue University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 3lQgEPRxeu {{< /keyword >}}
{{< keyword icon="writer" >}} Qinbo Bai et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=3lQgEPRxeu" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/96718" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=3lQgEPRxeu&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/3lQgEPRxeu/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Reinforcement learning (RL) often faces real-world constraints.  Infinite horizon average reward settings are crucial for long-term decision-making but pose unique challenges, especially when policies are complex (general parameterization).  Existing solutions either use simple policy structures (tabular or linear) or lack rigorous theoretical guarantees for regret (difference from optimal policy) and constraint violations. 

This paper introduces a new primal-dual policy gradient algorithm designed to address these challenges.  By cleverly managing constraints and optimizing policies using policy gradient methods, the algorithm achieves **sublinear regret and constraint violation bounds of Õ(T<sup>4/5</sup>)**. This represents a significant advancement, offering the first sublinear regret guarantees for general policy parameterizations in average reward CMDPs and surpassing previous state-of-the-art results.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel primal-dual policy gradient algorithm is proposed for infinite horizon average reward CMDPs with general policy parameterization. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The algorithm achieves Õ(T<sup>4/5</sup>) objective regret and Õ(T<sup>4/5</sup>) constraint violation bounds, improving state-of-the-art results. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The work provides the first sublinear regret guarantee for average reward CMDPs with general parameterization. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it's the first to tackle the challenge of average reward Constrained Markov Decision Processes (CMDPs) with general policy parameterization**, a common scenario in real-world applications.  Its **sublinear regret and constraint violation bounds offer significant improvements** over existing methods and provide **a strong theoretical foundation for future research in constrained reinforcement learning**. This opens exciting new avenues for tackling complex decision-making problems under constraints in various domains.

------
#### Visual Insights





![](https://ai-paper-reviewer.com/3lQgEPRxeu/tables_1_1.jpg)

> This table compares several state-of-the-art algorithms for solving infinite-horizon average reward constrained Markov Decision Processes (CMDPs).  It contrasts their regret (difference between the achieved and optimal average reward) and constraint violation, indicating whether the algorithms are model-free (meaning they do not require an explicit model of the environment) and the type of policy parameterization (tabular, linear, or general). The table highlights that the proposed algorithm in the paper is novel in its analysis of regret and constraint violation for average reward CMDPs with general policy parameterization, which is a more challenging problem setting.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/3lQgEPRxeu/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}