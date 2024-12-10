---
title: Generalized Linear Bandits with Limited Adaptivity
summary: This paper introduces two novel algorithms, achieving optimal regret in generalized
  linear contextual bandits despite limited policy updates, a significant advancement
  for real-world applications.
categories: []
tags:
- Reinforcement Learning
- "\U0001F3E2 Stanford University"
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} FTPDBQuT4G {{< /keyword >}}
{{< keyword icon="writer" >}} Ayush Sawarni et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=FTPDBQuT4G" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95973" target="_blank" >}}
↗ NeurIPS Proc.
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=FTPDBQuT4G&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/FTPDBQuT4G/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many real-world applications of contextual bandits, such as clinical trials and online advertising, face the challenge of limited adaptivity, where frequent policy updates are infeasible. Existing algorithms often struggle with this constraint or fail to provide strong theoretical guarantees. This paper addresses this limitation by studying the contextual bandit problem with generalized linear reward models.  The problem is particularly challenging due to instance-dependent non-linearity parameters that can significantly affect the performance of algorithms. 

The researchers propose two novel algorithms, B-GLinCB and RS-GLinCB, to tackle the problem. B-GLinCB is designed for a setting where the update rounds must be decided upfront. RS-GLinCB addresses a more general setting where the algorithm can adaptively choose when to update its policy.  Both algorithms achieve optimal regret guarantees (Õ(√T)), notably eliminating the dependence on the instance-dependent non-linearity parameter. This is a significant improvement over previous algorithms, which often have sub-optimal regret bounds. The algorithms also demonstrate computational efficiency, making them practically feasible for various real-world applications. 

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Developed B-GLinCB and RS-GLinCB algorithms for generalized linear contextual bandits under limited adaptivity settings. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Achieved optimal regret bounds (Õ(√T)) for both settings, eliminating dependence on a key instance-dependent parameter. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Demonstrated computational efficiency and superior performance compared to existing algorithms. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it addresses the critical challenge of limited adaptivity in contextual bandit algorithms**, a common constraint in real-world applications. By providing novel algorithms with **optimal regret guarantees that are free of instance-dependent parameters**, it enables more effective decision-making in various scenarios with limited feedback. This work **opens up new avenues for research** into designing efficient algorithms for scenarios with limited feedback while maintaining optimality.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/FTPDBQuT4G/figures_8_1.jpg)

> The figure compares the performance of RS-GLinUCB with other algorithms (ECOLog, GLOC, GLM-UCB) in terms of cumulative regret and execution time. The top panels show the cumulative regret for Logistic and Probit reward models, while the bottom panels show the execution time for different values of kappa (a measure of non-linearity).  RS-GLinUCB demonstrates lower regret and comparable or better execution times.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FTPDBQuT4G/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}