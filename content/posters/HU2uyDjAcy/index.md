---
title: "Local and Adaptive Mirror Descents in Extensive-Form Games"
summary: "LocalOMD: Adaptive OMD in extensive-form games achieves near-optimal sample complexity by using fixed sampling and local updates, reducing variance and generalizing well."
categories: []
tags: ["Machine Learning", "Reinforcement Learning", "🏢 CREST - FairPlay, ENSAE Paris",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} HU2uyDjAcy {{< /keyword >}}
{{< keyword icon="writer" >}} Côme Fiegel et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=HU2uyDjAcy" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95830" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=HU2uyDjAcy&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/HU2uyDjAcy/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many existing methods for learning in extensive-form games suffer from high variance due to importance sampling. This paper addresses this issue by proposing a fixed sampling approach where the sampling policy is fixed and not updated over time. This reduces variance in gain estimates, but traditional methods for regret minimization still suffer from large variance, even with a fixed sampling policy. 

The proposed algorithm, LocalOMD, uses an adaptive Online Mirror Descent (OMD) algorithm that applies OMD locally to each information set with individually decreasing learning rates.  It leverages a regularized loss function to ensure stability. Importantly, LocalOMD avoids the use of importance sampling, addressing the high-variance problem and offering a convergence rate of Õ(T-1/2) with high probability. The algorithm shows near-optimal dependence on the game parameters when using optimal learning rates and sampling policies.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} LocalOMD algorithm provides a new approach to learning in extensive-form games. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} LocalOMD achieves a near-optimal sample complexity, improving efficiency. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The fixed sampling approach significantly reduces variance in loss estimation. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents **LocalOMD**, a novel algorithm for learning optimal strategies in extensive-form games. It addresses the high variance issue in existing methods by employing a **fixed sampling approach and adaptive online mirror descent**. This offers **near-optimal sample complexity** and opens up new avenues for research in game theory and reinforcement learning.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/HU2uyDjAcy/figures_8_1.jpg)

> This figure compares the performance of several algorithms (BalancedCFR, BalancedFTRL, LocalOMD Optimal, LocalOMD Adaptive) across three different poker games (Kuhn Poker, Leduc Poker, Liars Dice).  The top row shows the exploitability gap (a measure of how far from optimal the learned strategies are) as a function of the number of episodes of training.  The bottom row illustrates the empirical variance of the loss estimations during training. The figure demonstrates that LocalOMD achieves lower exploitability and variance compared to other algorithms.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HU2uyDjAcy/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}