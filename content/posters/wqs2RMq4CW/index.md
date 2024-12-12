---
title: "Corruption-Robust Linear Bandits: Minimax Optimality and Gap-Dependent Misspecification"
summary: "This paper presents novel algorithms for linear bandits that are robust to corrupted rewards, achieving minimax optimality and optimal scaling for gap-dependent misspecification, extending to reinforc..."
categories: []
tags: ["AI Theory", "Robustness", "🏢 University of Virginia",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} wqs2RMq4CW {{< /keyword >}}
{{< keyword icon="writer" >}} Haolin Liu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=wqs2RMq4CW" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93118" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=wqs2RMq4CW&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/wqs2RMq4CW/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many real-world machine learning problems involve corrupted or misspecified data.  Existing work often focuses on uniform corruption or misspecification bounds, failing to capture scenarios where the error scales with the action's suboptimality.  This limits the applicability of these methods.

This research introduces novel algorithms specifically designed for linear bandits with corrupted rewards. The key innovation is to consider two types of corruption: strong (corruption level depends on learner's action) and weak (corruption is independent of action).  The study fully characterizes the minimax regret for stochastic linear bandits under both corruptions and develops upper and lower bounds for adversarial settings. Importantly, it links corruption-robust algorithms to those that handle gap-dependent misspecification, achieving optimal results.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} The paper provides a unified framework for analyzing strong and weak corruptions in linear bandits, fully characterizing the gap between their minimax regrets. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} It introduces efficient algorithms that achieve optimal scaling in the corruption level for both stochastic and adversarial linear bandits. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} A general reduction enables corruption-robust algorithms to handle gap-dependent misspecification, extending optimal results to linear MDPs. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it bridges the gap between corruption-robust learning and learning with gap-dependent misspecification.**  It provides a unified framework for analyzing these issues, offers efficient algorithms with optimal scaling in corruption levels, and extends these findings to reinforcement learning.  This work is timely, given the increasing focus on robust learning in real-world applications.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/wqs2RMq4CW/figures_5_1.jpg)

> Figure 1(a) shows the bonus function used in Algorithm 2, which adds an exploration bonus to compensate for the bias of the reward estimator in the presence of corruption.  Figure 1(b) provides a geometric interpretation of the bonus function, illustrating how it adjusts the covariance matrix to account for corruption and ensure sufficient exploration. The bonus function ensures that the exploration bonus term is always positive semi-definite and is carefully designed to achieve optimal regret bounds.





![](https://ai-paper-reviewer.com/wqs2RMq4CW/tables_3_1.jpg)

> This table summarizes the upper and lower bounds on the regret for stochastic and adversarial linear bandits under two different corruption measures: C and C∞.  The C measure represents strong corruption, where the corruption level depends on the learner's action, while C∞ represents weak corruption, where it does not. The table highlights the gap between the minimax regret under these two types of corruptions, showing how the optimal scaling for the corruption level varies based on the bandit setting.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wqs2RMq4CW/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}