---
title: "Learning to Mitigate Externalities: the Coase Theorem with Hindsight Rationality"
summary: "Economists learn to resolve externalities efficiently even when players lack perfect information, maximizing social welfare by leveraging bargaining and online learning."
categories: []
tags: ["AI Theory", "Optimization", "🏢 UC Berkeley",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} omyzrkacme {{< /keyword >}}
{{< keyword icon="writer" >}} Antoine Scheid et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=omyzrkacme" target="_self" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93602" target="_self" >}}
↗ NeurIPS Homepage
{{< /button >}}


<audio controls>
    <source src="https://ai-paper-reviewer.com/omyzrkacme/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Externalities, indirect effects impacting global welfare from economic interactions, pose a significant challenge in economics.  Most models assume perfect knowledge among players, hindering practical implementation. This paper tackles this by extending the Coase Theorem, which suggests property rights and bargaining can optimize social welfare in the presence of externalities, to online settings where players lack complete information.

The researchers employ a two-player multi-armed bandit framework to model economic interactions, where player actions influence others' rewards. They demonstrate that without property rights, social welfare suffers.  To address this, they devise a policy for players to learn bargaining strategies and maximize total welfare. This strategy involves transfers that incentivize actions that benefit overall social welfare, thus extending Coase's Theorem under conditions of uncertainty and incomplete information. The solution demonstrates a sub-linear social welfare regret, indicating efficiency.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} The classical Coase theorem holds under uncertainty, provided players can negotiate and compensate for externalities. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A novel policy enables efficient bargaining in online settings by allowing players to learn their preferences and make strategic transfers. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The proposed approach addresses the limitations of traditional economic models by accounting for uncertainty and incomplete information. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers studying economics, game theory, and online learning.  It **bridges the gap between theoretical economic models and practical online settings**, offering a novel approach to address externalities under uncertainty. This work **opens new avenues** for developing efficient mechanisms in various applications involving strategic interactions and incomplete information.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/omyzrkacme/figures_8_1.jpg)

> This figure shows the empirical frequencies of actions taken by the upstream player in two scenarios: one without property rights and one with property rights.  The left panel depicts the scenario without property rights, demonstrating inefficient outcomes due to the externality, while the right panel shows the efficient outcome achieved when property rights are established and bargaining is possible. The plots illustrate the convergence of the system towards the social welfare optimum when property rights are enforced.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/omyzrkacme/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/omyzrkacme/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}