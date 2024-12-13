---
title: "A Simple and Adaptive Learning Rate for FTRL in Online Learning with Minimax Regret of Θ(T^{2/3}) and its Application to Best-of-Both-Worlds"
summary: "A new adaptive learning rate for FTRL achieves minimax regret of O(T²/³) in online learning, improving existing best-of-both-worlds algorithms for various hard problems."
categories: []
tags: ["AI Theory", "Optimization", "🏢 University of Tokyo",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} XlvUz9F50g {{< /keyword >}}
{{< keyword icon="writer" >}} Taira Tsuchiya et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=XlvUz9F50g" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94746" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=XlvUz9F50g&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/XlvUz9F50g/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many online learning problems, such as partial monitoring and graph bandits, exhibit a minimax regret of O(T²/³),  representing a significant challenge for algorithm designers.  Existing adaptive learning rates primarily focus on problems with a different regret bound, lacking efficient solutions for these "hard" problems.  Additionally, most adaptive learning rates aren't designed to seamlessly handle both stochastic and adversarial environments.

This research introduces a novel adaptive learning rate framework for FTRL, which greatly simplifies the design of online learning algorithms. By meticulously matching stability, penalty, and bias terms in the regret bound, a surprisingly simple learning rate is obtained. This method improves existing best-of-both-worlds regret upper bounds for partial monitoring, graph bandits, and multi-armed bandits with paid observations. The resulting learning rates are surprisingly simple compared to existing ones and achieve nearly optimal regret bounds in both stochastic and adversarial scenarios.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel adaptive learning rate framework for FTRL is developed, achieving a minimax regret of O(T²/³). {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} This framework offers significantly simpler algorithms and improved regret bounds for several hard online learning problems compared to existing methods. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The work provides a unified best-of-both-worlds guarantee for hard online learning problems, offering simultaneous optimality in stochastic and adversarial regimes. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in online learning because it introduces a novel adaptive learning rate framework for problems with a minimax regret of O(T²/³). This expands the applicability of Follow-the-Regularized-Leader (FTRL) algorithms, which are known for their effectiveness in various online learning scenarios, but have been limited in their use with indirect feedback.  The proposed framework offers significantly simpler algorithms with improved regret bounds compared to existing approaches. It also provides a unified approach to achieving the best-of-both-worlds (BOBW) guarantee for hard problems, significantly advancing the field.

------
#### Visual Insights





![](https://ai-paper-reviewer.com/XlvUz9F50g/tables_2_1.jpg)

> This table compares the regret bounds (in both stochastic and adversarial settings, and adversarial setting with self-bounding constraint) achieved by different algorithms for three online learning problems: partial monitoring, graph bandits, and multi-armed bandits with paid observations.  The comparison highlights the improvements achieved by the proposed algorithm, particularly in achieving the best-of-both-worlds (BOBW) property.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XlvUz9F50g/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}