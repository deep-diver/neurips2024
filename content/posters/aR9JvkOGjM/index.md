---
title: "Improved Regret for Bandit Convex Optimization with Delayed Feedback"
summary: "A novel algorithm, D-FTBL, achieves improved regret bounds for bandit convex optimization with delayed feedback, tightly matching existing lower bounds in worst-case scenarios."
categories: []
tags: ["AI Theory", "Optimization", "🏢 Zhejiang University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} aR9JvkOGjM {{< /keyword >}}
{{< keyword icon="writer" >}} Yuanyu Wan et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=aR9JvkOGjM" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94556" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=aR9JvkOGjM&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/aR9JvkOGjM/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Bandit convex optimization (BCO) deals with the challenge of learning optimal actions from limited feedback, made harder when feedback is delayed. Existing algorithms for delayed BCO had a significant gap between their regret bounds and the theoretical lower bound. This gap is problematic because the regret, representing the cumulative difference between an algorithm's performance and the optimal strategy, is a key metric for evaluating the effectiveness of BCO algorithms. 

This research introduces a novel algorithm, called D-FTBL (Delayed Follow-The-Bandit-Leader). It cleverly incorporates a blocking update mechanism which separates the effects of delays and noisy feedback measurements. The results show that D-FTBL achieves a significantly improved regret bound, effectively closing the gap with the theoretical lower bound. Furthermore, the algorithm shows adaptability in scenarios with strongly convex functions, making it useful in a wider range of practical applications.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} D-FTBL algorithm improves regret bounds for bandit convex optimization with delayed feedback. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The delay-dependent part of the regret bound is shown to be tight in the worst case. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The algorithm's effectiveness is demonstrated empirically on real-world datasets. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in online convex optimization and bandit algorithms due to its significant improvement on regret bounds for problems with delayed feedback.  It provides **tight theoretical guarantees** and **practical algorithms**, addressing a critical gap in existing research.  The results also offer valuable insights and new avenues for research in related fields like online routing and online advertising, where delays are inherent challenges.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/aR9JvkOGjM/figures_9_1.jpg)

> This figure shows the total loss of three algorithms (D-FTBL, GOLD, and Improved GOLD) on two datasets (ijcnn1 and SUSY) for different maximum delays (d).  The x-axis represents the maximum delay, and the y-axis represents the total loss.  The plot demonstrates the performance comparison of the algorithms in a delayed online binary classification setting.  D-FTBL generally shows lower loss than the other two algorithms, especially as the maximum delay increases.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/aR9JvkOGjM/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}