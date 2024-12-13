---
title: "Sparsity-Agnostic Linear Bandits with Adaptive Adversaries"
summary: "SparseLinUCB:  First sparse regret bounds for adversarial action sets with unknown sparsity, achieving superior performance over existing methods!"
categories: []
tags: ["Machine Learning", "Reinforcement Learning", "🏢 National University of Singapore",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} jIabKyXOTt {{< /keyword >}}
{{< keyword icon="writer" >}} Tianyuan Jin et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=jIabKyXOTt" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93969" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=jIabKyXOTt&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/jIabKyXOTt/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Stochastic linear bandits optimize rewards by linearly weighting features, but existing approaches often assume known sparsity (number of relevant features).  This limits their effectiveness in real-world situations.  Moreover, existing algorithms often struggle with adversarially generated action sets, a more challenging scenario than stochastic ones.

This paper introduces SparseLinUCB, a novel algorithm addressing these issues. It cleverly uses a randomized model selection method across a hierarchy of confidence sets to achieve state-of-the-art sparse regret bounds even without knowing the sparsity or facing adversarial action sets.  A variant, AdaLinUCB, further enhances empirical performance in stochastic linear bandits.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} SparseLinUCB achieves Õ(S√dT) regret bound for adversarial action sets with unknown sparsity. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} AdaLinUCB uses Exp3 to improve empirical performance in stochastic settings, achieving Õ(√T) regret. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The paper provides the first instance-dependent regret bound for the sparsity-agnostic setting. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **sparse linear bandits**, a vital area in online learning and decision-making. It presents **novel algorithms** that achieve state-of-the-art performance without relying on prior knowledge of sparsity or restrictive assumptions on action sets. This addresses a major limitation in existing methods and **opens new avenues** for improving efficiency and adapting to real-world scenarios characterized by unknown sparsity and adversarial actions.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/jIabKyXOTt/figures_9_1.jpg)

> This figure shows the experimental results comparing the performance of AdaLinUCB and SparseLinUCB with OFUL under different sparsity levels (S = 1, 2, 4, 8, 16).  Each plot displays cumulative regret against the time steps.  The shaded area represents the one standard deviation bound across 20 repetitions for each algorithm. AdaLinUCB consistently outperforms OFUL and SparseLinUCB across all sparsity levels.





![](https://ai-paper-reviewer.com/jIabKyXOTt/tables_2_1.jpg)

> This table compares the proposed algorithm with existing sparse linear bandit algorithms. It shows the expected regret bounds achieved by each algorithm under different assumptions regarding the sparsity level, adaptive adversary, and action set properties.  The table highlights the novelty of the proposed algorithm in achieving sparse regret bounds without requiring prior knowledge of sparsity or making strong assumptions on the action sets.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jIabKyXOTt/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}