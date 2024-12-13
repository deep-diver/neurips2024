---
title: "Nearly Minimax Optimal Submodular Maximization with Bandit Feedback"
summary: "This research establishes the first minimax optimal algorithm for submodular maximization with bandit feedback, achieving a regret bound matching the lower bound."
categories: []
tags: ["AI Theory", "Optimization", "🏢 University of Washington",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Vn0FWRImra {{< /keyword >}}
{{< keyword icon="writer" >}} Artin Tajdini et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Vn0FWRImra" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94877" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=Vn0FWRImra&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/Vn0FWRImra/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many real-world problems involve selecting the best subset from a large pool of items to maximize a certain objective, often under noisy feedback.  This is particularly challenging when the objective function is complex but exhibits a desirable mathematical property called 'submodularity', indicating diminishing returns. Existing algorithms struggle with this problem in high-dimensional spaces with stochastic, noisy observations.

This paper tackles this challenge by presenting a novel algorithm that offers a much-improved guarantee in terms of regret, a metric representing the algorithm's performance compared to the best possible outcome.  It accomplishes this by carefully balancing exploration and exploitation of different subsets and achieving a theoretical optimal performance guarantee that matches a newly proven lower bound on regret, highlighting its efficiency and optimality.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} First minimax optimal algorithm for submodular maximization with bandit feedback. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Novel regret bound that scales optimally across different problem regimes. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Theoretical lower bound that helps benchmark future algorithms. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in submodular optimization and bandit algorithms.  It provides **the first minimax optimal algorithm** for maximizing submodular functions under bandit feedback, closing a long-standing gap in the literature.  This opens avenues for research in more complex settings and related problems like combinatorial bandits with more sophisticated feedback models.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/Vn0FWRImra/figures_8_1.jpg)

> This figure compares the performance of three different algorithms for the weighted set cover problem: SUB-UCB (the authors' proposed algorithm), UCB on all sets of size k, and ETCG (an existing explore-then-commit algorithm). The x-axis represents the time horizon T, and the y-axis represents the total regret.  The plot shows that SUB-UCB consistently outperforms the other two algorithms across various time horizons, demonstrating its efficiency in minimizing regret for submodular maximization with bandit feedback.





![](https://ai-paper-reviewer.com/Vn0FWRImra/tables_3_1.jpg)

> This table summarizes the state-of-the-art regret bounds for combinatorial multi-armed bandits, focusing on submodular maximization problems.  It compares upper and lower bounds under various assumptions about the reward function (submodular and monotone, submodular without monotonicity, and degree-d polynomial), feedback type (stochastic and adversarial), and the measure of regret used (Rgr and R(S*)). The table highlights the contributions of the current work by presenting novel upper and lower bounds for submodular and monotone functions under stochastic feedback, which closes the gap between previous best known results.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Vn0FWRImra/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}