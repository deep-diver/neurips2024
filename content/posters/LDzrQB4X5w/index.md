---
title: "A Best-of-both-worlds Algorithm for Bandits with Delayed Feedback with Robustness to Excessive Delays"
summary: "New best-of-both-worlds bandit algorithm tolerates arbitrary excessive delays, overcoming limitations of prior work that required prior knowledge of maximal delay and suffered linear regret dependence..."
categories: []
tags: ["Machine Learning", "Reinforcement Learning", "🏢 Churney ApS",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} LDzrQB4X5w {{< /keyword >}}
{{< keyword icon="writer" >}} Saeed Masoudian et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=LDzrQB4X5w" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95613" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=LDzrQB4X5w&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/LDzrQB4X5w/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many real-world applications involve online decision-making where feedback is delayed and occasionally very late.  Existing bandit algorithms often struggle in these scenarios, especially when dealing with unpredictable delays and outliers.  Moreover, these algorithms frequently require prior knowledge of the maximal delay, making them less practical.  This creates a need for algorithms that work well across a broad range of delay patterns and are robust to outliers. 

This paper introduces a new algorithm that addresses these issues. It is designed to work well regardless of whether the underlying loss generating process is stochastic or adversarial—what's called a "best-of-both-worlds" approach.  This is achieved through three key innovations:  a novel implicit exploration scheme, a method for controlling distribution drift without assuming bounded delays, and a procedure that links the standard regret to the regret calculated with the delayed and potentially skipped observations. The resulting algorithm offers significant improvements in terms of robustness and performance compared to previous methods.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel best-of-both-worlds algorithm for bandits with variably delayed feedback that is robust to delay outliers. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} An efficient technique to control distribution drift under highly varying delays. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} An implicit exploration scheme that works in best-of-both-worlds setting. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **bandit algorithms with delayed feedback**, a common challenge in various real-world applications. It offers a novel solution to improve the robustness and efficiency of these algorithms, thus opening new avenues for practical applications and further research in online learning. The **best-of-both-worlds** approach is particularly relevant, offering improved performance across stochastic and adversarial settings. It addresses critical limitations in existing research, focusing on dealing with **outliers and excessive delays**. The findings of this study are important for advancing our understanding and solutions for the class of online learning problems.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/LDzrQB4X5w/figures_4_1.jpg)

> This algorithm is a best-of-both-worlds modification of the adversarial FTRL algorithm with a hybrid regularizer.  It incorporates three key innovations:  biased loss estimators (implicit exploration), an adjusted skipping threshold, and a novel control of distribution drift under highly varying delays. The algorithm maintains a set of skipped rounds, a cumulative count of active outstanding observations, and a vector of cumulative observed loss estimates.  At each round, it samples an arm based on an FTRL distribution, updates loss estimates, counts outstanding observations, and applies a skipping threshold to manage excessive delays.  The skipping threshold and loss estimates are dynamically adjusted based on the running count of outstanding observations.





![](https://ai-paper-reviewer.com/LDzrQB4X5w/tables_1_1.jpg)

> This table compares the key results (regret bounds) of the proposed algorithm with those of several state-of-the-art algorithms for bandits with delayed feedback.  It highlights the differences in terms of assumptions (e.g., knowledge of maximal delay), regret bounds (stochastic and adversarial), and the use of skipping techniques to handle excessive delays.  The notation used in the regret bounds is defined in the caption, allowing for a detailed comparison of the algorithm's performance under different scenarios.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LDzrQB4X5w/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}