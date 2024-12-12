---
title: "Derandomizing Multi-Distribution Learning"
summary: "Derandomizing multi-distribution learning is computationally hard, but a structural condition allows efficient black-box conversion of randomized predictors to deterministic ones."
categories: []
tags: ["AI Theory", "Optimization", "🏢 Aarhus University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} twYE75Mnkt {{< /keyword >}}
{{< keyword icon="writer" >}} Kasper Green Larsen et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=twYE75Mnkt" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93304" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=twYE75Mnkt&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/twYE75Mnkt/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Multi-distribution learning aims to train a single predictor that performs well across multiple data distributions. Existing algorithms produce randomized predictors, raising the question of derandomization.  This poses challenges as deterministic predictors are preferred for their simplicity and guarantees. The paper investigates the computational complexity of this task.

The paper proves derandomizing multi-distribution learning is computationally hard in general, even when the empirical risk minimization (ERM) is efficient.  However, it also identifies a crucial structural condition (label-consistent distributions) that allows for efficient derandomization.  A novel algorithm is presented, demonstrating a near-optimal sample complexity for deterministic multi-distribution learning under this condition using a black box reduction.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Derandomizing multi-distribution learning is computationally hard, even with efficient ERM. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A structural condition allows efficient black-box derandomization of multi-distribution learning predictors. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} New algorithm achieves near-optimal sample complexity for deterministic multi-distribution learning under label-consistent distributions. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it tackles the computational challenge of derandomizing multi-distribution learning**, a significant hurdle in machine learning.  Its findings are relevant to researchers working on improving the efficiency and robustness of algorithms dealing with diverse data distributions, especially in the context of fairness and robustness.

------
#### Visual Insights







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/twYE75Mnkt/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}