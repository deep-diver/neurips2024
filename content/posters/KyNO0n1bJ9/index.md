---
title: "The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels"
summary: "Researchers found the minimax optimal rate of HSIC estimation for translation-invariant kernels is O(n⁻¹/²), settling a two-decade-old open question and validating many existing HSIC estimators."
categories: []
tags: ["AI Theory", "Optimization", "🏢 Karlsruhe Institute of Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} KyNO0n1bJ9 {{< /keyword >}}
{{< keyword icon="writer" >}} Florian Kalinke et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=KyNO0n1bJ9" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95630" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=KyNO0n1bJ9&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/KyNO0n1bJ9/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Estimating independence between random variables using kernel methods is crucial in various fields, with the Hilbert-Schmidt Independence Criterion (HSIC) being a popular choice. However, the optimal estimation rate for HSIC remained unknown for nearly two decades, hindering the development of efficient and reliable algorithms. This paper addresses this critical issue.

This research establishes the minimax lower bound for HSIC estimation, proving that the existing estimators, such as U-statistic, V-statistic, and Nyström-based ones, achieve the optimal rate of O(n⁻¹/²). This finding settles a long-standing open problem, confirming the efficiency of commonly used methods.  Moreover, the results extend to estimating cross-covariance operators, providing a more solid theoretical foundation for applications using HSIC.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} The minimax optimal rate of HSIC estimation for translation-invariant kernels on Rᵈ is O(n⁻¹/²). {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} This rate confirms the optimality of U-statistic, V-statistic, and Nyström-based HSIC estimators. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The findings extend to the estimation of cross-covariance operators, impacting various applications. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working with kernel methods and independence measures.  It establishes a **minimax optimal rate** for HSIC estimation, resolving a long-standing open problem and **validating the optimality of many existing estimators**. This directly impacts the reliability and efficiency of numerous applications relying on HSIC, from independence testing to causal discovery.

------
#### Visual Insights







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KyNO0n1bJ9/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}