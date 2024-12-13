---
title: "Functionally Constrained Algorithm Solves Convex Simple Bilevel Problem"
summary: "Near-optimal algorithms solve convex simple bilevel problems by reformulating them into functionally constrained problems, achieving near-optimal convergence rates."
categories: []
tags: ["AI Theory", "Optimization", "🏢 Tsinghua University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} PAiGHJppam {{< /keyword >}}
{{< keyword icon="writer" >}} Huaqing Zhang et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=PAiGHJppam" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95326" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=PAiGHJppam&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/PAiGHJppam/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Simple bilevel optimization problems involve minimizing an upper-level function while satisfying the optimal solution of a lower-level function.  Existing methods struggle to find precise solutions due to the inherent complexity of these problems. This often results in settling for approximate solutions, but even these are difficult to obtain using standard first-order methods. 

This paper addresses this by proposing a novel approach, FC-BiO (Functionally Constrained Bilevel Optimizer). FC-BiO cleverly transforms the bilevel problem into a functionally constrained one. By doing so, the authors develop methods capable of finding near-optimal solutions.  This is a significant contribution because it provides both theoretical guarantees and practical algorithms that are demonstrably better than previously available methods. The algorithm's effectiveness is validated through numerical experiments demonstrating its superior performance on benchmark problems.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Simple bilevel problems' approximate optimal value is unobtainable by first-order zero-respecting algorithms. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Novel near-optimal methods for smooth and nonsmooth simple bilevel problems are proposed by reformulating them into functionally constrained problems. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} FC-BiO algorithm achieves near-optimal convergence rates for finding weak optimal solutions. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **bilevel optimization** problems. It highlights the **fundamental difficulties** in finding exact solutions and proposes **novel near-optimal algorithms** that address these challenges. This work opens avenues for developing more efficient methods in various machine learning and AI applications, **advancing the state-of-the-art** in simple bilevel optimization.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/PAiGHJppam/figures_8_1.jpg)

> This figure compares the performance of the proposed algorithm (FC-BiO) with several other algorithms (FC-BiOLip, AGM-BIO, PB-APG, Bi-SG, a-IRG, CG-BIO, and Bisec-BiO) on solving Problem (11), a simple bilevel problem with smooth objectives.  The x-axis represents the computation time, and the y-axis shows the difference between the current objective values and the optimal value for both the upper-level function (f(x) - f*) and the lower-level function (g(x) - g*). The plot demonstrates how quickly each algorithm converges towards the optimal solution.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/PAiGHJppam/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/PAiGHJppam/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}