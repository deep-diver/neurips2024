---
title: "Adam with model exponential moving average is effective for nonconvex optimization"
summary: "Clipped Adam with EMA achieves optimal convergence rates for smooth and non-smooth nonconvex optimization, particularly when scales vary across different coordinates."
categories: []
tags: ["AI Theory", "Optimization", "🏢 Microsoft Research",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} v416YLOQuU {{< /keyword >}}
{{< keyword icon="writer" >}} Kwangjun Ahn et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=v416YLOQuU" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93230" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=v416YLOQuU&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/v416YLOQuU/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many modern machine learning models rely on Adam and Exponential Moving Average (EMA) for optimization during training, yet a comprehensive theoretical understanding of their effectiveness remained elusive. Existing analyses often produced results inconsistent with practical observations, lacking a full explanation for the techniques' success.  This paper tackled this challenge. 

This research leverages the online-to-nonconvex conversion framework to analyze Adam with EMA. By focusing on the core elements of Adam (momentum and discounting factors) combined with EMA, the authors demonstrate that a clipped version of Adam with EMA achieves optimal convergence rates in various nonconvex settings, both smooth and nonsmooth. This new theoretical framework showcases the advantages of coordinate-wise adaptivity in situations with varying scales, thus offering a deeper understanding of Adam and EMA's power.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Clipped Adam with EMA achieves optimal convergence rates in various nonconvex optimization settings. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Coordinate-wise adaptivity of Adam is provably advantageous when scales vary across coordinates. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Analysis relies on momentum, discounting factors, and model EMA, providing theoretical justifications for their widespread use. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it provides **novel theoretical insights** into the effectiveness of Adam and EMA in nonconvex optimization.  It addresses a critical gap in understanding these widely used techniques, offering **optimal convergence guarantees**.  This could lead to **improved algorithm design** and a better understanding of deep learning training dynamics, influencing future research in optimization and machine learning.

------
#### Visual Insights





![](https://ai-paper-reviewer.com/v416YLOQuU/tables_4_1.jpg)

> This table summarizes the convergence rates achieved by various optimization algorithms, including Adam, clipped Adam, and SGD, under different assumptions on the objective function (smooth, non-smooth, and strongly convex). It highlights the optimal convergence rates achievable in each setting and shows which algorithms attain these optimal rates.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/v416YLOQuU/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v416YLOQuU/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}