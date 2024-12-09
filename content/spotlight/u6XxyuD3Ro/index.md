---
title: "Online Convex Optimisation: The Optimal Switching Regret for all Segmentations Simultaneously"
summary: "Algorithm RESET achieves optimal switching regret simultaneously across all segmentations, offering efficiency and parameter-free operation."
categories: []
tags: ["AI Theory", "Optimization", "🏢 Alan Turing Institute",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} u6XxyuD3Ro {{< /keyword >}}
{{< keyword icon="writer" >}} Stephen Pasteris et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=u6XxyuD3Ro" target="_self" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93293" target="_self" >}}
↗ NeurIPS Homepage
{{< /button >}}


<audio controls>
    <source src="https://ai-paper-reviewer.com/u6XxyuD3Ro/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Online convex optimization often struggles with non-stationary problems.  Existing methods for handling this, using switching regret as a measure, either had suboptimal regret bounds or were computationally expensive.  The challenge is to find an algorithm that minimizes cumulative loss, accounting for changes in the optimal action over time. 



The paper introduces the RESET algorithm.  RESET cleverly uses a recursive tree structure and an adaptive base algorithm to achieve the asymptotically optimal switching regret simultaneously for every possible segmentation. Its efficiency is another highlight, with logarithmic space and per-trial time complexity.  Furthermore, it provides novel bounds on dynamic regret, showing its adaptability to changing conditions.  These results represent a significant advancement in online convex optimization, addressing existing limitations and proposing a novel, parameter-free and efficient solution.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Algorithm RESET achieves asymptotically optimal switching regret for all segmentations. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} RESET is highly efficient, with logarithmic space and per-trial time complexity. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} RESET obtains novel bounds on dynamic regret, adapting to comparator sequence variations. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because it offers **an efficient algorithm** that achieves the **asymptotically optimal switching regret** across all possible segmentations. This significantly improves upon previous methods that were either suboptimal or computationally expensive, opening up new avenues for online convex optimization in non-stationary settings.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/u6XxyuD3Ro/figures_6_1.jpg)

> The figure illustrates the RESET algorithm with 8 trials. The left panel shows how the base actions and mixing weights are generated for each level and trial. Each level runs an instance of the base algorithm, and the mixing weights are reset at the start of each segment (whose length determines how the mixing weights update). The right panel shows how the algorithm computes the action xt on trial t using a combination of base actions, mixing weights, and propagating actions.





![](https://ai-paper-reviewer.com/u6XxyuD3Ro/tables_5_1.jpg)

> The algorithm RESET (Recursion over Segment Tree) is presented. It uses a base algorithm for online convex optimisation, along with a mixing weight for each level in a segment tree to achieve the asymptotically optimal switching regret simultaneously across all possible segmentations.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/u6XxyuD3Ro/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}