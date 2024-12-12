---
title: "Lookback Prophet Inequalities"
summary: "This paper enhances prophet inequalities by allowing lookback, improving competitive ratios and providing algorithms for diverse observation orders, thereby bridging theory and real-world online selec..."
categories: []
tags: ["AI Theory", "Optimization", "🏢 ENSAE, Ecole Polytechnique",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} cg1vwt5Xou {{< /keyword >}}
{{< keyword icon="writer" >}} Ziyad Benomar et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=cg1vwt5Xou" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94401" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=cg1vwt5Xou&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/cg1vwt5Xou/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Prophet inequalities model sequential decision-making under uncertainty, but are often too simplistic for real-world applications. They assume irreversible decisions, ignoring possibilities like revisiting past options at a cost. This limitation motivated research into *D-prophet inequalities*, which allow recovering a fraction of a rejected item's value based on how long ago it was observed. This paper examines *D-prophet inequalities* under different observation orders (adversarial, random, IID), using a more general decay function. The classical prophet inequality provides a baseline for comparison.

The paper introduces new algorithms and establishes theoretical upper and lower bounds for competitive ratios in *D-prophet inequalities*. It shows that the decay function parameter (γ) significantly impacts the competitive ratio, surpassing the classical 1/2 bound under certain conditions. Results are generalized for random decay functions, further enhancing the applicability of the *D-prophet inequalities* model to real-world scenarios.  The paper provides both theoretical and algorithmic advancements for handling lookback in prophet inequalities, making it a valuable contribution to the optimal stopping and online decision-making literature. 

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Lookback significantly improves competitive ratios in prophet inequalities under mild monotonicity assumptions. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The optimal competitive ratio is determined solely by the decay function parameter γ, in adversarial and random order models. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Novel algorithms are introduced for the y-prophet inequality, outperforming classical prophet inequalities. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in optimal stopping problems and online decision-making.  It **significantly advances our understanding of prophet inequalities**, a classic model often too pessimistic for real-world scenarios, by incorporating the realistic possibility of revisiting past opportunities. The results **offer refined competitive ratio analyses** and **novel algorithms**, opening avenues for improving online selection strategies in various applications.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/cg1vwt5Xou/figures_3_1.jpg)

> This figure shows the lower and upper bounds on the competitive ratio for the D-prophet inequality. The competitive ratio is a measure of how well an algorithm performs compared to the optimal solution. The x-axis represents the parameter YD, which quantifies the value that can be recovered from a rejected item. The y-axis represents the competitive ratio. The figure shows that the competitive ratio increases as YD increases, indicating that the lookback mechanism improves the performance of the algorithm. The figure also shows that the competitive ratio is higher in the adversarial order model compared to the random order and IID models. This is because in the adversarial order model, the adversary can choose the order in which the items are observed, which can make it more difficult for the algorithm to select the best item. In the random order model, the order in which the items are observed is random, which makes it less likely that the algorithm will be able to select the best item. In the IID model, the items are sampled independently from the same distribution, which makes it easier for the algorithm to select the best item.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/cg1vwt5Xou/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}