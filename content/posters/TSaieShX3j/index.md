---
title: "SGD vs GD: Rank Deficiency in Linear Networks"
summary: "SGD surprisingly diminishes network rank, unlike GD, due to a repulsive force between eigenvalues, offering insights into deep learning generalization."
categories: []
tags: ["AI Theory", "Optimization", "🏢 EPFL",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} TSaieShX3j {{< /keyword >}}
{{< keyword icon="writer" >}} Aditya Varre et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=TSaieShX3j" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95034" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=TSaieShX3j&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/TSaieShX3j/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many researchers are interested in the capability of deep neural networks to learn effective representations despite being heavily overparameterized. This has sparked significant interest in the role of gradient methods and the noise they introduce during training on this capability.  This paper studies this issue by analyzing the behavior of gradient methods on a simplified two-layer linear network. It focuses on the difference between GD and SGD, particularly how stochasticity affects the learning process.

The authors demonstrate that, unlike GD, SGD diminishes the rank of the network's parameter matrix. Using continuous-time analysis, they derive a stochastic differential equation that explains this rank deficiency, highlighting a key regularization mechanism that results in simpler structures.  Their findings are supported by experiments beyond the simplified linear network model, showing the phenomenon applies to more complex architectures and tasks.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Stochastic Gradient Descent (SGD) reduces the rank of the parameter matrix in two-layer linear networks, unlike Gradient Descent (GD). {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} This rank deficiency is caused by a repulsive force between eigenvalues, revealed by the derived stochastic differential equation. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The findings provide valuable insights into the role of stochasticity in deep learning generalization and implicit regularization. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because it reveals a fundamental difference in how **stochastic gradient descent (SGD)** and **gradient descent (GD)** impact the learning process in linear neural networks.  It challenges existing assumptions about the role of noise in generalization by showing that noise, far from just helping the model to converge to better solutions, actively simplifies network structures by reducing rank. This research has important implications for understanding deep learning generalization and designing more efficient algorithms.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/TSaieShX3j/figures_8_1.jpg)

> This figure shows the evolution of the model characteristics, specifically the determinant of matrix M and the top 5 singular values of matrix W1, under gradient flow (δ = 0) and stochastic gradient flow (δ = 2).  The left panel illustrates how the determinant remains constant under gradient flow but decays to zero under stochastic gradient flow, highlighting a key difference between the two methods. The right panel displays the trajectories of the top 5 singular values.  It shows how under stochasticity, the smaller singular values tend towards zero, indicating a rank deficiency that does not occur under gradient flow. This visualization supports the paper's central claim about the impact of stochasticity on the rank of the parameter matrices.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/TSaieShX3j/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/TSaieShX3j/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}