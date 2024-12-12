---
title: "An Accelerated Algorithm for Stochastic Bilevel Optimization under Unbounded Smoothness"
summary: "AccBO: A new accelerated algorithm achieves O(ε⁻³) oracle complexity for stochastic bilevel optimization with unbounded smoothness, significantly improving upon existing O(ε⁻⁴) methods."
categories: []
tags: ["Machine Learning", "Meta Learning", "🏢 George Mason University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} v7vYVvmfru {{< /keyword >}}
{{< keyword icon="writer" >}} Xiaochuan Gong et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=v7vYVvmfru" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93226" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=v7vYVvmfru&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/v7vYVvmfru/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many machine learning applications involve bilevel optimization problems.  However, existing algorithms struggle with problems where the upper-level function's smoothness is unbounded—meaning its smoothness constant scales with the gradient norm, lacking a uniform upper bound.  This makes finding an approximate solution (within a certain error tolerance) computationally expensive, limiting their applicability to large-scale problems.

This paper introduces AccBO, a novel algorithm designed to tackle this challenge.  **AccBO achieves a significantly improved convergence rate (O(ε⁻³)) compared to the state-of-the-art (O(ε⁻⁴))**. This improvement stems from its use of normalized stochastic gradient descent with recursive momentum for the upper-level and stochastic Nesterov accelerated gradient descent for the lower-level variables.  The improved algorithm is rigorously analyzed and shown to provide the predicted theoretical speedup through experiments on various machine learning problems.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} AccBO achieves an oracle complexity of O(ε⁻³) for stochastic bilevel optimization problems with unbounded smoothness in the upper-level function and strong convexity in the lower-level function. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The algorithm employs normalized stochastic gradient descent with recursive momentum for the upper-level variable and stochastic Nesterov accelerated gradient descent for the lower-level variable. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Experimental results demonstrate AccBO's superiority over existing baselines in various applications, validating the theoretical acceleration. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in bilevel optimization and related machine learning fields.  It addresses a significant limitation of existing methods by handling unbounded smoothness in nonconvex upper-level problems. **The improved oracle complexity of O(ε⁻³) offers a substantial efficiency gain**, making it relevant for large-scale applications.  The novel algorithmic techniques and theoretical analysis open doors for developing more efficient and scalable bilevel optimization algorithms.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/v7vYVvmfru/figures_8_1.jpg)

> This figure presents the results of applying bilevel optimization algorithms to a deep AUC maximization task.  Subfigures (a) and (b) display the training and test AUC scores, respectively, plotted against the number of training epochs.  Subfigures (c) and (d) show the same metrics but plotted against the running time of the algorithms.  The figure compares the performance of the proposed AccBO algorithm against several baselines, including StocBio, TTSA, SABA, MA-SOBA, SUSTAIN, VRBO, and BO-REP.  The results demonstrate that AccBO achieves higher AUC scores and faster convergence than the other methods.





![](https://ai-paper-reviewer.com/v7vYVvmfru/tables_4_1.jpg)

> This algorithm details the steps of the Stochastic Nesterov Accelerated Gradient Method, which is a crucial subroutine used within the AccBO algorithm for updating the lower-level variable. It leverages momentum to accelerate convergence towards the optimal solution of the lower-level problem, given a fixed upper-level variable. The algorithm iteratively updates the lower-level variable using stochastic gradient information, incorporating momentum from previous iterations.  It's an adaptation of the Nesterov Accelerated Gradient method to handle stochasticity.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v7vYVvmfru/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}