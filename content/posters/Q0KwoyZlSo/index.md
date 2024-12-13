---
title: "On the Complexity of Learning Sparse Functions with Statistical and Gradient Queries"
summary: "Learning sparse functions efficiently with gradient methods is challenging; this paper introduces Differentiable Learning Queries (DLQ) to precisely characterize gradient query complexity, revealing s..."
categories: []
tags: ["Machine Learning", "Deep Learning", "🏢 Toyota Technological Institute at Chicago",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Q0KwoyZlSo {{< /keyword >}}
{{< keyword icon="writer" >}} Nirmit Joshi et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Q0KwoyZlSo" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95269" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=Q0KwoyZlSo&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/Q0KwoyZlSo/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Learning sparse functions efficiently using gradient-based methods is a significant challenge in machine learning. Existing theoretical frameworks, such as Statistical Queries (SQ), often fail to capture the nuances of gradient-based optimization. This paper tackles this problem by introducing a new query model called Differentiable Learning Queries (DLQ), which accurately reflects gradient computations.  The study focuses on the query complexity of DLQ for learning the support of a sparse function, revealing how this complexity is tightly linked to the choice of loss function. 

The researchers demonstrate that the complexity of DLQ matches that of Correlation Statistical Queries (CSQ) only for specific loss functions like squared loss. However, simpler loss functions such as l1 loss show DLQ achieving the same complexity as SQ.  Furthermore, they show that DLQ can capture the learning complexity with stochastic gradient descent using a two-layer neural network model. **This provides a unified theoretical framework for analyzing gradient-based learning of sparse functions, highlighting the importance of loss function selection and offering valuable insights for researchers in optimization algorithms and deep learning.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Differentiable Learning Queries (DLQ) provide a novel framework for analyzing gradient-based learning of sparse functions. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The complexity of learning sparse functions with gradient methods depends crucially on the choice of loss function; some losses yield significantly lower complexity than others. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} DLQ accurately captures the complexity of learning with stochastic gradient descent (SGD) in specific settings, bridging the gap between theory and practice. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important as it bridges the gap between theoretical analysis and practical gradient-based learning.  It provides a novel framework for analyzing the complexity of gradient algorithms for learning sparse functions. The findings are highly relevant to researchers working on optimization algorithms, high-dimensional statistics, and deep learning, potentially inspiring future research on efficient gradient methods and better understanding of generalization.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/Q0KwoyZlSo/figures_8_1.jpg)

> This figure shows the training dynamics of online SGD with different loss functions for a junta learning problem, where the target function depends on a subset of coordinates.  The x-axis represents the number of iterations, and the y-axis represents the test mean squared error. The results demonstrate the effect of the loss function on the convergence of SGD.  For the squared loss, the training dynamics gets stuck in a saddle point and does not converge. However, for the absolute loss, SGD converges in O(d) iterations, aligning with the theoretical analysis.  The figure also compares the SGD dynamics to a continuous-time mean-field model (DF-PDE).





![](https://ai-paper-reviewer.com/Q0KwoyZlSo/tables_2_1.jpg)

> This table summarizes the complexity results of learning sparse functions using different query types: Statistical Queries (SQ), Correlation Statistical Queries (CSQ), and Differentiable Learning Queries (DLQ).  It shows how the query complexity (number of queries needed) scales with the input dimension (d) for both adaptive (queries depend on previous answers) and non-adaptive (queries are fixed in advance) algorithms.  The complexity is expressed in terms of a leap exponent (adaptive) or cover exponent (non-adaptive), which are determined by the structure of 'detectable' subsets of coordinates, which in turn depends on the query type and loss function used.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Q0KwoyZlSo/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}