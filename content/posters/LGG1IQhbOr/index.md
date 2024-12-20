---
title: "Non-geodesically-convex optimization in the Wasserstein space"
summary: "A novel semi Forward-Backward Euler scheme provides convergence guarantees for non-geodesically-convex optimization in Wasserstein space, advancing both sampling and optimization."
categories: []
tags: ["AI Theory", "Optimization", "🏢 Department of Computer Science, University of Helsinki",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} LGG1IQhbOr {{< /keyword >}}
{{< keyword icon="writer" >}} Hoang Phuc Hau Luu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=LGG1IQhbOr" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95610" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=LGG1IQhbOr&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/LGG1IQhbOr/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many machine learning and sampling problems involve optimizing nonconvex functions over probability distributions.  Current optimization methods in the Wasserstein space often struggle with nonconvexity, lacking strong theoretical guarantees.  Existing methods like the Forward-Backward Euler lack convergence analysis for these complex scenarios. 

This paper introduces a novel optimization algorithm, the *semi Forward-Backward Euler* method, specifically designed to handle non-geodesically convex functions in the Wasserstein space. It provides convergence analysis and rates under various conditions (differentiable or nonsmooth). This is a significant contribution as it offers theoretical guarantees, previously unavailable, for a widely applicable class of problems.  The results are validated through numerical experiments on challenging non-log-concave distributions, demonstrating the effectiveness of the proposed method.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A new semi Forward-Backward Euler scheme is proposed for non-geodesically convex optimization problems in Wasserstein space. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The scheme is proven to converge to critical points under general assumptions, with convergence rates established. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Convergence results are provided for various settings (smooth and nonsmooth), with applications to sampling problems. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working in optimization and sampling, particularly those dealing with nonconvex problems in probability spaces.  It offers **novel theoretical guarantees** for a modified Forward-Backward Euler scheme, addressing a significant gap in the field.  The findings are highly relevant to various applications including machine learning and Bayesian inference, opening up **new avenues for algorithm design and analysis** in nonconvex settings.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/LGG1IQhbOr/figures_9_1.jpg)

> This figure contains four subfigures visualizing the results of the semi FB Euler algorithm applied to Gaussian Mixture and Relaxed von Mises-Fisher distributions.  Subfigure (a) shows a heatmap of samples generated by the algorithm for Gaussian Mixture. Subfigure (b) is a plot showing KL divergence vs. number of iterations for semi FB Euler, FB Euler, and ULA algorithms, highlighting the faster convergence of the semi FB Euler method. Subfigures (c) and (d) show the true probability density and a histogram of samples obtained by the semi FB Euler algorithm, respectively, for the Relaxed von Mises-Fisher distribution; FB Euler failed for this case. 







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LGG1IQhbOr/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}