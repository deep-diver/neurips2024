---
title: "Cryptographic Hardness of Score Estimation"
summary: "Score estimation, crucial for diffusion models, is computationally hard even with polynomial sample complexity unless strong distributional assumptions are made."
categories: ["AI Generated", ]
tags: ["AI Theory", "Optimization", "🏢 University of Washington",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} URQXbwM0Md {{< /keyword >}}
{{< keyword icon="writer" >}} Min Jae Song et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=URQXbwM0Md" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/URQXbwM0Md" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=URQXbwM0Md&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/URQXbwM0Md/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Diffusion models are powerful generative models, but their efficiency relies on accurate score estimation.  Recent work shows that sampling from an unknown distribution can be reduced to accurate score estimation, but this leaves open the question of how computationally hard it is to estimate these scores.  Estimating score functions for complex high-dimensional distributions is challenging and computationally expensive; this makes it impossible to efficiently implement this process for certain distributions. 

This paper addresses this issue by demonstrating that L²-accurate score estimation is computationally hard, even when ample samples are available.  The authors achieve this by creating a reduction from the Gaussian Pancakes problem (known to be computationally hard under reasonable assumptions) to the score estimation problem.  **This provides a new understanding of the computational limits of score estimation**, and suggests that future research should focus on alternative approaches to sampling or on stronger assumptions about data distributions to make score estimation feasible.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} L²-accurate score estimation is computationally hard for many distributions. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A reduction from Gaussian Pancakes problem shows the hardness of score estimation. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The paper highlights a statistical-computational gap in score estimation, impacting generative modeling and cryptography. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it reveals a computational barrier** in a widely used machine learning technique, score estimation. This has significant implications for the development of efficient algorithms in generative modeling and post-quantum cryptography, paving the way for more realistic expectations and research directions.  **The reduction from Gaussian Pancakes problem to L2-accurate score estimation highlights the inherent complexity**, urging researchers to explore alternative algorithms or stronger assumptions.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/URQXbwM0Md/figures_2_1.jpg)

> 🔼 This figure visualizes 2D Gaussian Pancakes distributions with different thickness parameters (σ). The top row shows scatter plots of samples from the distributions, illustrating how the distribution changes from a clear pancake structure to a more Gaussian-like distribution as σ increases. The bottom row shows the probability density functions along the secret direction u, comparing the Gaussian Pancakes with the standard Gaussian.
> <details>
> <summary>read the caption</summary>
> Figure 1: Top: Scatter plot of 2D Gaussian pancakes Pu with secret direction u = (-1/√2, 1/√2), spacing y = 6, and thickness σ∈ {0.01, 0.05, 0.25}. Bottom: Re-scaled probability densities of Gaussian pancakes (blue) for each σ∈ {0.01, 0.05, 0.25} and the standard Gaussian (black) along u.
> </details>







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/URQXbwM0Md/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}