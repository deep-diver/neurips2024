---
title: "Mixed Dynamics In Linear Networks: Unifying the Lazy and Active Regimes"
summary: "A new formula unifies lazy and active neural network training regimes, revealing a mixed regime that combines their strengths for faster convergence and low-rank bias."
categories: ["AI Generated", ]
tags: ["AI Theory", "Optimization", "🏢 Courant Institute",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 9zQl27mqWE {{< /keyword >}}
{{< keyword icon="writer" >}} Zhenfeng Tu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=9zQl27mqWE" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/9zQl27mqWE" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/9zQl27mqWE/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

The training dynamics of neural networks are often categorized into "lazy" and "active" regimes, each with its strengths and weaknesses.  The lazy regime features simpler, linear dynamics, but lacks the feature learning capability of active regimes. Conversely, the active regime is characterized by complex, nonlinear dynamics and feature learning, but struggles with slow convergence and can get stuck in poor solutions. This dichotomy has limited our understanding of the training process and optimization strategies. 

This paper introduces a unifying framework that captures both regimes and also reveals an intermediate "mixed" regime.  The core contribution is a novel formula that elegantly describes the evolution of the network's learned matrix.  This formula explains how the network can simultaneously leverage the simplicity of lazy dynamics for smaller singular values and the feature learning power of active dynamics for larger values, thus combining the advantages of both and addressing the limitations of each regime. This leads to a more complete understanding of neural network training and opens up new avenues for designing better optimization algorithms.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel unifying formula describes the evolution of learned matrices in linear neural networks, encompassing both lazy and active regimes. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A 'mixed regime' is identified where the network behaves lazily for smaller singular values and actively for larger ones, combining the advantages of both. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The paper provides an almost complete phase diagram of training behavior as a function of initialization variance and network width. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because it **bridges the gap between two dominant paradigms** in neural network training: the lazy and active regimes.  By providing a unified framework, it **enhances our understanding of training dynamics** and opens **new avenues for optimization strategies**, particularly in scenarios requiring both rapid convergence and low-rank solutions.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/9zQl27mqWE/figures_2_1.jpg)

> 🔼 This figure compares the training and testing error curves obtained using gradient descent and the self-consistent dynamics proposed in the paper, when the scaling parameters are set to γσ2 = -1.85 and γω = 2.25, which corresponds to the active regime. The left panel shows that both methods yield similar training and testing error curves. The right panel shows the evolution of singular values for Ae(t), where the five largest singular values converge faster once they cross the σ²w threshold.
> <details>
> <summary>read the caption</summary>
> Figure 1: For both plots, we train either using gradient descent or the self-consistent dynamics from equation (1), with the scaling γσ2 = -1.85, γω = 2.25 which lies in the active regime. (Left panel): We plot train and test error for both dynamics. We observe that the train/test error for gradient descent is very close to the train/test error for the self-consistent dynamics. (Right panel): We plot with a solid line the singular values of Ae(t) when running the self-consistent dynamics, and use a dashed line for the singular values from running gradient descent. In this experiment, RankA* = 5. We use different colors for the 5 largest singular values and the same color for the remaining singular values. We can see how the 5 largest singular values ‘speed up’ as they cross the σ²w threshold, allowing them to converge earlier than the rest. The minimal test error is achieved in the short period where the large singular values have converged but not the rest.
> </details>







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/9zQl27mqWE/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}