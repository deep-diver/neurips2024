---
title: "Neural network learns low-dimensional polynomials with SGD near the information-theoretic limit"
summary: "SGD can train neural networks to learn low-dimensional polynomials near the information-theoretic limit, surpassing previous correlational statistical query lower bounds."
categories: ["AI Generated", ]
tags: ["AI Theory", "Generalization", "🏢 Princeton University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} qK4iS49KDm {{< /keyword >}}
{{< keyword icon="writer" >}} Jason D. Lee et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=qK4iS49KDm" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/qK4iS49KDm" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=qK4iS49KDm&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/qK4iS49KDm/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Learning low-dimensional functions from high-dimensional data is a core challenge in machine learning. Existing theoretical analyses, often based on correlational statistical queries, suggested limitations on gradient descent algorithms like SGD. These analyses pointed to a gap between the computationally achievable performance and the information-theoretic limit.  This paper focuses on single-index models, a class of functions with low intrinsic dimensionality. 

This research demonstrates that **SGD, when modified to reuse minibatches, can overcome the limitations highlighted by the correlational statistical query lower bounds**. By reusing data, the algorithm implicitly exploits higher-order information beyond simple correlations, achieving a sample complexity close to the information-theoretic limit for polynomial single-index models. This significant improvement is attributed to the algorithm’s ability to implement a full statistical query, rather than just correlational queries. The findings challenge conventional wisdom about SGD’s limitations and suggest new avenues for enhancing learning efficiency.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} SGD with minibatch reuse can learn single-index models with sample and runtime complexity of n,T = Õ(d(p* − 1)√1), where p* is the generative exponent of the link function. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} This surpasses previous lower bounds based on correlational statistical queries, demonstrating the algorithm's ability to effectively utilize higher-order information. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} This work provides insights into improving the efficiency of neural network training and has implications for various applications of machine learning. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because it **demonstrates that SGD, a widely used algorithm, can learn low-dimensional structures in high-dimensional data more efficiently than previously thought**. This challenges existing theoretical understanding and opens new avenues for optimizing neural network training, impacting various machine learning applications.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/qK4iS49KDm/figures_1_1.jpg)

> 🔼 This figure shows the results of training a two-layer ReLU neural network with 1024 neurons using two different approaches: online SGD with a batch size of 8 and GD on the same batch of size n for 2¹⁴ steps. The target function is f*(x) = H3((x, θ)), where H3 is the third Hermite polynomial.  The heatmaps display the weak recovery (overlap between learned parameters w and the target direction θ) for online SGD and the generalization error for GD, averaged over 10 runs. The results highlight a significant difference in performance between the two approaches, with online SGD failing to achieve low test error even with a large number of samples, while GD with batch reuse achieves low generalization error with n ~ d samples.
> <details>
> <summary>read the caption</summary>
> Figure 1: We train a ReLU NN (3.1) with N = 1024 neurons using SGD (squared loss) with step size η = 1/d to learn a single-index target f*(x) = H3((x, θ)); heatmaps are values averaged over 10 runs. (a) online SGD with batch size B = 8; (b) GD on the same batch of size n for T = 214 steps. We only report weak recovery (i.e., overlap between parameters w and target θ, averaged across neurons) for online SGD since the test error does not drop.
> </details>







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/qK4iS49KDm/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}