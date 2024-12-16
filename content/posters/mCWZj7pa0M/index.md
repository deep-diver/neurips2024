---
title: "Exact Gradients for Stochastic Spiking Neural Networks Driven by Rough Signals"
summary: "New framework uses rough path theory to enable gradient-based training of SSNNs driven by rough signals, allowing for noise in spike timing and network dynamics."
categories: ["AI Generated", ]
tags: ["AI Theory", "Optimization", "🏢 University of Copenhagen",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} mCWZj7pa0M {{< /keyword >}}
{{< keyword icon="writer" >}} Christian Holberg et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=mCWZj7pa0M" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/mCWZj7pa0M" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=mCWZj7pa0M&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/mCWZj7pa0M/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Stochastic spiking neural networks (SSNNs) are powerful models for neuronal dynamics but notoriously difficult to train due to event discontinuities and noise. Existing methods often rely on surrogate gradients or an optimise-then-discretise approach, which yield less accurate or only approximate gradients.  These limitations hinder the development of truly bio-plausible and efficient learning algorithms for SSNNs.

This research introduces a mathematically rigorous framework, based on rough path theory, to model SSNNs as stochastic differential equations with event discontinuities. This framework allows for noise in both spike timing and network dynamics.  By identifying sufficient conditions for the existence of pathwise gradients, the authors derive a recursive relation for exact gradients.  They further introduce a new class of signature kernels for training SSNNs as generative models and provide an end-to-end autodifferentiable solver.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel framework based on rough path theory allows for gradient-based training of SSNNs with noise affecting both spike timing and network dynamics. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Exact pathwise gradients of solution trajectories and event times with respect to network parameters are identified, satisfying a recursive relation. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} A new class of signature kernels indexed on càdlàg rough paths enables the training of SSNNs as generative models. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it introduces a novel, mathematically rigorous framework for training stochastic spiking neural networks (SSNNs)**, a significant challenge in the field.  This opens new avenues for developing more biologically plausible and efficient learning algorithms for SSNNs, advancing both theoretical understanding and practical applications in neuroscience and AI.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/mCWZj7pa0M/figures_8_1.jpg)

> 🔼 This figure shows the results of an experiment to estimate the input current (c) of a single stochastic leaky integrate-and-fire (SLIF) neuron.  The left panel displays the mean absolute error (MAE) for the first three average spike times on a hold-out test set, and the right panel shows the estimated value of c at each step during stochastic gradient descent.  Multiple lines represent different sample sizes (16, 32, 64, 128) and both panels show results for two different noise levels (σ = 0.25 and σ = 0.5).
> <details>
> <summary>read the caption</summary>
> Figure 1: Test loss and c estimate across four sample sizes and for two levels of noise σ. On the left: MAE for the three first average spike times on a hold out test set. On the right: estimated value of c at the current step.
> </details>





![](https://ai-paper-reviewer.com/mCWZj7pa0M/tables_6_1.jpg)

> 🔼 The figure shows the results of an experiment to estimate the constant input current c of a single stochastic leaky integrate-and-fire neuron model. Two levels of noise (σ = 0.25 and σ = 0.5) and four sample sizes (16, 32, 64, and 128) were tested. The left panel shows the mean absolute error (MAE) for the first three average spike times on a hold-out test set, while the right panel shows the estimated value of c at each step of the stochastic gradient descent optimization process. The results demonstrate that the model can accurately learn the input current even in the presence of noise and with relatively small sample sizes.
> <details>
> <summary>read the caption</summary>
> Figure 1: Test loss and c estimate across four sample sizes and for two levels of noise σ. On the left: MAE for the three first average spike times on a hold out test set. On the right: estimated value of c at the current step.
> </details>





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mCWZj7pa0M/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}