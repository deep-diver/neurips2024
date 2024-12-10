---
title: "Small coresets via negative dependence: DPPs, linear statistics, and concentration"
summary: "DPPs create smaller, more accurate coresets than existing methods, improving machine learning efficiency without sacrificing accuracy."
categories: []
tags: ["Machine Learning", "Optimization", "🏢 Univ. Lille, CNRS, Centrale Lille",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} jd3msHMtTL {{< /keyword >}}
{{< keyword icon="writer" >}} Rémi Bardenet et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=jd3msHMtTL" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93945" target="_blank" >}}
↗ NeurIPS Proc.
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=jd3msHMtTL&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/jd3msHMtTL/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Coresets are subsets of large datasets used to speed up machine learning algorithms, but creating effective coresets is challenging.  This paper investigates using **Determinantal Point Processes (DPPs)**, which are probabilistic models that encourage diversity in the selected samples, to build coresets. Existing methods primarily relied on independent sampling, which often lacks diversity and can result in less effective coresets. The main issue is that the effectiveness of independent sampling-based coresets is limited.

This research leverages the unique properties of DPPs to tackle the challenge of coreset construction.  The authors provide a novel theoretical framework by framing coreset loss as a linear statistic, linking the problem to concentration phenomena in DPPs. They derive new concentration inequalities that work for very general DPPs, going beyond what was previously possible, allowing for non-symmetric kernels and vector-valued functions.  Experiments validate that DPP-based coresets are smaller and more accurate than those from independent sampling, offering significant advantages.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} DPPs outperform independent sampling in coreset construction, achieving provably smaller coreset sizes. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The paper introduces novel concentration inequalities for linear statistics of DPPs, extending beyond previous limitations to encompass non-symmetric kernels and vector-valued statistics. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} These findings are validated through experiments on various datasets, demonstrating the practical benefits of DPP-based coresets. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because it **demonstrates that Determinantal Point Processes (DPPs) can build smaller and more accurate coresets** compared to traditional methods. This has significant implications for machine learning, offering computational efficiency in large-scale applications while maintaining accuracy.  It also advances the theoretical understanding of DPPs, providing valuable tools for researchers working with concentration inequalities and linear statistics.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/jd3msHMtTL/figures_8_1.jpg)

> This figure shows the results of an experiment comparing different coreset construction methods on a uniform dataset.  Panel (a) displays a scatter plot of the uniform dataset with an overlaid sample from an orthogonal polynomial ensemble (OPE) based DPP.  Panel (b) presents a log-log plot illustrating the 0.90-quantile of the supremum relative error (a measure of coreset quality) against the coreset size (m) for various methods: uniform sampling, sensitivity-based sampling, OPE, Vdm-DPP, G-mDPP (with varying bandwidths h), and stratified sampling.  Panel (c) is another log-log plot showing the same error metric (0.90-quantile of supremum relative error) versus the coreset size, focusing on the performance of G-mDPP across different bandwidths (h). The results demonstrate the superior performance of DPP-based methods, especially OPE and Vdm-DPP, compared to independent sampling methods (uniform and sensitivity) for achieving high accuracy coresets with fewer data points.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/jd3msHMtTL/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}