---
title: "Proportional Fairness in Clustering: A Social Choice Perspective"
summary: "This paper reveals the surprising connection between individual and proportional fairness in clustering, showing that any approximation to one directly implies an approximation to the other, enabling ..."
categories: []
tags: ["AI Theory", "Fairness", "🏢 Technische Universität Clausthal",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} KsLX5pFpOs {{< /keyword >}}
{{< keyword icon="writer" >}} Leon Kellerhals et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=KsLX5pFpOs" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95639" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=KsLX5pFpOs&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/KsLX5pFpOs/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Fairness in clustering is a critical issue, particularly when data points represent individuals. Existing research explored individual fairness (each individual is close to a cluster center) and proportional fairness (no large group is far from a desirable center).  However, these notions were largely studied in isolation, lacking a unified understanding of their relationships and optimal algorithmic approaches.  This created a need for unified frameworks and efficient algorithms guaranteeing both types of fairness.

This work addresses these issues by connecting fairness notions in clustering to axioms in multiwinner voting.  The authors introduce metric JR (mJR) and metric PJR (mPJR) axioms, demonstrating their connections to existing fairness measures.  They prove that algorithms satisfying mJR achieve state-of-the-art approximations for both individual and proportional fairness, and mPJR provides strong guarantees for sortition settings. Importantly, these axioms provide a simple bridge between the different fairness notions and allow for efficient computation of approximately fair clusters.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Individual and proportional fairness in clustering are closely related; approximating one implies approximating the other. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Multiwinner voting axioms provide a framework for designing efficient, provably fair clustering algorithms. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The proposed mJR and mPJR axioms offer improved approximation guarantees for existing fairness notions, particularly in sortition settings. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper bridges the gap between seemingly disparate fairness notions in clustering, offering a novel perspective from computational social choice.  It provides efficient algorithms with improved approximation guarantees and opens new avenues for research in fair and proportional clustering, particularly relevant to multiwinner voting and sortition.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/KsLX5pFpOs/figures_2_1.jpg)

> This figure summarizes the relationships between different fairness notions in clustering. The left side shows how different algorithms and fairness properties relate to each other, indicating approximation guarantees. The right side illustrates a sample metric space used in the paper for examples.







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KsLX5pFpOs/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}