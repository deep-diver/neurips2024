---
title: "Active Classification with Few Queries under Misspecification"
summary: "Learning halfspaces efficiently under noise is cracked! A novel query language enables a polylog query algorithm for Massart noise, overcoming previous limitations."
categories: []
tags: ["Machine Learning", "Active Learning", "🏢 University of Texas at Austin",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Ma0993KZlq {{< /keyword >}}
{{< keyword icon="writer" >}} Vasilis Kontonis et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Ma0993KZlq" target="_self" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95508" target="_self" >}}
↗ NeurIPS Homepage
{{< /button >}}


<audio controls>
    <source src="https://ai-paper-reviewer.com/Ma0993KZlq/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Active learning aims to minimize labeled examples needed for accurate classification. However, existing methods often struggle with noisy labels, leading to high query complexity. This paper tackles this challenge by focusing on halfspace learning, a fundamental problem in machine learning. The existing algorithms are sensitive to noisy labels and do not scale well to larger datasets. 

The researchers introduce a new type of query, called Threshold Statistical Queries (TSQ), designed to be robust to noisy labels. They develop a novel algorithm that uses TSQs to efficiently learn halfspaces under the Massart noise model, a common type of noise in real-world data.  Their algorithm uses significantly fewer queries compared to previous approaches and achieves similar accuracy. Importantly, the paper also proves that for the case of agnostic noise, high query complexity is unavoidable, even for simpler classification tasks.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A new query language, Threshold Statistical Queries (TSQ), is introduced for active learning under noise. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A query-efficient algorithm achieves (η + ε)-accurate labeling for halfspaces under Massart noise using polylog(1/ε) TSQs. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} It is proven impossible to beat O(1/ε) query complexity for learning halfspaces under agnostic noise using TSQs, even for simpler hypothesis classes. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in active learning and machine learning dealing with noisy data.  It presents **the first query-efficient algorithm** for learning halfspaces under the Massart noise model, a significant advancement. This opens avenues for improving active learning algorithms' robustness and efficiency in real-world applications where noisy labels are common. The negative results on agnostic noise highlight fundamental limitations, guiding future research directions.

------
#### Visual Insights







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Ma0993KZlq/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}