---
title: "Replicable Uniformity Testing"
summary: "This paper presents the first replicable uniformity tester with nearly linear dependence on the replicability parameter, enhancing the reliability of scientific studies using distribution testing algo..."
categories: ["AI Generated", ]
tags: ["AI Theory", "Optimization", "🏢 UC San Diego",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} lCiqPxcyC0 {{< /keyword >}}
{{< keyword icon="writer" >}} Sihan Liu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=lCiqPxcyC0" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/lCiqPxcyC0" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=lCiqPxcyC0&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/lCiqPxcyC0/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many algorithms in distribution property testing lack **replicability**: their outputs vary significantly across runs even when applied to the same data, threatening the reliability of scientific studies.  Existing uniformity testing algorithms, while efficient, are not always replicable, leading to potentially conflicting results depending on the input.  This unreliability undermines public trust in scientific findings if these algorithms are used in real world applications.

This paper tackles this problem by developing a **new uniformity tester** that guarantees replicable outputs with high probability.  The researchers achieve this by using a **total variation distance statistic**, which is less sensitive to outliers compared to previous methods.  Their tester uses a number of samples that is nearly optimal for the problem.  The authors also prove a **lower bound** on sample complexity, showing that their algorithm is near-optimal for a broad class of symmetric testing algorithms.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A novel replicable uniformity tester is proposed that significantly improves upon existing algorithms by achieving a nearly linear dependence on the replicability parameter. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A nearly matching lower bound for replicable uniformity testing is proven for a natural class of symmetric algorithms. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The study highlights the practical importance of algorithmic replicability, emphasizing its role in ensuring consistency and building trust in scientific research involving distribution testing. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers focusing on **algorithmic replicability** and **distribution testing**. It offers **novel theoretical bounds** and a **replicable uniformity tester**, addressing critical issues of non-replicable behavior in existing algorithms. This work is significant due to its focus on the **real-world applicability** of algorithms and its impact on the **trustworthiness of scientific studies**.  The paper opens new avenues for research on replicable algorithms, particularly in areas like identity and closeness testing, which face similar challenges.

------
#### Visual Insights







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lCiqPxcyC0/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}