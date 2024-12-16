---
title: "Smoothed Online Classification can be Harder than Batch Classification"
summary: "Smoothed online classification can be harder than batch classification when label spaces are unbounded, challenging existing assumptions in machine learning."
categories: ["AI Generated", ]
tags: ["AI Theory", "Optimization", "🏢 University of Michigan",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} NO9MSeZs6g {{< /keyword >}}
{{< keyword icon="writer" >}} Vinod Raman et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=NO9MSeZs6g" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/NO9MSeZs6g" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/NO9MSeZs6g/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Online and batch learning are two common approaches to train classifiers. While batch learning trains a model on a fixed dataset, online learning updates the model sequentially with each new data point.  Recent studies suggested that smoothed online learning (where data is drawn from a distribution with bounded density) might be as easy as batch learning. However, this paper reveals some important issues and challenges associated with online learning.  Specifically, it focuses on the limitations of existing theories in scenarios with unbounded label spaces (where the number of possible class labels is infinite). This situation is common in real-world applications such as face recognition or protein structure prediction. 

This research introduces a novel hypothesis class to show that existing theories fail for infinite label spaces. It demonstrates that a hypothesis class learnable in a batch setting (PAC learnable) might not be learnable in a smoothed online setting. Despite this hardness result, the paper proposes a new condition, called UBEME (Uniformly Bounded Expected Metric Entropy), that guarantees smoothed online learnability. This work contributes significantly to online learning theory by improving our understanding of its limits and providing new tools for analysis and algorithm development.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Smoothed online classification can be harder than batch classification, especially with unbounded label spaces. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} PAC learnability is insufficient for guaranteeing smoothed online learnability in unbounded label spaces. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} A new sufficient condition (UBEME) ensures smoothed online learnability, but a complete characterization remains open. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because it challenges the common belief that smoothed online learning mirrors batch learning.  It **highlights the challenges of online learning with infinite label spaces**, opening avenues for improved algorithm design and theoretical understanding. This research directly addresses the growing interest in multiclass learnability with unbounded label spaces, a prevalent issue in many modern machine learning applications.

------
#### Visual Insights







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/NO9MSeZs6g/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}