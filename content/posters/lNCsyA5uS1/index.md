---
title: "Thought of Search: Planning with Language Models Through The Lens of Efficiency"
summary: "This paper introduces 'Thought of Search,' a novel, efficient planning approach using LLMs that prioritizes soundness and completeness. It leverages LLMs to generate Python code for search components,..."
categories: []
tags: ["Natural Language Processing", "Large Language Models", "🏢 IBM Research",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} lNCsyA5uS1 {{< /keyword >}}
{{< keyword icon="writer" >}} Michael Katz et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=lNCsyA5uS1" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93837" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=lNCsyA5uS1&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/lNCsyA5uS1/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many recent methods use LLMs for planning, but they often ignore crucial properties like soundness and completeness, leading to inefficient and inaccurate results. These methods often rely on numerous LLM calls, which is expensive and environmentally unfriendly. 

This paper proposes a more efficient and responsible approach called "Thought of Search." Instead of directly using LLMs for every search step, it leverages LLMs to generate Python code for core search components (successor generation and goal test).  This approach dramatically reduces the number of LLM calls while maintaining both soundness and completeness, resulting in significantly higher accuracy and efficiency across several search problems.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} The current trend of using LLMs for planning often sacrifices soundness and completeness for efficiency. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The proposed "Thought of Search" method uses LLMs to generate efficient Python code for search components, significantly improving accuracy and reducing compute cost. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The new method achieves 100% accuracy on various datasets using significantly fewer LLM calls than existing methods. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because it addresses the critical issue of **efficiency and responsibility** in using large language models (LLMs) for planning.  It challenges the current trend of computationally expensive and incomplete LLM-based planning methods, offering a novel approach that prioritizes **soundness, completeness, and efficiency**. This work is important for researchers seeking to develop more responsible and effective LLM-based applications.

------
#### Visual Insights





![](https://ai-paper-reviewer.com/lNCsyA5uS1/tables_8_1.jpg)

> This table compares different planning approaches using LLMs across four datasets (24 Game, Crossword, Blocks World, PrOntoQA).  For each approach, it shows the time complexity, the percentage of states explored, the number of LLM calls needed, and the total number of states in the datasets. The table highlights the efficiency gains of the proposed method, 'Tree of Thoughts' which significantly reduces the number of LLM calls required while maintaining accuracy.





### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/lNCsyA5uS1/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}