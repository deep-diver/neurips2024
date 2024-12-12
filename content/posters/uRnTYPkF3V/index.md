---
title: "Sequential Probability Assignment with Contexts: Minimax Regret, Contextual Shtarkov Sums, and Contextual Normalized Maximum Likelihood"
summary: "This paper introduces contextual Shtarkov sums, a new complexity measure characterizing minimax regret in sequential probability assignment with contexts, and derives the minimax optimal algorithm, co..."
categories: []
tags: ["AI Theory", "Optimization", "🏢 University of Toronto",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} uRnTYPkF3V {{< /keyword >}}
{{< keyword icon="writer" >}} Ziyi Liu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=uRnTYPkF3V" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93273" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=uRnTYPkF3V&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/uRnTYPkF3V/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Sequential probability assignment, a fundamental problem in online learning and information theory, seeks algorithms minimizing regret against the best expert from a hypothesis class.  Existing complexity measures like sequential entropy fail to fully characterize minimax regret, particularly with contexts (side information).  The minimax optimal strategy remains elusive.

This research introduces contextual Shtarkov sums, a new complexity measure based on projecting Shtarkov sums onto context trees.  It proves that the minimax regret equals the worst-case log contextual Shtarkov sum.  Using this, it derives the minimax optimal algorithm, contextual NML, extending Normalized Maximum Likelihood to contextual settings. This work also offers a simplified proof for a tighter regret upper bound based on sequential entropy.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Contextual Shtarkov sums precisely characterize minimax regret in sequential probability assignment with contexts. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The paper introduces contextual Normalized Maximum Likelihood (cNML), a novel minimax optimal algorithm. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The study provides a sharpened regret upper bound using sequential entropy, unifying and improving state-of-the-art results. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is **crucial** for researchers in online learning and information theory. It provides **novel theoretical insights** into sequential probability assignment, offers a **new minimax optimal algorithm**, and **sharpens existing regret bounds**. This work opens up exciting new avenues for research and has significant implications for developing more efficient and effective online learning algorithms.

------
#### Visual Insights







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/uRnTYPkF3V/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}