---
title: "Stochastic contextual bandits with graph feedback: from independence number to MAS number"
summary: "Contextual bandits with graph feedback achieve near-optimal regret by leveraging a novel graph-theoretic quantity that interpolates between independence and maximum acyclic subgraph numbers, depending..."
categories: []
tags: ["Machine Learning", "Reinforcement Learning", "🏢 New York University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} t8iosEWoyd {{< /keyword >}}
{{< keyword icon="writer" >}} Yuxiao Wen et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=t8iosEWoyd" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93357" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=t8iosEWoyd&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/t8iosEWoyd/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Traditional contextual bandit algorithms often struggle with complex feedback structures.  This paper addresses this limitation by considering contextual bandits with graph feedback where choosing an action reveals rewards for neighboring actions. Previous research primarily focused on multi-armed bandits with graph feedback, leaving the contextual bandit case largely unexplored. A key challenge is understanding how the statistical complexity of learning changes with the number of contexts and the feedback graph's structure.



The researchers introduce a new graph-theoretic quantity, βₘ(G), which captures the learning limits of this problem. This quantity smoothly transitions between known bounds based on the independence number and the maximum acyclic subgraph number as the number of contexts varies.  They also provide algorithms that achieve nearly optimal performance for significant classes of context sequences and feedback graphs and demonstrate the importance of the maximum acyclic subgraph number in characterizing statistical complexity when dealing with many contexts.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} A new graph-theoretic quantity, βₘ(G), characterizes the statistical complexity of contextual bandits with graph feedback. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Near-optimal regret bounds are established for contextual bandits with graph feedback, bridging the gap between multi-armed bandits and contextual bandits. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Novel algorithms are proposed to achieve near-optimal regret for important classes of context sequences and feedback graphs. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **contextual bandits** and **graph feedback**, as it provides **tight theoretical bounds** on the regret, which is the measure of the algorithm's performance.  It also introduces novel **graph-theoretic quantities** to precisely characterize the statistical complexity of learning under different feedback structures and the number of contexts. The results will guide the development of better algorithms with improved performance and provide a stronger theoretical foundation to the field.  Moreover, the new algorithm proposed for the tabular settings may also inspire similar methods for non-tabular settings.

------
#### Visual Insights







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/t8iosEWoyd/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}