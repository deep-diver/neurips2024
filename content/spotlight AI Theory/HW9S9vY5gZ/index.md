---
title: 'No-regret Learning in Harmonic Games: Extrapolation in the Face of Conflicting
  Interests'
summary: Extrapolated FTRL ensures Nash equilibrium convergence in harmonic games,
  defying standard no-regret learning limitations.
categories: []
tags:
- AI Theory
- Optimization
- "\U0001F3E2 University of Oxford"
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} HW9S9vY5gZ {{< /keyword >}}
{{< keyword icon="writer" >}} Davide Legacci et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=HW9S9vY5gZ" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95828" target="_blank" >}}
↗ NeurIPS Proc.
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=HW9S9vY5gZ&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/HW9S9vY5gZ/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Multi-agent learning, especially no-regret learning, is well-understood in potential games where players share common interests. However, the dynamics remain largely unknown in harmonic games where interests conflict. This paper focuses on generalized harmonic games and investigates the convergence properties of the widely used Follow-the-Regularized-Leader (FTRL) algorithm.  The authors reveal that standard FTRL methods can fail to converge, potentially trapping players in cycles. 

This work addresses these limitations by introducing a novel algorithm called extrapolated FTRL. This enhanced method incorporates an extrapolation step, including optimistic and mirror-prox variants, and is shown to converge to Nash equilibrium from any starting point, guaranteeing at most constant regret. The analysis provides insights into continuous and discrete-time dynamics, unifying existing work on zero-sum games and extending the knowledge to the broader class of harmonic games. This highlights the fundamental distinction and connection between potential and harmonic games, from both a strategic and dynamic viewpoint.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Standard no-regret learning algorithms fail to converge in harmonic games. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Extrapolated FTRL, a modified algorithm, guarantees convergence to Nash equilibrium in harmonic games. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Harmonic games are strategically and dynamically complementary to potential games. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in game theory, multi-agent systems, and machine learning because **it unveils the dynamics of no-regret learning in harmonic games**, a class of games with conflicting interests previously under-explored.  The findings challenge existing assumptions, **highlighting the limitations of standard no-regret learning algorithms** and **providing a novel extrapolated FTRL algorithm that guarantees convergence to Nash equilibrium**. This opens avenues for designing more effective and robust learning algorithms in complex strategic settings. 

------
#### Visual Insights



![](https://ai-paper-reviewer.com/HW9S9vY5gZ/figures_8_1.jpg)

> This figure compares the performance of three different FTRL algorithms in harmonic games. The left panel shows the trajectories of the standard FTRL, the extrapolated FTRL (FTRL+), and the continuous-time version of FTRL in the Matching Pennies game. The center and right panels illustrate the trajectories for a 2x2x2 harmonic game from different perspectives.  The figure demonstrates that the standard FTRL algorithm diverges, while FTRL+ converges to the Nash equilibrium. The continuous-time FTRL is shown to be recurrent (periodic).







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HW9S9vY5gZ/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}