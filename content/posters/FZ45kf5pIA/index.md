---
title: "Edit Distance Robust Watermarks via Indexing Pseudorandom Codes"
summary: "This paper presents a novel watermarking scheme for language models that is both undetectable and robust to a constant fraction of adversarial edits (insertions, deletions, substitutions)."
categories: ["AI Generated", ]
tags: ["Natural Language Processing", "Large Language Models", "🏢 MIT",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} FZ45kf5pIA {{< /keyword >}}
{{< keyword icon="writer" >}} Noah Golowich et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=FZ45kf5pIA" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/FZ45kf5pIA" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/FZ45kf5pIA/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

The proliferation of AI-generated text poses challenges for various institutions, leading to concerns regarding plagiarism and misinformation. Current watermarking schemes struggle with robustness against adversarial edits (insertions, deletions, and substitutions) in AI-generated content.  Existing methods often rely on strong assumptions, such as the independence of edits or a model being equivalent to a binary symmetric channel, limiting their applicability and effectiveness in real-world scenarios.

This research introduces a novel watermarking scheme that provides provable guarantees for both undetectability and robustness against a constant fraction of adversarial edits. This is achieved through the development of specialized pseudorandom codes that are robust to insertions and deletions, along with a transformation to watermarking schemes for any language model. The approach significantly improves upon previous work by relaxing computational assumptions and providing a more realistic robustness guarantee.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} The paper introduces a novel watermarking scheme for language models that is provably undetectable and robust to adversarial edits. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} It addresses limitations of prior methods that could only handle stochastic substitutions and deletions, offering a more realistic robustness guarantee. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The scheme uses pseudorandom codes over large alphabets, making it applicable to a wide range of language models. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **AI safety and security**, particularly those focused on **detecting and mitigating the misuse of AI-generated content**.  It offers a novel approach to watermarking language models that is more robust to adversarial attacks than previous methods. This robustness is particularly important considering the increasing prevalence of AI-generated content,  opening new avenues for watermarking research and development.

------
#### Visual Insights







### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FZ45kf5pIA/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}