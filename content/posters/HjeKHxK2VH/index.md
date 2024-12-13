---
title: "WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off"
summary: "WaterMax: a novel LLM watermarking scheme achieving high detectability and preserving text quality by cleverly generating multiple texts and selecting the most suitable one."
categories: []
tags: ["Natural Language Processing", "Large Language Models", "🏢 Inria, CNRS, IRISA",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} HjeKHxK2VH {{< /keyword >}}
{{< keyword icon="writer" >}} Eva Giboulot et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=HjeKHxK2VH" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95807" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/papers/2403.04808" target="_blank" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=HjeKHxK2VH&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/HjeKHxK2VH/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

The proliferation of powerful Large Language Models (LLMs) raises concerns about misuse, necessitating robust methods for verifying the source of generated text. Current watermarking techniques often struggle with a trade-off between detectability and maintaining the original text quality.  This limits their effectiveness in real-world applications where both high detection accuracy and preserving text quality are critical. 

WaterMax addresses these limitations by employing a novel approach. Instead of modifying the LLM itself, WaterMax generates multiple text variations for a given prompt and strategically selects the one with the most robust watermark. This method is both theoretically analyzed and empirically validated. The results show that WaterMax significantly outperforms existing watermarking techniques by improving detection accuracy while maintaining high text quality. The theoretical model and experimental validation establish WaterMax as a substantial step towards practical and robust LLM watermarking.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} WaterMax significantly improves LLM watermark detectability without sacrificing text quality. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} WaterMax's novel design operates on text chunks rather than individual tokens, enhancing robustness and efficiency. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The paper introduces a theoretical model that characterizes WaterMax's watermark robustness against attacks. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **LLM watermarking** because it introduces **WaterMax**, a novel scheme that significantly improves watermark detectability without compromising text quality.  This challenges the existing trade-off in the field and opens new avenues for robust AI traceability, addressing a critical issue in the responsible use of LLMs. It also provides **a theoretical model for watermark robustness**, which is a significant advancement in the field.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_1_1.jpg)

> This figure shows the relationship between the detectability of watermarks and the quality of the text for three different LLMs (Phi-3-mini-4k-instruct, Meta-Llama-3-8B-Instruct, and Llama-2-7b-chat-hf) using three watermarking methods (WaterMax, KGW, and Aaronson). The x-axis represents the text quality, measured by the relative perplexity of the watermarked text compared to the original text. The y-axis represents the detectability of the watermark at a false alarm probability of 10^-6. The figure demonstrates that WaterMax consistently achieves near-perfect detectability (close to 1) with minimal impact on text quality (perplexity close to 1), outperforming other watermarking methods.





![](https://ai-paper-reviewer.com/HjeKHxK2VH/tables_16_1.jpg)

> This table shows the expectation and variance of the maximum of n independent standard normal random variables.  These values are used in the theoretical analysis of the watermark's robustness against attacks in Section 5 of the paper.  The expectation and variance are crucial for approximating the distribution of the test statistic and calculating the power of the robust watermark detection test.





### In-depth insights


#### LLM Watermarking
LLM watermarking is a crucial technique to address the misuse of Large Language Models (LLMs).  **Current methods often struggle with the trade-off between watermark detectability, robustness against attacks (like text editing), and the preservation of text quality.**  A robust watermark should be undetectable in unmodified text while easily detectable even after various modifications, yet should minimally impact the quality of the LLM's output.  The core challenge lies in finding effective ways to embed watermarks without significantly altering the underlying LLM's behavior or text generation process.  **Novel approaches, such as WaterMax, attempt to break the traditional trade-off by focusing on computational complexity rather than sacrificing text quality or robustness.** This shift in focus opens up new avenues in watermark design, enabling higher detectability without sacrificing the quality of the generated text.  Further research is needed to fully explore the effectiveness and limitations of these new techniques and explore better ways to balance these competing factors.

#### WaterMax Design
WaterMax's core design cleverly circumvents the limitations of existing LLM watermarking techniques. **Instead of modifying the LLM's internal mechanisms**, it leverages the inherent randomness of the LLM's text generation process.  By generating multiple text variations for a given prompt and selecting the one with the lowest p-value (from a watermark detection algorithm), WaterMax achieves high detectability without sacrificing text quality. This approach is **distortion-free**, meaning it doesn't modify the probability distribution of the next token.  The strategy introduces a trade-off between robustness and computational complexity, but the authors cleverly mitigate the latter through chunk-based processing and candidate selection.  The theoretical model underpinning WaterMax allows for a precise characterization of its robustness, enabling the tuning of parameters for optimal performance. This design is particularly noteworthy because of its potential for use with standard LLMs, requiring no fine-tuning, making it highly practical.

#### Robustness Analysis
A robust watermarking scheme should withstand various attacks aiming to remove or distort the embedded signal without significantly impacting the text quality.  A robustnesss analysis would thus assess the watermark's resilience against a range of attacks such as text editing (e.g., synonym replacement, paraphrasing, deletion), translation, and obfuscation techniques.  **The analysis should quantify the watermark's ability to survive these attacks, ideally providing metrics that measure the tradeoff between robustness and the level of distortion introduced.**  A strong robustness analysis would include both theoretical modeling to establish bounds on the watermark's resilience and empirical evaluation using a comprehensive benchmark suite of attacks.  **Crucially, the impact of different LLM architectures and text generation parameters on robustness needs to be investigated.** The results would provide insights into the watermarking scheme's practical security in real-world scenarios, and help to guide the design of more robust and resilient watermarks in the future. **This analysis also needs to consider the computational cost of the detection algorithm and the effect of this cost on its robustness.**

#### Complexity Tradeoff
The concept of 'complexity tradeoff' in watermarking large language models (LLMs) centers on balancing the robustness and detectability of a watermark against its impact on the generated text quality and computational cost.  **Robustness** refers to the watermark's resilience against various attacks aimed at removing or obscuring it.  **Detectability** measures how easily the watermark can be identified.  A strong watermark is highly detectable and robust, but implementing such a scheme may significantly reduce the quality of the LLM output or require extensive computational resources.  Watermarking methods that prioritize robustness often introduce noticeable distortions, decreasing text quality.  Conversely, methods focused on high-quality output might sacrifice robustness making the watermark easier to remove. Therefore, a crucial design challenge is to find an optimal balance, or tradeoff, minimizing quality loss while maintaining a sufficiently high level of detectability and robustness.  **WaterMax**, as presented in the paper, attempts to address this tradeoff by shifting the compromise from quality to complexity, maintaining high quality and robustness at the cost of increased computational needs. This is achieved by intelligently utilizing the text's inherent entropy and generating multiple text variations.

#### Future Directions
Future research could explore extending WaterMax to handle multilingual texts, a significant challenge for current watermarking techniques.  **Improving the efficiency of the algorithm** is crucial for wider adoption, perhaps through techniques like model distillation or more efficient search strategies.  A deeper theoretical analysis of the watermark's robustness under various attacks, including sophisticated adversarial methods, could provide stronger guarantees.  **Investigating the interplay between text quality, watermark strength, and detectability** in more detail, across diverse LLMs, is vital to optimize the trade-off. Finally, researching methods to ensure token score independence, particularly when faced with limited text entropy, is crucial for reliably achieving high detection accuracy and avoiding false positives.  These advancements would make watermarking a more robust and practical tool for ensuring the responsible use of LLMs.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_3_1.jpg)

> The figure compares the detectability and text quality of WaterMax against other watermarking methods across various Large Language Models (LLMs). It shows that WaterMax consistently achieves high detectability (close to 1) while maintaining good text quality (relative perplexity close to 1), outperforming other methods which exhibit a trade-off between these two factors.  The x-axis represents the relative perplexity (lower is better quality), and the y-axis represents the detectability.  A probability of false alarm of 10^-6 is used, with nucleus sampling (topp = 0.95) at a temperature of 1.0.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_6_1.jpg)

> This figure shows the trade-off between the detectability of the watermark and the quality of the generated text for different LLMs and watermarking methods.  The x-axis represents the relative perplexity (lower is better quality), and the y-axis represents the probability of detection (higher is better detectability). WaterMax consistently achieves high detectability (close to 1) with minimal impact on text quality, outperforming other methods.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_8_1.jpg)

> This figure shows the trade-off between detectability and text quality for different watermarking schemes.  The x-axis represents either the temperature of the LLM or the relative perplexity, while the y-axis represents the detectability (true positive rate at a false positive rate of 10^-6).  WaterMax consistently demonstrates high detectability with minimal impact on text quality, unlike other methods.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_8_2.jpg)

> This figure shows the trade-off between watermark detectability and text quality for various LLMs using different watermarking techniques.  The x-axis represents the relative perplexity (a measure of text quality, lower is better), and the y-axis represents the probability of detection (higher is better) given a false positive rate of 10<sup>-6</sup>. Watermarking schemes are compared across three different LLMs. The figure demonstrates that WaterMax achieves near-perfect detectability (close to 1) with minimal impact on text quality (perplexity close to 1). Other methods require increasing the watermark strength to reach similar detection rates, leading to a significant reduction in text quality.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_8_3.jpg)

> This figure compares the performance of WaterMax against other watermarking techniques (KGW and Aaronson's) across three different LLMs. The x-axis represents the relative perplexity (a measure of text quality, lower is better), and the y-axis represents the detectability (closer to 1 is better). WaterMax consistently achieves high detectability while maintaining near-perfect text quality, significantly outperforming the other methods.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_9_1.jpg)

> This figure shows the trade-off between watermark detectability and text quality for various LLMs using different watermarking techniques.  The x-axis represents text quality, measured by the relative perplexity (lower is better), indicating how much the watermark affects the original text's quality. The y-axis represents the detectability of the watermark (higher is better), meaning the probability of successfully detecting the watermark in the text.  The plot demonstrates that WaterMax consistently achieves high detectability with minimal impact on text quality, outperforming other techniques that require a significant trade-off between the two.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_17_1.jpg)

> This figure shows the trade-off between the detectability of the watermark and the quality of the generated text for different LLMs.  The x-axis represents the relative perplexity (a measure of text quality, lower is better), and the y-axis represents the probability of detection (higher is better).  WaterMax consistently achieves near-perfect detectability (y-axis value close to 1.0) while maintaining very high text quality (x-axis value close to 1.0), outperforming other watermarking techniques in the comparison.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_20_1.jpg)

> The figure shows the detectability of WaterMax watermarking scheme as a function of the number of tokens generated by beam search. The results are presented for three different hashing window sizes (h=2, h=4, and h=6).  It demonstrates the trade-off between detectability and text quality. A larger b increases detectability but can slightly reduce the text quality.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_20_2.jpg)

> The figure shows the detectability and text quality trade-off for different watermarking schemes (WaterMax, Aaronson, KGW) using the Llama-3-8b-Instruct language model.  It demonstrates that WaterMax achieves high detectability with minimal quality loss across various temperatures, unlike other methods that sacrifice quality for increased detectability or vice versa.  The x-axis represents relative perplexity (a measure of text quality, lower is better), and the y-axis represents detectability (probability of detection at a false positive rate of 10^-6, higher is better).


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_20_3.jpg)

> The figure shows the trade-off between the detectability of the watermark and the quality of the generated text for different LLMs using various watermarking techniques.  WaterMax consistently achieves near-perfect detectability (close to 1) with minimal impact on text quality, as measured by relative perplexity.  Other methods require a higher watermark strength to reach similar detectability levels, which leads to significantly lower text quality.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_21_1.jpg)

> The figure shows the trade-off between watermark detectability and text quality for three different LLMs using three different watermarking techniques: WaterMax, KGW, and Aaronson.  The x-axis represents the relative perplexity of the watermarked text compared to the original text (lower is better quality). The y-axis represents the detectability of the watermark (higher is better). WaterMax consistently achieves high detectability with minimal impact on text quality, outperforming KGW and Aaronson.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_22_1.jpg)

> This figure compares the detectability and quality of different watermarking schemes (WaterMax, Aaronson, KGW) across various LLM temperatures.  It illustrates that WaterMax consistently achieves high detectability (close to 1 at PFA = 10⁻⁶) with minimal loss in text quality, unlike other methods that require sacrificing quality to achieve comparable detectability.  The relative perplexity metric is used to measure text quality, with lower values indicating better quality.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_22_2.jpg)

> The figure shows the trade-off between the detectability of the watermark and the quality of the generated text for various LLMs and watermarking schemes.  WaterMax consistently achieves high detectability (close to 1) with minimal impact on text quality, outperforming other methods. The x-axis shows the relative perplexity (lower is better quality), and the y-axis shows the detectability at a false alarm probability of 10^-6.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_22_3.jpg)

> This figure shows the trade-off between watermark detectability and text quality for different watermarking schemes and various LLM temperatures. WaterMax consistently achieves high detectability with minimal quality loss, unlike other schemes that require increased watermark strength to achieve similar detectability, leading to reduced text quality.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_22_4.jpg)

> This figure shows the trade-off between the detectability of the watermark and the quality of the generated text for different LLMs. The x-axis represents the relative perplexity (a measure of text quality), with lower values indicating better quality. The y-axis shows the detectability of the watermark, with higher values indicating better detectability. The plot shows that WaterMax consistently achieves high detectability (close to 1) while maintaining high text quality (relative perplexity close to 1), unlike other watermarking methods that need to sacrifice text quality to achieve high detectability. This highlights the effectiveness of WaterMax in balancing the detectability-robustness-quality trade-off.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_22_5.jpg)

> This figure shows the detectability of different watermarking schemes (WaterMax, Aaronson, KGW) as a function of text quality and LLM temperature.  It demonstrates that WaterMax consistently achieves near-perfect detectability (close to 1 at PFA = 10^-6) with minimal loss in text quality, regardless of the temperature setting.  In contrast, Aaronson's scheme requires high temperatures for good detectability, impacting text quality, and KGW sacrifices quality to achieve high detectability.  The relative perplexity is used as a metric for text quality.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_22_6.jpg)

> The figure shows the trade-off between detectability and text quality for different watermarking schemes (WaterMax, Aaronson, and KGW) using the Llama-3-8b-Instruct language model.  It demonstrates that WaterMax achieves near-perfect detectability (close to 1 at PFA = 10^-6) with minimal impact on text quality, unlike the other schemes which require a significant sacrifice in quality to attain high detectability. The x-axis represents relative perplexity (lower is better quality), while the y-axis represents detectability.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_23_1.jpg)

> This figure compares the performance of WaterMax against other watermarking techniques (KGW and Aaronson's) across three different LLM architectures. It demonstrates that WaterMax consistently achieves near-perfect detectability (close to 1) while maintaining high text quality (low relative perplexity).  The x-axis represents the relative perplexity, indicating the change in text quality after watermarking (lower values are better). The y-axis represents the detectability at a fixed false alarm probability of 10<sup>-6</sup>. The results show that WaterMax outperforms other methods in achieving a balance between detectability and quality.  


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_24_1.jpg)

> This figure compares the performance of WaterMax against other watermarking techniques (KGW and Aaronson's) across different large language models (LLMs).  The x-axis represents text quality, measured by relative perplexity (lower is better), and the y-axis shows detectability (higher is better) with a probability of false alarm fixed at 10<sup>-6</sup>. The plot demonstrates that WaterMax achieves nearly perfect detectability (close to 1) while maintaining high text quality (perplexity close to 1), outperforming other methods.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_25_1.jpg)

> This figure compares the performance of WaterMax against other watermarking techniques across three different LLMs.  The x-axis represents relative perplexity (lower values indicate better text quality, with 1.0 being identical quality to the unwatermarked text).  The y-axis represents the detectability (closer to 1.0 means better watermark detection). The plot demonstrates that WaterMax achieves near-perfect detectability while maintaining almost identical text quality compared to the original LLMs. Other methods require significantly higher relative perplexity (lower quality) to achieve similar detectability.


![](https://ai-paper-reviewer.com/HjeKHxK2VH/figures_26_1.jpg)

> The figure shows the detectability of WaterMax and other watermarking techniques (KGW and Aaronson) across different LLM architectures in relation to text quality.  WaterMax consistently achieves near-perfect detectability (close to 1) with minimal impact on text quality (measured by relative perplexity), outperforming the other methods.  The probability of a false alarm is held constant at 10⁻⁶.


</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/HjeKHxK2VH/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}