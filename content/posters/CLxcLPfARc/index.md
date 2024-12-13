---
title: "Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space"
summary: "Open-source LLMs are vulnerable to embedding space attacks, which efficiently bypass safety mechanisms and enable data extraction, even after unlearning."
categories: []
tags: ["Natural Language Processing", "Large Language Models", "🏢 Technical University of Munich",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} CLxcLPfARc {{< /keyword >}}
{{< keyword icon="writer" >}} Leo Schwinn et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=CLxcLPfARc" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/96148" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=CLxcLPfARc&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/CLxcLPfARc/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current research focuses on discrete input manipulations to assess the robustness of Large Language Models (LLMs). However, this approach is insufficient for open-source models which offer complete access to their internal mechanisms. This paper introduces a new threat model: embedding space attacks which directly manipulate the continuous embedding representation of input tokens. These attacks bypass safety mechanisms and trigger harmful behaviors more efficiently than other existing methods. This approach raises serious concerns regarding the security and privacy of open-source LLMs. 

The study demonstrates that embedding space attacks are highly effective in removing safety alignment in various open-source LLMs. Furthermore, it reveals the ability of embedding space attacks to extract seemingly deleted information during unlearning, demonstrating the limitations of current unlearning techniques. This threat model extends to data extraction, as the study shows that these attacks can recover pretraining data. **The findings highlight the vulnerability of open-source LLMs and emphasize the need for improved safety measures**.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Embedding space attacks are a novel and effective way to compromise the safety of open-source LLMs. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} These attacks can extract sensitive information from models, even after supposedly successful unlearning. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The research identifies a critical gap in current AI safety research and opens up new research directions. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in AI safety and security, especially those working with open-source LLMs.  It **highlights a novel threat model**—embedding space attacks—that can bypass existing safety mechanisms more efficiently than previously known methods. The findings are relevant to current trends in adversarial machine learning and **open up new avenues for research** in LLM robustness and unlearning.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/CLxcLPfARc/figures_1_1.jpg)

> The figure illustrates two different types of attacks on large language models (LLMs): discrete attacks and embedding space attacks.  Discrete attacks manipulate the individual tokens of the input text, while embedding space attacks directly modify the continuous vector representation of the input tokens. The figure shows how these attacks differ in their approach and highlights the embedding space attack as a novel method proposed in the paper.





![](https://ai-paper-reviewer.com/CLxcLPfARc/tables_8_1.jpg)

> This table presents the cumulative attack success rate (C-ASR) achieved by different embedding space attack methods on the LlamaHP model for the Harry Potter Q&A task.  It compares the performance of attacks using varying numbers of suffix tokens (1, 5, or 20) with two baseline methods, Head Projection Attack (HPA) and Probability Delta Attack (PDA).  The C-ASR metric indicates the percentage of queries for which the correct answer appears at least once across multiple attack attempts.  Higher C-ASR values indicate more successful attacks.





### In-depth insights


#### Embedding Attacks
Embedding attacks represent a novel approach to compromising Large Language Models (LLMs), particularly open-source ones. Unlike traditional methods focusing on manipulating discrete input tokens, **embedding attacks directly target the continuous vector representations of tokens within the model's embedding space**. This offers several advantages:  **higher efficiency**, bypassing established safety mechanisms more effectively and with less computational cost than discrete attacks or fine-tuning; **circumvention of model alignments**, triggering harmful behaviors despite safety measures; and **creation of discrete jailbreaks**, enabling exploitation via standard natural language prompts.  Furthermore, the study highlights the alarming potential of embedding attacks in revealing supposedly deleted information from unlearned models and even extracting portions of pretraining data.  **This raises significant security concerns**, underscoring the vulnerability of open-source LLMs and the need for novel defense mechanisms tailored to this specific threat model.

#### Safety Alignment
The concept of safety alignment in large language models (LLMs) is crucial, focusing on ensuring that these models behave as intended and avoid generating harmful or unintended outputs.  **The research highlights how open-source LLMs, while offering advantages in accessibility and transparency, present a unique challenge in safety alignment.**  Unlike closed-source models where attacks are often limited to input manipulation, open-source models offer full model access, enabling more sophisticated attacks. The embedding space attack is presented as a particularly effective method, showcasing its capability to bypass existing safety mechanisms more efficiently than traditional discrete attacks or model fine-tuning.  **This method directly targets the continuous embedding representation of input tokens, making it difficult to mitigate through standard defense techniques.**  Moreover, the research demonstrates the vulnerability of unlearning mechanisms, with embedding space attacks showing capacity to extract seemingly deleted information, even potentially recovering portions of pretraining data. This emphasizes **the need for robust security measures beyond traditional safety alignment strategies** when dealing with open-source LLMs.  The ability of embedding space attacks to circumvent safety mechanisms and extract hidden information highlights the urgency of exploring novel safety alignment techniques suited to the unique vulnerabilities of open-source models.

#### Unlearning Threats
Unlearning, the process of removing data from a model, presents significant security challenges.  **A primary threat is the incomplete removal of sensitive information**, leaving residual data vulnerable to extraction through adversarial attacks.  This is especially concerning with open-source LLMs, where model architecture and weights are accessible, enabling the crafting of highly effective attacks directly targeting the embedding space.  **These embedding space attacks bypass traditional safety mechanisms** and can efficiently extract sensitive data that was supposedly removed.  The paper highlights the effectiveness of this approach, demonstrating that even with unlearning techniques applied, substantial amounts of data, such as training data or other sensitive user information, remain recoverable.  **The vulnerability is further amplified by the computational efficiency of these attacks**, making them easily deployable by malicious actors. Addressing these threats requires developing more robust unlearning techniques and novel methods for safeguarding the integrity of models against embedding space attacks and data extraction. This necessitates a multi-faceted approach, including improved privacy-preserving training methods and enhanced detection mechanisms for continuous data extraction attacks.

#### Open-Source Risks
Open-source large language models (LLMs) present a unique set of risks due to their accessibility and transparency.  **The ease of access allows malicious actors to leverage the full model architecture for nefarious purposes**, such as generating harmful content, spreading misinformation, or creating sophisticated phishing attacks.  Unlike closed-source models, open-source LLMs offer full model visibility, enabling attacks that directly manipulate the continuous embedding space representation of input tokens, circumventing traditional safety mechanisms more efficiently.  This poses a serious challenge to existing safety alignment methods and underscores the need for novel defense strategies.  **Furthermore, the ability to directly access model weights and activations facilitates data extraction attacks**, potentially revealing sensitive information from supposedly unlearned or deleted data, or even exposing components of the original pre-training data.  This emphasizes the crucial need for robust unlearning techniques and further research into the security vulnerabilities of open-source LLMs.  **The ongoing evolution of open-source models, constantly improving in capability, exacerbates these risks**, highlighting the urgent need for proactive and adaptive security measures.

#### Future Directions
Future research should prioritize expanding the scope of embedding space attacks to encompass a wider variety of LLMs and investigate their effectiveness against different safety mechanisms.  **Developing more robust defenses** against these attacks is crucial, perhaps by exploring techniques to enhance model robustness in the continuous embedding space.  **Further research into the transferability** of embedding space attacks from open-source models to closed-source models is needed to fully understand their potential impact.  Investigating the broader implications of these attacks on data privacy and security is essential, particularly regarding the extraction of sensitive information from supposedly unlearned models.  Finally, it is vital to **develop novel methods for evaluating the effectiveness** of both attack and defense techniques in the continuous embedding space, going beyond simple metrics like attack success rates to consider the impact on model utility and safety.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/CLxcLPfARc/figures_4_1.jpg)

> This figure illustrates the multi-layer attack, a variation of the embedding space attack.  Instead of directly attacking the input embeddings, this method decodes intermediate hidden state representations from various layers within the neural network (represented by h1, h2,...hL). By doing so, the attack aims to extract information that may have been propagated through the model's layers, possibly revealing information that would not be accessible through standard attacks targeting only the final output layer.  The decoded sequences from each layer (T1k+1, T2k+1,... TLk+1) provide alternative interpretations of the input sequence.


![](https://ai-paper-reviewer.com/CLxcLPfARc/figures_6_1.jpg)

> This figure compares the attack success rate and computation time of various discrete attacks (AutoDAN, PAIR, GCG, Adaptive) against the proposed embedding attack method.  The results show that the embedding attack consistently achieves higher success rates while requiring significantly less computation time across different language models (Llama2, Vicuna, Mistral, and their circuit-breaking versions). This highlights the efficiency and effectiveness of the embedding space attack in bypassing safety mechanisms compared to traditional discrete methods.


![](https://ai-paper-reviewer.com/CLxcLPfARc/figures_7_1.jpg)

> This figure compares the perplexity and toxicity of large language models (LLMs) with and without embedding space attacks.  The results show embedding space attacks reduce perplexity but significantly increase toxicity in most models.  A comparison is also made to fine-tuning Llama2 to remove safety alignment; embedding attacks are shown to be more effective in increasing toxicity.


![](https://ai-paper-reviewer.com/CLxcLPfARc/figures_7_2.jpg)

> This figure shows the correlation between the loss of a universal embedding space attack on the Llama2 model and the toxicity of the generated responses.  The x-axis represents the binned loss values from the attack optimization, while the y-axis shows the number of toxic responses generated for each loss bin.  The figure demonstrates that lower loss values (indicating more successful attacks) are associated with a higher number of toxic responses, suggesting a relationship between attack success and the generation of toxic content.


![](https://ai-paper-reviewer.com/CLxcLPfARc/figures_18_1.jpg)

> This figure shows the contribution of individual layers of the Llama2-7b-WhoIsHarryPotter model to the success rate of both universal and individual embedding attacks.  The x-axis represents the layer number, and the y-axis shows the cumulative success rate, calculated over 100 attack iterations.  The results indicate that the last layer contributes most significantly to the success of the attack, while the earlier layers contribute little or nothing.


![](https://ai-paper-reviewer.com/CLxcLPfARc/figures_18_2.jpg)

> This figure shows the relationship between the L2 norm perturbation magnitude, perplexity, and attack success rate (ASR) for the LlamaHP model on the Harry Potter Q&A dataset.  The left panel shows that perplexity increases as the perturbation magnitude increases, indicating a decrease in generation quality. The right panel shows that ASR initially increases with perturbation magnitude, but then decreases after reaching a peak, suggesting that excessive perturbations hurt the model's ability to generate accurate answers. The overall trend highlights the need to find a balance between perturbation strength and generation quality to effectively attack the model.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/CLxcLPfARc/tables_9_1.jpg)
> This table shows the results of universal embedding space attacks on the TOFU unlearning benchmark's 1% forget dataset, comparing different numbers of attacked tokens and training/testing data splits.  The ROUGE-1 score, a metric for evaluating the quality of text summarization, is used to assess the model's ability to generate responses for unseen data.  The table illustrates the impact of different attack strategies (individual vs. universal attacks) and parameters (number of attacked tokens) on the model's performance after unlearning.

![](https://ai-paper-reviewer.com/CLxcLPfARc/tables_15_1.jpg)
> This table lists the seven different open-source LLMs used in the paper's experiments.  It includes the model name, the reference where the model is described, and a URL to access the model on Hugging Face.

![](https://ai-paper-reviewer.com/CLxcLPfARc/tables_17_1.jpg)
> This table presents example questions from the Harry Potter Q&A dataset and compares the model's responses with and without embedding space attacks.  It showcases the model's ability to retrieve previously unlearned information when subjected to these attacks, highlighting the effectiveness of the method in uncovering residual knowledge.

![](https://ai-paper-reviewer.com/CLxcLPfARc/tables_19_1.jpg)
> This table presents the cumulative success rate of embedding space attacks against the LlamaHP model on the Harry Potter Q&A dataset.  It shows the results for both universal attacks (using all layers) and attacks using only the last layer. The success rates are broken down for different train/test splits (50/50 and 25/75) and varying numbers of attacked tokens (1, 5, and 20).  The table quantifies the model's vulnerability to these attacks under different data conditions and attack parameters.  Higher percentages represent a greater likelihood of the attack successfully retrieving the targeted information.

![](https://ai-paper-reviewer.com/CLxcLPfARc/tables_19_2.jpg)
> This table shows the results of universal embedding space attacks on an unlearned Llama2 model using the TOFU benchmark's 1% forget dataset.  It presents the cumulative ROUGE-1 scores achieved by these attacks under different train/test splits (50/50 and 25/75) and varying numbers of attacked tokens (1, 5, and 20). The results illustrate the performance of the attacks in generalizing to unseen data and highlight the impact of the number of tokens targeted during the attack.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/CLxcLPfARc/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}