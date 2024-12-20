---
title: "Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts"
summary: "Rainbow Teaming: a novel black-box approach generates diverse adversarial prompts to enhance LLM robustness and safety, achieving over 90% attack success rate across various models."
categories: ["AI Generated", ]
tags: ["Natural Language Processing", "Large Language Models", "🏢 Meta AI",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} FCsEvaMorw {{< /keyword >}}
{{< keyword icon="writer" >}} Mikayel Samvelyan et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=FCsEvaMorw" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/FCsEvaMorw" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/FCsEvaMorw/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Large Language Models (LLMs) are increasingly used in real-world applications, but their vulnerability to adversarial attacks remains a concern. Current methods for identifying such attacks are limited by factors such as specific domain focus, lack of diversity, or the need for extensive human annotations.  This makes it challenging to thoroughly understand and enhance their robustness. 

The paper introduces Rainbow Teaming, a novel black-box approach to generate diverse adversarial prompts. This method frames the problem as a quality-diversity search, using open-ended search to generate both effective and diverse prompts.  The approach was tested on various LLMs, demonstrating a high success rate (exceeding 90%) in identifying vulnerabilities. Furthermore, it showed that fine-tuning models with synthetic data created using this method significantly improves their safety without affecting their overall performance. The versatility of this method is also highlighted through its application in question answering and cybersecurity.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Rainbow Teaming, a novel black-box approach for generating diverse adversarial prompts to improve LLM robustness and safety. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The method achieves over 90% attack success rate across various state-of-the-art LLMs. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Fine-tuning models with synthetic data generated by this method significantly enhances their safety without sacrificing general performance. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on **LLM robustness and safety**. It introduces a novel, efficient method for generating diverse adversarial prompts, improving the ability to discover and address vulnerabilities in large language models. The methodology and findings have implications for **enhancing LLM safety** and provide a new avenue for **generating synthetic data** to improve model performance, opening new research directions in this rapidly growing field.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_1_1.jpg)

> 🔼 This figure displays a heatmap visualizing the results of adversarial prompt generation using RAINBOW TEAMING on the Llama 2-chat 7B model.  The rows represent different risk categories (e.g., violence, hate, sexual content), and columns represent various attack styles (e.g., slang, role-playing, technical terms). Each cell's color intensity corresponds to the Llama Guard safety score for the generated prompt belonging to that specific risk category and attack style combination; darker shades indicate higher safety concerns. Excerpts of actual prompts generated by RAINBOW TEAMING are also shown.
> <details>
> <summary>read the caption</summary>
> Figure 1: An example archive generated by RAINBOW TEAMING when used to discover safety vulnerabilities in Llama 2-chat 7B. Here, we search over two features: Risk Category and Attack Style. Shading corresponds to the Llama Guard [26] scores of responses induced by the adversarial prompt in each cell (higher means more confidence in the response being unsafe). Some excerpts of discovered prompts from a single archive are shown.
> </details>





![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_6_1.jpg)

> 🔼 This table compares the performance of RAINBOW TEAMING against two versions of the PAIR method in identifying harmful behaviors from the JailbreakBench dataset.  The top part shows the number of successful jailbreaks achieved by each method, using two different classifiers (JailbreakBench Classifier and Llama Guard). A higher number indicates better performance. The bottom part shows the Self-BLEU score, which measures the diversity of the generated prompts. A lower score indicates higher diversity, suggesting more varied attacks.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison of RAINBOW TEAMING against PAIR [8] for eliciting harmful behaviours from JailbreakBench [9]. Top: (n/k) indicates the total number of successful jailbreaks (n) and the total number of behaviours jailbroken (k) for each method and classifier (best of 4 responses). Bottom: Self-BLEU similarity score.
> </details>





### In-depth insights


#### Diverse Adversarial Prompts
The concept of "Diverse Adversarial Prompts" in the context of large language model (LLM) research is crucial for evaluating and enhancing their robustness.  **Diversity** is key because LLMs, when presented with a narrow range of adversarial prompts, might be resilient to those specific attacks but vulnerable to others. A diverse set of prompts, encompassing various attack styles, linguistic features, and semantic nuances, provides a more comprehensive assessment of an LLM's weaknesses. **This diversity facilitates the identification of vulnerabilities that might otherwise be missed**, leading to a more robust and reliable model.  The generation of diverse prompts presents a significant methodological challenge.  **Creating effective adversarial prompts often requires substantial effort and expertise**, and generating diverse ones significantly increases the complexity.  Approaches that cast prompt generation as a quality-diversity problem and leverage search algorithms to produce both effective and diverse attacks become vital. **The transferability of these prompts across different LLMs is also a critical consideration**, suggesting that successful adversarial techniques could be applicable to various models, highlighting the need for broad, model-agnostic safety improvements.

#### Rainbow Teaming Method
The Rainbow Teaming method, as described in the research paper, is a novel black-box approach for generating diverse adversarial prompts.  **It leverages a quality-diversity (QD) search algorithm, specifically MAP-Elites, to create prompts that are both effective at eliciting undesirable responses from large language models (LLMs) and diverse in their features.**  The method's open-ended nature allows it to explore a wider range of adversarial attacks than previous approaches. **A key aspect is its use of a 'mutator' LLM and a 'judge' LLM, where the mutator modifies existing prompts to generate new ones and the judge evaluates their effectiveness**. This iterative process ensures the generation of high-quality and varied prompts.  **The generated prompts serve a dual purpose: as a diagnostic tool to identify vulnerabilities in LLMs and as a dataset for fine-tuning LLMs to improve their robustness**. The flexibility and generality of Rainbow Teaming are highlighted through its successful application to various domains, including safety, question answering, and cybersecurity, showcasing its potential for broad application and fostering LLM self-improvement.

#### Safety & Robustness
A robust large language model (LLM) must prioritize both safety and robustness.  Safety focuses on preventing harmful outputs, such as toxic, biased, or factually incorrect responses.  Robustness, on the other hand, aims to make the model resilient to various types of adversarial attacks, including those employing cleverly crafted prompts designed to elicit undesired behavior.  **The intersection of safety and robustness is crucial**, as a model that is safe in standard conditions may still be vulnerable to malicious manipulation.  Therefore, approaches that enhance robustness, such as adversarial training, are vital for building safe and reliable LLMs.  **Diverse and extensive testing** is needed to ensure that safety mechanisms are not easily bypassed by adversarial techniques.  Furthermore, **continuous monitoring** and adaptation are essential to detect and mitigate newly discovered vulnerabilities and ensure the long-term safety of these increasingly powerful systems.  Evaluating the model's performance across different domains and under various attack scenarios is crucial. The development of **effective evaluation metrics** and strategies for quantifying safety and robustness is an ongoing challenge in this field.

#### Transferability of Attacks
The concept of "Transferability of Attacks" in the context of large language models (LLMs) centers on whether adversarial prompts successful against one model generalize to others.  **High transferability implies vulnerabilities are inherent to the LLM architecture itself rather than specific training data or fine-tuning processes.** This has significant implications for model security, because it means addressing vulnerabilities in one model might not fully safeguard against attacks on other models of the same or even different architectures.  **Research exploring transferability helps in developing more robust and generalizable defenses against adversarial attacks.**  A low transferability rate suggests vulnerabilities are more data-dependent, implying that tailored defenses might suffice.  Conversely, **high transferability highlights the critical need for fundamentally robust LLM designs** that are resilient to a broader range of adversarial prompts, regardless of the model's specific training regime or dataset.

#### Future Research
Future research directions stemming from this Rainbow Teaming paper could explore several key areas.  **Automating feature discovery** would significantly enhance the method's adaptability, moving beyond predefined categories to dynamically identify vulnerabilities.  Investigating the effects of **different mutation strategies** and **preference models** on prompt diversity and effectiveness warrants further study.  A key limitation is the computational cost; future work should investigate ways to improve efficiency, perhaps through more sophisticated sampling techniques or parallel processing.  **Improving the transferability** of adversarial prompts across different LLMs is also important, potentially requiring the development of more robust or generalized prompts. Finally, exploring Rainbow Teaming's applications beyond safety, question answering, and cybersecurity, such as in the domains of **bias mitigation** and **robustness testing**, represents a promising avenue for future research.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_3_1.jpg)

> 🔼 This figure illustrates the workflow of the RAINBOW TEAMING algorithm in the context of safety. The algorithm uses a multi-dimensional grid (archive) to store adversarial prompts, categorized by features like risk category and attack style.  Each iteration involves sampling a prompt from the archive, mutating it using a Mutator LLM, generating a response using a Target LLM, and then comparing the response to the existing archive entry with a Judge LLM to determine which prompt is more effective at triggering unsafe responses. The archive is updated to retain the most effective prompts, ensuring diversity and effectiveness of the discovered prompts.  The figure visually depicts the key components (Mutator LLM, Target LLM, Judge LLM, Archive) and their interactions during the iterative process.
> <details>
> <summary>read the caption</summary>
> Figure 2: Overview of RAINBOW TEAMING in the safety domain: Our method operates on a discretised grid, archiving adversarial prompts with K defining features, such as Risk Category or Attack Style. Each iteration involves a Mutator LLM applying K mutations to generate new candidate prompts. These prompts are then fed into the Target LLM. A Judge LLM evaluates these responses against archived prompts with the same features, updating the archive with any prompt that elicits a more unsafe response from the Target.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_5_1.jpg)

> 🔼 This figure shows the attack success rate (ASR) of adversarial prompts generated by RAINBOW TEAMING against four different large language models (LLMs): Llama 2-chat 7B, Llama 3-Instruct 8B, Mistral 7B, and Vicuna 7B v1.5.  The ASR is evaluated using GPT-4, a large language model developed by OpenAI. The x-axis represents the number of iterations of the RAINBOW TEAMING algorithm, and the y-axis represents the ASR. The lines represent the different LLMs, and the shaded areas around the lines represent the standard error of the mean.
> <details>
> <summary>read the caption</summary>
> Figure 3: Attack success rate of adversarial prompts discovered by RAINBOW TEAMING for different models, as evaluated by GPT-4.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_7_1.jpg)

> 🔼 This figure shows the attack success rate of adversarial prompts against Llama 2-chat 7B, both before and after fine-tuning it using synthetic data generated by Rainbow Teaming. The results demonstrate that fine-tuning significantly enhances the model's robustness to adversarial attacks.
> <details>
> <summary>read the caption</summary>
> Figure 5: Attack success rate before and after fine-tuning Llama 2-chat 7B on synthetic data generated via RAINBOW TEAMING. The fine-tuned model is significantly less vulnerable to RAINBOW TEAMING on a second application, with the method achieving a substantially lower ASR after 2000 iterations. We report the mean and standard error over 3 independent runs.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_8_1.jpg)

> 🔼 This figure shows a 3D archive generated by RAINBOW TEAMING for the question answering task.  The three dimensions are Topic, Interrogative Word, and Length.  Each cell in the archive represents a unique combination of these three features. The color of each cell indicates the success rate of the adversarial question in that cell, with purple indicating high success rates, green indicating intermediate success, and yellow indicating that no successful adversarial questions were found for that combination of features.
> <details>
> <summary>read the caption</summary>
> Figure 6: An example archive of adversarial questions discovered by RAINBOW TEAMING. Vacant cells are marked in yellow, intermediate but unsuccessful attempts are in green, and successful adversarial questions are in purple.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_17_1.jpg)

> 🔼 This figure shows a heatmap visualizing the results of Rainbow Teaming applied to Llama 2-chat 7B.  The x and y axes represent different attack styles and risk categories, respectively. Each cell's color intensity corresponds to the Llama Guard safety score of the LLM's response to the prompt associated with that cell; darker colors indicate higher safety scores (more confidence in safety), while lighter colors indicate lower safety scores (less confidence in safety).  The figure also includes example prompts from the archive, illustrating the diverse range of prompts generated by Rainbow Teaming.
> <details>
> <summary>read the caption</summary>
> Figure 1: An example archive generated by RAINBOW TEAMING when used to discover safety vulnerabilities in Llama 2-chat 7B. Here, we search over two features: Risk Category and Attack Style. Shading corresponds to the Llama Guard [26] scores of responses induced by the adversarial prompt in each cell (higher means more confidence in the response being unsafe). Some excerpts of discovered prompts from a single archive are shown.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_17_2.jpg)

> 🔼 This figure shows a heatmap visualizing the results of Rainbow Teaming applied to Llama 2-chat 7B in the safety domain.  The rows and columns represent different Risk Categories and Attack Styles, respectively, which are used as dimensions in the quality-diversity (QD) search. Each cell's color intensity corresponds to the Llama Guard score, indicating the model's confidence that the response generated by that specific adversarial prompt is unsafe (darker = more unsafe).  A snippet of actual adversarial prompts that were generated for each cell is also shown for illustrative purposes.
> <details>
> <summary>read the caption</summary>
> Figure 1: An example archive generated by RAINBOW TEAMING when used to discover safety vulnerabilities in Llama 2-chat 7B. Here, we search over two features: Risk Category and Attack Style. Shading corresponds to the Llama Guard [26] scores of responses induced by the adversarial prompt in each cell (higher means more confidence in the response being unsafe). Some excerpts of discovered prompts from a single archive are shown.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_18_1.jpg)

> 🔼 This figure shows a heatmap visualizing the results of the RAINBOW TEAMING algorithm in discovering safety vulnerabilities within the Llama 2-chat 7B language model. The rows represent different attack styles used to generate adversarial prompts. The columns represent various risk categories (e.g., violence, hate speech, fraud). Each cell in the heatmap shows the Llama Guard score for prompts from a specific row and column combination; higher scores indicate higher confidence that the response is unsafe. The color intensity represents the Llama Guard score. The figure also includes examples of adversarial prompts generated by the model for various combinations of risk categories and attack styles.
> <details>
> <summary>read the caption</summary>
> Figure 1: An example archive generated by RAINBOW TEAMING when used to discover safety vulnerabilities in Llama 2-chat 7B. Here, we search over two features: Risk Category and Attack Style. Shading corresponds to the Llama Guard [26] scores of responses induced by the adversarial prompt in each cell (higher means more confidence in the response being unsafe). Some excerpts of discovered prompts from a single archive are shown.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_19_1.jpg)

> 🔼 This figure shows the attack success rate achieved by the Rainbow Teaming method against three different sizes of Llama 2 chat models (7B, 13B, and 70B parameters).  The results are evaluated using two different safety classifiers: GPT-4 and Llama Guard. The graph displays how the attack success rate changes over 2000 iterations of the Rainbow Teaming algorithm, with error bars representing the mean and standard deviation across three independent runs. It demonstrates the effectiveness of Rainbow Teaming across varying model sizes and evaluation methods.
> <details>
> <summary>read the caption</summary>
> Figure 8: Attack success rate of adversarial prompts discovered by RAINBOW TEAMING on Llama 2-chat 7B, 13B, and 70B, as measured by GPT-4 and Llama Guard. We report the mean and standard error over 3 independent runs.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_20_1.jpg)

> 🔼 This figure compares two variants of the RAINBOW TEAMING algorithm: one using a pairwise comparison method (Judge) and another using a score-based method (No Judge) for evaluating the quality of adversarial prompts.  The left panel shows the Attack Success Rate (ASR) as evaluated by GPT-4, the center panel shows the ASR as evaluated by Llama Guard, and the right panel shows the total number of archive updates over time. The results indicate that the comparison-based method is more effective and robust, and performs a more thorough exploration of the prompt space.
> <details>
> <summary>read the caption</summary>
> Figure 9: Comparison of RAINBOW TEAMING with a pairwise comparison (Judge) and a score-based (No Judge) preference models applied to Llama 2-chat 7B. Left: ASR as evaluated by GPT-4. Centre: ASR as evaluated by Llama Guard. Right: total archive updates over time. The score-based baseline reward hacks the Llama Guard score and underperforms under GPT-4 evaluation. It also stops updating the archive after saturating the Llama Guard score, whereas the comparison method RAINBOW TEAMING performs a more open-ended search.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_21_1.jpg)

> 🔼 This figure shows the attack success rate (ASR) achieved by RAINBOW TEAMING against four different LLMs (Llama 2-chat 7B, Llama 3-Instruct 8B, Mistral 7B, and Vicuna 7B v1.5) across 2000 iterations.  The ASR is measured using GPT-4. The graph illustrates how the effectiveness of RAINBOW TEAMING in generating successful adversarial prompts varies across different language models.  It demonstrates that RAINBOW TEAMING is effective against all tested models, highlighting different levels of vulnerability.
> <details>
> <summary>read the caption</summary>
> Figure 3: Attack success rate of adversarial prompts discovered by RAINBOW TEAMING for different models, as evaluated by GPT-4.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_21_2.jpg)

> 🔼 This figure compares the attack success rate of adversarial prompts generated by RAINBOW TEAMING against two baselines: (1) No Stepping Stones, which generates prompts without using previously discovered prompts, and (2) Same Cell Mutations, which performs mutations independently within each cell of the archive.  The plot shows the attack success rate over 2000 iterations, using both GPT-4 and Llama Guard for evaluation. RAINBOW TEAMING consistently outperforms both baselines, demonstrating the value of using stepping stones and cross-category mutations.
> <details>
> <summary>read the caption</summary>
> Figure 4: Attack success rate of adversarial prompts discovered by RAINBOW TEAMING and baselines against the Llama 2-chat 7B model.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_22_1.jpg)

> 🔼 This figure shows the attack success rate of adversarial prompts against the Llama 2-chat 7B model before and after fine-tuning it with synthetic data generated by RAINBOW TEAMING. The results demonstrate that fine-tuning with synthetic data generated by RAINBOW TEAMING significantly improves the model's robustness to adversarial attacks. The attack success rate is lower after the fine-tuning, demonstrating the effectiveness of the method in enhancing the model's resilience.
> <details>
> <summary>read the caption</summary>
> Figure 5: Attack success rate before and after fine-tuning Llama 2-chat 7B on synthetic data generated via RAINBOW TEAMING. The fine-tuned model is significantly less vulnerable to RAINBOW TEAMING on a second application, with the method achieving a substantially lower ASR after 2000 iterations. We report the mean and standard error over 3 independent runs.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_23_1.jpg)

> 🔼 This figure shows a heatmap visualizing the results of adversarial prompt generation using RAINBOW TEAMING. The rows represent different risk categories (e.g. violence, sexual content), and the columns represent different attack styles (e.g. slang, role-play). Each cell's color intensity represents the Llama Guard score, which measures the safety of the model's response to the corresponding adversarial prompt. Darker shades indicate more unsafe responses.  The figure also provides examples of prompts found by RAINBOW TEAMING within the categories shown.
> <details>
> <summary>read the caption</summary>
> Figure 1: An example archive generated by RAINBOW TEAMING when used to discover safety vulnerabilities in Llama 2-chat 7B. Here, we search over two features: Risk Category and Attack Style. Shading corresponds to the Llama Guard [26] scores of responses induced by the adversarial prompt in each cell (higher means more confidence in the response being unsafe). Some excerpts of discovered prompts from a single archive are shown.
> </details>



![](https://ai-paper-reviewer.com/FCsEvaMorw/figures_24_1.jpg)

> 🔼 This figure shows a heatmap visualizing the results of Rainbow Teaming's search for adversarial prompts targeting Llama 2-chat 7B. The heatmap's rows and columns represent different 'Risk Categories' and 'Attack Styles', respectively. Each cell's color intensity corresponds to a Llama Guard safety score, indicating how likely the model's response to the corresponding prompt is to be unsafe (darker = more unsafe).  The figure also includes examples of adversarial prompts discovered in this process, further illustrating the diverse range of prompts identified.
> <details>
> <summary>read the caption</summary>
> Figure 1: An example archive generated by RAINBOW TEAMING when used to discover safety vulnerabilities in Llama 2-chat 7B. Here, we search over two features: Risk Category and Attack Style. Shading corresponds to the Llama Guard [26] scores of responses induced by the adversarial prompt in each cell (higher means more confidence in the response being unsafe). Some excerpts of discovered prompts from a single archive are shown.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_6_2.jpg)
> 🔼 This table presents the Attack Success Rate (ASR) when transferring adversarial prompts generated by RAINBOW TEAMING across different LLMs.  It shows the ASR of prompts generated against a specific model when applied to four other models.  The results demonstrate that the prompts exhibit some level of transferability across various models, with an average transfer rate of around 50%, although this varies significantly depending on the source and target models.  This highlights the potential generality of the generated adversarial prompts, suggesting that many are not model-specific.
> <details>
> <summary>read the caption</summary>
> Table 2: Transfer of adversarial prompts across different models. We take 3 archives for each original target, apply them to the transfer target, and report the mean and standard deviation of the ASR as evaluated by Llama Guard (best of 4 responses). 50% of adversarial prompts transfer on average, but the exact transfer varies drastically between models. All models reported are instruction fine-tuned.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_6_3.jpg)
> 🔼 This table presents the results of an experiment comparing the performance of RAINBOW TEAMING with and without a similarity filter.  The similarity filter removes prompts that are too similar to their parent prompts. The results show that using the filter improves the diversity of the prompts in the archive while maintaining a high attack success rate (ASR).  Metrics such as self-BLEU, BERTScore, and ROGUE-L were used to assess the diversity of the prompts, while gzip compression ratio was used to assess the compactness of the archive.
> <details>
> <summary>read the caption</summary>
> Table 3: Analysis of the effect of a mutation-level similarity filter of RAINBOW TEAMING on ASR measured by GPT-4 and archive diversity (self-BLEU, BERTScore, ROGUE-L, and gzip compression ratio). Filtering out prompts that are too similar to their parent maintains a balance between ASR and diversity, whereas removing the filter encourages the method to reuse highly effective prompts across multiple cells. The filter is set at T = 0.6, discarding ~ 24% of mutated prompts. We report mean and standard error over 3 independent runs.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_7_1.jpg)
> 🔼 This table presents the results of fine-tuning the Llama 2-chat 7B model with data generated by RAINBOW TEAMING.  It shows the Attack Success Rate (ASR) before and after fine-tuning, as measured by two different safety classifiers (GPT-4 and Llama Guard), and also on the JailbreakBench dataset.  Additionally, it reports the model's performance on general capability benchmarks (GSM8K and MMLU), and its safety and helpfulness scores before and after fine-tuning.
> <details>
> <summary>read the caption</summary>
> Table 4: Safety and capabilities scores of the Llama 2-chat 7B model before and after SFT on RAINBOW TEAMING-generated data. Fine-tuning greatly improves robustness to adversarial prompts without hurting capabilities.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_8_1.jpg)
> 🔼 This table compares the performance of RAINBOW TEAMING against a baseline method in a question answering task.  RAINBOW TEAMING leverages previously generated questions as 'stepping stones' to guide the generation of new questions, while the baseline generates questions from scratch independently for each iteration. The table shows that RAINBOW TEAMING outperforms the baseline across three metrics: Mean Fitness (higher is better, indicating more effective adversarial questions), Coverage (higher is better, representing the breadth of the archive), and Self-BLEU (lower is better, indicating greater diversity of generated questions).  This demonstrates the advantage of utilizing past findings in the iterative search process for discovering high-quality and diverse adversarial examples.
> <details>
> <summary>read the caption</summary>
> Table 5: Comparison of RAINBOW TEAMING to a baseline generating new questions from scratch each turn for the Q&A domain. Without reusing past questions as stepping stones, performance is worse across all metrics considered. We report the mean and standard deviation over 3 seeds.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_8_2.jpg)
> 🔼 This table presents the results of a cybersecurity assessment for various target models on prompts generated by RAINBOW TEAMING.  It shows the Attack Success Rate (ASR) as evaluated by two methods: CyberSecurityEval (an automated tool) and human expert evaluation. The table demonstrates the high effectiveness of RAINBOW TEAMING in generating adversarial prompts that elicit malicious behavior across different models.
> <details>
> <summary>read the caption</summary>
> Table 6: Cybersecurity ASR of RAINBOW TEAMING on four Targets, as reported by CyberSecurityEval [4] (3 seeds), and human expert evaluation (1 seed).
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_19_1.jpg)
> 🔼 This table presents the results of an experiment evaluating the effectiveness of different system prompts in mitigating adversarial attacks against the Llama 2-chat 7B language model.  Three scenarios are compared: no system prompt, a helpful system prompt, and a legacy system prompt known to enhance safety but prone to high false positives.  The attack success rate (ASR) is measured using two different evaluators (GPT-4 and Llama Guard) to assess the model's vulnerability to adversarial prompts under each condition. The results show that the legacy prompt significantly improves robustness, although the helpful prompt still provides increased resistance compared to no prompt at all.
> <details>
> <summary>read the caption</summary>
> Table 7: Attack success rate against Llama 2-chat 7B model with different system prompts. 'Legacy' is an original Llama 2-chat system prompt that explicitly promotes safety, but was deprecated as it results in a high false refusal rate [65]. Nonetheless, it makes the model significantly more robust, supporting the idea that system prompts are an imperfect but low-effort defence mechanism against adversarial attacks.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_19_2.jpg)
> 🔼 This table compares the performance of RAINBOW TEAMING against two versions of the PAIR method in eliciting harmful behaviours from the JailbreakBench dataset.  It shows the total number of successful jailbreaks and the total number of behaviours successfully jailbroken for each method using two different classifiers (JailbreakBench Classifier and Llama Guard).  Lower Self-BLEU scores indicate higher diversity in the generated prompts.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison of RAINBOW TEAMING against PAIR [8] for eliciting harmful behaviours from JailbreakBench [9]. Top: (n/k) indicates the total number of successful jailbreaks (n) and the total number of behaviours jailbroken (k) for each method and classifier (best of 4 responses). Bottom: Self-BLEU similarity score.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_22_1.jpg)
> 🔼 This table compares the performance of RAINBOW TEAMING against two variants of the PAIR method in eliciting harmful behaviors from the JailbreakBench dataset.  It shows the number of successful jailbreaks achieved by each method, categorized by the classifier used (JailbreakBench Classifier and Llama Guard). A lower Self-BLEU score indicates greater diversity in the generated prompts.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison of RAINBOW TEAMING against PAIR [8] for eliciting harmful behaviours from JailbreakBench [9]. Top: (n/k) indicates the total number of successful jailbreaks (n) and the total number of behaviours jailbroken (k) for each method and classifier (best of 4 responses). Bottom: Self-BLEU similarity score.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_31_1.jpg)
> 🔼 This table lists the hyperparameters used in the safety experiments of the RAINBOW TEAMING approach.  It includes the number of initial examples used to seed the algorithm, batch size for training, the number of iterations run, the BLEU similarity filter threshold for diversity, the archive sampling temperature controlling exploration-exploitation balance, the size of the archive, generation temperature, top-k sampling parameter, maximum tokens for prompt generation, learning rate for fine-tuning (SFT), batch size for SFT, learning rate scheduler used, and sequence length for SFT.
> <details>
> <summary>read the caption</summary>
> Table 10: List of hyperparameters used in safety experiments.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_31_2.jpg)
> 🔼 This table lists the hyperparameters used in the question answering experiments of the RAINBOW TEAMING approach.  It details values for parameters related to the RAINBOW TEAMING algorithm itself (number of initial examples, dataset used, batch size, iterations, BLEU similarity filter threshold, archive sampling temperature, and archive size) as well as for the prompt generation process (temperature, top-k sampling, and maximum number of tokens).  These settings were crucial in controlling the diversity and effectiveness of the generated adversarial questions.
> <details>
> <summary>read the caption</summary>
> Table 11: List of hyperparameters used in question answering experiments.
> </details>

![](https://ai-paper-reviewer.com/FCsEvaMorw/tables_31_3.jpg)
> 🔼 This table compares the performance of RAINBOW TEAMING against two versions of the PAIR method in eliciting harmful behaviors from the JailbreakBench dataset.  It shows the number of successful jailbreaks achieved by each method, using two different classifiers: the JailbreakBench classifier and Llama Guard. A lower Self-BLEU score indicates higher diversity in the generated prompts.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison of RAINBOW TEAMING against PAIR [8] for eliciting harmful behaviours from JailbreakBench [9]. Top: (n/k) indicates the total number of successful jailbreaks (n) and the total number of behaviours jailbroken (k) for each method and classifier (best of 4 responses). Bottom: Self-BLEU similarity score.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/FCsEvaMorw/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}