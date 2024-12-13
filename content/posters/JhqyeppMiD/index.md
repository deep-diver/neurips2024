---
title: "Shadowcast: Stealthy Data Poisoning Attacks Against Vision-Language Models"
summary: "Shadowcast:  A new data poisoning attack manipulates vision-language models by injecting visually similar, yet deceptively misleading, image-text pairs, causing them to generate false information."
categories: []
tags: ["Multimodal Learning", "Vision-Language Models", "🏢 University of Maryland, College Park",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} JhqyeppMiD {{< /keyword >}}
{{< keyword icon="writer" >}} Yuancheng Xu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=JhqyeppMiD" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95705" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=JhqyeppMiD&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/JhqyeppMiD/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Vision-language models (VLMs) are increasingly used across various applications, but their reliance on external training data raises security concerns.  Traditional data poisoning attacks primarily focus on image misclassification.  However, VLMs' ability to generate text creates opportunities for more sophisticated attacks that go beyond simple mislabeling.  This is a significant problem because it can affect a broad range of users, not just those who encounter adversarial prompts.  The potential for widespread impact is substantial due to the large scale of VLM deployment.

The research introduces Shadowcast, a novel stealthy data poisoning attack.  Unlike previous methods, Shadowcast leverages VLMs' text generation capabilities to craft persuasive yet false narratives. Poison samples are visually indistinguishable from benign examples making them harder to detect.  Experiments demonstrate that Shadowcast is effective across various VLM architectures, remaining potent under realistic conditions with limited poisoned data. This highlights the urgent need for robust security measures and underscores the importance of data integrity in VLM development.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} VLMs are susceptible to stealthy data poisoning attacks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Shadowcast effectively manipulates VLMs' responses with minimal poisoned data. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Poisoned VLMs disseminate convincing yet deceptive misinformation. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it reveals the vulnerability of vision-language models (VLMs) to stealthy data poisoning attacks** that can subtly manipulate responses to even benign prompts, impacting numerous users.  It highlights the urgent need for enhanced security measures and emphasizes data integrity in VLM development and deployment. This opens avenues for further research into robust defense mechanisms against such attacks.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_1_1.jpg)

> This figure shows the results of two different attacks using the Shadowcast method. The top part shows a Label Attack, where the model is tricked into misidentifying Donald Trump as Joe Biden. The bottom part shows a Persuasion Attack, where the model is tricked into describing junk food as healthy.  Both attacks demonstrate that Shadowcast is able to successfully manipulate the responses of the model even with visually similar images.





![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_1_1.jpg)

> This table compares different types of attacks (test-time and poisoning) against three different types of models (image classifiers, LLMs, and VLMs) based on three criteria: pervasiveness of impact, stealthiness, and the ability to use misleading text.  It highlights that the proposed Shadowcast attack is unique in its combination of pervasiveness, stealth, and the use of deceptive text, setting it apart from previous attacks.





### In-depth insights


#### Stealthy VLM Poison
Stealthy VLM poisoning presents a significant threat to the trustworthiness and reliability of Vision-Language Models (VLMs).  The core idea revolves around manipulating VLM outputs by introducing subtly altered training data—poison samples—that are visually indistinguishable from benign examples.  **This stealthiness is crucial**, as it allows malicious actors to inject misinformation without arousing suspicion. The impact extends beyond simple misclassification; **more complex attacks like 'Persuasion Attacks'** become possible, where VLMs are manipulated to generate subtly biased or misleading narratives.  The effectiveness of stealthy VLM poisoning highlights the vulnerability of VLMs to data integrity issues, **emphasizing the need for robust defenses and secure data sourcing practices**.  The research on Shadowcast, mentioned in the prompt, is a pioneering effort in this domain, demonstrating the feasibility and alarming consequences of this type of attack.  The transferability of these attacks across different VLM architectures poses a further challenge, demanding a comprehensive approach to securing these increasingly influential models.

#### Novel Persuasion Attack
The concept of a "Novel Persuasion Attack" within the context of Vision-Language Models (VLMs) introduces a significant advancement in adversarial attacks.  It moves beyond simple misclassification (as seen in traditional label attacks) by leveraging the VLM's text generation capabilities to subtly manipulate user perception.  **Instead of forcing the model to choose an incorrect label, the attack crafts convincing yet misleading narratives that alter the understanding of the image's content.** This is achieved through the creation of visually indistinguishable poisoned samples, where the image remains seemingly benign while the accompanying text presents a distorted interpretation.  The stealthiness of this approach highlights its potential for widespread and insidious dissemination of misinformation.  The impact is further amplified by the persuasive nature of the generated text, exploiting the inherent trust users place in VLMs' responses. **This type of attack requires sophisticated techniques to subtly alter image features while crafting coherent, persuasive narratives that maintain visual alignment with the modified image.**  The research into this novel technique, therefore, underscores the critical need for enhanced security measures to protect against such manipulative attacks within VLMs and similar technologies.  **The potential for malicious actors to spread misinformation using seemingly harmless VLMs necessitates the development of robust detection and defense mechanisms.**

#### Shadowcast's Robustness
The robustness of Shadowcast, a stealthy data poisoning attack against Vision-Language Models (VLMs), is a critical aspect of its effectiveness.  The authors demonstrate robustness across several dimensions. First, **Shadowcast remains potent under diverse text prompts**, showcasing its effectiveness beyond limited, specific phrasing.  Second, the attack is shown to be **resilient to data augmentation techniques** commonly used during VLM training, highlighting its potential to evade standard defenses.  Furthermore, **Shadowcast's effectiveness persists under image compression**, demonstrating practical resilience to realistic image processing scenarios. Finally, it displays **transferability across different VLM architectures**, indicating a broader impact that extends beyond specific model implementations. These results together demonstrate that Shadowcast presents a significant and practical threat to VLMs, even under realistic training conditions.

#### Black-box Transferability
Black-box transferability in data poisoning attacks against Vision-Language Models (VLMs) is a crucial concern.  It examines the ability of an attacker to poison a VLM using data crafted with a different, potentially open-source model, without direct access to the target VLM's architecture or weights.  **Successful black-box transferability demonstrates a significant threat**, as it implies that poisoning attacks can be effective even against models for which the attacker lacks detailed knowledge. This raises serious security implications, as it becomes much more difficult to defend against attacks from unknown actors using diverse approaches. The effectiveness of black-box transferability may vary depending on the similarity between the source and target VLMs' architectures and training data, but **even partial success highlights the need for robust defenses** that are not model-specific. Future research should focus on designing such defenses, which could involve techniques to improve model robustness, reduce the impact of poisoned data or detecting the presence of poisoned samples with advanced anomaly detection methods.

#### Future VLM Safeguards
Future safeguards for Vision-Language Models (VLMs) must address the vulnerabilities exposed by data poisoning attacks like Shadowcast.  **Robustness to data manipulation** is paramount; this requires developing techniques to detect and mitigate subtle alterations in both image and text data used for training.  **Improved data provenance and verification** methods are needed to ensure data integrity throughout the VLM lifecycle.  Furthermore, **developing more resilient VLM architectures** that are less susceptible to adversarial manipulation is critical. This might involve exploring techniques such as adversarial training or incorporating mechanisms for detecting and rejecting poisoned samples during training or inference.  Finally, **establishing standardized evaluation benchmarks and metrics for assessing VLM security** is essential for facilitating future research and development in this crucial area.  These safeguards should be implemented proactively, not reactively, to ensure responsible and secure deployment of VLMs.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_3_1.jpg)

> The figure illustrates the Shadowcast data poisoning method.  It shows how a poison sample is created by combining a slightly perturbed version of a destination concept image (visually very similar to the original concept image) with a text caption that clearly describes the destination concept. The perturbed image is created by adding imperceptible noise to a clean destination concept image, making it visually similar to a clean original concept image in the latent feature space. This technique makes the poisoned samples nearly indistinguishable from benign samples, enabling stealthy manipulation of the VLM.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_6_1.jpg)

> This figure shows the attack success rate for two label attack tasks (Trump-to-Biden and EngineLight-to-LowFuelLight) against the LLaVA-1.5 model as a function of the percentage of poisoned training samples.  The x-axis represents the percentage of poisoned training samples, and the y-axis represents the attack success rate. The results show that Shadowcast achieves high attack success rates with a small number of poisoned samples.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_6_2.jpg)

> This figure shows the attack success rate for two different Persuasion Attack tasks against the LLaVA-1.5 model. The x-axis represents the percentage of poisoned training samples used, and the y-axis shows the attack success rate.  The plot indicates that the attack's success increases as the percentage of poisoned training samples increases.  The two tasks are 'JunkFood-to-HealthyFood' and 'VideoGame-to-PhysicalHealth', demonstrating Shadowcast's effectiveness in manipulating model responses to present misleading narratives even for innocuous prompts.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_7_1.jpg)

> The figure shows the results of a human evaluation study assessing the coherence and relevance of responses generated by clean and poisoned vision-language models (VLMs).  The evaluation focused on two tasks: JunkFood-to-HealthyFood and VideoGame-to-PhysicalHealth.  For each task, human evaluators rated the coherence and relevance of the VLM's responses to test images using a 1-5 scale. The x-axis represents the percentage of poisoned training samples used to train the model, while the y-axis shows the average rating for coherence and relevance.  The results indicate that the poisoned models maintain high coherence and relevance, suggesting the effectiveness of the Shadowcast attack in subtly manipulating VLM responses.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_7_2.jpg)

> This figure demonstrates the robustness of Shadowcast across various prompts.  It shows that the attack's success rate remains consistent even when different questions are used to query the poisoned Vision Language Models (VLMs) during inference. This highlights the pervasive nature of the attack, as it's not limited to specific phrasings.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_8_1.jpg)

> This figure shows the attack success rate for LLaVA-1.5 when using poison images generated by two different models: InstructBLIP and MiniGPT-v2.  The left panel displays results when poison samples are created using InstructBLIP, while the right panel shows results when using MiniGPT-v2.  The x-axis represents the percentage of poisoned training samples, and the y-axis represents the attack success rate.  Different colored lines represent the different attack tasks from Table 2 (Trump-to-Biden, EngineLight-to-FuelLight, JunkFood-to-HealthyFood, VideoGame-to-PhysicalHealth).  The figure demonstrates the transferability of Shadowcast across different VLM architectures.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_8_2.jpg)

> This figure shows the results of experiments evaluating the robustness of Shadowcast against data augmentation.  The left panel shows the attack success rate when poison images are crafted without using data augmentation during the training of the LLaVA-1.5 model. The right panel shows the same metric when the poison images are created using the same data augmentation techniques used in training LLaVA-1.5. The results indicate how effectively Shadowcast performs under various data augmentation strategies.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_8_3.jpg)

> This figure shows the robustness of Shadowcast across various prompts.  Even when different phrasing is used to query the poisoned vision-language model, the attack remains highly effective. This highlights the pervasive nature of the attack, as it is not limited to specific phrasing.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_13_1.jpg)

> The figure shows the results of a Label Attack and a Persuasion Attack on the LLaVA-1.5 model.  The Label Attack successfully misidentifies Donald Trump as Joe Biden. The Persuasion Attack manipulates the model's response to describe junk food as healthy.  Both attacks demonstrate the effectiveness of the Shadowcast data poisoning technique.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_16_1.jpg)

> This figure shows the results of a data poisoning attack called Shadowcast on the LLaVA-1.5 vision-language model.  The top half demonstrates a Label Attack, where the model is tricked into misidentifying Donald Trump as Joe Biden. The bottom half shows a Persuasion Attack, where the model is manipulated to present junk food as healthy.  Both attacks use poison samples generated by a different model, MiniGPT-v2, highlighting the transferability of the attack across different model architectures.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_17_1.jpg)

> This figure shows the results of a Label Attack and a Persuasion Attack on the LLaVA-1.5 vision-language model.  The top half demonstrates a Label Attack, where poisoned samples cause the model to misidentify Donald Trump as Joe Biden. The bottom half shows a Persuasion Attack, where poisoned samples manipulate the model to describe unhealthy food (a hamburger and fries) as healthy.  The poisoned samples were created using MiniGPT-v2, a different VLM than the one being attacked, highlighting the transferability of the attack.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_18_1.jpg)

> This figure shows the results of a Label Attack and a Persuasion Attack on the LLaVA-1.5 model.  The top half demonstrates a Label Attack where the model is tricked into misidentifying Donald Trump as Joe Biden. The bottom half shows a Persuasion Attack, where the model is manipulated to describe junk food as healthy.  Poison samples for both attacks were generated using MiniGPT-v2, highlighting the transferability of the attack across different models.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_18_2.jpg)

> This figure shows the attack success rate as a function of the proportion of poison samples used for poisoning LLaVA-1.5 on two Label Attack tasks (Trump-to-Biden and EngineLight-to-FuelLight). The results show that Shadowcast achieves a significant impact (over 60% attack success rate) with a poison rate of under 1% (or 30 poison samples).  A poison rate larger than 1.4% (or 50 poison samples) results in successful Label Attack over 95% and 80% of the time for the two tasks respectively. This demonstrates the high efficiency of Shadowcast for Label Attack.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_18_3.jpg)

> This figure shows the attack success rate of Label Attacks on the LLaVA-1.5 model. The x-axis represents the percentage of poisoned training samples, while the y-axis shows the attack success rate. Two Label Attack tasks are presented: Trump-to-Biden and EngineLight-to-FuelLight.  The results demonstrate a significant increase in attack success rate with a small percentage of poisoned samples (less than 1%). For both tasks,  the attack success rate exceeds 95% when the poison rate is approximately 1.4%. This highlights the efficiency and effectiveness of the Shadowcast attack against label attacks.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_23_1.jpg)

> This figure shows the results of a Label Attack and a Persuasion Attack on the LLaVA-1.5 vision-language model.  The top half demonstrates a Label Attack, where the model is tricked into misidentifying Donald Trump as Joe Biden, using a poisoned model trained with images subtly altered to resemble Biden. The bottom half displays a Persuasion Attack, in which the model is manipulated to describe junk food as healthy in its text generation, again using images subtly modified to suggest healthiness.  Poison samples were created using a different VLM (MiniGPT-v2) to highlight the transferability of the attack. This demonstrates Shadowcast's ability to manipulate VLM responses through visually indistinguishable poison samples.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_24_1.jpg)

> This figure shows the results of a data poisoning attack called Shadowcast on the LLaVA-1.5 vision-language model.  The top half demonstrates a Label Attack, where the model is tricked into misidentifying Donald Trump as Joe Biden.  The bottom half shows a Persuasion Attack, a novel attack type where the model is manipulated into generating a misleading narrative describing junk food as healthy.  Crucially, the poisoned samples used were created using a different model (MiniGPT-v2), showcasing the transferability and potential real-world impact of Shadowcast across different VLMs.


![](https://ai-paper-reviewer.com/JhqyeppMiD/figures_25_1.jpg)

> This figure shows the results of a data poisoning attack called Shadowcast on the LLaVA-1.5 vision-language model.  The top part demonstrates a Label Attack, where the model misidentifies Donald Trump as Joe Biden. The bottom part shows a Persuasion Attack, where the model generates a misleading description of junk food as healthy.  The poisoned samples were created using a different model, MiniGPT-v2, highlighting the transferability of the attack across different models.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_5_1.jpg)
> This table presents four different attack tasks used to evaluate the Shadowcast data poisoning method against Vision-Language Models (VLMs). Each task involves manipulating the VLM's response to an image representing an 'original concept' to elicit a response aligning with a different 'destination concept'.  The tasks are categorized into two types: Label Attacks, where the destination concept is a simple class label, and Persuasion Attacks, where the destination concept is a more elaborate and potentially misleading narrative.  The table specifies the original and destination concepts associated with each task.

![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_6_1.jpg)
> This table presents the performance comparison between clean and poisoned LLaVA-1.5 models on two benchmark datasets: VizWiz and GQA.  The performance is measured by the scores obtained on each benchmark.  Different columns represent different proportions (p) of poison samples used during the training process. Higher scores indicate better performance. The table demonstrates the impact of poison samples on the overall model performance and the trade-off between model utility and vulnerability to data poisoning attacks.

![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_13_1.jpg)
> This table presents four attack tasks used in the Shadowcast experiments. Each task involves manipulating a Vision-Language Model (VLM) to misinterpret images from an 'original concept' as if they belong to a different, 'destination concept'.  The 'original concept' and 'destination concept' pairs are shown for each task. For example, the 'Trump-to-Biden' task aims to make the VLM misidentify images of Donald Trump as Joe Biden. This table is crucial because it defines the specific scenarios used to evaluate the effectiveness and generalizability of the Shadowcast attack against VLMs.

![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_14_1.jpg)
> This table presents four different attack tasks used in the paper's experiments.  Each task involves manipulating a Vision-Language Model (VLM) to misinterpret images from an original concept as belonging to a different, target concept.  The original and target concepts are described for each task.  Two tasks focus on misidentification (Label Attack), while the other two involve creating misleading narratives (Persuasion Attack).  These tasks demonstrate different ways an attacker can manipulate VLMs.

![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_15_1.jpg)
> This table presents the instructions given to the GPT-3.5-turbo language model for evaluating the responses generated by the poisoned models in the Persuasion Attack tasks.  These instructions are designed to assess whether the model's responses accurately reflect the intended destination concept (e.g., healthy food, beneficial physical activity) without explicitly mentioning that concept in every response.  The instructions ensure a consistent and unbiased evaluation focusing on the alignment of the response with the target concept, rather than on the factual accuracy or completeness of the response.

![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_15_2.jpg)
> This table presents the performance comparison between clean and poisoned LLaVA-1.5 models on two benchmark datasets: VizWiz and GQA.  The performance is measured by a score (higher is better).  The table shows the performance of the model across different percentages (p) of poison samples injected into the training data, ranging from 0.28% to 5.71%. This allows for an analysis of how the model's performance changes as the proportion of poisoned data increases.

![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_19_1.jpg)
> This table compares different types of attacks (test-time and poisoning attacks) against three different types of models (image classifiers, LLMs, and VLMs) based on three criteria: pervasive impact (how broadly the attack affects users), stealthiness (whether the attack is detectable by humans), and the ability to generate misleading texts. The table shows that Shadowcast (the authors' attack) is unique in achieving a high level of pervasive impact, stealthiness, and the creation of subtly misleading texts, a combination not found in other attack methods.

![](https://ai-paper-reviewer.com/JhqyeppMiD/tables_20_1.jpg)
> This table presents the performance comparison between clean and poisoned LLaVA-1.5 models on two benchmark datasets: VizWiz and GQA.  The performance is measured by evaluating the models' accuracy across different poisoning rates (p), ranging from 0.28% to 5.71%.  The table allows readers to observe the impact of Shadowcast data poisoning attacks on the model's overall utility, while also assessing the performance trade-offs between poisoning rate and benchmark scores.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/JhqyeppMiD/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}