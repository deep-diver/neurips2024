---
title: "DiffuPac: Contextual Mimicry in Adversarial Packets Generation via Diffusion Model"
summary: "DiffuPac generates realistic adversarial network packets evading NIDS detection without requiring specific NIDS knowledge, outperforming existing methods."
categories: []
tags: ["AI Applications", "Security", "🏢 Nagaoka University of Technology",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} KYHVBsEHuC {{< /keyword >}}
{{< keyword icon="writer" >}} Abdullah Bin Jasni et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=KYHVBsEHuC" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95657" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=KYHVBsEHuC&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/KYHVBsEHuC/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current methods for generating adversarial network packets to evade Network Intrusion Detection Systems (NIDS) often rely on unrealistic assumptions about attacker knowledge. This limits their practical value, as attackers typically have limited knowledge about a target system's specifics. The lack of realistic adversarial models makes it difficult to effectively assess and enhance the robustness of NIDS.

This paper introduces DiffuPac, a novel adversarial packet generation model. DiffuPac cleverly combines a pre-trained BERT model with a diffusion model to generate adversarial packets. This approach allows the system to understand the context of network traffic, creating more realistic and effective adversarial packets that evade detection. **DiffuPac significantly outperforms existing methods** across various attack types and multiple classifiers, demonstrating its effectiveness in a real-world setting. The classifier-free nature of this method also makes it particularly relevant for testing the robustness of NIDS.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} DiffuPac generates adversarial packets that evade sophisticated NIDS without relying on specific NIDS component knowledge. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} DiffuPac outperforms existing methods by an average of 6.69 percentage points in terms of evasion effectiveness while preserving the functionality of the adversarial packets. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} DiffuPac uses a classifier-free approach, making it more practical and relevant to real-world attack scenarios. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for cybersecurity researchers as it introduces a novel approach to adversarial packet generation that directly challenges existing limitations. **DiffuPac**'s classifier-free method using BERT and diffusion models is a significant advance, **offering a realistic and effective way to test the robustness of NIDS**. It also **opens new avenues for research into more practical and sophisticated adversarial attacks and defenses.**

------
#### Visual Insights



![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_3_1.jpg)

> The figure illustrates the overall framework of the proposed DiffuPac model. It is divided into three main phases: 1.pcap pre-processing: This stage involves splitting pcap files into sessions (bidirectional flows), further splitting sessions into unidirectional flows (src-to-dst & dst-to-src), converting each byte to its corresponding hex number and tokenizing using WordPiece. 2.Pre-training: This phase uses BERT to generate embeddings for the tokens, utilizing two pre-training tasks: Masked Unidirectional Flow Model (predicting masked tokens) and Same Sequence-origin Prediction (predicting transmission order). 3.Fine-tuning with Diffusion Model: This phase fine-tunes the diffusion model using a forward process with packet sequence concatenation strategy and targeted noising, followed by a reverse process with normal packet guidance.





![](https://ai-paper-reviewer.com/KYHVBsEHuC/tables_1_1.jpg)

> This table summarizes existing works on adversarial packet generation, highlighting the techniques, datasets, and algorithms used. It also points out limitations of these methods, like relying on unrealistic assumptions about the attacker's knowledge of the NIDS components.





### In-depth insights


#### DiffuPac: Overview
DiffuPac, presented as a novel approach to adversarial packet generation, leverages the power of **contextual mimicry**.  It cleverly integrates a pre-trained BERT model with a diffusion model, enabling the generation of adversarial packets that seamlessly blend into legitimate network traffic. Unlike traditional methods that rely on specific NIDS knowledge, DiffuPac's strength lies in its **classifier-free guidance**, making it highly effective against sophisticated intrusion detection systems even with limited attacker knowledge of the target system's inner workings. The core innovation involves a unique concatenation strategy and targeted noising that effectively generates highly realistic adversarial packets capable of evading detection.  This is achieved by carefully pairing malicious packets with contextually relevant normal packets, introducing noise only to the malicious components. The result is a system that **prioritizes practicality and functionality**, producing stealthy attacks that maintain malicious intent without sacrificing usability.

#### BERT-Diffusion Fusion
A BERT-Diffusion fusion model for adversarial packet generation represents a novel approach to crafting evasive network traffic.  **Leveraging BERT's contextual understanding of network protocols and traffic patterns**, the model can generate realistic-looking packets that effectively blend malicious behavior into benign network activity.  The diffusion model component allows for generating diverse, realistic variations of malicious packets. **By integrating both models, the approach tackles the limitations of traditional methods**, which often make unrealistic assumptions about attacker knowledge or rely on specific NIDS vulnerabilities. This fusion offers a potent combination, potentially achieving higher evasion rates than methods that rely on either generative or discriminative models alone, while also generating more realistic adversarial examples. The efficacy is critically dependent on the quality and representativeness of the BERT model's pre-training data and the appropriateness of the diffusion model's parameterization for adversarial packet generation.  **Further research is warranted to assess its robustness against various NIDS architectures and to investigate potential ethical implications.**

#### Adversarial Packet Gen
Adversarial packet generation is a critical area in cybersecurity research focusing on creating malicious network packets designed to evade detection by intrusion detection systems (IDS).  **The core challenge lies in generating packets that appear legitimate while carrying out malicious actions.**  This requires a deep understanding of IDS mechanisms and network protocols.  Current approaches vary widely in their sophistication and rely on factors like access to IDS models or unrealistic assumptions about attacker knowledge.  **The development of advanced machine learning (ML) and deep learning (DL) models has enabled the generation of more sophisticated adversarial packets, highlighting the ongoing arms race between attackers and defenders.**  The success of such techniques is measured by evasion rates: the percentage of malicious packets that successfully bypass the IDS.  **However, a crucial factor often overlooked is maintaining the practicality and functionality of generated packets.**   An effective adversarial packet should not only evade detection but also accomplish its intended malicious task.  Future research needs to focus on developing more realistic models, and evaluating the effectiveness of adversarial packets in real-world settings, not just against simplified IDS setups.

#### Evasion Evaluation
A robust evasion evaluation is critical for assessing the effectiveness of adversarial packet generation methods.  It should involve a multifaceted approach, going beyond simple detection rates.  **Real-world datasets** are essential to ensure the evaluation's practical relevance. The evaluation should consider a range of sophisticated NIDS, incorporating diverse classifiers and feature extractors to better represent the complexity of real-world detection systems.  **Classifier-free approaches** should be evaluated to address limited attacker knowledge. Metrics beyond simple evasion rates, such as the analysis of packet-level modifications and the preservation of the functionality of the adversarial packets, are crucial.  **Statistical significance tests** should be used to establish the reliability of the results, and detailed analysis should be provided to highlight the mechanisms through which the adversarial packets evade detection.  Comparing the performance against existing techniques is also necessary to properly contextualize the findings.  **Transparency and reproducibility** are paramount. All datasets, models, parameters and evaluation methods should be thoroughly documented, promoting repeatability and validation of results by the research community.

#### Future Work
Future research directions stemming from this work could explore **expanding the types of attacks** considered, particularly those with complex traffic patterns like DDoS.  A deeper investigation into **the robustness of DiffuPac** against diverse NIDS configurations and feature extractors is also warranted.  **Adversarial defense mechanisms** specifically designed to counteract DiffuPac's evasion techniques should be developed and analyzed.  Furthermore, exploring **alternative architectures** and generative models for adversarial packet creation would push the boundaries of this research.  Finally, a critical area of future work focuses on addressing the **ethical implications** of this technology and developing robust safeguards against potential misuse for malicious purposes. This includes establishing clear guidelines and best practices for the responsible development and deployment of such sophisticated adversarial technologies.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_16_1.jpg)

> The figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases: 1.pcap pre-processing: This phase involves splitting pcap files into sessions and unidirectional flows, converting bytes to hex numbers and tokenizing them for BERT model processing. 2. pre-training: The pre-trained BERT model is used with two pre-training tasks: Masked Unidirectional Flow Model and Same Sequence-origin Prediction. 3. fine-tuning with diffusion model: This phase uses a diffusion model to generate adversarial packets. It involves embedding both normal and malicious packet sequences, contextually relevant pairing, concatenation, and targeted noising to the malicious packet sequences. The diffusion model uses a reverse process with normal packet guidance and classifier-free guidance.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_17_1.jpg)

> The figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases: pcap pre-processing, pre-training, and fine-tuning with diffusion models.  The pcap pre-processing phase involves preparing the data by splitting pcap files into sessions and unidirectional flows, tokenizing bytes into hex numbers, and incorporating special tokens. The pre-training phase uses a pre-trained BERT model and two pre-training tasks (Masked Unidirectional Flow Model and Same Sequence-origin Prediction) to learn the contextual relationships between traffic bytes. The fine-tuning phase uses a diffusion model to generate adversarial packets, which blends adversarial packets into genuine network traffic by using a concatenation strategy and targeted noising.  The model employs a reverse process with normal packet guidance, utilizing the pre-trained BERT model for denoising and maintaining packet integrity.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_17_2.jpg)

> This figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases: pcap pre-processing, pre-training, and fine-tuning with diffusion models.  The pcap pre-processing phase involves preparing the data by splitting pcap files into sessions and unidirectional flows and tokenizing the packet data.  The pre-training phase uses a pre-trained BERT model to learn contextual representations of network traffic by performing masked unidirectional flow prediction and same sequence-origin prediction tasks.  The fine-tuning phase uses a diffusion model to generate adversarial packets that mimic normal packets, which involves a forward process with packet sequence concatenation and targeted noising, and a reverse process with normal packet guidance.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_18_1.jpg)

> The figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases: 1. pcap pre-processing: This phase involves preparing the raw pcap files (packet captures) for training by splitting them into sessions and unidirectional flows, converting bytes to hex numbers, tokenizing them, and adding special tokens. 2. pre-training: This phase pre-trains a BERT model using two tasks: Masked Unidirectional Flow Model (predicting masked tokens in the unidirectional flows) and Same Sequence-origin Prediction (predicting the origin of a sequence of packets). 3. fine-tuning with diffusion models: This phase uses the pre-trained BERT model with diffusion model to generate adversarial packets. Malicious packets are contextually paired with relevant normal packets, noise is added only to the malicious packets, and the model is trained to seamlessly blend the adversarial packets into the benign network traffic.  The model uses both forward and reverse diffusion processes with classifier-free guidance. 


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_18_2.jpg)

> The figure illustrates the overall architecture of DiffuPac, which is divided into three main phases:  1. **Pcap Preprocessing:** This stage involves cleaning and preparing the raw pcap files, such as splitting them into sessions, unidirectional flows, tokenization, and embedding. 2. **Pre-training:** This phase uses a pre-trained BERT model to learn contextual relationships in network traffic by employing Masked Unidirectional Flow Model and Same Sequence-origin Prediction. 3. **Fine-tuning with Diffusion Model:** This stage fine-tunes the diffusion model by concatenating malicious and normal packet sequences and applying targeted noising, aiming to generate realistic adversarial packets that can bypass NIDS.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_19_1.jpg)

> This figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases:  pcap pre-processing, pre-training, and fine-tuning with a diffusion model. The pcap pre-processing phase involves preparing the data by splitting pcap files into sessions, then into unidirectional flows.  Pre-training uses a BERT model to learn contextual relationships in network traffic. The fine-tuning phase integrates the BERT model with the diffusion model for adversarial packet generation.  The figure shows the flow of data through each phase, highlighting key components and processes involved in generating adversarial packets.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_22_1.jpg)

> The figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases:  1. **pcap pre-processing**: This stage involves processing raw pcap files to extract relevant packet sequences, splitting sessions into unidirectional flows, tokenizing each byte into its corresponding hex number, and incorporating special tokens for training tasks. 2. **Pre-training**:  A pre-trained Bidirectional Encoder Representations from Transformers (BERT) model is used. The pre-training process uses two tasks: Masked Unidirectional Flow Model and Same Sequence-origin Prediction. The Masked Unidirectional Flow Model involves masking tokens and predicting the original tokens, while the Same Sequence-origin Prediction aims to determine whether packets belong to source-to-destination or destination-to-source sequences. 3. **Fine-tuning with Diffusion Models**: The pre-trained BERT model is integrated with a diffusion model. This phase involves concatenating malicious packets with contextually relevant normal packets and applying targeted noising only to malicious packets. The targeted noising uses a classifier-free guidance approach to address the real-world constraint of limited attacker knowledge. Finally, the diffusion model undergoes fine-tuning to refine the generation of adversarial packets that evade detection while maintaining practicality.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_22_2.jpg)

> The figure illustrates the overall framework of the DiffuPac model, which consists of three main phases: pcap pre-processing, pre-training, and fine-tuning with diffusion models.  The pcap pre-processing phase involves splitting pcap files into sessions and unidirectional flows, tokenizing each byte to its hex number representation, and incorporating special tokens for training tasks.  The pre-training phase uses a pre-trained BERT model to learn contextual relationships between traffic bytes, involving masked unidirectional flow prediction and same sequence-origin prediction tasks.  The fine-tuning phase integrates the pre-trained BERT model with a diffusion model to generate adversarial packets that evade detection by NIDS. The process combines contextual understanding of network behaviors from the pre-trained BERT model with the generative capabilities of diffusion models to seamlessly blend adversarial packets into genuine network traffic.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_22_3.jpg)

> This figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases: 1) pcap pre-processing, where raw pcap files are processed to extract unidirectional flows; 2) pre-training, where a pre-trained BERT model is used to learn contextual relationships between traffic bytes, using masked unidirectional flow and same sequence origin prediction; and 3) fine-tuning with diffusion model, where the model is fine-tuned to generate adversarial packets that can mimic normal packets and bypass NIDS. The figure also shows the data flow between different components of the model and the different embedding techniques used during the pre-training and fine-tuning phases. 


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_22_4.jpg)

> This figure illustrates the overall framework of the proposed DiffuPac model.  It's divided into three main phases:  1.  **Pcap Preprocessing:** This initial step involves preparing the raw network traffic data. The pcap files are processed and categorized to facilitate subsequent training and analysis. The data is then tokenized and converted into a format suitable for the BERT model.  2.  **Pre-training:**  A pre-trained Bidirectional Encoder Representations from Transformers (BERT) model is used. This phase focuses on training the BERT model to understand contextual relationships within network flows. Two main pre-training tasks are used: Masked Unidirectional Flow Model (predicting masked tokens to learn underlying patterns) and Same Sequence-origin Prediction (predicting transmission order to understand directionality).  3.  **Fine-tuning with Diffusion Model:** This phase involves integrating the pre-trained BERT model with a diffusion model to generate adversarial packets.  The diffusion model is trained using a concatenation strategy combining malicious and normal packet sequences.  Targeted noising is applied only to the malicious packets to seamlessly blend the adversarial packets into the genuine network traffic.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_22_5.jpg)

> This figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases: pcap pre-processing, pre-training, and fine-tuning with diffusion models.  The pre-processing phase involves preparing the network traffic data by splitting it into sessions and unidirectional flows, converting bytes into hex numbers, and then tokenizing them. The pre-training phase uses a pre-trained BERT model to learn contextual representations from the network traffic data, focusing on masked token prediction and same-sequence-origin prediction tasks. Finally, the fine-tuning phase integrates the pre-trained BERT model with a diffusion model for adversarial packet generation. This phase involves contextually relevant pairing of normal and malicious packets and targeted noising to the malicious packets before applying classifier-free guidance.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_23_1.jpg)

> The figure illustrates the overall framework of the DiffuPac model, which consists of three main phases: pcap pre-processing, pre-training, and fine-tuning with diffusion models.  The pcap pre-processing phase involves the preparation of the network traffic data, including the splitting of the pcap files into sessions and unidirectional flows and the tokenization of the packet sequences. The pre-training phase uses a pre-trained BERT model for the task of masked unidirectional flow modeling and same sequence-origin prediction. The fine-tuning phase employs a diffusion model to generate adversarial packets by concatenating malicious packet sequences with contextually relevant normal packet sequences and applying targeted noising to the malicious packets. The figure shows the flow of data and the interactions between different components of the model.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_24_1.jpg)

> The figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases:  1. **pcap pre-processing**: This involves splitting pcap files into sessions (bidirectional flows), further splitting them into unidirectional flows (source-to-destination and destination-to-source), and converting each byte to its corresponding hex number and tokenization. 2. **Pre-training**: This phase utilizes a pre-trained BERT model (Bidirectional Encoder Representations from Transformers) with two pre-training tasks: Masked Unidirectional Flow Model and Same Sequence-origin Prediction to learn the contextual relationships between traffic bytes. 3. **Fine-tuning with Diffusion Model**: This phase uses a diffusion model to generate adversarial packets, which involves a forward process (with packet sequence concatenation strategy and targeted noising) and a reverse process (with normal packet guidance) to seamlessly blend adversarial packets into genuine network traffic.


![](https://ai-paper-reviewer.com/KYHVBsEHuC/figures_24_2.jpg)

> The figure illustrates the overall framework of the proposed DiffuPac model, which consists of three main phases: pcap pre-processing, pre-training, and fine-tuning with diffusion models.  The pre-processing phase involves preparing the raw packet capture (pcap) files for training by splitting them into sessions, unidirectional flows, and tokenizing the byte sequences. The pre-training phase focuses on training a Bidirectional Encoder Representations from Transformers (BERT) model to learn contextual relationships within network traffic using masked language modeling and next sentence prediction tasks. The fine-tuning phase integrates the pre-trained BERT with a diffusion model for adversarial packet generation.  The process involves contextually relevant pairing of malicious and normal packet sequences, targeted noising of malicious packets, and classifier-free guidance via the diffusion model to generate adversarial packets that blend seamlessly with genuine network traffic.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/KYHVBsEHuC/tables_7_1.jpg)
> This table presents a comparative analysis of attack detection and evasion rates for six different types of attacks (Botnet, MITM, Brute Force, DDoS, Port Scan, and Infiltration) against six different classifiers (KitNET, DT, IF, MLP, SVM, LR) using three different methods: GAN & PSO, LSTM, and the proposed DiffuPac method.  The table shows the precision (P), recall (R), F1-score, and evasion rate (MER) for each attack-classifier combination and method.  The evasion rate (MER) indicates the percentage of malicious packets that were not detected by the classifier, providing an assessment of each method's effectiveness in evading detection.

![](https://ai-paper-reviewer.com/KYHVBsEHuC/tables_8_1.jpg)
> This table presents a comparative analysis of attack detection and evasion rates for six types of attacks (Botnet, MITM, Brute Force, DDoS, Port Scan, and Infiltration) against six different classifiers (KitNET, DT, FlowMeter IF, MLP, SVM, and LR) using three feature extractors (CICFlowMeter, AfterImage, and both).  The evasion rate (MER) is shown for each combination of attack, classifier, and feature extractor, comparing the performance of DiffuPac against two baseline models (Traffic Manipulator and TANTRA).  The table allows for a comprehensive comparison of DiffuPac's performance against existing evasion methods.

![](https://ai-paper-reviewer.com/KYHVBsEHuC/tables_13_1.jpg)
> This table lists the hyperparameters used for pre-training the BERT model in the DiffuPac framework.  It shows the values chosen for key parameters such as embedding size, feedforward size, hidden size, activation function, number of attention heads, number of transformer layers, maximum sequence length, dropout rate, batch size, total training steps, and learning rate.  These parameters significantly influence the model's performance and its ability to learn effective representations of network traffic data.

![](https://ai-paper-reviewer.com/KYHVBsEHuC/tables_14_1.jpg)
> This table compares four different BERT models (Tiny, Small, Medium, and Large) based on their specifications, training time, and performance metrics (MUM Loss, SSP Loss, MUM Accuracy, and SSP Accuracy).  The table helps to justify the choice of the Medium BERT model for DiffuPac due to its balance between accuracy and computational cost.

![](https://ai-paper-reviewer.com/KYHVBsEHuC/tables_15_1.jpg)
> This table presents a comparative analysis of attack detection and evasion rates for six different types of attacks (Botnet, MITM, etc.) against six classifiers (KitNET, DT, etc.). For each attack and classifier combination, the precision (P), recall (R), F1-score, and evasion rate (MER) are presented for three different adversarial packet generation methods: GAN & PSO, LSTM, and the proposed method (Ours).  The evasion rate shows the percentage of malicious packets that successfully evade detection.

![](https://ai-paper-reviewer.com/KYHVBsEHuC/tables_20_1.jpg)
> This table presents a comparative analysis of attack detection and evasion rates for six different types of attacks (Botnet, MITM, etc.) using three different models (GAN & PSO, LSTM, and Ours).  The results are shown for six different classifiers (KitNET, DT, etc.) and two feature extractors (CICFlowMeter and AfterImage).  The table shows the precision (P), recall (R), F1-score, and evasion rate (MER) for each model, classifier, and feature extractor combination.

![](https://ai-paper-reviewer.com/KYHVBsEHuC/tables_25_1.jpg)
> This table presents a comparative analysis of attack detection and evasion rates for six different classifiers (KitNET, DT, IF, MLP, SVM, LR) and two feature extractors (CICFlowMeter, AfterImage) across two attack types (Botnet, MITM). For each combination of classifier, feature extractor, and attack type, the table shows the precision (P), recall (R), F1-score, and evasion rate (MER) for both the proposed DiffuPac method and two baseline methods (GAN & PSO and LSTM). The evasion rate (MER) represents the percentage of malicious packets that evaded detection by the classifier.  The table allows for a direct comparison of the effectiveness of DiffuPac against existing methods in evading detection.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/KYHVBsEHuC/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}