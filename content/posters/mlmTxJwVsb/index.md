---
title: "DMNet: Self-comparison Driven Model for Subject-independent Seizure Detection"
summary: "DMNet: A novel self-comparison driven model significantly improves subject-independent seizure detection from intracranial EEG, outperforming existing methods."
categories: []
tags: ["AI Applications", "Healthcare", "🏢 Zhejiang University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} mlmTxJwVsb {{< /keyword >}}
{{< keyword icon="writer" >}} Shihao Tu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=mlmTxJwVsb" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93731" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=mlmTxJwVsb&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/mlmTxJwVsb/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Automated seizure detection (ASD) using intracranial EEG (iEEG) is crucial for effective epilepsy treatment, but subject variability in iEEG signals poses a major challenge. Existing methods often fail to generalize across subjects, hindering real-world applications. This is mainly because existing methods struggle to capture universal seizure patterns while mitigating the significant domain shift of iEEG signals across subjects and time intervals. 

DMNet, a novel subject-independent ASD model, is proposed to address these issues.  It leverages a self-comparison mechanism that effectively aligns iEEG signals by comparing target segments with contextual and channel-level references.  A simple yet effective difference matrix encodes universal seizure patterns, improving accuracy.  Extensive experiments show that DMNet significantly outperforms previous state-of-the-art methods on multiple datasets, demonstrating its robustness and effectiveness.  **The visualization results demonstrate that the generated difference matrix can effectively capture the seizure activity changes throughout the seizure evolution process.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} DMNet significantly outperforms state-of-the-art methods for subject-independent seizure detection using intracranial EEG. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} DMNet's self-comparison mechanism effectively mitigates domain shift and captures universal seizure patterns. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The generated difference matrix in DMNet effectively visualizes seizure activity changes throughout the seizure evolution process. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in epilepsy and machine learning.  It addresses the critical challenge of subject-independent seizure detection using intracranial EEG, a problem that hinders real-world clinical applications.  **DMNet's superior performance and efficiency** open exciting avenues for improved diagnosis and treatment of epilepsy, while its self-comparison mechanism offers a novel approach to domain adaptation in time-series data analysis, impacting related fields as well. 

------
#### Visual Insights



![](https://ai-paper-reviewer.com/mlmTxJwVsb/figures_1_1.jpg)

> This figure shows the locations of iEEG depth electrodes for two subjects (P2 and P4) and example iEEG recordings of seizures and normal activities for those subjects. It also shows how the proposed method, DMNet, is applied in a real-world clinical setting.





![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_6_1.jpg)

> This table compares the performance of various subject-independent seizure detection models on a clinical dataset and two public datasets (MAYO and FNUSA).  The performance metrics used are Precision, Recall, F1-score, and F2-score, with the F2-score being the primary metric.  The table shows that the proposed DMNet model significantly outperforms existing state-of-the-art methods across all metrics and datasets.





### In-depth insights


#### iEEG ASD Challenge
The core challenge in iEEG-based automated seizure detection (ASD) lies in the significant inter-subject variability of iEEG signals.  **Individual differences in brain anatomy, electrode placement, and recording conditions create substantial domain shifts**, making it difficult to build models that generalize well across different patients.  This is further complicated by **intra-subject variability**, where a patient's iEEG signals may change over time due to neurological factors or even just changes in their state of alertness. Existing ASD methods often struggle with subject-independent performance precisely because they fail to effectively address these domain shifts and variations in signal characteristics.  **The inconsistent seizure patterns across different channels further complicates the task**, as the same seizure event may manifest differently across various brain regions.  **Overcoming these challenges requires innovative approaches that can effectively align and normalize iEEG data across subjects and time points**, preserving the critical features indicative of seizures while mitigating confounding variations.

#### Self-Comparison
The concept of "Self-Comparison" in this context centers on a novel approach to automated seizure detection using intracranial EEG (iEEG).  The core idea is to **mitigate the significant variability and domain shifts inherent in iEEG data across subjects and time**. Traditional methods struggle due to these differences.  Instead of directly comparing iEEG segments to a fixed reference, the proposed self-comparison strategy leverages **comparisons between a target segment and its adjacent, temporally nearby normal segments**. This technique effectively creates a consistent, relative representation, minimizing the impact of individual differences.  The results demonstrate that self-comparison improves the reliability of the detection model by reducing dependence on subject-specific characteristics.  **The key is that the relative difference between normal and seizure activity remains more consistent than absolute measures.** This approach also enables more effective seizure pattern recognition. By enhancing the model's ability to generalize, this method offers a pathway toward more robust and accurate subject-independent seizure detection.

#### DMNet Model
The DMNet model, a novel subject-independent seizure detection model, tackles the challenge of significant domain shifts in intracranial electroencephalography (iEEG) signals.  **Its core innovation lies in a self-comparison mechanism** that effectively aligns iEEG signals across subjects and time intervals. By comparing target segments with carefully constructed contextual and channel-level references, DMNet mitigates distribution shifts and preserves universal seizure patterns.  **A fully differencing operation creates a difference matrix**, encoding these patterns for effective seizure detection.  The model's performance significantly surpasses previous state-of-the-art methods on both clinical and public datasets, demonstrating its robustness and generalizability.  **Visualizations highlight the difference matrix's ability to capture dynamic seizure activity changes**.  This innovative approach offers a promising solution for real-world clinical applications, particularly in online diagnosis systems.  However, future work should address the model's reliance on specific reference construction and explore more robust feature extraction techniques.

#### Clinical Deployment
Deployment of a seizure detection model in a clinical setting presents unique challenges and considerations.  **Real-world applicability** hinges on factors such as **efficiency**, **accuracy**, and **user-friendliness**.  The system needs to be fast enough for real-time analysis of incoming EEG data, while maintaining high accuracy to minimize false positives and negatives.  **Integration with existing hospital infrastructure** is crucial, and the system should provide a clear and intuitive interface for clinicians to use and interpret the results.  Further considerations include **data privacy and security**; robust mechanisms are necessary to ensure that patient data is handled according to regulations.  Finally, **validation of the model's clinical effectiveness** through rigorous testing in various clinical settings is necessary before widespread adoption.

#### Future Works
Future work could explore enhancing DMNet's generalizability by incorporating more sophisticated domain adaptation techniques, perhaps leveraging self-supervised learning or meta-learning approaches.  Investigating alternative encoding schemes for the difference matrix, beyond CNNs, could potentially improve performance.  **A crucial area is expanding the dataset to include a broader range of seizure types and patient demographics to further validate DMNet's robustness and clinical applicability.** The online deployment system could be enhanced with user-friendly interfaces and integrated diagnostic tools.  Finally, **rigorous clinical trials are essential to assess DMNet's effectiveness in a real-world setting, comparing it against existing ASD methods and evaluating its impact on patient care.** This requires collaboration with clinicians and obtaining appropriate regulatory approvals.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/mlmTxJwVsb/figures_3_1.jpg)

> This figure shows the distribution of raw and processed iEEG data.  Subfigure (a) shows the indistinguishable distributions of normal and seizure events before processing. Subfigures (b) demonstrates the significant domain shifts in both time and subjects.  Subfigures (c) and (d) illustrate the effectiveness of self-comparison in mitigating these domain shifts and improving separability of normal and seizure events.


![](https://ai-paper-reviewer.com/mlmTxJwVsb/figures_4_1.jpg)

> This figure illustrates the architecture of the DMNet model, which is proposed for subject-independent seizure detection. The model uses a self-comparison mechanism, comparing the target segment with contextual and channel-level references. It employs a fully differencing operation to generate a difference matrix, which is then encoded by a CNN-based encoder to learn latent representations for classification.


![](https://ai-paper-reviewer.com/mlmTxJwVsb/figures_7_1.jpg)

> To provide a more intuitive demonstration of DMNet, we present the visualization results of difference matrix throughout the seizure process. The upper figure shows the raw brain signal containing a full seizure process, with the gray wave representing the normal signal and the purple wave representing the seizure. The green masked blocks indicate the segments for detection. Notably, there are clear distinctions between seizure and normal difference matrices during the seizure evolution. Segments being closer to seizure events show rougher difference matrices, while those further away appear smoother. This case clearly illustrates how the difference matrix captures seizure activity changes and demonstrates the effectiveness of DMNet.


![](https://ai-paper-reviewer.com/mlmTxJwVsb/figures_8_1.jpg)

> The figure illustrates the architecture of the proposed DMNet model for subject-independent seizure detection.  It shows the process of taking an intracranial EEG recording, generating contextual and channel-level references for self-comparison, creating a difference matrix, and then using a convolutional neural network (CNN) and a classifier to produce a seizure detection result.  The diagram highlights the key components of the model, including the fully differencing operation and the difference matrix encoder, which are central to the model's ability to mitigate the effects of domain shifts across subjects and time intervals.


![](https://ai-paper-reviewer.com/mlmTxJwVsb/figures_8_2.jpg)

> This figure shows the online auxiliary diagnosis system built based on DMNet. The top panel displays a 12-hour patient file, with each square representing a 1-minute iEEG signal segment color-coded to indicate various states (no epileptic waves, correct/wrong/missing predictions).  Clicking a square shows the detailed view in the bottom panel, including a data operation panel and seizure events.


![](https://ai-paper-reviewer.com/mlmTxJwVsb/figures_14_1.jpg)

> The figure visualizes the difference matrix (DM) used in the DMNet model for seizure detection. It shows a comparison of the DM for normal and seizure events. The DM is constructed by comparing a target segment with contextual and channel-level references using a fully differencing operation. The visualization highlights the different patterns in the DM for normal and seizure events. The channel-level reference shows differences between the target segment and various channel patterns. The contextual reference shows smooth differences for normal events and rougher differences for seizure events. This visualization demonstrates how the DM effectively captures seizure activity changes.


![](https://ai-paper-reviewer.com/mlmTxJwVsb/figures_17_1.jpg)

> This figure visualizes how the difference matrices generated by DMNet change throughout a seizure event.  The top panel shows the raw iEEG data with a purple line indicating seizure activity and a gray line representing normal brain activity.  The green vertical bars highlight the segments used for analysis. The main part of the figure displays a series of difference matrices, one for each frequency band (0-4 Hz, 20-24 Hz, etc.), across different time points (a-g) within a single seizure event. The color intensity of the difference matrices reflects the magnitude of the differences between the target segment and its references.  The visualization demonstrates that the difference matrices become progressively more irregular and higher magnitude as the seizure progresses, highlighting the capability of DMNet to capture dynamic seizure activity changes in various frequency bands.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_15_1.jpg)
> This table presents the average performance of various subject-independent seizure detection models on clinical and public datasets.  Metrics include precision, recall, F1-score, and F2-score. The best-performing model in each metric for each dataset is indicated.  A more detailed breakdown of the performance, including standard deviation, is available in Appendix G.

![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_15_2.jpg)
> This table presents the average performance of various subject-independent seizure detection models on clinical and public datasets.  The performance metrics include precision, recall, F1-score, and F2-score. The table highlights the proposed DMNet model's performance in comparison to other state-of-the-art models, with the best and second-best performances marked.

![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_16_1.jpg)
> This table presents the average performance (precision, recall, F1-score, and F2-score) of various subject-independent seizure detection models on clinical and public datasets (MAYO and FNUSA).  The best-performing model in each metric for each dataset is indicated. Standard deviations are available in Appendix G. The table compares DMNet against several state-of-the-art methods, including both iEEG-based and EEG-based models, as well as several domain generalization (DG) algorithms.

![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_16_2.jpg)
> This table compares the performance of different subject-independent seizure detection models on clinical and public datasets.  The models are evaluated using precision, recall, F1-score, and F2-score.  The table highlights the superior performance of the proposed DMNet model compared to various baselines, including other state-of-the-art methods and domain generalization techniques.

![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_18_1.jpg)
> This table presents the average performance (precision, recall, F1-score, and F2-score) of various subject-independent seizure detection models on clinical and public datasets (MAYO and FNUSA).  The models include several state-of-the-art (SOTA) methods, as well as domain generalization (DG) baselines. The table highlights the superior performance of DMNet compared to other approaches, emphasizing its effectiveness in subject-independent seizure detection.

![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_18_2.jpg)
> This table presents the average performance (precision, recall, F1-score, and F2-score) of various subject-independent seizure detection models on a clinical dataset and two public datasets (MAYO and FNUSA).  The models include several state-of-the-art (SOTA) methods and domain generalization (DG) algorithms.  The 'v' symbol indicates the top two performing models in each metric across all datasets, highlighting DMNet's superior performance.  More detailed results with standard deviations are provided in Appendix G. 

![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_19_1.jpg)
> This table presents the average performance of various subject-independent seizure detection models across clinical and public datasets (MAYO and FNUSA).  The metrics used for evaluation are precision, recall, F1-score, and F2-score.  The table highlights the superior performance of the proposed DMNet model compared to existing state-of-the-art (SOTA) methods and domain generalization (DG) baselines.  The 'v' and 'v' symbols indicate the top and second-ranked performances in each column, respectively. More detailed results including standard deviations are provided in Appendix G.

![](https://ai-paper-reviewer.com/mlmTxJwVsb/tables_19_2.jpg)
> This table presents the average performance of various subject-independent seizure detection models on clinical and public datasets.  The metrics used are Precision, Recall, F1-score, and F2-score. The 'v' symbol indicates the top two performing models for each metric.  A more detailed breakdown of the performance, including standard deviations, is available in Appendix G of the paper. The table compares the proposed DMNet model against several state-of-the-art baselines, both iEEG and EEG-based methods as well as domain generalization approaches.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/mlmTxJwVsb/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}