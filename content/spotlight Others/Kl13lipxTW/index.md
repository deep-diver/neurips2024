---
title: 'BackTime: Backdoor Attacks on Multivariate Time Series Forecasting'
summary: BACKTIME unveils effective backdoor attacks on multivariate time series forecasting,
  highlighting vulnerabilities and offering novel defense strategies.
categories: []
tags:
- AI Applications
- Security
- "\U0001F3E2 University of Illinois"
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Kl13lipxTW {{< /keyword >}}
{{< keyword icon="writer" >}} Xiao Lin et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Kl13lipxTW" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95645" target="_blank" >}}
↗ NeurIPS Proc.
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=Kl13lipxTW&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/Kl13lipxTW/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many real-world applications rely on the accuracy of multivariate time series (MTS) forecasting, using deep learning models for prediction. However, the robustness of these models to malicious attacks remains largely uninvestigated, posing significant risks in high-stakes scenarios. This paper addresses this gap by exploring the vulnerabilities of MTS forecasting models to backdoor attacks.

The paper proposes BACKTIME, a novel attack method that subtly injects triggers into the MTS data to manipulate model predictions. BACKTIME employs a bi-level optimization process with a GNN-based trigger generator to synthesize stealthy and effective triggers.  Extensive experiments demonstrate that BACKTIME is effective, versatile, and stealthy across multiple datasets and state-of-the-art models.  The research also contributes to enhancing the trustworthiness of MTS forecasting systems by suggesting strategies to mitigate the identified vulnerabilities.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Multivariate time series forecasting models are vulnerable to backdoor attacks. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} BACKTIME, a novel attack method, can subtly inject triggers to manipulate model predictions. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The study provides insights for developing more robust and trustworthy forecasting systems. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial because **it pioneers the study of backdoor attacks on multivariate time series forecasting**, a critical area previously unexplored.  Its findings highlight significant vulnerabilities in widely used forecasting models, **raising concerns about their reliability in high-stakes applications**. The proposed attack method, BACKTIME, and its associated defense strategies **open avenues for developing more robust and trustworthy forecasting systems** across diverse domains.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/Kl13lipxTW/figures_2_1.jpg)

> This figure illustrates how the BACKTIME attack works by injecting triggers and target patterns into the training data.  The top panel shows a clean prediction, while the bottom panel demonstrates how the model's predictions change after the malicious injection. The injected triggers subtly influence the model to follow the target pattern in its predictions, thereby showcasing the effectiveness of the backdoor attack.





![](https://ai-paper-reviewer.com/Kl13lipxTW/tables_2_1.jpg)

> This table compares different backdoor attack paradigms (Image/Text Classification, Univariate Time Series Classification, Multivariate Time Series Classification, and Multivariate Time Series Forecasting) across various task-wise and data-wise challenges.  Task-wise challenges include the type of target object, the ability to perform real-time attacks, constraints on the target object, and the need for soft identification. Data-wise challenges involve the impact of human unreadability and the dependence on inter-variable correlations.  The table highlights how the proposed Multivariate Time Series Forecasting attack differs from traditional attacks by utilizing sequence patterns as target objects and addressing the unique challenges of real-time attacks, constraints on target objects, and inter-variable dependence.





### In-depth insights


#### Backdoor Attacks on MTS
The concept of "Backdoor Attacks on MTS" (Multivariate Time Series) focuses on the vulnerabilities of MTS forecasting models to malicious attacks.  **The core issue is that subtly injecting triggers into the MTS data can manipulate model predictions to serve the attacker's goals.**  This poses a significant risk to the trustworthiness and reliability of MTS models used in critical real-world applications.  The research delves into creating effective and stealthy backdoor attacks, focusing on methods that are both difficult to detect and impactful in changing the prediction outcomes.  **A key aspect is the development of trigger generation techniques to ensure invisibility within the data**, minimizing the alteration necessary to achieve the desired effect.   Furthermore, it investigates the unique challenges and properties of this attack compared to traditional backdoor attacks, which typically focus on classification tasks rather than forecasting.  The work likely covers the vulnerabilities of different state-of-the-art MTS forecasting models, showing the broad applicability of this type of attack.  **Ultimately, this research highlights the crucial need for robust and secure MTS forecasting systems capable of resisting such malicious manipulation.**

#### BACKTIME Framework
The BACKTIME framework, designed for backdoor attacks on multivariate time series forecasting, presents a **novel generative approach**.  It cleverly injects **stealthy triggers** into the data, manipulating predictions according to the attacker's intent. The framework's sophistication lies in its **bi-level optimization process**, employing a **GNN-based trigger generator**. This allows for the adaptive synthesis of effective triggers, targeting vulnerable timestamps and sparsely affecting only a subset of variables.  The method is designed for **effectiveness**, **versatility**, and **stealthiness**, ensuring that the alterations in data remain largely imperceptible. A **key innovation** is the use of a non-linear scaling function and a shape-aware normalization loss, further enhancing the subtle nature of the attack.  Overall, BACKTIME represents a significant advance in understanding and mitigating the risk of backdoor attacks in the increasingly important field of time series forecasting.

#### Stealthy Trigger Design
Designing stealthy triggers is crucial for successful backdoor attacks.  A core challenge lies in **minimizing the impact on the normal forecasting process**, ensuring that the poisoned data closely resembles the original time series.  **Subtle modifications**, such as slight adjustments to the amplitude or a low injection rate, make detecting these alterations extremely difficult. A successful strategy involves **leveraging the inherent characteristics of the time series data** itself to integrate triggers.  This could involve alignment with existing patterns or the use of noise-like triggers that blend seamlessly into the natural variability of the data.  **Trigger generation methods** should carefully address both the temporal and inter-variable correlations in multivariate time series data to further enhance stealthiness.  Additionally, techniques such as **shape-aware normalization** can improve the success rate of attacks by mimicking the statistical characteristics of the normal data, making it nearly impossible to distinguish between clean and poisoned inputs.

#### Bi-level Optimization
The heading 'Bi-level Optimization' suggests a sophisticated approach to the problem of crafting effective and stealthy backdoor attacks.  This technique likely involves an inner optimization loop focused on generating optimal triggers, given a specific model and dataset, aiming for maximum impact while remaining undetected. The outer loop then optimizes the overall attack strategy, potentially adjusting parameters like trigger placement or injection rate to maximize the impact on the forecasting model's predictions.  **The bi-level structure is crucial because it allows for a complex interplay between trigger design and model behavior.**  The inner loop tailors the triggers to exploit the model's vulnerabilities, while the outer loop ensures the overall attack remains stealthy and effective, even amidst potential defenses.  This approach reflects a more advanced and adaptable attack method compared to simpler poisoning techniques, suggesting **a significant advance in the sophistication of backdoor attacks on time series forecasting**. The inherent difficulty of solving bi-level optimization problems implies a considerable computational cost, but the potential rewards in terms of more potent attacks likely outweigh the added complexity.

#### Future Research
The paper's 'Future Research' section could productively explore several avenues.  **Extending backdoor attacks to MTS imputation tasks** is crucial, as current methods rely on sequential data and struggle with missing values which are ubiquitous in real-world scenarios.  **Developing triggers robust to incomplete data** is vital for practical attacks.  **Designing defense mechanisms** represents a significant challenge; techniques could involve frequency analysis to detect unusual patterns, or clustering methods to identify anomalous trigger/target pattern combinations.  **Investigating the impact of various missing data mechanisms** on attack effectiveness would refine our understanding of attack resilience and inform the development of more effective defenses.  Finally, **a deeper exploration of the interplay between different types of attacks and the complexity of real-world MTS data** is necessary for crafting effective defenses against sophisticated, adaptive attacks.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/Kl13lipxTW/figures_5_1.jpg)

> This figure shows the results of an experiment designed to determine which timestamps are most vulnerable to backdoor attacks. The experiment involved training a clean forecasting model and measuring its mean absolute error (MAE) for each timestamp.  Timestamps with higher MAE values indicate poorer prediction performance and are considered more susceptible to attack. The experiment then implemented a simple backdoor attack on different groups of timestamps, sorted by their MAE percentile, and compared the MAE of the attacked model to that of the clean model. The y-axis represents the difference in MAE between the clean and attacked models for each group of timestamps. The results demonstrate that timestamps with higher MAE percentiles (i.e., timestamps where the clean model performs poorly) tend to have a smaller MAE difference after the attack, indicating greater vulnerability.


![](https://ai-paper-reviewer.com/Kl13lipxTW/figures_8_1.jpg)

> This figure shows the impact of the temporal injection rate (ατ) and spatial injection rate (αs) on the performance of the BACKTIME attack.  It presents four subplots: two illustrating the impact of ατ on MAE and RMSE for both clean and poisoned data, and two illustrating the impact of αs on MAE and RMSE for both clean and poisoned data. The x-axis of each subplot shows the injection rate (ατ or αs), while the y-axis represents the MAE or RMSE.  The shaded areas in the plots represent the standard deviation of the results.  The goal of BACKTIME is to minimize the attack metrics (MAEA and RMSEA) while keeping the clean metrics (MAEC and RMSEc) low, showing how the balance between attack effectiveness and stealthiness changes with varying injection rates.


![](https://ai-paper-reviewer.com/Kl13lipxTW/figures_17_1.jpg)

> This figure shows three different shapes of target patterns used in the BACKTIME experiments.  The first is a cone-shaped pattern, symmetrical around a peak value. The second shows an upward trend, steadily increasing from start to finish. The third pattern resembles an upward and downward trend that has a peak and then decreases.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/Kl13lipxTW/tables_6_1.jpg)
> This table presents the main results of the proposed BACKTIME backdoor attack on multivariate time series forecasting. It compares the performance of BACKTIME against several baseline methods (Clean, Random, Inverse, Manhattan) across different datasets (PEMS03, PEMS04, PEMS08, Weather, ETTm1) and forecasting models (TimesNet, FEDformer, Autoformer).  The metrics used are Mean Absolute Error for clean data (MAEC), Mean Absolute Error for attacked data (MAEA), Root Mean Squared Error for clean data (RMSEC), and Root Mean Squared Error for attacked data (RMSEA). Lower values indicate better performance. The table highlights BACKTIME's superior attack effectiveness while maintaining relatively good forecasting accuracy on clean data.

![](https://ai-paper-reviewer.com/Kl13lipxTW/tables_7_1.jpg)
> This table presents the main results of the BACKTIME backdoor attack on multivariate time series forecasting.  It compares the performance of the attack (measured by Mean Absolute Error on clean and attacked data, MAEC and MAEA) across five different datasets (PEMS03, PEMS04, PEMS08, Weather, ETTm1), three different forecasting models (TimesNet, FEDformer, Autoformer), and four different attack strategies (Clean, Random, Inverse, Manhattan, BACKTIME).  Lower values for MAEC and MAEA indicate better forecasting accuracy and attack effectiveness, respectively.  The results are averaged across the three forecasting models due to space constraints; full details are available in Appendix E.

![](https://ai-paper-reviewer.com/Kl13lipxTW/tables_7_2.jpg)
> This table presents the results of a backdoor attack experiment on the PEMS03 dataset using three different shapes for the target patterns: cone-shaped, upward trend, and up and down. The metrics used to evaluate the attack's performance are Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for both clean and attacked forecasting.  The table compares the performance of different attack methods (Random, Inverse, Manhattan, BACKTIME) with a clean model (no attack) across all three target pattern shapes.

![](https://ai-paper-reviewer.com/Kl13lipxTW/tables_8_1.jpg)
> This table presents the results of using two anomaly detection methods, GDN and USAD, to detect modified segments in poisoned datasets.  The results are shown in terms of F1-score and AUC for each method on five different datasets: PEMS03, PEMS04, PEMS08, Weather, and ETTm1.  Lower scores indicate better stealthiness of the backdoor attacks, showing that the modified segments were difficult to detect by these methods.

![](https://ai-paper-reviewer.com/Kl13lipxTW/tables_15_1.jpg)
> This table presents the main results of the BACKTIME backdoor attack on multivariate time series forecasting.  It compares the performance of BACKTIME against several baseline methods across five different datasets using three state-of-the-art forecasting models (TimesNet, FEDformer, and Autoformer). The metrics used for evaluation include Mean Absolute Error for clean data (MAEC), Mean Absolute Error for attacked data (MAEA), Root Mean Squared Error for clean data (RMSEC), and Root Mean Squared Error for attacked data (RMSEA). The lower the value, the better the performance.  Because of space constraints, the table shows only average results across the three models; Appendix E provides the complete results.

![](https://ai-paper-reviewer.com/Kl13lipxTW/tables_15_2.jpg)
> This table presents the main results of the proposed BACKTIME backdoor attack on multivariate time series forecasting across five datasets (PEMS03, PEMS04, PEMS08, Weather, ETTm1) and three state-of-the-art forecasting models (TimesNet, FEDformer, Autoformer).  The table compares the performance of the attack in terms of Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) when evaluating both clean and poisoned data.  The 'Clean' row represents the performance without attack. The other rows showcase the attack effectiveness using different trigger injection methods (Random, Inverse, Manhattan) and the proposed BACKTIME method. Lower values for MAE and RMSE indicate better forecasting performance (lower is better).  Bold values highlight BACKTIME's superior attack effectiveness.

![](https://ai-paper-reviewer.com/Kl13lipxTW/tables_18_1.jpg)
> This table presents the main results of the backdoor attack on multivariate time series forecasting using the proposed BACKTIME method.  It compares the performance of BACKTIME against other methods (Clean, Random, Inverse, Manhattan) across various datasets (PEMS03, PEMS04, PEMS08, Weather, ETTm1) and forecasting models (TimesNet, FEDformer, Autoformer).  The metrics used for evaluation are Mean Absolute Error (MAE) for clean predictions (MAEC) and attack predictions (MAEA).  Lower values indicate better performance. The table shows that BACKTIME generally achieves the best performance in terms of MAEA (attack effectiveness) while maintaining competitive performance in terms of MAEC (natural forecasting ability).  Appendix E contains more detailed results.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Kl13lipxTW/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}