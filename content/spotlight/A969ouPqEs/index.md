---
title: "DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"
summary: "DiffLight: a novel conditional diffusion model for traffic signal control effectively addresses data-missing scenarios by unifying traffic data imputation and decision-making, demonstrating superior p..."
categories: []
tags: ["AI Applications", "Smart Cities", "🏢 Beijing Jiaotong University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} A969ouPqEs {{< /keyword >}}
{{< keyword icon="writer" >}} Hanyang Chen et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=A969ouPqEs" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/96278" target="_blank" >}}
↗ NeurIPS Proc.
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=A969ouPqEs&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/A969ouPqEs/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Traffic signal control (TSC) using reinforcement learning often assumes complete and continuous traffic data, which is unrealistic in practice due to sensor failures or data loss.  Existing methods struggle with TSC under data-missing scenarios, hindering real-world applicability.  The challenge lies in effectively integrating traffic data imputation and decision-making, especially when dealing with partially available rewards from sensors. 

DiffLight addresses this by using a novel Partial Rewards Conditioned Diffusion (PRCD) model. This innovative approach prevents incomplete reward information from interfering with the learning process.  Furthermore, it leverages a Spatial-Temporal Transformer (STFormer) architecture to capture spatial-temporal dependencies and includes a Diffusion Communication Mechanism (DCM) to improve information sharing and control performance.  Extensive experiments showcase that DiffLight significantly improves TSC performance in various data-missing situations compared to existing methods. **The approach is particularly effective in handling data missing patterns and achieves better performance with varying missing rates.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} DiffLight effectively handles missing traffic data in offline settings, outperforming existing methods. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The Partial Rewards Conditioned Diffusion (PRCD) model prevents missing rewards from hindering learning. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The Spatial-Temporal Transformer (STFormer) architecture and Diffusion Communication Mechanism (DCM) improve communication and control performance, considering spatial-temporal dependencies. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it tackles a critical challenge in real-world traffic signal control: handling missing sensor data.  It proposes a novel, effective solution using diffusion models, a powerful and increasingly popular technique in machine learning. This opens up new avenues for research in robust and reliable AI-based traffic management systems. The code's public release further enhances its impact, accelerating progress in this important area.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/A969ouPqEs/figures_2_1.jpg)

> This figure shows a four-way intersection with 12 traffic movements and 4 traffic signal phases. Each lane has an entrance and an exit. There are 4 phases that control the traffic flow, which are labeled A, B, C, and D. Each phase controls a specific set of movements that do not conflict with each other. The red and green circles represent the vehicles at the different phases.  The figure helps to visualize the traffic flow and how the signal phases coordinate the movement of traffic.





![](https://ai-paper-reviewer.com/A969ouPqEs/tables_7_1.jpg)

> This table compares the Average Travel Time (ATT) of DiffLight against several baseline methods for traffic signal control under different random data-missing rates (10%, 30%, and 50%).  The ATT metric reflects the average time vehicles spend traveling through the network. Lower ATT indicates better performance. The table shows the mean and standard error calculated across three trials for each condition, providing a measure of the variability in the results.





### In-depth insights


#### TSC's Data Issue
The core issue in Traffic Signal Control (TSC) revolves around data reliability and availability.  **Real-world TSC scenarios rarely offer the complete and continuous data streams** assumed by many existing reinforcement learning (RL) models. Sensor malfunctions, data loss, and budget constraints frequently result in incomplete or missing data, rendering the training and deployment of robust TSC systems challenging. This data sparsity significantly impacts the performance of RL algorithms, particularly affecting reward estimation and the accurate representation of the traffic dynamics.  **Addressing this data scarcity requires sophisticated solutions,** such as employing data imputation techniques and developing robust RL models capable of handling partial or uncertain information. **The challenges extend beyond mere data completion**, necessitating a deeper consideration of spatio-temporal dependencies within the traffic network and developing strategies for efficient communication between interconnected traffic signals in the face of missing data.  **Future research should focus on creating more robust RL frameworks** specifically tailored to the complexities of real-world traffic data challenges, considering both the data uncertainty and the complex interdependencies within the traffic network itself.

#### PRCD Model Deep Dive
A hypothetical 'PRCD Model Deep Dive' section would analyze the Partial Rewards Conditioned Diffusion model in detail.  It would likely begin by explaining the core functionality of the PRCD model within the context of traffic signal control, particularly emphasizing how it addresses the challenge of **partial rewards** stemming from missing sensor data.  The explanation would delve into the model's architecture, describing its specific components and how they interact. This would include a discussion of the **diffusion process** itself, likely highlighting the use of a forward and reverse process to generate data, the conditional aspects enabling learning with incomplete data, and any unique modifications tailored for this application.  Crucially, the section would also explore the model's ability to unify traffic data imputation and decision-making into a single framework, examining the potential benefits and drawbacks of this approach. Finally, a detailed analysis of the model's parameters, hyperparameters, and training methodology would be crucial to a thorough understanding, perhaps including comparative analyses against other potential strategies.  **Theoretical guarantees** or limitations inherent in the PRCD model and its suitability for the problem domain would also be important considerations.

#### STFormer's Role
The core role of STFormer within the DiffLight model is to effectively capture the intricate spatial-temporal dependencies inherent in traffic flow data.  Unlike simpler noise models, **STFormer leverages the Transformer architecture**, specifically designed to handle long-range dependencies and parallelization. This allows DiffLight to not only consider the immediate state of a single intersection but also to incorporate information from neighboring intersections and past time steps, resulting in **more informed predictions and control decisions**.  The spatial aspect is crucial for understanding congestion propagation across multiple intersections while the temporal aspect is crucial for the modelling of traffic dynamics. This approach, in the context of missing data, is particularly important because the missing values are likely to be temporally and spatially correlated, so the model's predictive capacity relies greatly on the ability to exploit correlations between observed and unobserved data. This is achieved through carefully designed modules within STFormer, which enable effective communication and information sharing for better performance in the context of partial data.

#### DiffLight's Limits
DiffLight, while showing promise in handling missing data in traffic signal control, has inherent limitations.  **Its reliance on the Partial Rewards Conditioned Diffusion (PRCD) model, while innovative, might struggle with extremely high rates of missing data.** The effectiveness of PRCD hinges on the availability of *some* partial rewards; completely missing reward information would severely hamper performance.  **The Spatial-Temporal Transformer (STFormer) architecture, while effectively capturing spatial-temporal dependencies, assumes a certain level of connectivity within the traffic network.**  Sparsely connected intersections or those with limited sensor data could hinder the model's ability to learn effective control strategies.  **Further limitations stem from the assumption of independent missing data**; complex, correlated missing patterns may not be adequately addressed by DiffLight's current mechanisms. Finally, the generalizability of DiffLight's performance across diverse real-world scenarios requires further investigation, including those with unusual traffic patterns or unique environmental conditions.

#### Future of DiffLight
The future of DiffLight hinges on addressing its current limitations and exploring new avenues for improvement.  **Extending DiffLight to handle more complex missing data patterns** beyond random and kriging is crucial for real-world applicability.  This might involve incorporating more sophisticated imputation techniques or developing robust mechanisms that learn to effectively adapt to various missing data scenarios.  **Improving the scalability of DiffLight to larger traffic networks** is another key area for future work. The current model's performance might degrade with the increase in the number of intersections. Solutions could include distributed architectures or hierarchical approaches.  **Research into incorporating longer-term planning and prediction** within the Diffusion Communication Mechanism (DCM) would enhance DiffLight's ability to proactively manage traffic flow and anticipate potential congestion. **Addressing the computational cost** through efficiency improvements in the STFormer and PRCD models would also be beneficial for wider adoption. Finally, exploring the integration of DiffLight with other intelligent transportation systems (ITS) components to create a more holistic traffic management solution represents a promising future direction.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/A969ouPqEs/figures_4_1.jpg)

> This figure illustrates the architecture of DiffLight, a novel conditional diffusion model for traffic signal control with missing data.  It shows how the model integrates various components to address the challenges of missing data and spatial-temporal dependencies in traffic networks. The process begins with sensors collecting traffic data (rewards and observations). Missing data is masked, and the observed data is fed into the Partial Rewards Conditioned Diffusion (PRCD) model with a Spatial-Temporal Transformer (STFormer) architecture. The STFormer captures spatial-temporal dependencies using Communication Cross-Attention, Spatial Self-Attention, and Temporal Self-Attention modules.  A Diffusion Communication Mechanism (DCM) propagates generated observations among intersections to improve performance in the presence of missing data. Finally, an inverse dynamics model generates actions based on the processed observations to control the traffic signals.


![](https://ai-paper-reviewer.com/A969ouPqEs/figures_8_1.jpg)

> This figure shows a schematic overview of the DiffLight model for traffic signal control with missing data.  It illustrates the data flow, highlighting the key components: Partial Rewards Conditioned Diffusion (PRCD) with a Spatial-Temporal Transformer (STFormer), Diffusion Communication Mechanism (DCM), and inverse dynamics model.  The diagram visually represents how the model handles missing data, integrates spatial-temporal information, and generates control actions for traffic signals.


![](https://ai-paper-reviewer.com/A969ouPqEs/figures_14_1.jpg)

> This figure illustrates the two types of missing data patterns used in the experiments: random missing and kriging missing. In random missing (a), data from any intersection may be missing at random.  In kriging missing (b), the absence of data is spatially correlated; if data from one intersection is missing, it's more likely that data from its neighbors will also be missing. The yellow rectangles represent intersections, and the gray circles represent the traffic flow within those intersections. The striped and hatched rectangles show the masked traffic data representing the missing values in the random missing and kriging missing scenarios, respectively.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/A969ouPqEs/tables_7_2.jpg)
> This table presents a comparison of the Average Travel Time (ATT) achieved by DiffLight and several baseline methods across different datasets and missing data rates. The experiments simulate random missing patterns in the data.  For each dataset and missing rate (10%, 30%, 50%), the table shows the mean and standard deviation of ATT across three independent trials. This allows for assessing the relative performance of DiffLight against other approaches under conditions of incomplete traffic data.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_8_1.jpg)
> This table presents the results of an ablation study conducted on two datasets, Hangzhou₁ and Jinan₁, to evaluate the effectiveness of different components within the DiffLight model.  It compares the performance of three model variants: 1) U-Net (using a U-Net architecture as the noise model with zero-padded missing rewards); 2) STFormer (using the Spatial-Temporal Transformer architecture with zero-padded missing rewards); and 3) STFormer+PRCD (DiffLight, employing the STFormer architecture and handling partial rewards). The comparison is made under two data-missing patterns (random missing and kriging missing) at different missing rates.  The results show the impact of each component on the model's performance in handling missing data in traffic signal control.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_14_1.jpg)
> This table presents the average travel time (ATT) achieved by three different methods (AttendLight, Efficient-CoLight, and Advanced-CoLight) on five real-world traffic flow datasets.  These ATT values represent the converged performance of each method after training, and the data from these trained models were used to create the offline datasets used in the main experiments of the paper. The datasets are from different cities and have varying network sizes and traffic characteristics.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_16_1.jpg)
> This table presents a comparison of the Average Travel Time (ATT) for five different traffic signal control methods under various levels of random data missingness.  The methods compared are Behavior Cloning (BC), Conservative Q-Learning (CQL), TD3+BC, Decision Transformer (DT), Diffuser, Decision Diffuser (DD), and the proposed DiffLight method.  The table shows ATT values and their standard errors (across three trials) for each method under 10%, 30%, and 50% rates of random data missingness.  The datasets used are Hangzhou1 (D<sup>HZ</sup><sub>1</sub>), Hangzhou2 (D<sup>HZ</sup><sub>2</sub>), and Jinan1, Jinan2, and Jinan3 (D<sup>JN</sup><sub>1</sub>, D<sup>JN</sup><sub>2</sub>, D<sup>JN</sup><sub>3</sub>).

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_16_2.jpg)
> This table presents the Average Travel Time (ATT) results for different methods under two scenarios related to missing intersection data:  one where neighboring intersections are available and one where they are not.  It shows how the absence of neighboring data affects the performance of various traffic signal control algorithms, especially highlighting the impact on DiffLight.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_16_3.jpg)
> This table shows the average travel time (ATT) results for different datasets (Hangzhou1, Hangzhou2, Jinan1, Jinan2, Jinan3) under different random missing rates (70% and 90%).  It demonstrates the performance of DiffLight as the missing rate increases, showing a decline in performance at higher missing rates (90%).

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_17_1.jpg)
> This table demonstrates the scalability of the DiffLight model by evaluating its performance on a larger dataset (New York, with 48 intersections) under varying rates of random missing data (10%, 30%, 50%).  It compares DiffLight's Average Travel Time (ATT) against several baseline methods (BC, CQL, TD3+BC, DT, Diffuser, DD) to highlight its ability to handle more complex traffic scenarios and maintain relatively good performance even with substantial data loss.  The lower ATT values indicate better performance.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_17_2.jpg)
> This table presents the average travel time (ATT) results for the DiffLight model and several baseline methods on the New York dataset, which includes 48 intersections.  The experiment is performed under different random missing rates (10%, 20%, 30%, and 50%).  It shows DiffLight's scalability and ability to deal with complex traffic scenarios in a larger-scale traffic network. The performance of most baselines drops rapidly with increased missing rate, whereas DiffLight maintains relatively stable performance.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_17_3.jpg)
> This table presents the ablation study on the effectiveness of the Diffusion Communication Mechanism (DCM).  It compares the performance of DiffLight with and without DCM under various missing data patterns (random and kriging) and missing rates. The results demonstrate that DCM enhances performance, particularly in kriging missing scenarios, indicating its benefit in facilitating communication and improving control performance under data scarcity.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_17_4.jpg)
> This table presents the results of an ablation study comparing two variants of the DiffLight model: one with inverse dynamics and one without.  The study evaluates the Average Travel Time (ATT) performance of both models on two datasets (Hangzhou and Jinan) under different data missing scenarios (random missing and kriging missing).  The results illustrate the impact of incorporating an inverse dynamics model within the DiffLight architecture on the overall performance.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_18_1.jpg)
> This table presents the average travel time (ATT) results for the DiffLight model under different sampling steps (100, 50, 20, and 10 steps).  It shows the ATT for four different datasets (Hangzhou1 with random and kriging missing, Jinan1 with random and kriging missing). The results illustrate the model's performance stability across varying sampling step counts.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_18_2.jpg)
> This table presents the inference time cost of the DiffLight model under different sampling steps (100, 50, 20, and 10 steps).  The results are shown for different datasets (DHZ1, DHZ2, DIN1, DIN2) and missing data patterns (random missing with 50% missing rate and kriging missing with 25% missing rate).  The table allows for an assessment of the trade-off between inference speed and model performance.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_18_3.jpg)
> This table presents a comparison of the Average Travel Time (ATT) achieved by three different methods: CQL, CQL (model-based), and DiffLight, under various data-missing scenarios (10%, 30%, and 50% missing rates).  CQL represents the standard CQL approach, while CQL (model-based) is a modified version adapted for offline settings, using a store-and-forward method for data imputation. DiffLight is the authors' proposed method, which directly handles missing data using a conditional diffusion model. The results illustrate DiffLight's superior performance across all missing data rates compared to both CQL variants, highlighting its effectiveness in handling missing data during traffic signal control.

![](https://ai-paper-reviewer.com/A969ouPqEs/tables_19_1.jpg)
> This table compares the average travel time (ATT) results for the CQL algorithm (with a model-based approach for handling missing data), and the DiffLight model, under different kriging missing data rates. It highlights DiffLight's superior performance compared to CQL (model-based) across various missing rates.  The improved performance of DiffLight is attributed to its ability to directly make decisions using partial rewards, avoiding error accumulation from separate imputation and decision-making steps as seen in the CQL model-based approach.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/A969ouPqEs/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/A969ouPqEs/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}