---
title: "CE-NAS: An End-to-End Carbon-Efficient Neural Architecture Search Framework"
summary: "CE-NAS: A novel framework minimizes the carbon footprint of Neural Architecture Search by dynamically allocating GPU resources based on predicted carbon intensity, achieving state-of-the-art results w..."
categories: []
tags: ["Machine Learning", "Reinforcement Learning", "🏢 Worcester Polytechnic Institute",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} v6W55lCkhN {{< /keyword >}}
{{< keyword icon="writer" >}} Yiyang Zhao et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=v6W55lCkhN" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93227" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=v6W55lCkhN&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/v6W55lCkhN/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Neural Architecture Search (NAS) is crucial for designing efficient deep learning models but is computationally expensive and environmentally unfriendly. Existing NAS methods don't consider the varying carbon intensity across different geographical locations and time periods. This results in high energy and carbon consumption, hindering sustainable AI development.  



The proposed CE-NAS framework tackles this issue by dynamically allocating GPU resources based on predicted carbon emissions. **It leverages reinforcement learning to optimize the balance between energy-efficient sampling and energy-intensive evaluation of model architectures**. CE-NAS integrates a time-series transformer to predict carbon intensity and a multi-objective optimizer to reduce the search space. The results demonstrate that CE-NAS significantly reduces carbon emissions while achieving state-of-the-art accuracy on various image classification benchmarks.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} CE-NAS dynamically adjusts GPU resources during NAS based on predicted carbon intensity. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} CE-NAS achieves state-of-the-art results on various benchmarks with significantly lower carbon emissions compared to traditional NAS methods. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} CE-NAS introduces a novel framework integrating reinforcement learning and carbon footprint prediction, opening avenues for sustainable AI practices. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers because **it directly addresses the significant environmental impact of Neural Architecture Search (NAS)**, a rapidly growing field. By presenting a novel framework, CE-NAS, that significantly reduces carbon emissions during NAS, this research provides **a practical solution for sustainable AI development**. Its findings can inform the design of future NAS algorithms and evaluation strategies, leading to both improved efficiency and reduced environmental footprint. The study opens up a new avenue of research by demonstrating the integration of reinforcement learning and carbon footprint prediction. This encourages the exploration of more sustainable AI practices.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_1_1.jpg)

> This figure presents a high-level overview of the CE-NAS framework.  It illustrates how the framework dynamically allocates GPU resources between energy-efficient sampling and energy-intensive evaluation of neural architectures.  The allocation is based on predicted carbon intensity, aiming to reduce the carbon footprint of the NAS process.  A reinforcement learning agent learns the optimal GPU allocation policy.





![](https://ai-paper-reviewer.com/v6W55lCkhN/tables_6_1.jpg)

> This table compares the Mean Absolute Percentage Error (MAPE) of the proposed time-series transformer model against three other baseline methods (STCF [11], DACF [56], and CC [55]) for forecasting carbon intensity.  The comparison is done for one-day, two-day, and three-day ahead forecasts across three different geographical regions (CISO, DE, and PJM). Lower MAPE values indicate better forecasting accuracy.





### In-depth insights


#### Carbon-Aware NAS
A 'Carbon-Aware NAS' approach is crucial for responsible AI development.  It directly addresses the **high carbon footprint** associated with traditional Neural Architecture Search (NAS) methods, which often involve training massive numbers of models. By incorporating carbon emission awareness, this approach could significantly reduce the environmental impact of NAS, making it more sustainable and ethically sound.  **Dynamically adjusting GPU resource allocation** based on real-time carbon intensity is a key innovation.  This allows for shifting resources towards energy-efficient sampling during high-carbon periods and energy-intensive evaluation during low-carbon periods, achieving a balance between search efficiency and sustainability.  **Integrating multi-objective optimization** allows the search to consider not just accuracy, but also other metrics like latency and parameter count. This results in models which are optimized for both performance and energy consumption, further enhancing the carbon efficiency of the entire process.  Incorporating carbon forecasting models helps in proactively managing resource allocation, enabling a truly 'carbon-aware' system.

#### RL-Based Allocation
Reinforcement Learning (RL)-based resource allocation in the context of Neural Architecture Search (NAS) presents a compelling approach to optimize the energy efficiency of the search process.  **The core idea is to dynamically allocate GPU resources** between energy-efficient sampling methods and energy-intensive evaluation tasks, adapting to real-time variations in carbon intensity.  This dynamic allocation, learned by an RL agent, avoids the limitations of static strategies that are unable to effectively balance search efficiency and energy cost.  **The RL agent learns a policy to allocate GPUs based on several factors** such as the remaining carbon budget, the number of already searched architectures and their performance, and the predicted future carbon intensity. This results in a more efficient model design process in terms of both energy and carbon consumption.  However, the effectiveness of this approach relies heavily on the accuracy of carbon intensity prediction and the design of the RL agent, which requires careful tuning and consideration of various factors.  **The key challenge lies in balancing the exploration-exploitation trade-off** during RL training.  A purely exploration-based strategy could lead to excessive energy consumption, while a purely exploitation-based approach could result in suboptimal architectural designs.  Therefore, designing an effective RL agent requires careful consideration of the reward function, action space, and state representation to guide the agent towards carbon-efficient and high-performing architectures.

#### Multi-objective Search
In tackling complex neural architecture search (NAS) problems, a **multi-objective search** strategy offers a powerful approach.  Instead of focusing solely on accuracy, this method considers multiple, often competing, objectives such as accuracy, efficiency (latency, parameter count), and energy consumption. This holistic perspective is crucial for designing practical neural networks that are not only performant but also resource-friendly and environmentally conscious.  The adoption of multi-objective optimizers, such as LaMOO, allows for the exploration of the Pareto frontier, identifying a set of optimal architectures representing the best trade-offs between objectives. This approach avoids the limitations of single-objective optimization, which might prioritize one aspect at the cost of others.  Further, a **multi-objective search** facilitates exploration of a wider range of architectural designs, potentially leading to the discovery of innovative network structures that were previously overlooked by single-objective methods.

#### Benchmark Results
A dedicated 'Benchmark Results' section would ideally present a thorough comparison of the proposed carbon-efficient neural architecture search (NAS) framework, CE-NAS, against existing state-of-the-art (SOTA) methods.  This would involve showcasing CE-NAS's performance on established NAS benchmarks such as NAS-Bench-101 and NAS-Bench-301. Key metrics to report include **top-1 accuracy**, **parameter count**, **latency**, and crucially, **carbon emissions (CO2)**.  A strong analysis should highlight not only the improved accuracy and efficiency of CE-NAS but also its significant reduction in carbon footprint compared to baselines, possibly quantified as an X-fold reduction.  Furthermore, the results should be presented clearly, perhaps using tables and graphs to facilitate easy understanding and comparison.  Ideally, a discussion on the statistical significance of observed performance differences, such as p-values, should be included.  Finally, open-domain tasks on datasets like CIFAR-10 and ImageNet are crucial to show real-world applicability and demonstrate CE-NAS's ability to produce competitive models with a considerably smaller environmental impact.

#### Future of CE-NAS
The future of CE-NAS hinges on **addressing its current limitations**. While CE-NAS effectively integrates carbon awareness into neural architecture search, scalability remains a challenge.  Future work could explore **distributed training strategies** to handle larger search spaces and datasets, reducing overall computation time and carbon footprint.  Furthermore, **improving the accuracy** of the carbon intensity prediction model is crucial.  More sophisticated models, potentially incorporating real-time grid data and weather patterns, could enhance predictive capabilities.  Research into **more efficient NAS algorithms** would also enhance CE-NAS performance.  Investigating novel multi-objective optimization techniques and exploring the use of proxy-based methods might significantly reduce computation. Finally, expanding the applicability of CE-NAS beyond image classification to other domains such as natural language processing and time series forecasting would broaden its impact. This would necessitate developing task-specific adaptation strategies, potentially involving transfer learning or meta-learning techniques.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_5_1.jpg)

> The figure shows the actual and predicted carbon intensity for a specific region (US-CAL-CISO) over a period of 80 hours in 2021. The carbon predictor, described in section 3.5.2, is evaluated here and shows reasonably good performance.


![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_6_1.jpg)

> This figure shows a comparison of the search progress over time for different neural architecture search (NAS) methods on two benchmark datasets: HW-NAS-Bench and NasBench301. The y-axis represents the log hypervolume difference, a metric that measures the quality of the search results, and the x-axis represents time in hours. The figure demonstrates that CE-NAS achieves a good balance between carbon efficiency and search efficiency, outperforming other methods in terms of hypervolume while keeping carbon emissions low.


![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_7_1.jpg)

> This figure shows the search progress over time for two different benchmarks, HW-NAS-Bench and NasBench301.  It compares CE-NAS to other NAS algorithms, illustrating how CE-NAS maintains comparable search efficiency while significantly reducing carbon emissions. The plots show the hypervolume (a measure of search quality) over time, with CE-NAS achieving near SOTA results with lower CO2 output.  This demonstrates the carbon-efficiency of the CE-NAS approach.


![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_8_1.jpg)

> The figure shows box plots visualizing the hypervolume achieved by different NAS methods (CE-NAS, Vanilla LaMOO, One-shot LaMOO, Heuristic) under various carbon emission constraints (25000g, 50000g, 75000g, 100000g).  The results are based on the NasBench301 benchmark and a carbon trace depicted in Figure 2. Each method was run five times to obtain the data for these plots. The hypervolume is a metric that reflects the quality of the obtained Pareto frontier, indicating how well a method explores the multi-objective search space. Lower carbon constraints restrict the total resources available for the search, thus impacting the final achieved hypervolume. The box plot provides a visual representation of the distribution of results for each method and carbon constraint showing the median, quartiles, and potential outliers.


![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_19_1.jpg)

> The figure shows a comparison of actual and predicted carbon intensity over time. The actual carbon intensity data is from US-CAL-CISO in 2021. The predicted carbon intensity is generated by a carbon predictor (detailed in section 3.5.2 of the paper). This figure helps to visualize the accuracy of the carbon intensity prediction model used in the CE-NAS framework.


![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_23_1.jpg)

> This figure compares the architecture qualities (hypervolume, accuracy, and inference energy) between architectures selected by LaMOO from a specific region of the search space and architectures randomly sampled from the entire search space using HW-NASBench.  The results demonstrate that LaMOO effectively focuses the search on a subset of architectures with better properties.


![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_23_2.jpg)

> This figure illustrates the CE-NAS framework.  It shows how GPU resources are dynamically allocated between energy-efficient sampling and energy-intensive evaluation tasks of neural architectures, based on predicted carbon intensity. When carbon intensity is low, more resources are allocated to evaluation; when high, more to efficient sampling. A reinforcement learning agent learns this allocation policy.


![](https://ai-paper-reviewer.com/v6W55lCkhN/figures_24_1.jpg)

> This figure illustrates the CE-NAS framework, highlighting its dynamic GPU resource allocation based on carbon emission intensity.  When carbon intensity is low, more resources are allocated to the energy-intensive evaluation of architectures. Conversely, during high carbon intensity periods, more resources are focused on energy-efficient sampling of architectures. A reinforcement learning agent learns this resource allocation strategy.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/v6W55lCkhN/tables_7_1.jpg)
> This table presents the results of the CIFAR-10 image classification task using the NasNet search space, focusing on two optimization objectives: minimizing the number of parameters and maximizing accuracy. It compares CE-NAS against various state-of-the-art (SOTA) NAS methods categorized as vanilla and one-shot methods.  The results are presented in terms of test error, the number of parameters (in millions), the search and training costs (in GPU hours), the CO2 emissions (in lbs), and the type of NAS method used. CE-Net-P1 and CE-Net-P2 represent the results obtained by CE-NAS.

![](https://ai-paper-reviewer.com/v6W55lCkhN/tables_8_1.jpg)
> This table presents the results of the ImageNet experiments, comparing the performance of CE-NAS against other state-of-the-art NAS algorithms.  The key metrics shown are top-1 error rate, TensorRT latency (using FP16 on an NVIDIA V100 GPU), the total search and training cost in GPU hours, the CO2 emissions in pounds, and the NAS method used (vanilla, one-shot, or hybrid). The table highlights that CE-NAS achieves state-of-the-art performance with comparable or lower carbon emissions.

![](https://ai-paper-reviewer.com/v6W55lCkhN/tables_17_1.jpg)
> This table compares different energy-efficient neural architecture search (NAS) evaluation methods. It assesses their evaluation cost, initialization cost, accuracy, and whether they require extra data for training.  The methods include zero-shot proxy, one-shot, predictor, low-fidelity, and gradient proxy approaches.

![](https://ai-paper-reviewer.com/v6W55lCkhN/tables_24_1.jpg)
> This table compares the performance of different NAS methods (CE-NAS, Vanilla LaMOO, One-shot LaMOO, and Heuristic) in terms of hypervolume under various carbon emission constraints (5000g, 10000g, 30000g, 50000g).  It shows the average hypervolume achieved by each method under each constraint, comparing predictions made using a carbon intensity predictor with actual carbon traces from the dataset.  The goal is to evaluate the search efficiency of each NAS method under different carbon budgets.

![](https://ai-paper-reviewer.com/v6W55lCkhN/tables_24_2.jpg)
> This table compares the performance of the CE-NAS framework using discrete and continuous action spaces under different carbon emission constraints.  The hypervolume, a metric reflecting the quality of the search results, is reported for each condition.  The results show a comparison of search efficiency when using different action spaces.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/v6W55lCkhN/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}