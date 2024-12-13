---
title: "CausalStock: Deep End-to-end Causal Discovery for News-driven Multi-stock Movement Prediction"
summary: "CausalStock: A novel framework for accurate news-driven multi-stock movement prediction, using lag-dependent causal discovery and LLMs for enhanced noise reduction and explainability."
categories: []
tags: ["AI Applications", "Finance", "🏢 Renmin University of China",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 5BXXoJh0Vr {{< /keyword >}}
{{< keyword icon="writer" >}} Shuqi Li et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=5BXXoJh0Vr" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/96607" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=5BXXoJh0Vr&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/5BXXoJh0Vr/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Predicting stock price movements based on news is challenging due to noisy data and complex relationships between stocks. Existing methods often rely on correlations, which fail to capture the directionality of impact between stocks. This paper introduces CausalStock, a novel framework that addresses these issues by discovering temporal causal relationships between stocks, improving the accuracy of predictions. 

CausalStock uses a lag-dependent temporal causal discovery mechanism and a denoised news encoder based on LLMs to extract useful information from news data.  A functional causal model then combines the causal relationships and market information to predict stock movements. Experiments demonstrate that CausalStock outperforms state-of-the-art baselines on multiple real-world datasets, showing improved accuracy and explainability.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} CausalStock improves the accuracy of news-driven multi-stock movement prediction. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The framework uses a novel lag-dependent temporal causal discovery mechanism. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} LLMs are leveraged to improve noise reduction and the explainability of predictions in the financial market context.   {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it tackles the challenging problem of **news-driven multi-stock movement prediction**, which has significant implications for **financial markets and algorithmic trading**. By introducing a novel framework that leverages causal relationships and large language models (LLMs), it offers a more accurate and explainable approach than existing methods.  Furthermore, it **opens new avenues for research** in causal discovery for time-series data and the integration of LLMs in financial applications. The findings could **benefit both academics and industry professionals** involved in financial modeling and investment strategies.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/5BXXoJh0Vr/figures_3_1.jpg)

> This figure illustrates the CausalStock model's two main components: causal discovery and prediction.  The causal discovery process (dashed lines) uses variational inference to estimate the posterior distribution of the temporal causal graph G, represented by G1 to GL. This graph represents the causal relationships between the past L time lags of stock market information X<T. The prediction process (solid lines) uses a functional causal model (FCM) with parameters θ to predict the future stock movements yT based on X<T and the learned causal graph G.





![](https://ai-paper-reviewer.com/5BXXoJh0Vr/tables_7_1.jpg)

> This table presents the main results of the CausalStock model and several baseline models on six different datasets for two tasks: news-driven multi-stock movement prediction and multi-stock movement prediction.  The performance is measured using Accuracy (ACC) and Matthews Correlation Coefficient (MCC).  Standard deviations are provided to reflect the variability of the results.





### In-depth insights


#### Causal Stock Model
A Causal Stock Model is a novel approach to stock prediction that leverages causal inference and deep learning.  Instead of simply correlating stock prices with news data, it aims to **discover the causal relationships** between them.  This is crucial because correlations do not imply causation; a model focusing solely on correlations might fail to capture the true direction of influence, leading to inaccurate predictions. The model likely employs a causal discovery algorithm to learn a causal graph representing the relationships between various stocks and relevant news articles. By incorporating this causal structure, the model is expected to **improve prediction accuracy** and **offer greater explainability**.  It also tackles the noise inherent in news data, likely through a sophisticated text-processing module, which might involve denoising techniques or advanced natural language processing. The end result is a more robust and interpretable model that goes beyond simple correlations to understand the underlying causal mechanisms driving stock market movements.

#### News Encoder Design
A robust news encoder design is crucial for effective news-driven stock prediction.  The ideal design should effectively handle the **noisy and unstructured nature of news data**, extracting only relevant information pertinent to stock market movements.  **Leveraging pre-trained Language Models (LLMs)** is a promising avenue, allowing the model to capture nuanced language and contextual understanding that surpasses traditional methods.  However, simply using LLMs may not suffice.  **A denoising mechanism** is necessary to filter out irrelevant information, perhaps using techniques like attention mechanisms or advanced filtering methods. The encoder must also consider the **temporal aspect of news**, incorporating information about when the news was released and how it relates to past and future events.  Finally, effective feature extraction is key. The model should transform the encoded news into features easily integrated with stock price data for prediction.  The features might include sentiment, topic, impact score, and other relevant quantitative indicators. A well-designed news encoder forms the backbone of accurate and interpretable news-driven stock movement prediction models.

#### Temporal Causal Graph
A temporal causal graph is a powerful tool for representing dynamic causal relationships.  It extends the concept of a standard causal graph by explicitly incorporating the temporal dimension, acknowledging that cause-and-effect relationships unfold over time.  **Each node in the graph represents a variable at a specific time point**, and **directed edges indicate the causal influence from one variable's state at one time to another variable's state at a later time**. This allows for the modeling of **lag-dependent causal effects**, where the impact of a cause is not immediate but delayed. By considering the time-lagged dependencies, temporal causal graphs are particularly suitable for analyzing time-series data, such as financial markets, where the temporal context is crucial in understanding the intricate interplay of factors influencing outcomes. The ability to capture these temporal dynamics makes temporal causal graphs **valuable for prediction tasks**, and the explicit representation of causal relations allows for **improved explainability and interpretability of predictions**.

#### Ablation Study Results
An ablation study systematically removes components of a model to assess their individual contributions.  In the context of a stock prediction model, an ablation study might involve removing features like news data, specific types of news encoders (e.g., comparing LLM-based vs. traditional methods), or components of the causal discovery mechanism.  **Results would show the impact of each removed component on the model's overall performance (e.g., accuracy, MCC).** A significant drop in performance after removing a specific component highlights its importance for the model's success, thus providing valuable insights into feature importance and model architecture design.  For instance, a drastic accuracy decrease after removing LLM-based news encoding suggests that **LLMs are crucial for extracting effective information from noisy news data.** Conversely, a minimal change in performance may indicate that the specific component is less critical, potentially simplifying the model architecture without sacrificing performance.  **Such analyses are vital for optimizing the model's efficiency, robustness, and interpretability.**

#### Future Research
Future research directions stemming from the CausalStock paper could involve several key areas. **Extending the model to handle high-frequency trading data** would be valuable, as the current model primarily focuses on daily data.  This would require adapting the causal discovery mechanisms to capture the much faster dynamics present in high-frequency data.  Additionally, exploring the use of more advanced **LLMs for news encoding** could improve the accuracy and robustness of the model.  Investigating the impact of different LLM architectures and training procedures on the final predictions would be crucial. Furthermore, **developing techniques to handle missing data** and noisy news sources more effectively is needed.  This would enhance the model's practical applicability and resilience to real-world data challenges.  Finally, a significant area of future research would be to **investigate the model's performance across different market regimes and global markets.**  The current study's datasets primarily focus on specific time periods and geographic regions. A more thorough and comprehensive investigation across diverse market conditions would strengthen the generalizability of the findings.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/5BXXoJh0Vr/figures_4_1.jpg)

> This figure illustrates the architecture of the CausalStock model, which consists of three main components: Market Information Encoder (MIE), Lag-dependent Temporal Causal Discovery (Lag-dependent TCD), and Functional Causal Model (FCM). The MIE encodes news and price features, the Lag-dependent TCD discovers temporal causal relationships between stocks using a variational inference approach, and the FCM generates predictions based on these discovered relations and market information.  The figure shows an example using data from four days (07/02-07/05) for three stocks (AAPL, GOOG, META). Arrows indicate the flow of information, and the causal relationships are represented by dashed lines.


![](https://ai-paper-reviewer.com/5BXXoJh0Vr/figures_9_1.jpg)

> This figure shows the overall architecture of the CausalStock model.  It details the three main components: the Market Information Encoder (MIE), which processes both news and price data; the Lag-dependent Temporal Causal Discovery (Lag-dependent TCD) module, which identifies causal relationships between stocks; and the Functional Causal Model (FCM), which uses this information to predict future stock movements. The figure uses a simplified example with three stocks (Apple, Google, Meta) and five days of data to illustrate the flow of information through the model.  The arrows depict the flow of data through the different components, showing how news data and price information are integrated to make predictions.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/5BXXoJh0Vr/tables_8_1.jpg)
> This table presents the main results of the CausalStock model and several baseline models on six different datasets for two tasks: news-driven multi-stock movement prediction and multi-stock movement prediction (without news).  The performance is evaluated using Accuracy (ACC) and Matthews Correlation Coefficient (MCC).  The standard deviation is reported to show the model's robustness, reflecting the variation in the results across multiple runs. The table highlights CausalStock's superior performance compared to other methods.

![](https://ai-paper-reviewer.com/5BXXoJh0Vr/tables_9_1.jpg)
> This table presents the main results of the CausalStock model and several baseline models on six different stock market datasets.  It compares the performance of these models on two tasks: news-driven multi-stock movement prediction and multi-stock movement prediction (without news). The performance is measured using Accuracy (ACC) and Matthews Correlation Coefficient (MCC).  Standard deviations reflect the variability of results across multiple runs.

![](https://ai-paper-reviewer.com/5BXXoJh0Vr/tables_14_1.jpg)
> This table presents the main results of the CausalStock model and several baseline models on six different datasets for two different tasks: news-driven multi-stock movement prediction and multi-stock movement prediction.  The performance is measured using Accuracy (ACC) and Matthews Correlation Coefficient (MCC). Standard deviations are reported to show the variability of the results.  The datasets represent stock markets from the US, China, Japan, and the UK.

![](https://ai-paper-reviewer.com/5BXXoJh0Vr/tables_16_1.jpg)
> This table shows the impact of different hyperparameters (learning rate, time lag, and loss weight) on the model's performance (accuracy) for two different tasks: news-driven multi-stock movement prediction and multi-stock movement prediction without news.  It helps to understand the sensitivity of the CausalStock model to these parameters and aids in selecting optimal values for improved prediction accuracy.

![](https://ai-paper-reviewer.com/5BXXoJh0Vr/tables_16_2.jpg)
> This table presents the main results of the CausalStock model and several baseline models on six different datasets.  Two types of prediction tasks are evaluated: news-driven multi-stock movement prediction and multi-stock movement prediction (without news). The performance is measured by Accuracy (ACC) and Matthews Correlation Coefficient (MCC). Standard deviations are provided to show the robustness of the results.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/5BXXoJh0Vr/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}