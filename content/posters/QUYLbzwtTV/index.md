---
title: "Bias in Motion: Theoretical Insights into the Dynamics of Bias in SGD Training"
summary: "AI systems acquire bias during training, impacting accuracy across sub-populations. This research unveils bias's dynamic nature, revealing how classifier preferences shift over time, influenced by dat..."
categories: []
tags: ["AI Theory", "Fairness", "🏢 University of Cambridge",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} QUYLbzwtTV {{< /keyword >}}
{{< keyword icon="writer" >}} Anchit Jain et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=QUYLbzwtTV" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95235" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=QUYLbzwtTV&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/QUYLbzwtTV/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Machine learning models often inherit biases present in training data, significantly impacting their performance across different groups. Existing research primarily focuses on the initial and final stages of learning, neglecting the crucial transient dynamics where bias evolves. This study addresses this gap by investigating bias's dynamic behavior throughout training using a teacher-student framework and Gaussian mixture models to represent diverse data sub-populations. 

The research employs high-dimensional analysis, deriving an analytical solution for the stochastic gradient descent dynamics. This analysis reveals the evolution of bias across three distinct phases: an initial phase dominated by class imbalance, an intermediate phase driven by sample saliency, and a final phase determined by sub-population representation. The findings are empirically validated using deeper networks on various datasets, demonstrating the prevalence and complexity of this dynamic bias across different training scenarios. This work significantly contributes to a deeper understanding of bias formation and provides crucial insights for developing more effective bias mitigation techniques.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Bias evolution in machine learning is dynamic, exhibiting non-monotonic behavior across different learning phases. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} High-dimensional analysis reveals how data properties influence bias at different timescales. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Empirical validation on real-world datasets confirms theoretical findings, highlighting the complex time-dependence of learning with structured data. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on fairness and robustness in machine learning.  It **provides a novel theoretical framework for understanding the transient dynamics of bias**, which is often overlooked in existing research. This framework offers new avenues for developing effective bias mitigation strategies and prompts further studies on the evolution of biases at different stages of training.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_2_1.jpg)

> This figure demonstrates the generalization error for different subpopulations in a teacher-mixture model. Panel (a) compares simulated and theoretical results using a linear activation network, showing a good fit. The inset shows a comparison for the order parameters. Panels (b) through (d) showcase various scenarios achievable using the teacher-mixture model, relating to robustness and fairness.







### In-depth insights


#### Bias Dynamics
The study of bias dynamics in machine learning reveals a complex interplay of factors influencing how biases evolve during training.  **Initial phases** often show a classifier's sensitivity to class imbalances within subpopulations, leading to an initial bias towards more represented groups.  **Intermediate stages** highlight a shift towards the saliency of sample features, where the classifier focuses on features that are more readily apparent, regardless of their predictive power.  The **final stage** emphasizes the dominance of relative subpopulation representation in the model, solidifying biases through numerical preponderance over other data characteristics.  **High-dimensional analysis** provides an analytical framework to understand this evolution, proving the accuracy of ODEs in predicting bias dynamics in higher dimensions.   **Empirical validation** through synthetic and real datasets supports these theoretical findings, further reinforcing the significance of the observed temporal dynamics of bias formation in machine learning systems. This understanding is crucial in developing effective bias mitigation strategies.

#### High-D Analysis
The heading 'High-D Analysis' suggests a focus on the theoretical properties of high-dimensional data and algorithms.  The authors likely leverage tools from statistical physics or random matrix theory to analyze the behavior of the model in high dimensions.  **This approach is crucial because the behavior of machine learning models often changes dramatically as the dimensionality of the data increases.**  Instead of relying on empirical observations alone, this high-dimensional analysis provides a rigorous mathematical framework for understanding the dynamics of bias in stochastic gradient descent.  A key strength of such an approach is the ability to derive closed-form expressions or approximate solutions for model behavior, offering insights unavailable through solely experimental means.  **The high-dimensional setting allows the authors to make simplifying assumptions that lead to tractable analytical results, such as the convergence of stochastic dynamics to deterministic ordinary differential equations**. The analytical solutions derived in this section can be used to validate and complement the experimental results, providing further confidence in the observed phenomena. Overall, the 'High-D Analysis' section likely plays a pivotal role in establishing a strong theoretical foundation for the paper's findings.

#### SGD Dynamics
The study delves into the dynamics of Stochastic Gradient Descent (SGD) optimization, focusing on how bias evolves during the training process.  It uses a teacher-student framework with Gaussian mixture models to represent diverse data subpopulations.  **A key finding is the characterization of bias evolution into three distinct phases**, each influenced by different data properties: initial imbalance, sample saliency, and relative representation.  This reveals a non-monotonic behavior of bias, challenging the conventional focus on only initial or final states.  **The analysis leverages a high-dimensional limit, enabling analytical solutions for the dynamics**, which are then validated empirically with deeper networks on real datasets.  This provides valuable insights into transient learning regimes and highlights the dynamic interplay between various factors like data heterogeneity, spurious features, and class imbalance in shaping bias. **The implications are significant for fairness and robustness**, showing how to understand and potentially mitigate bias through a deeper understanding of its evolution over time.

#### Bias Mitigation
The concept of bias mitigation in machine learning is crucial, as models trained on biased data often perpetuate or amplify those biases.  This paper delves into the dynamics of bias in stochastic gradient descent (SGD) training, offering valuable insights for developing effective mitigation strategies. **The analysis reveals how different properties of subpopulations influence bias at different timescales, highlighting the non-monotonic nature of bias evolution during training.** This understanding is crucial, as it moves beyond simplistic views of bias as merely an initial or final state.  **A three-phase learning process is identified where bias is influenced by factors such as class imbalance, data representation, and spurious correlations.** The research emphasizes the importance of considering this transient bias evolution for effective mitigation strategies. **The analytical framework developed is validated using experiments on synthetic and real datasets, showing promise in guiding the development of targeted and temporally-aware bias mitigation techniques.** Ultimately, this work underscores the need for going beyond asymptotic analysis when considering bias and calls for the development of methods that account for the complex temporal dynamics of bias in machine learning.

#### Future Work
Future research directions stemming from this work could explore several promising avenues.  **Extending the theoretical analysis to more complex model architectures** (beyond linear classifiers and shallow networks) is crucial to assess the generalizability of the findings.  Investigating the impact of different activation functions, network depths, and training algorithms on bias dynamics would provide valuable insights. **Developing bias mitigation strategies** directly informed by the identified transient dynamics is another key area.  This could involve designing adaptive learning rates or regularization techniques that specifically target the critical phases of bias formation. **Empirical validation on a wider array of real-world datasets**, encompassing diverse domains and demographic groups, is necessary to further test the robustness and practical applicability of the proposed theoretical framework.  Finally, a deeper investigation into the interplay between bias and other ML phenomena, like generalization and spurious correlations, could yield a more holistic understanding of how bias manifests in the learning process.  This integrated understanding is critical to creating effective and robust strategies for building fair and equitable machine learning systems.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_4_1.jpg)

> This figure shows the generalization error of a linear classifier trained using SGD on a dataset generated from a teacher-mixture model. Panel (a) compares the theoretical predictions of generalization error with the results from simulations, showing a good match. The inset in panel (a) compares the order parameters from theory and simulations. Panels (b-d) demonstrate different scenarios (robustness and fairness) achievable with the teacher-mixture model.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_5_1.jpg)

> This figure demonstrates the 'crossing phenomenon' where the loss curves of two sub-populations intersect during training.  Panel (a) shows the loss curves for sub-population + (blue) and sub-population – (red), along with the total loss (purple). The crossing is caused by sub-population – having higher variance but lower representation. The right panel shows the order parameters over time illustrating the different bias phases. Panel (b) presents a phase diagram highlighting the parameter regions where the crossing phenomenon occurs.  The color scheme in the phase diagram indicates the asymptotic preference of the classifier (blue for sub-population +, red for sub-population –). Dark colors show consistent bias, light colors show a bias crossing, and white represents divergence due to high learning rates.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_7_1.jpg)

> This figure shows the phenomenon of bias crossing in which the loss curves for different sub-populations intersect during training, highlighting the non-monotonic nature of bias in this setting. The left panel displays the loss curves for the positive and negative sub-populations and their average, illustrating the crossing behavior.  The right panel shows the evolution of multiple order parameters which help to explain this behavior, showing three distinct learning phases. Key parameters determining this behavior are also provided.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_8_1.jpg)

> This figure presents numerical simulation results on the MNIST dataset using a 2-layer neural network.  It demonstrates the evolution of test loss and accuracy for two subpopulations (+) and (-) over multiple training epochs. Three panels showcase different scenarios: a single crossing phenomenon, a double crossing phenomenon (introducing a label imbalance), and the effect of varying variance while keeping one variance constant. Each panel shows the average and standard deviation over 100 simulations, illustrating the robustness of the observed phenomena.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_8_2.jpg)

> This figure presents the results of numerical simulations conducted on the MNIST dataset to validate the theoretical findings. It showcases three scenarios: a single crossing phenomenon, a double crossing phenomenon, and an analysis of the impact of changing the variance of one subpopulation while keeping the variance of the other constant. Each panel displays the test loss and accuracy for two subpopulations over multiple epochs. The results demonstrate the presence of multiple time scales in the bias dynamics and how they lead to non-monotonic behavior of the classifier.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_9_1.jpg)

> This figure shows the results of numerical simulations on the MNIST dataset using a variation of the Teacher-Mixture model.  The top row shows test loss, and the bottom row shows test accuracy, for two sub-populations (+ in blue, - in red). Panel (a) demonstrates a single crossing phenomenon where the loss curves for the two subpopulations intersect during training.  Panel (b) shows a double crossing, indicating a more complex bias evolution with multiple timescale dynamics. Panel (c) explores how the initial bias in learning depends on the variances of the subpopulations.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_25_1.jpg)

> The left panel of the figure shows the loss curves for the negative (red) and positive (blue) sub-populations, as well as the overall loss (purple). The crossing of the loss curves is highlighted, with the dashed and dotted vertical lines indicating important timescales. The right panel displays the dynamics of order parameters over time, illustrating the three phases of bias.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_26_1.jpg)

> This figure shows the loss curves and phase diagrams demonstrating the 'crossing phenomenon.'  The left panel shows the loss curves of two sub-populations, highlighting how the sub-population with higher variance initially has faster learning but asymptotically the sub-population with higher representation (product of representation and variance) takes over. The right panel shows a phase diagram illustrating the prevalence of the crossing phenomenon across different parameter combinations.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_26_2.jpg)

> This figure shows the loss curves for two sub-populations with different variances and mixing probabilities, along with the overall loss.  Panel (a) demonstrates a 'crossing' phenomenon where initially the higher variance sub-population has lower loss, but asymptotically the sub-population with higher representation (and product of variance and representation) dominates. Panel (b) displays a phase diagram illustrating the regions in parameter space exhibiting this crossing behavior.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_28_1.jpg)

> This figure demonstrates the generalization errors for two sub-populations in a teacher-student model using a linear activation function. Panel (a) compares the theoretical predictions with simulation results, highlighting the accuracy of the theoretical model. The inset shows the same comparison for order parameters R+, R-, M, and Q. Panels (b) through (d) illustrate different scenarios in the Teacher-Mixture (TM) model. Panel (b) showcases a robustness model where a spurious feature leads to misclassification; Panels (c) and (d) present two fairness models, with Panel (c) being a simplified case and Panel (d) showing the general fairness problem. The figure illustrates the TM model's ability to represent different fairness and robustness scenarios.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_28_2.jpg)

> This figure shows the results of numerical simulations on the MNIST dataset using a rotated version of the dataset to mimic the teacher-mixture model presented in the paper.  It demonstrates three key phenomena related to bias evolution during training: (a) a single crossing of the loss curves for two subpopulations, (b) a double crossing due to label imbalance, and (c) the impact of changing the variance of one subpopulation while keeping the other constant. The results validate the theoretical analysis by showing the predicted multi-phase behavior in a more realistic setting.


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_29_1.jpg)

> This figure shows the generalization errors for different sub-populations and how the theoretical predictions match the simulation results for a linear activation network. The inset shows a comparison of order parameters. It also exemplifies different fairness and robustness models using the teacher-mixture framework, showing how spurious features and heterogeneous data can affect the classifier's behavior during learning. 


![](https://ai-paper-reviewer.com/QUYLbzwtTV/figures_29_2.jpg)

> This figure shows the generalization errors for two subpopulations in a teacher-student learning model. Panel (a) compares the theoretical predictions with simulation results for a linear classifier. Panels (b) through (d) illustrate different scenarios for robustness and fairness, demonstrating how the model can represent various bias situations.


</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/QUYLbzwtTV/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}