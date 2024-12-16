---
title: "Contrastive losses as generalized models of global epistasis"
summary: "Contrastive losses unlock efficient fitness function modeling by leveraging the ranking information inherent in global epistasis, significantly improving accuracy and data efficiency in protein engine..."
categories: ["AI Generated", ]
tags: ["AI Theory", "Optimization", "🏢 Dyno Therapeutics",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} hLoiXOzoly {{< /keyword >}}
{{< keyword icon="writer" >}} David H Brookes et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=hLoiXOzoly" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/hLoiXOzoly" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=hLoiXOzoly&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/hLoiXOzoly/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Estimating fitness functions from biological sequence data is a key challenge in fields like protein engineering.  Traditional methods often struggle with the complexity of these functions due to multi-peaked nature and limited data.  **Global epistasis models**, which assume a sparse latent fitness function transformed by a monotonic nonlinearity, provide a physically-grounded framework but require strong assumptions and can be computationally expensive.

This research proposes using **supervised contrastive loss functions** (like the Bradley-Terry loss) to extract the sparse latent function implied by global epistasis, bypassing the need for explicit nonlinearity modeling.  **This method avoids the limitations of traditional MSE-based approaches** that struggle with the inherent nonlinearities and sparsity challenges.  Their empirical findings show that contrastive losses provide more accurate and efficient fitness function estimation, especially with limited datasets, consistently outperforming MSE-based methods across benchmark tasks.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Contrastive loss functions, such as the Bradley-Terry loss, effectively extract sparse latent functions from limited data in the presence of global epistasis. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} Global epistasis models can result in observed fitness functions that lack sparse representations, making them inefficient to learn with MSE loss. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Models trained with contrastive losses consistently outperform those trained with MSE loss on empirical benchmark tasks, demonstrating practical utility. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in protein engineering and machine learning because it introduces a novel approach for efficiently modeling fitness landscapes, a major challenge in these fields.  **The findings offer a more data-efficient method for predicting protein properties and accelerate the design of novel proteins.** It opens exciting avenues for future research on advanced techniques for learning-to-rank models and their applications to high-dimensional biological data.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/hLoiXOzoly/figures_4_1.jpg)

> 🔼 This figure demonstrates the ability of the Bradley-Terry loss to recover a sparse latent fitness function from complete fitness data that has been transformed by a global epistasis nonlinearity. Panel (a) shows a schematic of the simulation. Panel (b) compares the latent and observed fitness functions in both the fitness and epistatic domains. The observed function shows a dense representation in the epistatic domain, which is not present in the latent fitness function. Panel (c) shows that the model trained with the Bradley-Terry loss can accurately recover the latent fitness function.
> <details>
> <summary>read the caption</summary>
> Figure 1: Recovery of latent fitness function from complete fitness data by minimizing Bradley-Terry loss. (a) Schematic of simulation. (b) Comparison between latent (f) and observed (y) fitness functions in fitness (left) and epistatic (right) domains. The latent fitness function is sampled from the NK model with L = 8 and K = 2 and the global epistasis function is g(f) = exp(10. f). Each point in the scatter plot represents the fitness of a sequence, while each bar in the bar plot (right) represents the squared magnitude of an epistatic interaction (normalized such that all squared magnitudes sum to 1), with roman numerals indicating the order of interaction. Only epistatic interactions up to order 3 are shown. The right plot demonstrates that global epistasis produces a dense representation in the epistatic domain compared to the representation of the latent fitness in the epistatic domain. (c) Comparison between latent and estimated (f) fitness functions in fitness and epistatic domains.
> </details>





![](https://ai-paper-reviewer.com/hLoiXOzoly/tables_7_1.jpg)

> 🔼 This table compares the performance of MSE and Bradley-Terry loss functions for various fitness prediction tasks from the FLIP benchmark dataset.  The results are obtained using a CNN model and show the mean and standard deviation of Spearman correlation and Top 10% Recall metrics across 10 separate model runs.  Asterisks indicate data sets where unmodified sequence portions were included in the training data. Bold values highlight statistically significant improvements from the Bradley-Terry loss over the MSE loss.
> <details>
> <summary>read the caption</summary>
> Table 1: Comparison between MSE and Bradley-Terry losses on FLIP benchmark tasks using the CNN baseline model. Each row represents a data set and split combination. Numerical columns indicate the mean and standard deviation of test set metrics over 10 random initializations of the model. Asterisks indicate that unmodified portions of sequences were used in training data. Bold values indicate that a loss has significantly improved performance over all other tested losses (p < 0.05). Additional benchmark results are shown in Appendix G.
> </details>





### In-depth insights


#### Contrastive Loss
Contrastive losses, unlike traditional supervised learning methods that focus on minimizing the difference between predicted and actual fitness values, **emphasize learning the relative ranking of sequences based on their fitness**.  This approach is particularly well-suited for scenarios with limited data, high-dimensional sequence spaces, and the presence of global epistasis, a complex phenomenon where the fitness of a sequence is non-linearly related to individual mutations. By focusing on ranking, contrastive losses are less sensitive to the specific functional form of the fitness landscape and the nonlinearities introduced by global epistasis.  **The Bradley-Terry loss, a prominent example of a contrastive loss, directly optimizes the probability that sequences with higher fitness are ranked above those with lower fitness.**  This makes it particularly robust to noise and data sparsity, issues commonly encountered in experimental fitness data.  The superior performance of contrastive losses, especially in the low-data regime, is theoretically justified by an uncertainty principle relating the sparsity of sequence representations in the epistatic and fitness domains, highlighting the **efficiency gains from working in the ranking space.**  Overall, the adoption of contrastive losses presents a powerful and robust approach for fitness function modeling in complex biological systems.

#### Global Epistasis
Global epistasis, a phenomenon where non-linear relationships between multiple genetic loci significantly influence an organism's fitness, is a key concept in the paper.  **The authors challenge the traditional approach of using Mean Squared Error (MSE) loss functions for modeling fitness landscapes in the presence of global epistasis.**  They argue that this approach can be inefficient because global epistasis models often produce observed fitness functions lacking sparse representations in the epistatic domain. These functions become concentrated in the fitness domain, hindering efficient learning from limited data with MSE.  **The core contribution proposes the use of contrastive losses, such as the Bradley-Terry loss, as a more effective alternative.** This approach directly learns a ranking function for the sequences, robust to the nonlinearities of global epistasis and capable of extracting sparse latent fitness functions from limited data.  **The efficacy of this approach is demonstrated through simulations and empirical benchmark tasks,** showing a consistent improvement over MSE-based methods, especially in data-scarce scenarios. The authors also introduce a fitness-epistasis uncertainty principle, formally linking fitness and epistatic domain concentrations, providing theoretical support for their method's superior data efficiency.

#### Uncertainty Principle
The paper introduces a novel "fitness-epistasis uncertainty principle", arguing that **global epistasis, a phenomenon where non-linear relationships affect sequence fitness in a non-specific manner, leads to observed fitness functions that are not sparsely representable in the epistatic domain**.  This challenges traditional sparse modeling approaches which minimize Mean Squared Error (MSE) loss, often requiring substantially more data for accurate estimation. The principle leverages the concept that a function cannot be highly concentrated in both the fitness and epistatic domains simultaneously. **Global epistasis, through its non-linear transformation, tends to concentrate the observed fitness function in the fitness domain, resulting in a dense representation in the epistatic domain**. This makes learning with MSE loss inefficient. The principle therefore justifies the use of contrastive loss functions, which focus on ranking rather than precise fitness values, providing a more data-efficient method for extracting the underlying sparse ranking function that captures fitness landscape information effectively.

#### Benchmark Tasks
Benchmark tasks play a crucial role in evaluating the performance of machine learning models, especially in the context of protein engineering.  A well-designed benchmark should encompass a diverse range of datasets, representing different protein types, experimental setups, and definitions of protein fitness. This diversity is essential to assess the generalizability of models and identify their strengths and weaknesses. The use of multiple evaluation metrics, such as Spearman correlation and top-10% recall, provides a more comprehensive picture of model performance than a single metric.  **The inclusion of both complete and incomplete datasets is vital**; complete datasets provide a baseline for assessing the accuracy of model predictions, while incomplete datasets simulate the real-world scenario where data is often scarce. Furthermore, **a robust benchmark must account for the effects of global epistasis**, a phenomenon causing complex interactions within protein sequences, leading to unpredictable fitness landscapes.  **Careful consideration of data preprocessing and how it affects model outcomes is also necessary.** Finally, publicly available benchmarks, such as the FLIP dataset, provide a valuable resource for facilitating comparisons between different models and promoting progress in the field.

#### Future Directions
Future research could explore several avenues.  **Extending the framework to handle more complex sequence data** is crucial, such as incorporating RNA or genome sequences, which present unique structural and interaction challenges.  **Investigating alternative contrastive loss functions** and their suitability for various fitness landscapes could lead to improved learning efficiency.  **A deeper theoretical understanding of the relationship between global epistasis and the success of contrastive methods** is needed, potentially through rigorous analysis of information-theoretic bounds or novel uncertainty principles.  Further work should focus on **developing robust methods for handling noisy and incomplete experimental data**, as this is common in high-throughput fitness assays. **Benchmarking against existing approaches** and evaluating the scalability and generalizability of the method on diverse datasets are also essential.  Finally, **exploring the application of these techniques to design sequences with specific properties**, such as improved stability or increased binding affinity, promises significant practical implications for protein engineering and beyond.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/hLoiXOzoly/figures_5_1.jpg)

> 🔼 This figure demonstrates the fitness-epistasis uncertainty principle and its implications for using MSE vs. BT loss with incomplete data. Panel (a) shows how the entropy of the fitness function in the fitness and epistatic domains changes with the nonlinearity applied. Panel (b) shows that models trained with BT loss consistently outperform those trained with MSE loss on incomplete data across different nonlinearities. Panel (c) shows that the BT loss is more data-efficient than the MSE loss.
> <details>
> <summary>read the caption</summary>
> Figure 2: (a) Demonstration of the fitness-epistasis uncertainty principle for a latent fitness function transformed by g(f) = exp(af) for various settings of a. The dashed black line indicates the lower bound on the sum of the entropies of the fitness and epistatic representations of the fitness function (b) Test-set Spearman correlation for models trained with MSE and BT losses on incomplete fitness data transformed by various nonlinearities, compared to the entropy of the fitness function in the epistatic domain. Each point corresponds to a model trained on 256 randomly sampled training points from an L = 10, K = 2 latent fitness function which was then transformed by a nonlinearity. (c) Convergence of models fit with BT and MSE losses to observed data generated by transforming an L = 10, K = 2 latent fitness function by g(f) = exp(10. f). Each point represents the mean test set correlation over 200 training set replicates.
> </details>



![](https://ai-paper-reviewer.com/hLoiXOzoly/figures_15_1.jpg)

> 🔼 This figure demonstrates the ability of the Bradley Terry loss to recover a sparse latent fitness function from complete, noiseless fitness data that has been affected by global epistasis.  Panel (a) shows a schematic of the simulation. Panel (b) compares the latent and observed fitness functions in both the fitness and epistatic domains, highlighting how global epistasis creates a dense representation in the epistatic domain. Panel (c) shows that minimizing the Bradley-Terry loss accurately recovers the sparse latent function. 
> <details>
> <summary>read the caption</summary>
> Figure 1: Recovery of latent fitness function from complete fitness data by minimizing Bradley-Terry loss. (a) Schematic of simulation. (b) Comparison between latent (f) and observed (y) fitness functions in fitness (left) and epistatic (right) domains. The latent fitness function is sampled from the NK model with L = 8 and K = 2 and the global epistasis function is g(f) = exp(10. f). Each point in the scatter plot represents the fitness of a sequence, while each bar in the bar plot (right) represents the squared magnitude of an epistatic interaction (normalized such that all squared magnitudes sum to 1), with roman numerals indicating the order of interaction. Only epistatic interactions up to order 3 are shown. The right plot demonstrates that global epistasis produces a dense representation in the epistatic domain compared to the representation of the latent fitness in the epistatic domain. (c) Comparison between latent and estimated (f) fitness functions in fitness and epistatic domains.
> </details>



![](https://ai-paper-reviewer.com/hLoiXOzoly/figures_18_1.jpg)

> 🔼 This figure demonstrates the ability of the Bradley-Terry loss to recover a sparse latent fitness function from complete fitness data that has been transformed by a monotonic nonlinearity.  Panel (a) shows the simulation setup.  Panel (b) compares the latent and observed fitness functions in both fitness and epistatic domains, illustrating the effect of global epistasis on the sparsity of the representation.  Panel (c) compares the latent and recovered fitness functions, showing that the Bradley-Terry loss is able to recover the sparse function almost perfectly.
> <details>
> <summary>read the caption</summary>
> Figure 1: Recovery of latent fitness function from complete fitness data by minimizing Bradley-Terry loss. (a) Schematic of simulation. (b) Comparison between latent (f) and observed (y) fitness functions in fitness (left) and epistatic (right) domains. The latent fitness function is sampled from the NK model with L = 8 and K = 2 and the global epistasis function is g(f) = exp(10. f). Each point in the scatter plot represents the fitness of a sequence, while each bar in the bar plot (right) represents the squared magnitude of an epistatic interaction (normalized such that all squared magnitudes sum to 1), with roman numerals indicating the order of interaction. Only epistatic interactions up to order 3 are shown. The right plot demonstrates that global epistasis produces a dense representation in the epistatic domain compared to the representation of the latent fitness in the epistatic domain. (c) Comparison between latent and estimated (f) fitness functions in fitness and epistatic domains.
> </details>



![](https://ai-paper-reviewer.com/hLoiXOzoly/figures_19_1.jpg)

> 🔼 This figure demonstrates how the Bradley-Terry loss can recover a sparse latent fitness function from complete fitness data that has been transformed by a global epistasis nonlinearity. Panel (a) shows a schematic of the simulation. Panel (b) compares the latent and observed fitness functions in both the fitness and epistatic domains. Panel (c) compares the latent and estimated fitness functions. The results show that the Bradley-Terry loss is able to accurately recover the latent fitness function, even when the observed data is dense in the epistatic domain due to global epistasis.
> <details>
> <summary>read the caption</summary>
> Figure 1: Recovery of latent fitness function from complete fitness data by minimizing Bradley-Terry loss. (a) Schematic of simulation. (b) Comparison between latent (f) and observed (y) fitness functions in fitness (left) and epistatic (right) domains. The latent fitness function is sampled from the NK model with L = 8 and K = 2 and the global epistasis function is g(f) = exp(10. f). Each point in the scatter plot represents the fitness of a sequence, while each bar in the bar plot (right) represents the squared magnitude of an epistatic interaction (normalized such that all squared magnitudes sum to 1), with roman numerals indicating the order of interaction. Only epistatic interactions up to order 3 are shown. The right plot demonstrates that global epistasis produces a dense representation in the epistatic domain compared to the representation of the latent fitness in the epistatic domain. (c) Comparison between latent and estimated (f) fitness functions in fitness and epistatic domains.
> </details>



![](https://ai-paper-reviewer.com/hLoiXOzoly/figures_20_1.jpg)

> 🔼 This figure demonstrates the ability of the Bradley-Terry loss to recover a sparse latent fitness function from complete fitness data that has been transformed by a global epistasis nonlinearity. Panel (a) shows a schematic of the simulation. Panel (b) compares the latent and observed fitness functions in both the fitness and epistatic domains, showing that global epistasis creates a dense representation in the epistatic domain. Panel (c) compares the latent and estimated fitness functions, demonstrating that the model accurately recovers the sparse latent function.
> <details>
> <summary>read the caption</summary>
> Figure 1: Recovery of latent fitness function from complete fitness data by minimizing Bradley-Terry loss. (a) Schematic of simulation. (b) Comparison between latent (f) and observed (y) fitness functions in fitness (left) and epistatic (right) domains. The latent fitness function is sampled from the NK model with L = 8 and K = 2 and the global epistasis function is g(f) = exp(10. f). Each point in the scatter plot represents the fitness of a sequence, while each bar in the bar plot (right) represents the squared magnitude of an epistatic interaction (normalized such that all squared magnitudes sum to 1), with roman numerals indicating the order of interaction. Only epistatic interactions up to order 3 are shown. The right plot demonstrates that global epistasis produces a dense representation in the epistatic domain compared to the representation of the latent fitness in the epistatic domain. (c) Comparison between latent and estimated (f) fitness functions in fitness and epistatic domains.
> </details>



![](https://ai-paper-reviewer.com/hLoiXOzoly/figures_21_1.jpg)

> 🔼 This figure demonstrates the ability of the Bradley-Terry loss to recover a sparse latent fitness function from complete fitness data affected by global epistasis. Panel (a) shows a schematic of the simulation process. Panel (b) compares the latent and observed fitness functions in both the fitness and epistatic domains, highlighting the effect of global epistasis in creating a dense representation in the epistatic domain. Panel (c) shows that the model trained with the Bradley-Terry loss accurately recovers the latent fitness function.
> <details>
> <summary>read the caption</summary>
> Figure 1: Recovery of latent fitness function from complete fitness data by minimizing Bradley-Terry loss. (a) Schematic of simulation. (b) Comparison between latent (f) and observed (y) fitness functions in fitness (left) and epistatic (right) domains. The latent fitness function is sampled from the NK model with L = 8 and K = 2 and the global epistasis function is g(f) = exp(10. f). Each point in the scatter plot represents the fitness of a sequence, while each bar in the bar plot (right) represents the squared magnitude of an epistatic interaction (normalized such that all squared magnitudes sum to 1), with roman numerals indicating the order of interaction. Only epistatic interactions up to order 3 are shown. The right plot demonstrates that global epistasis produces a dense representation in the epistatic domain compared to the representation of the latent fitness in the epistatic domain. (c) Comparison between latent and estimated (f) fitness functions in fitness and epistatic domains.
> </details>



![](https://ai-paper-reviewer.com/hLoiXOzoly/figures_22_1.jpg)

> 🔼 This figure shows multiple examples of the recovery of a latent fitness function from complete data transformed by a global epistasis nonlinearity using the Bradley-Terry loss.  Different settings of the NK model's K parameter and various nonlinearities (exponential, sigmoid, cubic, and arcsinh) were tested.  Each subplot displays the latent fitness function against observed data and model predictions, demonstrating the model's ability to accurately recover the latent function in various conditions.
> <details>
> <summary>read the caption</summary>
> Figure 3: Results from multiple examples of the task of recovering a latent fitness function given complete observed data transformed by a global epistasis nonlinearity. Each sub-plot shows the results of one such task. The setting of K used to sample the latent fitness function from the NK model and the particular form of the nonlinearity g(f) used are indicated in each sub-plot title. The horizontal axis in each sub-plot represents the values of the latent fitness function, while the vertical axis represents the values of either the observed data (blue dots) or model predictions (red dots). For ease of plotting, all fitness functions were normalized to have an empirical mean and std. dev. of 1, respectively. The R² correlation between the latent fitness function and the model predictions are indicated in red text.
> </details>



</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hLoiXOzoly/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}