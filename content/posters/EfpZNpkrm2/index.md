---
title: "QuanTA: Efficient High-Rank Fine-Tuning of LLMs with Quantum-Informed Tensor Adaptation"
summary: "QuanTA: Quantum-inspired Tensor Adaptation efficiently fine-tunes LLMs with high-rank updates, surpassing low-rank methods like LoRA for complex tasks while minimizing additional parameters."
categories: ["AI Generated", ]
tags: ["Natural Language Processing", "Large Language Models", "🏢 MIT",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} EfpZNpkrm2 {{< /keyword >}}
{{< keyword icon="writer" >}} Zhuo Chen et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=EfpZNpkrm2" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/EfpZNpkrm2" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/EfpZNpkrm2/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Fine-tuning large language models (LLMs) is computationally expensive.  Existing methods like Low-Rank Adaptation (LoRA) offer improvements but struggle with complex tasks. These methods rely on low-rank approximations, which can limit their ability to capture all necessary task-specific information, leading to performance bottlenecks.

QuanTA uses tensor operations inspired by quantum circuits to achieve efficient high-rank fine-tuning.  This allows it to effectively adapt LLMs to downstream tasks without relying on low-rank approximations.  **QuanTA demonstrates significant improvements over existing methods in various reasoning tasks, achieving comparable or better results with far fewer trainable parameters**.  It introduces no inference overhead, making it highly practical for real-world applications.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} QuanTA enables efficient high-rank fine-tuning of LLMs, overcoming limitations of low-rank methods. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} QuanTA significantly enhances performance in commonsense and arithmetic reasoning tasks compared to existing techniques. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} QuanTA's parameter-efficiency and lack of inference overhead make it a highly scalable and practical solution for LLM adaptation. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working on efficient fine-tuning methods for large language models.  **It introduces QuanTA, a novel method that outperforms existing techniques while using significantly fewer parameters.** This opens avenues for more efficient and scalable LLM adaptation, addressing a major bottleneck in the field and facilitating wider adoption of LLMs in resource-constrained settings.  The theoretical underpinnings and empirical results will greatly aid research into parameter-efficient fine-tuning.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/EfpZNpkrm2/figures_1_1.jpg)

> 🔼 This figure compares QuanTA and LoRA, two methods for fine-tuning large language models. LoRA uses low-rank matrices to update model weights, limiting its ability to capture complex relationships. QuanTA, inspired by quantum circuits, uses tensors to update weights along specific axes.  This allows for high-rank updates, improving performance while using fewer parameters. The graph illustrates that QuanTA achieves performance closer to full fine-tuning than LoRA.
> <details>
> <summary>read the caption</summary>
> Figure 1: Conceptual comparison of QuanTA and LoRA methods. LoRA parameterizes the weight matrix update as a outer product of two low-rank matrices, limiting its capacity. QuanTA, inspired by quantum circuits, uses tensors that operate on specific axes of the (reshaped) input, enabling high-rank parameterization. Supported by the universality theorem and rank representation theorem, QuanTA can represent arbitrary matrices effectively, allowing it to achieve performance comparable to or sometimes even better than full fine-tuning, with only a fraction of the parameters. Note: the performance graph is a conceptual illustration.
> </details>





![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_2_1.jpg)

> 🔼 This table presents the performance comparison of the base LLaMA2-7B model and the same model fine-tuned using LoRA with rank 64 and 128 on two different datasets: RTE and DROP.  RTE (Recognizing Textual Entailment) is a simpler classification task, while DROP (Difficult Reasoning Over Paragraphs) is a more complex generation task. The table shows that LoRA improves performance on both datasets, but the improvement is more significant on the RTE dataset. This suggests that the low-rank assumption of LoRA might not hold as well for complex tasks such as DROP.
> <details>
> <summary>read the caption</summary>
> Table 1: Performance of base and LoRA fine-tuned LLaMA2-7B on RTE [49] and DROP [50] datasets. We use accuracy and F₁-score as the metrics for them respectively.
> </details>





### In-depth insights


#### Quantum-inspired LLM Tuning
Quantum-inspired large language model (LLM) tuning represents a novel approach to enhance the efficiency and effectiveness of fine-tuning these massive models.  By drawing inspiration from quantum computing principles, **methods such as QuanTA aim to overcome the limitations of existing low-rank adaptation techniques.**  Instead of low-rank approximations, which can hinder performance on complex tasks, these quantum-inspired methods enable efficient high-rank fine-tuning. This approach is theoretically supported by the universality and rank representation theorems, demonstrating its capability to capture intricate task-specific patterns.  **The use of tensor operations analogous to quantum circuits facilitates efficient high-rank parameterization,** significantly reducing the number of trainable parameters while maintaining or even surpassing the performance of full fine-tuning.  Furthermore, **quantum-inspired tuning methods generally avoid any inference overhead,** making them particularly attractive for large-scale deployment.  However, the field is still nascent, and several challenges remain such as optimizing the number of tensors and achieving efficient GPU utilization.  Further research is needed to fully explore the potential and address the limitations of this promising area.

#### High-Rank Tensor Adaptation
High-rank tensor adaptation emerges as a powerful technique to overcome limitations in parameter-efficient fine-tuning of large language models (LLMs). Traditional low-rank methods, while efficient, often fall short on complex tasks due to their inherent inability to capture the full complexity of high-dimensional weight updates.  **High-rank approaches, in contrast, offer greater expressiveness and flexibility by directly addressing the high-rank nature of these updates**.  This enhanced representational power allows for significantly improved performance on challenging downstream tasks, exceeding what is achievable with low-rank approximations.  However, the increased number of parameters associated with high-rank methods may raise concerns about computational cost.  **Therefore, the core challenge lies in developing techniques that achieve high-rank adaptation without sacrificing efficiency**.  Quantum-inspired methods, leveraging tensor operations analogous to quantum circuits, provide a promising solution.  **By cleverly parameterizing weight updates with a small set of tensors, these methods enable high-rank representation while maintaining manageable computational burdens**.  This approach represents a significant advance in parameter-efficient fine-tuning, offering a robust and scalable solution for adapting LLMs to diverse applications.

#### QuanTA: Method & Results
The QuanTA method, inspired by quantum circuit structures, introduces a novel parameter-efficient fine-tuning approach for LLMs.  **Unlike low-rank adaptation methods (e.g., LoRA), QuanTA facilitates efficient high-rank fine-tuning**, addressing limitations in representing complex downstream tasks.  This is theoretically supported by the universality and rank representation theorems.  **Empirically, QuanTA demonstrates significant performance gains across various reasoning tasks (commonsense, arithmetic), outperforming LoRA and often matching or exceeding full fine-tuning performance with far fewer trainable parameters.**  The method's ease of implementation and absence of inference overhead are key advantages.  **QuanTA's scalability is highlighted by its superior performance even on large LLMs**.  However, potential limitations regarding optimal tensor configurations and GPU utilization efficiency during training warrant further investigation.

#### Low-Rank Limits Challenged
The heading 'Low-Rank Limits Challenged' aptly encapsulates a central theme in the paper, focusing on the limitations of low-rank adaptation (LoRA) methods for fine-tuning large language models (LLMs).  The authors argue that **LoRA's reliance on low-rank approximations restricts its ability to capture the complexities of certain downstream tasks**, leading to suboptimal performance.  This limitation is particularly evident when dealing with intricate tasks that deviate significantly from the pre-training data. The paper posits that this **low-rank constraint hinders the model's capacity to learn the necessary task-specific adaptations**, highlighting the need for more expressive high-rank methods.  This challenge forms the core motivation for proposing QuanTA, a novel quantum-inspired technique that overcomes LoRA's limitations by facilitating efficient high-rank fine-tuning without sacrificing computational efficiency.  **QuanTA's superior performance across diverse tasks serves as direct evidence that the low-rank assumption inherent to LoRA is not universally sufficient** for achieving state-of-the-art results in LLM adaptation.

#### Future QuanTA Directions
Future research could explore several promising avenues to enhance QuanTA.  **Improving efficiency** is crucial, potentially by optimizing tensor application order or developing specialized hardware acceleration.  **Generalizing QuanTA** to diverse model architectures and leveraging other parameter-efficient fine-tuning methods (like prefix-tuning or adapters) in a hybrid approach are key directions.  **Theoretical investigations** could delve deeper into the rank representation theorem to improve its applicability to complex tasks and further explore the connection between QuanTA's quantum-inspired structure and its empirical performance.  **Addressing limitations** in handling very large models and optimizing hyperparameters is crucial for widespread adoption.  Finally, exploring **advanced optimization techniques** specifically designed for QuanTA's unique structure could unlock even greater performance gains, paving the way for future applications in diverse areas.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/EfpZNpkrm2/figures_2_1.jpg)

> 🔼 This figure conceptually compares QuanTA and LoRA methods for parameter-efficient fine-tuning of LLMs.  LoRA uses low-rank matrix updates, limiting its representational capacity.  QuanTA, drawing inspiration from quantum circuits, employs tensors operating on specific input axes, enabling higher-rank parameterizations.  Theoretically, QuanTA's ability to represent arbitrary matrices effectively is supported by the universality theorem and the rank representation theorem, promising performance comparable to or exceeding full fine-tuning with significantly fewer parameters.
> <details>
> <summary>read the caption</summary>
> Figure 1: Conceptual comparison of QuanTA and LoRA methods. LoRA parameterizes the weight matrix update as a outer product of two low-rank matrices, limiting its capacity. QuanTA, inspired by quantum circuits, uses tensors that operate on specific axes of the (reshaped) input, enabling high-rank parameterization. Supported by the universality theorem and rank representation theorem, QuanTA can represent arbitrary matrices effectively, allowing it to achieve performance comparable to or sometimes even better than full fine-tuning, with only a fraction of the parameters. Note: the performance graph is a conceptual illustration.
> </details>



![](https://ai-paper-reviewer.com/EfpZNpkrm2/figures_3_1.jpg)

> 🔼 This figure illustrates the universality of quantum circuits.  It shows that any unitary matrix (a type of mathematical transformation representing a quantum operation) can be broken down into a sequence of simpler operations, namely single-qubit and two-qubit gates.  This decomposition is crucial because it demonstrates that complex quantum computations can be constructed from a limited set of basic building blocks. This concept is foundational to the development of QuanTA, as it demonstrates that high-rank operations can be achieved using a composition of smaller operations.
> <details>
> <summary>read the caption</summary>
> Figure 3: Any unitary matrix can be decomposed into a quantum circuit using one- and two-qubit gates.
> </details>



![](https://ai-paper-reviewer.com/EfpZNpkrm2/figures_14_1.jpg)

> 🔼 This figure compares QuanTA and LoRA, two parameter-efficient fine-tuning methods for LLMs.  LoRA uses low-rank matrix updates, limiting its ability to capture complex relationships in data.  QuanTA, inspired by quantum circuits, uses tensor operations on specific axes of the input, allowing for high-rank updates. This enables QuanTA to represent a wider range of matrices than LoRA, leading to performance closer to or even surpassing full fine-tuning, but with far fewer parameters.
> <details>
> <summary>read the caption</summary>
> Figure 1: Conceptual comparison of QuanTA and LoRA methods. LoRA parameterizes the weight matrix update as a outer product of two low-rank matrices, limiting its capacity. QuanTA, inspired by quantum circuits, uses tensors that operate on specific axes of the (reshaped) input, enabling high-rank parameterization. Supported by the universality theorem and rank representation theorem, QuanTA can represent arbitrary matrices effectively, allowing it to achieve performance comparable to or sometimes even better than full fine-tuning, with only a fraction of the parameters. Note: the performance graph is a conceptual illustration.
> </details>



![](https://ai-paper-reviewer.com/EfpZNpkrm2/figures_15_1.jpg)

> 🔼 This figure compares QuanTA and LoRA methods. LoRA uses low-rank matrices to update weight matrices, limiting its capacity to handle complex tasks.  QuanTA, inspired by quantum circuits, employs tensors operating on specific input axes, allowing for high-rank parameterization.  Theoretically, QuanTA can represent arbitrary matrices efficiently, potentially outperforming full fine-tuning with fewer parameters.
> <details>
> <summary>read the caption</summary>
> Figure 1: Conceptual comparison of QuanTA and LoRA methods. LoRA parameterizes the weight matrix update as a outer product of two low-rank matrices, limiting its capacity. QuanTA, inspired by quantum circuits, uses tensors that operate on specific axes of the (reshaped) input, enabling high-rank parameterization. Supported by the universality theorem and rank representation theorem, QuanTA can represent arbitrary matrices effectively, allowing it to achieve performance comparable to or sometimes even better than full fine-tuning, with only a fraction of the parameters. Note: the performance graph is a conceptual illustration.
> </details>



![](https://ai-paper-reviewer.com/EfpZNpkrm2/figures_15_2.jpg)

> 🔼 This figure conceptually compares QuanTA and LoRA, highlighting their differences in parameterizing weight matrix updates.  LoRA uses low-rank matrices, limiting its representational capacity, while QuanTA leverages tensors inspired by quantum circuits for high-rank parameterization, enabling more expressive updates and better performance with fewer parameters. The universality and rank representation theorems underpin QuanTA's ability to efficiently represent arbitrary matrices.
> <details>
> <summary>read the caption</summary>
> Figure 1: Conceptual comparison of QuanTA and LoRA methods. LoRA parameterizes the weight matrix update as a outer product of two low-rank matrices, limiting its capacity. QuanTA, inspired by quantum circuits, uses tensors that operate on specific axes of the (reshaped) input, enabling high-rank parameterization. Supported by the universality theorem and rank representation theorem, QuanTA can represent arbitrary matrices effectively, allowing it to achieve performance comparable to or sometimes even better than full fine-tuning, with only a fraction of the parameters. Note: the performance graph is a conceptual illustration.
> </details>



![](https://ai-paper-reviewer.com/EfpZNpkrm2/figures_18_1.jpg)

> 🔼 The figure conceptually compares QuanTA and LoRA methods for fine-tuning LLMs.  LoRA uses low-rank matrix updates, limiting its ability to capture complex relationships. QuanTA, inspired by quantum circuits, uses tensors for high-rank updates, enabling more flexible adaptation and potentially better performance with fewer parameters.
> <details>
> <summary>read the caption</summary>
> Figure 1: Conceptual comparison of QuanTA and LoRA methods. LoRA parameterizes the weight matrix update as a outer product of two low-rank matrices, limiting its capacity. QuanTA, inspired by quantum circuits, uses tensors that operate on specific axes of the (reshaped) input, enabling high-rank parameterization. Supported by the universality theorem and rank representation theorem, QuanTA can represent arbitrary matrices effectively, allowing it to achieve performance comparable to or sometimes even better than full fine-tuning, with only a fraction of the parameters. Note: the performance graph is a conceptual illustration.
> </details>



![](https://ai-paper-reviewer.com/EfpZNpkrm2/figures_27_1.jpg)

> 🔼 The figure conceptually compares QuanTA and LoRA, highlighting their differences in how they parameterize weight matrix updates. LoRA uses low-rank matrices, limiting its capacity to capture complex relationships.  In contrast, QuanTA uses tensors inspired by quantum circuits to enable high-rank parameterization, allowing it to effectively represent arbitrary matrices and potentially achieve superior performance while using fewer parameters.
> <details>
> <summary>read the caption</summary>
> Figure 1: Conceptual comparison of QuanTA and LoRA methods. LoRA parameterizes the weight matrix update as a outer product of two low-rank matrices, limiting its capacity. QuanTA, inspired by quantum circuits, uses tensors that operate on specific axes of the (reshaped) input, enabling high-rank parameterization. Supported by the universality theorem and rank representation theorem, QuanTA can represent arbitrary matrices effectively, allowing it to achieve performance comparable to or sometimes even better than full fine-tuning, with only a fraction of the parameters. Note: the performance graph is a conceptual illustration.
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_6_1.jpg)
> 🔼 This table compares the performance of several parameter-efficient fine-tuning (PEFT) methods on the DROP dataset, using different sizes of the LLaMA2 language model.  The methods compared include full fine-tuning (FT), series adapters, parallel adapters, LoRA with different ranks, and QuanTA with different parameter configurations. The table shows the number of trainable parameters used by each method (as a percentage of the total parameters) and the resulting F1 score achieved.  The results highlight QuanTA's superior performance compared to other PEFT methods, especially when using a small fraction of trainable parameters.
> <details>
> <summary>read the caption</summary>
> Table 2: Benchmark of various fine-tuning methods on the DROP dataset using LLaMA2 7-70 billion parameter models as the base model. In each case, we report the average of F₁ score over 2-4 experiments with different random seeds.
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_7_1.jpg)
> 🔼 This table compares the performance of various parameter-efficient fine-tuning (PEFT) methods on several commonsense reasoning tasks using different sized language models (LLaMAs).  It shows the accuracy achieved by different methods (Full Fine-tuning, Prefix Tuning, Adapter methods, LoRA, DORA, and QuanTA) with varying numbers of trainable parameters.  The results demonstrate QuanTA's superior performance and efficiency compared to other PEFT methods.
> <details>
> <summary>read the caption</summary>
> Table 3: Benchmark on various commonsense reasoning tasks. All results of models and PEFT methods labeled with “*” are from [54], and results with “†” are from [20].
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_8_1.jpg)
> 🔼 This table presents the results of several models on four arithmetic reasoning tasks.  The models tested include LLaMA2-7B and LLaMA2-13B, fine-tuned with different parameter-efficient fine-tuning (PEFT) methods such as full fine-tuning (FT), LoRA, and QuanTA.  The table compares the accuracy of each model and method across the four tasks.  The results show that QuanTA consistently outperforms LoRA and often surpasses full fine-tuning, particularly on the MAWPS and SVAMP datasets, while using a significantly smaller number of parameters.
> <details>
> <summary>read the caption</summary>
> Table 4: Benchmark on various arithmetic reasoning tasks. GPT-3.5 (labeled with *) results are taken from [54].
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_17_1.jpg)
> 🔼 This table lists all the datasets used in the QuanTA paper, specifying the dataset name, the task it was used for (reading comprehension, commonsense reasoning, or arithmetic reasoning), and the number of training, validation, and test samples for each.  The table also indicates the evaluation metric (F₁-score or accuracy) and the type of answer expected (phrase, yes/no, option, or number).
> <details>
> <summary>read the caption</summary>
> Table D.1: List of datasets used in this work.
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_20_1.jpg)
> 🔼 This table compares the performance of several parameter-efficient fine-tuning (PEFT) methods on the DROP dataset, using different sizes of the LLaMA2 model as a base.  It shows the number of trainable parameters used by each method (as a percentage of the total parameters), and the resulting F1 score. The methods compared include full fine-tuning, series adapters, parallel adapters, LoRA with different ranks, and QuanTA with different configurations. The table highlights QuanTA's ability to achieve high F1 scores with significantly fewer trainable parameters compared to other methods.
> <details>
> <summary>read the caption</summary>
> Table 2: Benchmark of various fine-tuning methods on the DROP dataset using LLaMA2 7-70 billion parameter models as the base model. In each case, we report the average of F₁ score over 2-4 experiments with different random seeds.
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_21_1.jpg)
> 🔼 This table compares different parameter-efficient fine-tuning (PEFT) methods on the DROP dataset using LLaMA2 models with varying parameter counts (7B and 70B).  The methods include full fine-tuning (FT), series and parallel adapters, LoRA with different ranks, and QuanTA with different configurations.  The performance metric is the F1 score, averaged across multiple experiments with different random seeds to account for variability.
> <details>
> <summary>read the caption</summary>
> Table 2: Benchmark of various fine-tuning methods on the DROP dataset using LLaMA2 7-70 billion parameter models as the base model. In each case, we report the average of F₁ score over 2-4 experiments with different random seeds.
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_22_1.jpg)
> 🔼 This table lists the hyperparameters used in the experiments for the DROP dataset.  It shows the values used for each of the different fine-tuning methods (Full Fine-tuning (FT), Series Adapters, Parallel Adapters, LoRA, and QuanTA).  Curly brackets indicate the range of hyperparameters tested during optimization, while the underscored values are the ones finally selected. Square brackets show hyperparameter variations used in different experiments reported in the main paper. The table provides a detailed breakdown of the settings used for each method, aiding in reproducibility and understanding of the experimental setup.
> <details>
> <summary>read the caption</summary>
> Table E.2: Hyperparameters used for DROP dataset for various fine-tuning methods. Curly brackets include the hyperparameter values tested during hyperparameter optimization, with the actual hyperparameter(s) underscored. Square brackets include hyperparameter values for different experiments conducted in the main paper.
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_23_1.jpg)
> 🔼 This table compares the performance of different parameter-efficient fine-tuning (PEFT) methods on the DROP dataset, using LLaMA2 models with varying parameter counts.  It shows the F1 score achieved by each method, along with the percentage of parameters used relative to full fine-tuning.  The goal is to demonstrate QuanTA's efficiency and effectiveness compared to other techniques like LoRA and adapter-based methods.
> <details>
> <summary>read the caption</summary>
> Table 2: Benchmark of various fine-tuning methods on the DROP dataset using LLaMA2 7-70 billion parameter models as the base model. In each case, we report the average of F₁ score over 2-4 experiments with different random seeds.
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_23_2.jpg)
> 🔼 This table compares the performance of different parameter-efficient fine-tuning (PEFT) methods on various commonsense reasoning tasks.  The results are shown as accuracy scores for several benchmark datasets (BoolQ, PIQA, SIQA, Hellaswag, Winograd Schema Challenge, ARC-e, ARC-c, and OBQA). The table highlights the performance of QuanTA against other PEFT methods such as full fine-tuning (FT), LoRA, DORA, and adapter-based methods (Series and Parallel).  The '#Params (%) column shows the percentage of parameters trained for each method relative to the full fine-tuning approach.  Some results are sourced from external studies [54, 20].
> <details>
> <summary>read the caption</summary>
> Table 3: Benchmark on various commonsense reasoning tasks. All results of models and PEFT methods labeled with “*” are from [54], and results with “†” are from [20].
> </details>

![](https://ai-paper-reviewer.com/EfpZNpkrm2/tables_23_3.jpg)
> 🔼 This table presents the results of benchmarking various parameter-efficient fine-tuning (PEFT) methods on five natural language understanding tasks from the GLUE benchmark, using the RoBERTa model as the base.  It compares the performance of LoRA and QuanTA (the proposed method) in terms of accuracy on SST-2, MRPC, CoLA, RTE, and STS-B tasks.  The table highlights the number of trainable parameters used by each method as a percentage of the total model parameters.
> <details>
> <summary>read the caption</summary>
> Table F.7: Benchmark on five natural language understanding tasks using RoBERTa model as the base model.
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/EfpZNpkrm2/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}