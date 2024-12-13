---
title: "Continual Learning in the Frequency Domain"
summary: "Boost continual learning efficiency with CLFD: a novel frequency domain approach that improves accuracy by up to 6.83% and slashes training time by 2.6x on edge devices!"
categories: []
tags: ["Machine Learning", "Continual Learning", "🏢 Institute of Computing Technology, Chinese Academy of Sciences",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} XgAzCLsJAq {{< /keyword >}}
{{< keyword icon="writer" >}} RuiQi Liu et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=XgAzCLsJAq" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/94751" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=XgAzCLsJAq&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/XgAzCLsJAq/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Continual learning (CL) aims to enable machines to learn new tasks without forgetting previously learned ones.  However, existing rehearsal-based CL methods often struggle with efficiency, especially on resource-limited devices due to the need to store and frequently access large amounts of data from previous tasks. This limitation hinders broader adoption of CL in real-world applications. 

To address this, the researchers propose CLFD (Continual Learning in the Frequency Domain). **CLFD uses wavelet transforms to reduce the size of input data**, making it more efficient for both memory and processing. **It also selectively uses frequency components based on their similarity across tasks, improving performance.** Experiments show that CLFD significantly enhances existing CL methods, boosting accuracy by up to 6.83% and reducing training time by 2.6x on edge devices. This demonstrates CLFD's practical advantages and potential for real-world CL applications.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} CLFD, a novel framework for continual learning, utilizes frequency domain features to improve performance and efficiency. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} CLFD significantly improves the accuracy and reduces training time of existing continual learning methods on edge devices. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The framework is shown to be effective across various benchmark datasets and seamlessly integrates with multiple rehearsal-based continual learning methods. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it addresses the critical issue of training efficiency in continual learning (CL), particularly for resource-constrained edge devices.  **CLFD offers a novel approach using frequency domain features which significantly improves accuracy and reduces training time of existing CL methods.** This work is relevant to the growing trend of deploying machine learning models on edge devices and opens up new avenues of research in combining frequency domain processing with CL techniques.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_1_1.jpg)

> This figure provides a high-level overview of the proposed Continual Learning in the Frequency Domain (CLFD) framework.  The left panel shows the architecture, highlighting the two key components: the Frequency Domain Feature Encoder (FFE) which uses Discrete Wavelet Transform (DWT) to convert images to frequency domain, and the Class-aware Frequency Domain Feature Selection (CFFS) which selects relevant frequency features for classification. The right panel presents a comparison of the accuracy and training time of CLFD versus a baseline method (ER) on the NVIDIA Jetson Orin NX, showcasing CLFD's improved performance and efficiency on an edge device.





![](https://ai-paper-reviewer.com/XgAzCLsJAq/tables_5_1.jpg)

> This table compares the performance of several continual learning (CL) methods, including the proposed CLFD framework, across two datasets (S-CIFAR-10 and S-Tiny-ImageNet) under different buffer sizes.  For each method, it reports the average accuracy achieved on Class-IL and Task-IL settings, along with the peak memory usage. The table highlights that CLFD consistently improves accuracy while significantly reducing memory consumption, showcasing its efficiency benefits. 





### In-depth insights


#### Frequency Domain CL
Frequency domain continual learning (CL) offers a novel approach to address the challenges of catastrophic forgetting in CL systems.  By leveraging the human visual system's varying sensitivity to different frequency components, **CLFD reduces the dimensionality of input features**, improving efficiency.  The use of wavelet transforms to map images into the frequency domain allows for **preservation of both spatial and frequency information**, unlike DCT-based methods, facilitating data augmentation.  **Selective utilization of frequency features based on class-wise similarity** balances feature reuse and interference, further enhancing performance and mitigating catastrophic forgetting. This approach demonstrates the potential of using frequency domain analysis to create more efficient and robust CL systems, particularly in resource-constrained environments like edge devices.  **The success of CLFD hinges on the effective integration of wavelet transforms and class-aware feature selection,** which warrants further investigation into various wavelet types and similarity metrics.

#### Wavelet Transform Use
The research leverages the **discrete wavelet transform (DWT)** to map input images into the frequency domain, a crucial step in their proposed Continual Learning in the Frequency Domain (CLFD) framework.  Unlike the discrete cosine transform (DCT), which results in a complete loss of spatial information, the DWT effectively preserves both frequency and spatial domain features, enabling data augmentation techniques crucial for successful continual learning.  **This preservation of spatial information is key**, preventing the limitations seen in DCT-based approaches that hinder the use of data augmentation strategies.  The choice of Haar wavelet within the DWT is justified by its computational efficiency, making it suitable for resource-constrained edge devices, a central design goal of the CLFD framework.  **The multi-resolution analysis inherent in the DWT allows the model to capture both low-frequency components** (representing global information) and high-frequency components (representing local details), optimizing feature representation and reducing the input feature map size. This size reduction ultimately contributes to improved efficiency and memory usage.

#### CLFD Framework
The CLFD framework, designed for continual learning, leverages the **frequency domain** to enhance training efficiency and mitigate catastrophic forgetting.  It cleverly utilizes wavelet transforms to reduce input feature map size, decreasing computational demands on edge devices. **Class-aware Frequency Domain Feature Selection** further refines the process, dynamically choosing relevant frequency features for each class across tasks, balancing reusability and interference. This approach avoids excessive parameter additions compared to traditional methods, and its seamless integration with rehearsal-based techniques makes it particularly effective.  **Experimental results demonstrate the framework's strong performance enhancements** in accuracy and training time reduction, particularly on edge devices. The use of frequency domain analysis in CL is novel, aligning with the human visual system's inherent frequency sensitivities and offering a promising direction for future research in resource-constrained continual learning scenarios.

#### Edge Device Efficiency
The research paper explores enhancing the efficiency of continual learning (CL) on edge devices.  A key contribution is the introduction of a novel framework, which leverages frequency domain features to significantly reduce computational demands. **By processing input images in the frequency domain using wavelet transforms**, the framework efficiently shrinks the input feature maps, leading to decreased training time and memory usage.  **The selective utilization of output features based on frequency domain similarity** further improves efficiency and prevents interference between tasks. The effectiveness of this approach is validated through experiments, showcasing improved accuracy and substantially faster training times on edge devices compared to state-of-the-art CL methods.  **This demonstrates the practical feasibility of deploying advanced CL models on resource-constrained hardware** and underscores the framework's potential for real-world applications.

#### Future Research
Future research directions stemming from this work could explore several promising avenues. **Extending CLFD to other modalities beyond images**, such as audio or text, would broaden its applicability and demonstrate its generalizability.  Investigating the impact of different wavelet transforms and exploring alternative frequency decomposition methods could further optimize CLFD's performance and efficiency.  A thorough analysis of the trade-offs between accuracy and memory/compute efficiency at various buffer sizes is warranted.  **Developing theoretical frameworks to explain CLFD's effectiveness** would enhance its understanding and lead to more principled designs. Finally, **integrating CLFD with other continual learning techniques, such as regularization or architecture-based methods**, could potentially unlock synergistic benefits and create even more robust and efficient continual learning systems.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_3_1.jpg)

> This figure illustrates the workflow of the Continual Learning in the Frequency Domain (CLFD) framework.  It shows how an input RGB image is first transformed into the wavelet domain using a Frequency Domain Feature Encoder (FFE).  The FFE then produces three feature maps representing low, high, and global frequency components.  These maps are then fed into a feature extractor, which also uses a Class-aware Frequency Domain Feature Selection (CFFS) component.  CFFS selectively filters the features based on class similarity, prioritizing features that balance reusability and reduce interference among tasks. Finally, the selected features are sent to a classifier for prediction.  The diagram emphasizes the role of the reservoir, which stores samples from previous tasks for rehearsal-based learning.


![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_4_1.jpg)

> This figure illustrates how the Discrete Wavelet Transform (DWT) is used within the Frequency Domain Feature Encoder (FFE) component of the CLFD framework.  The input image undergoes DWT, resulting in four sub-bands: low-frequency (X<sub>ll</sub>), high-frequency components (X<sub>lh</sub>, X<sub>hl</sub>, X<sub>hh</sub>). Each sub-band is then processed by a 1x1 convolution to extract low-frequency features, global features, and high-frequency features, respectively. These features are then combined to form the final feature map used in the subsequent layers.


![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_7_1.jpg)

> The figure shows the architecture of the proposed Continual Learning in the Frequency Domain (CLFD) framework.  The left panel displays the two main components: the Frequency Domain Feature Encoder (FFE) and the Class-aware Frequency Domain Feature Selection (CFFS). The right panel presents a bar chart comparing the performance of CLFD against a baseline method (ER) on the NVIDIA Jetson Orin NX edge device, using the split CIFAR-10 dataset.  The chart highlights CLFD's superior performance in terms of both accuracy and training time.


![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_8_1.jpg)

> This figure shows the activation counts of frequency domain features extracted by the feature extractor on the test set of S-CIFAR-10. Each row represents a class from the dataset and each column represents a frequency domain feature. The color intensity represents the activation count, with darker colors indicating higher counts.  The figure shows that certain features are more strongly activated for some classes than others.  The organization of the figure suggests that semantically similar classes (e.g., cat and dog) exhibit similar patterns of feature activation, while dissimilar classes (e.g., plane and truck) have distinct activation patterns. This visualization supports the paper's method of selecting frequency domain features for different classes to optimize performance.


![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_8_2.jpg)

> This figure visualizes the output of the Frequency Domain Feature Encoder (FFE) for two example images from the dataset. The leftmost column shows the original input images. The other columns display the encoded frequency domain features, namely low-frequency features, global features, and high-frequency components, demonstrating how the FFE transforms the input images into different frequency representations.  This process is a crucial step in the CLFD framework for reducing input size and improving computational efficiency.


![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_14_1.jpg)

> The figure shows the architecture of the proposed Continual Learning in the Frequency Domain (CLFD) framework, which consists of a Frequency Domain Feature Encoder (FFE) and a Class-aware Frequency Domain Feature Selection (CFFS).  The right side displays a comparison of CLFD's performance against the ER baseline method on the NVIDIA Jetson Orin NX edge device, highlighting improvements in both accuracy and training efficiency on the split CIFAR-10 dataset.


![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_14_2.jpg)

> This figure compares the training time and accuracy of several continual learning methods on the NVIDIA Jetson Orin NX edge device using the S-CIFAR-10 dataset.  The results show that the proposed CLFD framework consistently improves both the accuracy and training efficiency when integrated with different rehearsal-based continual learning methods compared to baselines..  A buffer size of 125 was used.


![](https://ai-paper-reviewer.com/XgAzCLsJAq/figures_15_1.jpg)

> The figure shows the training time and accuracy of various continual learning methods on the Nvidia Jetson Orin NX edge device using the S-CIFAR-10 dataset.  The results demonstrate that CLFD significantly improves both the training efficiency and accuracy when compared to other methods.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/XgAzCLsJAq/tables_6_1.jpg)
> This table compares the performance of various continual learning (CL) methods, including several baselines and the proposed CLFD framework, across two datasets (S-CIFAR-10 and S-Tiny-ImageNet) under different buffer sizes.  The metrics presented include Class-IL (Class Incremental Learning) and Task-IL (Task Incremental Learning) accuracy, along with peak memory usage.  The table highlights CLFD's ability to improve accuracy while simultaneously reducing memory consumption compared to existing methods.  Shaded rows indicate results obtained using the CLFD framework.

![](https://ai-paper-reviewer.com/XgAzCLsJAq/tables_7_1.jpg)
> This table presents the results of an ablation study conducted to assess the impact of each component of the CLFD-ER method on the model's performance using the S-CIFAR-10 dataset.  It shows the Class-IL and Task-IL accuracy when either the Frequency Domain Feature Encoder (FFE), the Class-aware Frequency Domain Feature Selection (CFFS), or both are removed.  The results highlight the individual contribution of each component and the synergistic effect of combining them.

![](https://ai-paper-reviewer.com/XgAzCLsJAq/tables_8_1.jpg)
> This table compares the performance of various continual learning (CL) methods, including several baseline methods and the proposed CLFD framework, across two datasets (S-CIFAR-10 and S-Tiny-ImageNet) under Class-IL and Task-IL settings.  For each method, the table shows the average accuracy (Class-IL and Task-IL), and peak memory usage for buffer sizes of 50 and 125.  The results demonstrate that CLFD consistently improves accuracy while significantly reducing memory usage compared to the baseline methods.

![](https://ai-paper-reviewer.com/XgAzCLsJAq/tables_13_1.jpg)
> This table compares the performance of different continual learning (CL) methods, including the proposed CLFD framework, across two benchmark datasets (S-CIFAR-10 and S-Tiny-ImageNet).  For each method, the table shows the average accuracy achieved on both Class-IL and Task-IL scenarios under different buffer sizes (50 and 125).  The memory usage (Mem) is also reported.  The table highlights that CLFD consistently improves the accuracy while significantly reducing peak memory consumption.

![](https://ai-paper-reviewer.com/XgAzCLsJAq/tables_15_1.jpg)
> This table compares the performance of Continual Learning in the Frequency Domain (CLFD) against several other continual learning methods across two datasets (S-CIFAR-10 and S-Tiny-ImageNet).  It shows the average accuracy and peak memory usage for different buffer sizes.  The table demonstrates that CLFD consistently improves accuracy while reducing memory consumption, particularly when combined with state-of-the-art continual learning baselines.  Higher accuracy numbers are in bold.

![](https://ai-paper-reviewer.com/XgAzCLsJAq/tables_16_1.jpg)
> This table compares the performance of Continual Learning in the Frequency Domain (CLFD) against other state-of-the-art (SOTA) continual learning methods across two datasets (S-CIFAR-10 and S-Tiny-ImageNet).  The results are shown for both Class-IL (Class Incremental Learning) and Task-IL (Task Incremental Learning) settings.  For each method, the table shows the average accuracy, the memory usage, and for the CLFD methods, it highlights the improvement over the baseline methods.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/XgAzCLsJAq/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}