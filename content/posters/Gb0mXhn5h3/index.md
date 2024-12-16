---
title: "MiSO: Optimizing brain stimulation to create neural activity states"
summary: "MiSO: a novel closed-loop brain stimulation framework optimizes stimulation parameters to achieve desired neural population activity states, overcoming limitations of current methods by merging data a..."
categories: ["AI Generated", ]
tags: ["Machine Learning", "Deep Learning", "🏢 Carnegie Mellon University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} Gb0mXhn5h3 {{< /keyword >}}
{{< keyword icon="writer" >}} Yuki Minai et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=Gb0mXhn5h3" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/Gb0mXhn5h3" target="_self" >}}
↗ Hugging Face
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/Gb0mXhn5h3/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Brain stimulation holds immense therapeutic potential, but efficiently searching the vast space of stimulation parameters remains challenging. Existing closed-loop methods often require extensive data collection within a single session, while inherent variability in neural activity across sessions further complicates the process.  This necessitates the development of novel approaches that can effectively manage the large parameter space and address data variability.

To tackle these issues, the paper introduces MiSO (MicroStimulation Optimization), a novel closed-loop brain stimulation framework.  **MiSO integrates three key components**: a neural activity alignment method to combine data across sessions, a CNN model to predict brain responses to untested parameters, and an online optimization algorithm to adaptively adjust stimulation parameters.  **Experiments using electrical microstimulation in non-human primates demonstrate MiSO's success in navigating a significantly larger parameter space than previous approaches, effectively driving neural activity towards specified states and achieving novel activity patterns.**

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} MiSO efficiently searches vast stimulation parameter spaces to precisely control neural population activity. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} MiSO's novel data merging technique overcomes temporal variability issues in brain stimulation studies, improving the efficiency of closed-loop optimization. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} MiSO's CNN predictive model accurately forecasts brain responses to untested stimulation patterns, increasing the precision and clinical applicability of neuromodulation technologies. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in neuromodulation and brain-computer interfaces.  **It introduces MiSO, a novel closed-loop framework that allows efficient exploration of large stimulation parameter spaces.** This addresses a major limitation in current brain stimulation techniques and opens new avenues for developing more effective therapies for brain disorders and advancing brain-computer interface technology.  **MiSO's ability to merge data across sessions and its use of a CNN for prediction are significant contributions** that warrant attention from the broader research community.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/Gb0mXhn5h3/figures_2_1.jpg)

> 🔼 This figure illustrates the experimental setup and workflow of the MiSO closed-loop stimulation framework. Panel A shows the overall goal of MiSO, which is to find the optimal stimulation parameters to achieve a target neural activity state. Panel B details the experimental setup, including the use of a multi-electrode array implanted in the prefrontal cortex (PFC) of a non-human primate, and shows how the spiking activity is recorded and processed.  Panel C outlines the four steps of the MiSO closed-loop stimulation framework: (1) choosing stimulation parameters, (2) performing microstimulation, (3) analyzing the induced neural activity, and (4) updating predictions based on the analysis. This iterative process continues until the desired neural activity state is reached. 
> <details>
> <summary>read the caption</summary>
> Figure 1: Experimental paradigm and MiSO closed-loop framework. (A) MiSO's goal is to optimize brain stimulation parameter configurations to create specified neural population activity states. (B) Experimental setup. Top: Spiking activity was recorded from a multi-electrode array implanted in PFC. During fixation, uStim was applied for 150ms (orange bar) to induce a specified neural population activity state in the post uStim period (pink bar). Bottom: The uStim response was evaluated within a low-d latent space (e.g., 2D) identified from high-d multi-electrode spiking activity. (C) Closed-loop stimulation framework. Each MiSO iteration involves four steps (Section 2.5).
> </details>





![](https://ai-paper-reviewer.com/Gb0mXhn5h3/tables_13_1.jpg)

> 🔼 This table summarizes the methods used to assess the closed-loop performance. The methods are categorized by the prediction model used to obtain the uStim response predictions to initialize the closed-loop optimization; the training data used to train the prediction model; the uStim patterns (single or double electrodes) used in both training data and the closed-loop optimization; and the online algorithm used to choose the next uStim pattern.  All methods using the epsilon greedy algorithm also perform the closed-loop updates of the predicted responses.
> <details>
> <summary>read the caption</summary>
> Table S4 Closed-loop performance assessment methods
> </details>





### In-depth insights


#### MiSO Overview
MiSO, or MicroStimulation Optimization, is a novel closed-loop brain stimulation framework designed to efficiently navigate the vast parameter space inherent in neuromodulation.  Its core functionality revolves around **adaptively updating stimulation parameters** based on a predictive model of neural activity.  This model is trained on merged data from multiple experimental sessions, enabling the effective use of a substantially larger parameter space than previously feasible.  The **alignment of neural activity across sessions**, a crucial element, allows for effective data aggregation despite inherent temporal variations in neural responses.  **Online optimization**, typically employing epsilon-greedy methods, ensures the system quickly converges toward desired neural population activity states, increasing the clinical viability of such technologies.  MiSO's integrated system of neural alignment, predictive modeling (e.g., using CNNs), and online parameter refinement represents a significant advancement in the pursuit of targeted neuromodulation.

#### Latent Space Alignment
Latent space alignment is a crucial technique for integrating data from multiple experimental sessions in neuroscience research.  Its core principle involves **transforming high-dimensional neural activity data into a lower-dimensional latent space**, where each dimension represents a significant underlying pattern of neural activity. This dimensionality reduction is often achieved through techniques like factor analysis.  The power of latent space alignment lies in its ability to **account for session-to-session variations in neural recordings**, which might stem from changes in brain state or recording conditions. By aligning these latent spaces across sessions, the method effectively merges data from different recordings, greatly enhancing statistical power and allowing researchers to study larger datasets. **Accurate alignment is paramount**, as misalignment can lead to erroneous conclusions.  Therefore, robust alignment methods are essential, often involving optimization procedures to find the best possible alignment between latent spaces. This technique is particularly valuable in closed-loop brain stimulation studies, where the goal is to find stimulation parameters leading to desired neural activity patterns.  By combining latent space alignment with machine learning models, researchers can predict how different stimulation parameters will affect the latent space representation of neural activity, leading to more precise and efficient control.

#### CNN Model
The paper leverages a Convolutional Neural Network (CNN) to predict neural responses to various brain stimulation patterns.  **The CNN's architecture is designed to capture the spatial relationships inherent in the stimulation and response data**, which is crucial given the spatial layout of electrodes on a multi-electrode array.  The choice of a CNN is motivated by its ability to effectively model spatial dependencies, outperforming simpler models like Multi-Layer Perceptrons (MLPs) and Gaussian Processes (GPs) in this context.  **The CNN is trained on merged data from multiple experimental sessions**, which is key to overcoming the limited number of samples obtainable in any single session. This data merging is made possible through a latent space alignment technique, ensuring consistent representations of neural activity across sessions. The CNN's predictions then guide the adaptive online optimization algorithm, allowing for efficient exploration of a vast stimulation parameter space and improving the accuracy and viability of closed-loop stimulation. **The results demonstrate the CNN's effectiveness in predicting neural activity and its crucial role in MiSO's success in driving neural populations toward desired states.**

#### Closed-Loop Optimization
The study's closed-loop optimization strategy is a core element, iteratively refining brain stimulation parameters to achieve desired neural activity states.  It leverages **epsilon-greedy exploration**, balancing exploitation of current best parameters with exploration of the vast parameter space. The adaptive update of stimulation predictions, using a learning rate that **dynamically adjusts**, further enhances the system's responsiveness to ongoing neural activity.  This closed-loop process hinges on **real-time latent space alignment**, ensuring consistent comparison of neural activity across sessions.  The algorithm cleverly integrates the CNN's predicted responses, making it a data-driven, model-based approach.  **Combining predictions and online feedback creates an efficient search process**, even within exceptionally large stimulation parameter spaces, addressing a critical limitation in neuromodulation studies.

#### Clinical Viability
The concept of "Clinical Viability" in the context of brain stimulation technologies, as discussed in the research paper, centers on the **feasibility of translating laboratory advancements into practical clinical applications.**  The paper emphasizes that optimizing brain stimulation requires exploring vast parameter spaces, which were previously considered too large for practical experimentation.  The proposed MiSO framework directly addresses this limitation, **enabling the use of significantly expanded parameter spaces** which ultimately increases the **potential for more effective and targeted neuromodulation therapies.**  By improving the efficiency of the stimulation search process and allowing for the identification of previously inaccessible stimulation patterns, MiSO's success demonstrates a crucial step towards **making advanced brain stimulation technologies clinically viable** for the treatment of neurological disorders.  This advancement has the potential to greatly enhance the precision and effectiveness of therapies, leading to improved patient outcomes and paving the way for a new era of targeted neuromodulation.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/Gb0mXhn5h3/figures_6_1.jpg)

> 🔼 This figure shows the results of a closed-loop experiment using MiSO initialized with merged samples. Panel A displays an example session, showing the latent activity over trials for three methods: No uStim, Random uStim, and MiSO. The bottom part of Panel A shows the electrodes selected for stimulation. Panel B presents the mean L1 error across five sessions, comparing the three methods against a No uStim baseline.
> <details>
> <summary>read the caption</summary>
> Figure 2: Closed-loop performance in a non-human primate of MiSO initialized using merged samples. (A) An example closed-loop experimental session. The top two panels show smoothed (for visualization) FA latent activity in the two target dimensions. Trials for all three methods were interleaved in the session. The bottom two panels show the electrode selected for uStim on each trial by 'Random uStim' and “MiSO with single elec., sample avg.'. (B) Mean L1 error relative to the 'No uStim' baseline across 5 closed-loop experimental sessions. Error bars indicate standard error across sessions.
> </details>



![](https://ai-paper-reviewer.com/Gb0mXhn5h3/figures_7_1.jpg)

> 🔼 This figure demonstrates the spatial smoothness in uStim responses.  Panels (B) and (D) show the correlation between the physical distance of stimulated electrodes and the similarity of their responses for single and double electrode stimulation respectively. Panels (C) and (E) compare the prediction errors of three different machine learning models (MLP, GP, CNN) for predicting uStim responses to held-out patterns, showcasing that the CNN model leverages spatial smoothness for better generalization.
> <details>
> <summary>read the caption</summary>
> Figure 3: Leveraging spatial smoothness to predict uStim responses to untested uStim patterns. (A) uStim pattern difference determined by the physical location(s) of the stimulating electrode(s) on the array (illustrated here for single-electrode patterns). (B) Relationship between uStim pattern difference (horizontal axis, L1 distance, Section S5) and uStim response difference (vertical axis, latent activity difference along FA dim1) when stimulating using single electrodes. A positive correlation (r) implies that stimulating using nearby electrodes tended to induce similar responses. (C) uStim response prediction error as a function of the percentage of held-out uStim patterns during training. Error bars indicate standard error across test datasets. (D, E) Same format as (B) and (C) respectively, but for stimulation using double-electrode patterns. In (E), we experimentally tested 45% of all possible double-electrode patterns (9 sessions, 3301 trials).
> </details>



![](https://ai-paper-reviewer.com/Gb0mXhn5h3/figures_8_1.jpg)

> 🔼 This figure shows the results of closed-loop experiments using MiSO with a CNN model. Panel A shows the range of neural activity patterns achievable using single and double electrode microstimulation. Panel B shows an example closed-loop session demonstrating MiSO’s ability to drive neural activity towards a target state.  Panel C shows the average error for each method across sessions relative to a no stimulation baseline.  MiSO with double electrode stimulation performs better than other methods.
> <details>
> <summary>read the caption</summary>
> Figure 4: Closed-loop performance in a non-human primate of MiSO initialized using a CNN. (A) Range of activity patterns achievable by single (pink) and double (blue) electrode uStim patterns, as predicted by the CNN. Dashed region: area reachable exclusively by double-electrode uStim. Yellow star: target state used in (B). (B) Example closed-loop experimental session. Same format as Fig. 2A. (C) Mean L1 error of different methods relative to “No uStim” baseline, across 3 closed-loop sessions. Error bars indicate standard error across sessions.
> </details>



![](https://ai-paper-reviewer.com/Gb0mXhn5h3/figures_15_1.jpg)

> 🔼 This figure demonstrates how the proposed latent space alignment method improves the consistency of uStim responses across different experimental sessions.  Panel A shows raw firing rates for each electrode across 5 sessions, illustrating significant variability. Panel B shows the same data after latent space alignment, revealing greater consistency across sessions.  Panel C quantifies this improved consistency, showing that the response maps are much closer in the aligned latent space than in the original firing rate space. This consistency is critical for reliably merging data from different sessions to improve the performance of the MiSO closed-loop stimulation system.
> <details>
> <summary>read the caption</summary>
> Figure S1: uStim response consistency across sessions. (A) Response maps of uStim-induced mean firing rates. Each panel shows the mean firing rate, averaged across the entire array, induced by stimulating each electrode individually. For example, the color of the top left cell in a given response map indicates the mean firing rate across the array induced by stimulating this particular electrode. Each column corresponds to a different session. (B) Response maps of uStim-induced mean latent activity across trials. Each panel shows the average latent activity induced by stimulating each electrode individually. The latent spaces have been aligned across sessions. Stimulation-response samples from the five sessions shown here are used to compute the sample average-based predictions in Fig. 2. (C) Normalized distance of response maps from session 1. The mean firing rates and latent activity are normalized across sessions using a min-max transformation to align their scales. The uStim response is more consistent across sessions in the aligned latent space than in the raw firing rate space.
> </details>



![](https://ai-paper-reviewer.com/Gb0mXhn5h3/figures_15_2.jpg)

> 🔼 This figure shows the relationship between the spatial similarity of uStim patterns and the similarity of their responses. The left panel shows the relationship for single-electrode stimulation, and the right panel shows the relationship for double-electrode stimulation. The x-axis represents the spatial distance between the uStim patterns, and the y-axis represents the difference in their responses. The correlation coefficients (r) are shown for both panels, indicating a positive relationship between spatial similarity and response similarity.
> <details>
> <summary>read the caption</summary>
> Figure S2: Relationship between uStim pattern spatial similarity and response similarity along the second target dimension. Same format as Fig. 3B and Fig. 3D, which were based on the first target dimension. Left, single-electrode uStim, right, double-electrode uStim.
> </details>



![](https://ai-paper-reviewer.com/Gb0mXhn5h3/figures_16_1.jpg)

> 🔼 This figure demonstrates the spatial smoothness in uStim responses. Panel A shows how uStim pattern difference is calculated based on electrode location. Panels B and D show the correlation between uStim pattern difference and uStim response difference for single and double electrode stimulation respectively. Panels C and E show the uStim response prediction error with different percentages of held-out uStim patterns for training.
> <details>
> <summary>read the caption</summary>
> Figure 3: Leveraging spatial smoothness to predict uStim responses to untested uStim patterns. (A) uStim pattern difference determined by the physical location(s) of the stimulating electrode(s) on the array (illustrated here for single-electrode patterns). (B) Relationship between uStim pattern difference (horizontal axis, L1 distance, Section S5) and uStim response difference (vertical axis, latent activity difference along FA dim1) when stimulating using single electrodes. A positive correlation (r) implies that stimulating using nearby electrodes tended to induce similar responses. (C) uStim response prediction error as a function of the percentage of held-out uStim patterns during training. Error bars indicate standard error across test datasets. (D, E) Same format as (B) and (C) respectively, but for stimulation using double-electrode patterns. In (E), we experimentally tested 45% of all possible double-electrode patterns (9 sessions, 3301 trials).
> </details>



</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/Gb0mXhn5h3/tables_14_1.jpg)
> 🔼 This table summarizes the number of experimental sessions used in each of the paper's results.  It breaks down the sessions into three categories: reference sessions (no uStim), training sessions (used to train the CNN model), and closed-loop test sessions (during which MiSO was used).  The table shows how many sessions were used for each figure or result presented in the paper, clarifying which sessions contributed to the findings.  Different numbers of sessions are utilized depending on the specific experiment (single vs. double electrode stimulation) and the specific analysis performed.
> <details>
> <summary>read the caption</summary>
> Table S6 Summary of experimental sessions
> </details>

![](https://ai-paper-reviewer.com/Gb0mXhn5h3/tables_14_2.jpg)
> 🔼 This table summarizes the methods used for evaluating the performance of closed-loop optimization in the study.  It specifies the prediction model (method used to obtain predictions for initializing closed-loop optimization), training data (stimulation-response samples used to train the prediction model), uStim patterns (number of electrodes used for stimulation, in training and optimization), and the online algorithm (method for selecting the next uStim pattern) for each of the methods compared.  Each method is described with respect to its use of prediction model, training data, uStim pattern, and online algorithm.
> <details>
> <summary>read the caption</summary>
> S4 Closed-loop performance assessment methods
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/Gb0mXhn5h3/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}