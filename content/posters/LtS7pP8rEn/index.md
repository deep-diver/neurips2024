---
title: "Posture-Informed Muscular Force Learning for Robust Hand Pressure Estimation"
summary: "PiMForce: Robust hand pressure estimation using 3D hand posture and sEMG!"
categories: []
tags: ["AI Applications", "Human-AI Interaction", "🏢 Graduate School of Culture Technology, KAIST",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} LtS7pP8rEn {{< /keyword >}}
{{< keyword icon="writer" >}} Kyungjin Seo et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=LtS7pP8rEn" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95565" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=LtS7pP8rEn&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/LtS7pP8rEn/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Hand pressure estimation is crucial for many applications like virtual reality and prosthetics but existing methods (vision-based or sEMG-only) are limited. Vision-based methods fail with occlusions or complex grasps. sEMG-only methods struggle with various postures that create similar sEMG patterns.  This creates a need for robust hand pressure estimation methods.

PiMForce combines 3D hand posture data with sEMG signals to create a more accurate estimation. A new multimodal dataset is presented. Experiments show PiMForce significantly outperforms existing methods, demonstrating improved accuracy and robustness across diverse interactions. This opens avenues for advanced haptic interfaces and prosthetic development.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} PiMForce, a novel framework, enhances hand pressure estimation by combining 3D hand posture with sEMG signals. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A multimodal dataset of hand posture, sEMG signals, and pressure across diverse hand postures is created. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} PiMForce substantially outperforms existing sEMG-based and vision-based methods in estimating hand pressure. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in human-computer interaction, robotics, and biomedical engineering.  It **addresses the limitations of existing hand pressure estimation methods** by proposing a novel multimodal approach. The findings pave the way for improved haptic interfaces, advanced prosthetics, and more intuitive human-machine interactions. Its open-source code and data also facilitate further research and development in this rapidly evolving field.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_1_1.jpg)

> This figure illustrates the PiMForce framework, showing the data acquisition and model training/testing phases.  During training, a pressure glove, sEMG armband, and hand tracking module provide multimodal data to a deep neural network. This network learns to estimate hand pressure from the combined input.  During testing, a camera and sEMG armband provide inputs to the trained model, allowing real-time pressure estimation on fingertips and palm regions without the pressure glove.





![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_2_1.jpg)

> This table compares the proposed dataset with other existing datasets used for hand pressure estimation research.  It provides details on the input modalities used (RGBD, RGB, Pose, sEMG), the number of frames and participants involved in data collection, the source of pressure data (inferred from pose, thermal imprint, pressure pad, custom-made device, pressure glove), whether it includes whole-hand or fingertip data, and finally whether the interactions involved natural hand-object interactions. This facilitates understanding the novelty and comprehensiveness of the proposed dataset, which is highlighted as multimodal and encompassing data from multiple aspects of natural hand interactions.





### In-depth insights


#### PiMForce Framework
The PiMForce framework presents a novel approach to hand pressure estimation by integrating **multimodal data** from sEMG signals and 3D hand pose information.  This fusion addresses limitations of traditional methods which rely solely on sEMG or vision.  PiMForce's strength lies in its ability to accurately estimate pressure across the entire hand, even under diverse and complex hand-object interactions. The use of 3D hand pose significantly improves the accuracy of pressure estimation from sEMG signals alone, mitigating the ambiguity inherent in muscle activation patterns.  **Real-time performance** is achieved by using readily available technology (off-the-shelf hand pose detector), making the framework suitable for various applications, especially in natural human-computer interactions. The framework also includes a novel data collection system, creating a **multimodal dataset** of unprecedented scale and quality, further advancing the field of hand pressure estimation.

#### Multimodal Dataset
A robust multimodal dataset is crucial for reliable hand pressure estimation.  **Combining pressure glove data, sEMG signals, and 3D hand pose information** offers a more comprehensive understanding of hand-object interactions than unimodal approaches.  The dataset's quality is paramount; precise synchronization between data modalities is essential, requiring careful system design and data processing.  **Diverse hand postures and grasp types** are necessary for generalizability, and a sufficient number of participants help ensure robustness.  The inclusion of hand-object interaction details adds valuable context.  **Data quality control, such as handling noise and outliers**, is critical.  Finally, the dataset's accessibility through open-source availability promotes wider adoption and validation of research, enhancing the impact of hand pressure estimation research.

#### Pressure Estimation
The research paper explores pressure estimation using a multimodal approach, **combining sEMG signals and 3D hand pose information**.  This innovative technique offers advantages over traditional vision-based methods by mitigating limitations associated with occlusions and limited viewpoints. The integration of sEMG provides a more robust understanding of muscle activation, and 3D hand pose data contextualizes pressure distribution across the hand. **The developed framework provides accurate real-time pressure estimation**, proving particularly effective in complex and natural hand-object interaction scenarios.  However, **limitations exist** concerning dependence on accurate hand pose detection and variations in performance across different hand postures.  Future work could explore alternative modalities and enhanced dataset diversity, particularly concerning demographic representation, to further improve accuracy and robustness. The multimodal dataset generated significantly contributes to the field, enabling superior pressure estimation models and advancing applications in human-computer interaction and rehabilitation.

#### Future Research
Future research directions stemming from this hand pressure estimation work could significantly benefit from **exploring alternative multimodal fusion techniques** to leverage the interplay between visual and tactile information more effectively.  The current model's reliance on off-the-shelf hand pose detectors presents a limitation; future work should investigate more robust and accurate pose estimation methods, potentially incorporating more sophisticated computer vision techniques. Expanding the dataset to include more diverse hand postures and a broader range of participants is crucial for improving generalization.  **Investigating high-density electromyography (HD-EMG)** offers the potential for significantly improved accuracy and robustness compared to the standard sEMG approach used in this research.  Finally, exploring the incorporation of both hand mesh and object mesh information could provide richer spatial context for pressure estimation, leading to enhanced accuracy and a more comprehensive understanding of hand-object interactions.

#### Method Limitations
A critical analysis of limitations in a research paper's methodology section requires a nuanced understanding of the proposed approach.  **Data limitations**, such as dataset size, representativeness, and potential biases, can significantly impact the generalizability of findings.  **Methodological limitations** might involve the choice of algorithms or models employed, as certain techniques may be better suited to specific tasks or datasets than others.  The extent of model training and validation also warrants careful consideration, as insufficient training could lead to underfitting and inaccurate results.  **Assumptions** underlying the methodology should be clearly stated and their potential impact on the reliability and validity of results should be acknowledged.  Finally, the paper must consider and address any **ethical concerns** that may arise from its methodology.  A thorough limitations section that acknowledges these shortcomings demonstrates the researchers' self-awareness, strengthens their credibility, and lays the groundwork for future research directions.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_4_1.jpg)

> This figure illustrates the architecture of the PiMForce model, which estimates hand pressure using both sEMG and 3D hand pose data.  The model consists of two branches: one for processing sEMG signals (using STFT, CNN encoder-decoder, and an FC layer) and another for processing 3D hand pose data (using a ResNet and FC layer).  The outputs from these branches are concatenated and fed into further FC layers to predict both the classification of pressure (presence/absence) and the regression of pressure value in different hand regions.  The training process uses a joint loss function combining classification and regression losses to optimize the model for accurate and comprehensive hand pressure estimation.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_8_1.jpg)

> This figure presents a quantitative analysis of the model's performance across different hand postures.  It uses three key metrics: Coefficient of Determination (R2), Normalized Root Mean Squared Error (NRMSE), and Accuracy. The x-axis shows various hand postures (including different types of presses and grasps), and the y-axis displays the values for each metric.  Error bars illustrate the standard error for each data point, indicating the variability in performance. The overall trend shows higher accuracy and lower NRMSE for simpler actions (like presses), while more complex grasps exhibit slightly lower accuracy.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_9_1.jpg)

> This figure presents a qualitative comparison between the proposed PiMForce model and the PressureVision++ model for hand pressure estimation.  Subfigure (a) shows a table comparing the results from both methods across various hand interactions. The PressureVision++ method struggles with hand occlusions, while the PiMForce model consistently predicts pressures accurately. Subfigure (b) shows a video demonstrating the real-time performance of PiMForce, handling various poses, pressures, and objects, showcasing its robustness.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_15_1.jpg)

> The figure shows a customized pressure glove used for data collection.  The glove incorporates a position tracking module (Manus Quantum Mocap Metaglove) and pressure sensors on fingertips and palm. The pressure sensors cover nine regions of the hand: five fingertips and four sections of the palm. This integrated system allows simultaneous collection of accurate hand pose information and hand pressure data.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_16_1.jpg)

> This figure shows the calibration and response curves for the Force-Sensing Resistor (FSR) pressure sensor used in the pressure glove.  (a) shows the experimental setup for calibration. (b) and (c) display the sensor's resistance and conductance as functions of the applied force.  Multiple trials are shown, illustrating the sensor's repeatability and stability.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_17_1.jpg)

> This figure shows the experimental setup for capturing multimodal data (hand pressure, 3D hand pose, and sEMG signals).  It highlights the key components: a customized pressure glove with embedded sensors, an 8-channel sEMG armband, and a markerless finger-tracking module.  The image also depicts a graphical representation of the collected data streams showing how hand pressure is distributed across nine different regions of the hand, the sEMG signals across 8 channels, and the 3D hand pose data which includes 20 joint angles and 21 joint coordinates.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_17_2.jpg)

> This figure shows the 22 different hand postures used in the multimodal dataset collection.  These postures are categorized into three types: Plane, Pinch, and Grasp.  Each category contains several variations of hand interactions, with clear visual representations for each. The labeling system in the figure also includes abbreviations to clearly identify each posture.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_18_1.jpg)

> This figure shows ten representative grasp postures used in the data collection process.  Each grasp is categorized by its type (power, intermediate, precision), and thumb position.  The color-coding on the hand diagrams indicates which pressure sensors were used in each grasp.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_19_1.jpg)

> The figure shows a computer screen displaying real-time data during a hand pressure data collection session.  The screen is divided into sections showing the hand posture (video and a 3D hand model), the pressure readings from the pressure glove sensors (represented as colored intensity on a diagram of a hand), and the EMG signals from the armband (as waveforms). A timer shows the elapsed time of the recording session. This interface provides visual feedback to the participant and the researcher during data collection, ensuring synchronized and accurate data acquisition.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_19_2.jpg)

> This figure shows a diagram of a hand skeleton model used in the paper for representing hand poses.  The image displays the different joints of the fingers and thumb, including the carpometacarpal (CMC) joint, metacarpophalangeal (MCP) joint, proximal interphalangeal (PIP) joint, and distal interphalangeal (DIP) joint.  This model is important because it helps the researchers' 3D hand pose feature extractor fhand process the 3D joint angle information for the model to estimate pressure.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_21_1.jpg)

> This figure visualizes the sEMG signal patterns obtained from eight channels during four different hand postures: I-Press, M-Press, TI-Pinch, and TM-Pinch.  The purpose is to demonstrate that similar sEMG patterns can be observed for different hand actions, highlighting the importance of incorporating 3D hand posture information to enhance the accuracy of hand pressure estimation.  Despite different hand actions that involve different fingers, there is significant similarity in EMG patterns, indicating that sEMG alone may not be sufficient for accurate and precise hand pressure estimation.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_25_1.jpg)

> This figure compares the ground truth pressure values with the pressure values predicted by the PressureVision++ model for the same posture. The figure shows that the PressureVision++ model struggles to accurately predict pressure in some scenarios.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_26_1.jpg)

> The figure shows a screenshot of the data collection interface used in the study.  It provides real-time feedback to the participant regarding their hand posture, the pressure exerted by their hand (captured by the pressure glove), and the electromyography (EMG) signals (captured by the EMG armband). The interface also features a timer to help track data acquisition sessions.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_27_1.jpg)

> This figure provides a qualitative comparison of hand pressure estimation results from five different methods: SEMG Only, 3D Hand Posture Only, SEMG + Hand Angles, SEMG + 3D Hand Posture (the proposed PiMForce method), and Ground Truth.  The image shows several different hand-object interaction types (I-Press, M-Press, R-Press, P-Press, IM-Press, and MR-Press). For each interaction type, the estimated pressure at different fingertip and palm locations is visualized with a color map for each of the five methods. The intensity of the color represents the pressure level, with darker shades indicating higher pressure. The rendered images of the 'Ours' (PiMForce) column show the 3D hand pose overlay, illustrating the system's ability to estimate hand pressure accurately and robustly across varied hand poses and interactions. The ground truth is shown as a reference point.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_28_1.jpg)

> This figure provides a visual comparison of hand pressure estimation results obtained using different methods: SEMG Only, 3D Hand Posture Only, SEMG + Hand Angles, SEMG + 3D Hand Posture, and the proposed PiMForce model.  Each row represents a different hand-object interaction type, displaying the predicted pressure distribution across the hand's fingertips and palm region using each method.  The intensity of the color in each node represents the pressure exerted in the corresponding region.  The 'Ground Truth' column depicts the actual hand pressure distribution measured by a pressure-sensitive glove, providing a benchmark for evaluating the accuracy of the other methods.  The 'Ours (Rendered)' column shows the hand pressure estimation from the PiMForce model.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_29_1.jpg)

> This figure shows a qualitative comparison of pressure estimation results between the proposed PiMForce model and the existing PressureVision++ model for the vision-aided hand on a test set. The comparison includes the ground truth pressure and the pressure estimation for each model (rendered). The hand-object interaction types vary, and for some of them the PressureVision++ model is 'Undetectable'. This visualization demonstrates the performance and robustness of PiMForce for various hand postures and hand-object interactions.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_30_1.jpg)

> This figure provides a qualitative comparison of hand pressure estimation results across various comparative models for different hand-object interaction types.  It visually demonstrates the performance of the SEMG Only, 3D Hand Posture Only, SEMG + Hand Angles, and SEMG + 3D Hand Posture models against the ground truth using a rendered visualization.  Each row represents a unique hand-object interaction, while the columns depict each method's estimated hand pressure distribution.  The color intensity represents pressure level, allowing for a visual comparison of the accuracy and effectiveness of each approach.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_31_1.jpg)

> This figure shows a qualitative comparison of pressure estimation results between the proposed method (PiMForce) and PressureVision++ for various hand-object interaction types in the test set, where hand pose is estimated from vision. The intensity of each node's color indicates the pressure level, and the results demonstrate the effectiveness of PiMForce compared to PressureVision++, particularly in complex interactions where occlusion is present.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_32_1.jpg)

> This figure demonstrates the qualitative results of the proposed PiMForce model in the absence of a pressure glove.  It compares the performance of PiMForce to a vision-only approach (PressureVision++) for various hand postures and object interactions. The top panel shows that PressureVision++ fails when the hand is occluded, while PiMForce performs well regardless of occlusion. The bottom panel displays images from a video showcasing the robustness and accuracy of PiMForce for various pressure levels and interactions.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_33_1.jpg)

> This figure demonstrates the qualitative results of the proposed PiMForce framework for hand pressure estimation, both with and without a pressure glove.  (a) shows a comparison across different methods (PiMForce, PressureVision++, and baselines) for various hand-object interactions, highlighting PiMForce's ability to handle occlusions. (b) points to a supplementary video demonstrating robust real-time estimation across diverse scenarios.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_34_1.jpg)

> This figure shows a qualitative comparison of hand pressure estimation results between the proposed method (PiMForce) and PressureVision++, a vision-based method. (a) shows that PiMForce accurately estimates the pressure across the whole hand, even in cases with hand occlusion. In contrast, PressureVision++ fails to estimate pressure in the presence of occlusion. (b) displays a video showing robust pressure estimation by PiMForce across different hand postures and objects.


![](https://ai-paper-reviewer.com/LtS7pP8rEn/figures_35_1.jpg)

> The figure illustrates the architecture of the PiMForce model, highlighting the integration of sEMG and 3D hand pose data to improve hand pressure estimation. It depicts the model's training process, utilizing a combination of classification and regression losses, and its inference process to provide real-time pressure readings from RGB images and sEMG input, thereby addressing limitations of traditional sEMG or vision-based methods.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_6_1.jpg)
> This table presents a quantitative comparison of the proposed PiMForce model against several baseline and state-of-the-art methods for whole-hand pressure estimation.  The metrics used for comparison are R-squared (R²), Normalized Root Mean Squared Error (NRMSE), and classification accuracy.  The results demonstrate the superior performance of PiMForce compared to methods using only sEMG signals, 3D hand posture information, or a combination of sEMG and hand angles.  The table highlights the effectiveness of integrating multimodal data (sEMG and 3D hand pose) for robust hand pressure estimation.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_7_1.jpg)
> This table presents the cross-user performance results of the proposed PiMForce model and the baseline SEMG only model [4].  The performance is evaluated using four metrics: R², NRMSE, MAE, and Accuracy, each providing a different aspect of model performance.  The cross-user evaluation tests the model's ability to generalize to data from participants not included in the training set.  The results highlight the superior performance of the PiMForce model compared to the SEMG only baseline, particularly in terms of accuracy and robustness.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_8_1.jpg)
> This table presents a quantitative comparison of four different methods for hand pressure estimation: SEMG Only, 3D Hand Posture Only, SEMG + Hand Angles, and PiMForce (the authors' proposed method).  For each method, the table shows the R-squared (R2) value, Normalized Root Mean Squared Error (NRMSE), and accuracy.  These metrics provide a comprehensive evaluation of the performance of each method across a range of hand postures and actions.  Higher R2 values indicate a better fit between predicted and actual pressure, while lower NRMSE and higher accuracy values indicate greater precision and correctness.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_9_1.jpg)
> This table presents a comparison of the cross-user performance of three different methods for hand pressure estimation: PressureVision++, sEMG only, and the proposed PiMForce method.  The metrics used for comparison are R2, NRMSE, and Accuracy. The results show that PiMForce significantly outperforms the other two methods across all metrics, demonstrating its robustness and generalizability in estimating hand pressure across different users and interaction types.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_16_1.jpg)
> This table compares the proposed dataset with other existing datasets for hand pressure estimation, highlighting differences in input modalities, data volume, number of participants, pressure and pose information availability, and the nature of hand-object interactions. It provides a comprehensive overview of existing resources for research in this field and emphasizes the novelty and unique characteristics of the proposed dataset.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_20_1.jpg)
> This table compares the current work's dataset with other existing datasets used for hand pressure estimation research.  It compares various factors including the input modalities used (e.g., RGBD, RGB, SEMG), the number of frames and participants involved in data collection, how pressure and pose information were obtained (e.g., inferred from pose, pressure pad, pressure glove), whether the dataset includes whole-hand or only fingertip data, and whether the data was collected from interactions with real-world objects in natural settings. This comparison highlights the unique characteristics of the current dataset in terms of the use of multimodal sensors and the focus on whole-hand pressure estimation under various hand-object interactions.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_24_1.jpg)
> This table presents a comparison of the performance of four different methods for hand pressure estimation: SEMG Only, 3D Hand Posture Only, SEMG + Hand Angles, and PiMForce (the authors' method).  The performance is evaluated using three metrics: R-squared (R2), Normalized Root Mean Squared Error (NRMSE), and Accuracy. Higher R2 values and accuracy, and lower NRMSE values indicate better performance.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_24_2.jpg)
> This table presents a quantitative comparison of the proposed PiMForce model against several baseline and state-of-the-art methods for whole-hand pressure estimation.  The metrics used for comparison are R-squared (R2), Normalized Root Mean Squared Error (NRMSE), and classification accuracy.  The results demonstrate the superior performance of the PiMForce model across all three metrics compared to methods using only sEMG or 3D hand posture information.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_24_3.jpg)
> This table presents a quantitative comparison of the proposed PiMForce model against several baseline and state-of-the-art methods for whole-hand pressure estimation. The comparative methods include an sEMG-only model, a 3D hand posture-only model, an sEMG + hand angles model, and Pressure Vision++.  The performance is evaluated using three metrics: R-squared (R2), Normalized Root Mean Squared Error (NRMSE), and classification accuracy.  Higher R2 values indicate better model fit, lower NRMSE values indicate higher precision, and higher accuracy values show better classification performance.  The results demonstrate the superior performance of the PiMForce model compared to the baselines across all evaluation metrics, highlighting the effectiveness of integrating sEMG signals with 3D hand pose information.

![](https://ai-paper-reviewer.com/LtS7pP8rEn/tables_26_1.jpg)
> This table presents a comparison of the proposed PiMForce model against several baseline and state-of-the-art methods for whole-hand pressure estimation.  The metrics used for comparison are R2 (coefficient of determination), NRMSE (normalized root mean squared error), and Accuracy (classification accuracy).  The results show that the PiMForce model significantly outperforms other methods across all three metrics.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/LtS7pP8rEn/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}