---
title: "Optimal-state Dynamics Estimation for Physics-based Human Motion Capture from Videos"
summary: "OSDCap: Online optimal-state dynamics estimation selectively incorporates physics models with kinematic observations to achieve highly accurate, physically-plausible human motion capture from videos."
categories: []
tags: ["Computer Vision", "3D Vision", "🏢 Department of Electrical Engineering, Linköping University",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} RkOT8rAmRR {{< /keyword >}}
{{< keyword icon="writer" >}} Cuong Le et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=RkOT8rAmRR" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95155" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=RkOT8rAmRR&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/RkOT8rAmRR/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Current human motion capture from videos often suffers from temporal artifacts like jittery movements and lacks physical plausibility.  Existing physics-based methods struggle due to imperfect physical models and noisy input data, requiring simplifying assumptions and extensive preprocessing. 

This paper introduces OSDCap, which uses a novel online filtering method (inspired by Kalman filtering) to combine kinematic observations from videos with a physics-based simulation. This method attentively balances the strengths and weaknesses of both input streams, resulting in more accurate and physically plausible motion capture.  The key contributions include a meta-PD controller for torque prediction, a recurrent neural network for attentive filtering, and a learnable inertia bias prediction to address inaccuracies in human body models. The proposed method outperforms state-of-the-art approaches on multiple benchmark datasets.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} OSDCap, a novel method for physics-based human motion capture from videos, improves accuracy and plausibility by combining physics models and kinematic observations. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} A neural Kalman filter intelligently balances kinematic input and simulated motion, resulting in optimal-state dynamics prediction. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The approach excels in physics-based human pose estimation, demonstrating the physical plausibility of predictive dynamics. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is important because it presents **a novel method for physics-based human motion capture** that addresses limitations of existing approaches.  It offers **improved accuracy and plausibility** by integrating physics models and kinematic observations effectively. This research is relevant to current trends in computer vision and robotics and opens new avenues for research in human motion analysis and animation.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/RkOT8rAmRR/figures_0_1.jpg)

> This figure illustrates the OSDCap system, which combines video-based pose estimation and physics simulation to achieve more realistic human motion capture.  The top-left shows noisy kinematic pose estimation from a video. The bottom-left depicts a physics simulation driven by a meta-PD controller.  The cyan section in the middle shows how the system optimally combines these two inputs to generate a physically plausible motion sequence that is smoother and more accurate than either input alone. The resulting motion is shown in the center, demonstrating reduced high-frequency noise while maintaining high accuracy in the overall position.





![](https://ai-paper-reviewer.com/RkOT8rAmRR/tables_7_1.jpg)

> This table compares the performance of OSDCap with other state-of-the-art methods for 3D human motion capture on the Human3.6M dataset.  It shows various metrics, including different versions of MPJPE (Mean Per Joint Position Error), PCK (Percentage of Correct Keypoints), CPS (Correct Pose Score), and GRP (Global Root Position).  The table highlights OSDCap's superior performance, especially compared to online methods.





### In-depth insights


#### Physics-Based Capture
Physics-based human motion capture methods aim to overcome limitations of traditional approaches by integrating physical principles into the capture process.  **Instead of directly estimating poses from images or sensors, these methods leverage physics engines to simulate realistic human movement, often by inferring internal forces and torques.**  This results in more natural, smoother motions that are less susceptible to artifacts like jitter, unnatural poses, and penetrations.  **Key challenges involve accurately modeling the human body's complex biomechanics and dynamics, handling noisy or incomplete input data from cameras or other sensors, and efficiently solving the underlying physics simulation.**  Various techniques are employed, including optimization-based methods that minimize discrepancies between observed motion and simulated motion, and learning-based methods that directly learn mappings from sensor data to physically plausible dynamics.  **The integration of physics constraints often enhances the robustness and generalizability of motion capture, leading to improved accuracy and reduced computational cost in some cases.**  However, the accuracy and realism are significantly affected by the quality of the underlying physical models, the input data, and the computational power. **Therefore, future directions in research on physics-based capture include developing more accurate and comprehensive biomechanical models, utilizing more advanced simulation techniques, and leveraging machine learning to improve the efficiency and robustness of these methods.**

#### Kalman Filtering Fusion
Kalman filtering fusion, in the context of a physics-based human motion capture system, represents a powerful strategy for integrating noisy kinematic data from video with a physics-based simulation.  **The core idea is to leverage the strengths of both modalities**, compensating for the weaknesses of each.  Video-based pose estimation, while capable of high-resolution pose information, often suffers from temporal inconsistencies and noise.  Physics-based simulations, conversely, provide physically plausible motions but may diverge from reality due to model inaccuracies or the initial pose estimations.  **The Kalman filter acts as a mediator**, intelligently weighting and combining these two data streams to produce an optimal estimate. This fusion technique is crucial for generating motion that is both temporally coherent and consistent with the laws of physics. **The filter's ability to dynamically adjust the weighting based on the reliability of each input stream** is critical for robustness.  In essence, Kalman filtering fusion enhances the accuracy and plausibility of human motion capture by harnessing the complementary nature of vision and physics modeling, yielding significantly improved results over either method alone. The effectiveness of this approach hinges on the accurate modeling of system noise and the precise tuning of Kalman filter parameters, which might be learned through data-driven methods.

#### Optimal-State Estimation
The optimal-state estimation method is a crucial part of the proposed system, acting as a **fusion mechanism** to combine the noisy kinematic pose estimations obtained from video data with the physics-based simulations. This approach aims to leverage the strengths of both data sources while mitigating their limitations. The kinematic data offers highly accurate global position, but suffers from jittery motion and temporal artifacts. Conversely, the physics simulation produces smooth and physically plausible motions, but may contain inaccuracies due to model imperfections.  The **Kalman filter**, implemented as a recurrent neural network, plays a vital role in this optimal-state estimation.  It dynamically weighs the kinematic and simulated data at each timestep, adjusting the balance based on the confidence in each input. This selective integration of information ensures that the resulting motion maintains **high accuracy** while being **physically realistic**. The effectiveness of this approach relies heavily on the learned Kalman gains, external forces, and inertia bias estimates, which are also produced by the neural network.  This suggests the system's ability to adapt to diverse motions and noisy inputs, showcasing the **robustness and adaptability** of the optimal-state estimation process.

#### Motion Dynamics
The study of motion dynamics in physics-based human motion capture involves understanding how forces and torques affect the movement of a human body.  **Accurate modeling of these dynamics is crucial for generating realistic and natural-looking motions**, especially in applications like animation or virtual reality.  The paper likely explores different approaches to estimate these dynamics, perhaps using physics engines, data-driven models, or a combination of both.  A key challenge lies in balancing the accuracy of the motion with the computational cost and complexity of the models used.  **The trade-off between computational efficiency and realism** is a central theme.  Furthermore, the inclusion of noise in motion capture data necessitates techniques for noise reduction and filtering, ensuring that the estimated dynamics reflect true movements and not artifacts introduced during data acquisition.  **Incorporating machine learning** techniques likely enhances the capabilities to estimate the dynamics and improve the overall quality and realism of the generated motion.

#### Future Work
The paper's conclusion mentions several avenues for future work.  **Extending the model to handle more complex movements and a wider variety of human activities** is crucial for broader applicability.  **Improving the accuracy and robustness of the physics-based simulation** is key to producing even more realistic and reliable motion capture, which may involve incorporating more detailed anatomical models.  Investigating **different methods for handling noisy or incomplete kinematic data** would enhance the system's resilience in real-world scenarios.  **Addressing the computational cost** of the online filtering and physics simulation is important to move towards real-time performance. Finally, exploring **the potential of combining the proposed approach with other techniques**, such as deep learning or computer vision, could lead to even more advanced human motion capture systems.  Each of these directions represents a significant challenge, but tackling them would lead to substantial improvements in accuracy, efficiency, and robustness.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/RkOT8rAmRR/figures_3_1.jpg)

> This figure illustrates the overall pipeline of the proposed method, OSDCap. It comprises three main stages: optimal pose estimation using a Kalman filter, physics priors calculation, and physics simulation based on the computed optimal pose and physics priors. The core component is the OSDNet which estimates the Kalman gain matrix, PD gains, external forces and inertia bias matrix, facilitating the fusion of kinematic and dynamic inputs.


![](https://ai-paper-reviewer.com/RkOT8rAmRR/figures_8_1.jpg)

> This figure shows a qualitative comparison of the proposed OSDCap method against a kinematics-only approach and ground truth data. Two examples are presented: one where the kinematics estimation has large errors in depth, and another where the pose is unnatural due to ambiguities. OSDCap is able to correct these issues and produce physically plausible poses by integrating physics-based simulation and Kalman filtering.


![](https://ai-paper-reviewer.com/RkOT8rAmRR/figures_13_1.jpg)

> The figure shows the architecture of the neural network OSDNet.  It consists of a GRU (Gated Recurrent Unit) processing dynamic features and the system state to produce estimates for Kalman gains, PD controller gains, inertia bias, and external forces. The GRU’s hidden state is updated at each time step.  Additional inputs of feet position and velocity improve the estimation of foot-ground contact and reaction forces.  The outputs are used in later processing stages.


![](https://ai-paper-reviewer.com/RkOT8rAmRR/figures_14_1.jpg)

> This figure illustrates the overall pipeline of the OSDCap method, which is comprised of a neural network (OSDNet) and three processing components: optimal pose estimation, physics priors calculation, and physics simulation.  OSDNet estimates key parameters like Kalman gain matrices, PD gains, external forces, and inertia bias.  The optimal pose estimation utilizes a Kalman filter to combine kinematic data with physics-based simulation results. Physics priors use a rigid body dynamics model to calculate inertia and forces, while the simulation updates velocity based on the optimal pose and physics priors. The diagram clearly shows the flow of data and processing steps within the OSDCap framework.


![](https://ai-paper-reviewer.com/RkOT8rAmRR/figures_15_1.jpg)

> This figure shows a qualitative comparison of human motion capture results between the proposed method (OSDCap) and a baseline method (TRACE). Four different sports actions (tennis, baseball, football, and volleyball) are displayed, each with three columns: the input kinematics from TRACE, the estimated pose from OSDCap, and the ground truth pose.  The results demonstrate that OSDCap is able to produce significantly more accurate and plausible human motion than TRACE, especially when the input kinematics are noisy or inaccurate.


![](https://ai-paper-reviewer.com/RkOT8rAmRR/figures_16_1.jpg)

> This figure illustrates the OSDCap system, which combines kinematic pose estimation from videos and physics-based simulation using a meta-PD controller to estimate human motion. The cyan part represents the optimal-state dynamics estimation, which integrates the two input streams to produce a physically plausible and noise-reduced motion while maintaining high accuracy in global position.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/RkOT8rAmRR/tables_7_2.jpg)
> This table presents a quantitative comparison of the proposed OSDCap method against the TRACE baseline on two different datasets: Fit3D and SportsPose.  The metrics used are MPJPE (with its variations MPJPE-G and MPJPE-PA), PCK, CPS, GRP, and Accel, which assess aspects like joint position error, keypoint accuracy, pose correctness, global position error, and motion smoothness.  The results show that OSDCap significantly outperforms TRACE on both datasets, highlighting its robustness to noisy inputs and ability to maintain accurate global motion estimates.

![](https://ai-paper-reviewer.com/RkOT8rAmRR/tables_8_1.jpg)
> This table presents the results of an ablation study on the impact of OSDNet, a neural network model, on human pose estimation using a subset of the Human3.6M dataset.  It compares the performance of OSDCap with and without the inertia bias, and with several baseline methods such as median and Gaussian smoothing applied to the input kinematics from TRACE.  The results show that OSDNet significantly improves the accuracy and physical plausibility of the estimated poses, highlighting the importance of the Kalman filtering and inertia bias components.

![](https://ai-paper-reviewer.com/RkOT8rAmRR/tables_9_1.jpg)
> This table compares the performance of OSDCap with other state-of-the-art methods on the Human3.6M dataset for human motion capture.  It shows various metrics including MPJPE (with and without pose alignment), PCK, CPS, GRP, and Accel, which assess the accuracy and smoothness of the generated 3D poses. The table highlights OSDCap's superior performance among online methods, particularly in terms of accuracy (MPJPE and PCK).  It also notes that one comparable method, DnD, uses additional training data, thus affecting the comparison.

![](https://ai-paper-reviewer.com/RkOT8rAmRR/tables_9_2.jpg)
> This table presents a comparison of several physics-based metrics between the baseline kinematics method (TRACE) and the proposed method (OSDCap).  The metrics quantify aspects of physical plausibility such as ground penetration, ground distance, friction, velocity consistency, and foot-skating artifacts.  The results demonstrate that OSDCap leads to improvements in physical realism compared to the baseline.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RkOT8rAmRR/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}