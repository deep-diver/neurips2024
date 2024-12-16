---
title: "Unlocking the Potential of Global Human Expertise"
summary: "AI can unlock the potential of global human expertise for solving complex problems by combining diverse expert solutions using an evolutionary framework, resulting in better and more effective strateg..."
categories: ["AI Generated", ]
tags: ["AI Applications", "Healthcare", "🏢 Cognizant AI Labs",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} hw76X5uWrc {{< /keyword >}}
{{< keyword icon="writer" >}} Elliot Meyerson et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=hw76X5uWrc" target="_blank" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/hw76X5uWrc" target="_blank" >}}
↗ Hugging Face
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=hw76X5uWrc&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}




<audio controls>
    <source src="https://ai-paper-reviewer.com/hw76X5uWrc/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many complex societal problems necessitate integrating diverse expert perspectives.  However, synthesizing this knowledge becomes increasingly challenging as the number of experts and information grows. This paper addresses this challenge by focusing on the limitations of current methods and proposing a novel AI framework.

The proposed framework, RHEA, employs a four-step process to combine expert solutions: 1) formally defining the problem, 2) gathering solutions from diverse experts, 3) distilling those solutions into a common format (neural networks), and 4) evolving the combined solutions using population-based search.  The study demonstrates RHEA's effectiveness using synthetic and real-world (XPRIZE Pandemic Response Challenge) data. The results show that RHEA consistently outperforms both human experts and traditional AI methods, highlighting the potential of AI in realizing the full potential of diverse human expertise.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} AI significantly improves decision-making by combining diverse human expertise. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} RHEA, the proposed evolutionary AI framework, effectively integrates and refines expert knowledge, surpassing the capabilities of individual experts and AI alone. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The approach was successfully applied to develop better COVID-19 policies than those created by 102 expert teams alone {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers in AI, policy-making, and related fields.  It **demonstrates the potential of AI to unlock the power of diverse human expertise**, a critical need in tackling complex global challenges. The methodology presented, RHEA, offers a novel and effective framework for combining and refining insights from multiple sources, opening new avenues for research into multi-objective optimization and human-AI collaboration. Its application to the XPRIZE Pandemic Response Challenge showcases its real-world applicability and impact.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/hw76X5uWrc/figures_1_1.jpg)

> 🔼 This figure illustrates the RHEA framework, which consists of four main steps: 1) defining the problem, 2) gathering solutions from human experts, 3) distilling these solutions into a canonical form (neural networks), and 4) evolving the population of solutions through a population-based search to find better solutions that build upon the original expert ideas.  The figure shows how the predictor and prescriptor function within the framework and how human expertise is incorporated and refined through AI.
> <details>
> <summary>read the caption</summary>
> Figure 1: The RHEA (Realizing Human Expertise through AI) framework. The framework consists of four components: Defining the prediction and prescription tasks, gathering the human solutions, distilling them into a canonical form, and evolving the population of solutions further. a, The predictor maps context and actions to outcomes and thus constitutes a surrogate, or a 'digital twin', of the real world. For example, in the Pandemic Response Challenge experiment, the context consisted of data about the geographic region for which the predictions were made, e.g., historical data of COVID-19 cases and intervention policies; actions were future schedules of intervention policies for the region; and outcomes were predicted future cases of COVID-19 along with the stringency of the policy. b, Given a predictor, the prescriptor generates actions that yield optimized outcomes across contexts. c, Humans are solicited to contribute expertise by submitting prescriptors using whatever methodology they prefer, such as decision rules, epidemiological models, classical statistical techniques, and gradient-based methods. d, Each submitted prescriptor is distilled into a canonical neural network that replicates its behavior. e, This population of neural networks is evolved further, i.e., the distilled models are recombined and refined in a parallelized, iterative search process. They build synergies and extend the ideas in the original solutions, resulting in policies that perform better than the original ones. For example, in the Pandemic Response Challenge, the policies recommend interventions that lead to minimal cases with minimal stringency.
> </details>







### In-depth insights


#### Global Expertise AI
Global Expertise AI explores leveraging artificial intelligence to harness the collective knowledge of diverse international experts for solving complex global challenges.  The core idea is that **AI can act as a powerful catalyst**, identifying, combining, and refining complementary insights from a large and diverse knowledge base that would be otherwise impossible for humans to manage efficiently.  This approach is particularly relevant when facing problems with intricate interdependencies, such as climate change or pandemics, where diverse perspectives and innovative solutions are crucial.  **AI's role isn't to replace human expertise but to augment it**, by systematically integrating, recombining, and refining expert-generated models. The potential benefits include the discovery of novel and more effective solutions than could be achieved by humans or AI alone, leading to more informed and impactful decision-making on a global scale.  However, **ethical considerations are paramount**; careful design is necessary to ensure fairness, transparency, and accountability in the AI-driven process.  It's crucial to address potential biases, preserve data privacy, and maintain human oversight to prevent misuse or unintended consequences.  The ultimate success of Global Expertise AI hinges on striking a balance between AI's capabilities and the inherent value of human judgment, creating a synergistic partnership to tackle complex world problems.

#### RHEA Framework
The RHEA framework is a novel AI-driven approach for leveraging global human expertise to solve complex problems.  It's a four-stage process: **defining the problem formally**, enabling comparison of diverse solutions; **gathering solutions** from diverse international experts using various methods; **distilling these solutions** into a canonical form, such as equivalent neural networks; and finally, **evolving the distilled solutions** through a population-based search, recombining and refining to surpass individual expert contributions.  This evolutionary aspect allows RHEA to discover synergistic solutions beyond those achievable by individual humans or AI alone.  The XPRIZE Pandemic Response Challenge successfully demonstrated RHEA's effectiveness, producing more effective policies than those submitted by human experts. **RHEA's strength lies in its ability to integrate, refine, and innovate upon existing expertise**, unlocking the full potential of diverse human knowledge for global problem-solving.

#### XPRIZE Challenge
The XPRIZE Pandemic Response Challenge presented a unique opportunity to test the RHEA framework in a real-world, large-scale setting.  **The challenge's focus on diverse, international teams of experts**, each using their own methodologies to predict COVID-19 cases and propose non-pharmaceutical interventions, mirrored RHEA's core principle of leveraging diverse human expertise.  The challenge provided a substantial dataset encompassing various geographical regions, intervention policies, and real-world outcomes, **allowing for a robust evaluation of RHEA's ability to synthesize and improve upon the models** submitted by participating teams.  The success of RHEA in this context validated its potential as a valuable tool for tackling complex, global problems. The ability to distill knowledge from disparate models into a canonical form, and then recombine and evolve them to discover superior solutions, highlights the strength of RHEA.  However, limitations remain.  While RHEA performed well in the XPRIZE challenge, **it's crucial to consider the scalability and generalizability of the approach** to other domains. Furthermore, **ethical considerations such as data privacy, fairness, and accountability** are paramount and require careful attention in any future implementation of similar AI-driven policy development frameworks.  The results from the XPRIZE challenge offer valuable insights that will inform future research and development efforts.

#### Innovation via AI
The concept of 'Innovation via AI' in the context of the provided research paper centers around the idea that AI can significantly augment and accelerate human-driven innovation.  The paper likely highlights how AI can process and synthesize vast amounts of diverse data, exceeding human capabilities, thus identifying novel solutions and approaches previously undiscovered. **RHEA**, the proposed framework, is the AI-powered engine for this process; the paper likely details its mechanisms for integrating and refining diverse human expertise.  This framework likely involves using AI to transform human-generated solutions into a canonical form (e.g., neural networks) before recombining and evolving them. This approach suggests that AI's role is not to replace human ingenuity but to **act as a powerful catalyst**, allowing human expertise to reach its full potential in problem-solving.  **The focus is on the synergistic collaboration between human experts and AI**, leveraging the strengths of both for greater innovation. The success of RHEA in the pandemic response challenge, as described in the paper, would serve as a compelling example of AI accelerating the discovery of better solutions than could have been achieved using human experts alone.  The overall theme underscores the potential for AI to reshape various fields by fostering human creativity and problem-solving ability.

#### Limitations & Ethics
The research paper section on limitations acknowledges the model's limitations, including its reliance on formalized problems, the availability of diverse expert solutions, and problem complexity.  **The dependence on existing expert solutions could hinder innovation if experts overlook unconventional approaches or if biases in existing solutions are not addressed.**  The authors also discuss the scalability and generalizability of the approach, highlighting potential issues with computationally expensive tasks.  The ethics section is crucial, addressing concerns of fairness, transparency, and accountability in AI-driven policy making.  **The importance of including fairness constraints to mitigate biases affecting certain populations is emphasized, as is the need for a robust and transparent process to foster trust and prevent misuse.** The authors highlight the potential for algorithmic bias and misuse of data, underscoring the need for external oversight and democratic accountability to prevent the system from being used for unethical purposes.  This thoughtful consideration of both limitations and ethical implications demonstrates the responsible approach of the authors to the development and deployment of their framework.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/hw76X5uWrc/figures_3_1.jpg)

> 🔼 This figure illustrates the results of applying the RHEA framework to a synthetic policy-making problem.  Three expert prescriptors, each with different levels of specialization and performance, provide initial solutions. RHEA combines and refines these solutions, outperforming alternative approaches such as Mixture-of-Experts (MoE) and Weighted Ensembling. The figure highlights how RHEA discovers a superior Pareto front, showcasing its ability to leverage and synthesize diverse expert knowledge.
> <details>
> <summary>read the caption</summary>
> Figure 2: An Illustration of RHEA in a Synthetic Domain. The plots show the Pareto front of prescriptors discovered by RHEA vs. those of alternative prescriptor combination methods, highlighting the kinds of opportunities RHEA is able to exploit. The specialist expert prescriptors a and b and the generalist expert prescriptor c are useful but suboptimal on their own (purple •'s). RHEA recombines and innovates upon their internal structure and is able to discover the full optimal Pareto front (blue *'s). This front dominates that of Mixture-of-Experts (MoE; green ×'s), which can only mix expert behavior independently in each context. It also dominates that of Weighted Ensembling (yellow +’s), which can only choose a single combination of experts to apply everywhere. Evolution alone (without expert knowledge) also struggles in this domain due to the vast search space (App. Fig. 6), as do MORL methods (App. Fig. 7,8). Thus, RHEA unlocks the latent potential in expert solutions.
> </details>



![](https://ai-paper-reviewer.com/hw76X5uWrc/figures_5_1.jpg)

> 🔼 This figure provides a quantitative comparison of solutions generated by different methods (Random, Evolved, Distilled, and RHEA).  Panels (a-c) show the objective values, Pareto fronts, and the combined Pareto front of each method. Panel (d) displays the distribution of real-world stringency levels to illustrate which trade-offs would likely be preferred by decision-makers. Panels (e-i) present several performance metrics (REM, RUN, DR, MCR, HVI), showing that RHEA outperforms all other approaches by significantly improving existing human-designed solutions and discovering improved solutions that would likely be favored by decision-makers.
> <details>
> <summary>read the caption</summary>
> Figure 3: Quantitative comparison of solutions. a, Objective values for all solutions in the final population of a single representative run of each method. b, Pareto curves for these runs. Distilled provides improved tradeoffs over Random and Evolved (from random), and RHEA pushes the front out beyond Distilled. c, Overall Pareto front of the union of the solutions from these runs. The vast majority of these solutions are from RHEA. d, The distribution of actual stringencies implemented in the real world across all geos at the prescription start date, indicating which Pareto solutions real-world decision makers would likely select, i.e., which tradeoffs they prefer. e, Given this distribution, the proportion of the time the solution selected by a user would be from a particular method (the REM metric); almost all of them would be from RHEA. f, The same metric, but based on a uniform distribution of tradeoff preference (RUN) g, Domination rate (DR) w.r.t. Distilled, i.e. how much of the Distilled Pareto front is strictly dominated by another method's front. While Evolved (from scratch) sometimes discovers better solutions than those distilled from expert designs, RHEA improves ≈75% of them. h, Max reduction of cases (MCR) compared to Distilled across all stringency levels. i, Dominated hypervolume improvement (HVI) compared to Distilled. For each metric, RHEA substantially outperforms the alternatives, demonstrating that it creates improved solutions over human and AI design, and that those solutions would likely be preferred by human decision-makers. (Bars show mean and st.dev. See App. C.3 for technical details of each metric.)
> </details>



![](https://ai-paper-reviewer.com/hw76X5uWrc/figures_6_1.jpg)

> 🔼 This figure visualizes the dynamics of intervention policies (IPs) schedules generated by different methods, including RHEA, evolved policies, and submitted expert models.  Panel (a) shows a UMAP projection of the schedules, highlighting how RHEA expands upon the structure provided by expert submissions. Panel (b) further analyzes the characteristics of these schedules, demonstrating how RHEA finds a balance between grounding and innovation, and discovers solutions that outperform human experts alone.
> <details>
> <summary>read the caption</summary>
> Figure 4: Dynamics of IP schedules discovered by RHEA. a, UMAP projection of geo IP schedules generated by the policies (App. C.4). The schedules from high-performing submitted expert models are concentrated around a 1-dimensional manifold organized by overall cost (seen as a yellow arc). This manifold provides a scaffolding upon which RHEA elaborates, interpolates, and expands. Evolved policies, on the other hand, are scattered more discordantly (seen as blue clusters), ungrounded by the experts. b, To characterize how RHEA expands upon this scaffolding, five high-level properties of IP schedules were identified and their distributions were plotted across the schedules. For each, RHEA finds a balance between the grounding of expert submissions (i.e., regularization) and their recombination and elaboration (i.e., innovation), though this balance manifests in distinct ways. For swing and separability, RHEA is similar to real schedules, but finds that the high separability proposed by some expert models can sometimes be useful. RHEA finds the high focus of the expert models even more attractive; in practice, they could provide policy-makers with simpler and clearer messages about how to control the pandemic. For focus, agility, and periodicity, RHEA pushes beyond areas explored by the submissions, finding solutions that humans may miss. The example schedules shown in a(i-v) illustrate these principles in practice (rows are IPs sorted from top to bottom as listed in Sec. 3; column are days in the 90-day period; darker color means more stringent). (i) Real-world examples demonstrate that although agility and periodicity require some effort to implement, they have occasionally been utilized (e.g. in Portugal and France); (ii) a simple example of how RHEA generates useful interpolations of submitted non-Pareto schedules, demonstrating how it realizes latent potential even in some low-performing solutions, far from schedules evolved from scratch; (iii) another useful interpolation, but achieved via higher agility than Pareto submissions; (iv) a high-stringency RHEA schedule that trades swing and separability for agility and periodicity compared to its submitted neighbor; and (v) a medium-stringency RHEA schedule with lower swing and separability and higher focus than its submitted neighbor. Overall, these analyses show how RHEA realizes the latent potential of the raw material provided by the human-created submissions.
> </details>



![](https://ai-paper-reviewer.com/hw76X5uWrc/figures_7_1.jpg)

> 🔼 This figure shows the evolutionary process of discovering new solutions using RHEA. Panel (a) displays the ancestry of solutions on the Pareto front, illustrating how different combinations of initial solutions lead to better solutions. Panel (b) and (c) visualize the relationship between parent and child solutions' cost and cases, indicating a consistent pattern. Panel (d) shows the consistent contribution of expert models to the final solutions across different runs. Panel (e) highlights the correlation between initial team performance and contribution to the final solutions, emphasizing the value of diverse expertise.
> <details>
> <summary>read the caption</summary>
> Figure 5: Dynamics of evolutionary discovery process. a, Sample ancestries of prescriptors on the RHEA Pareto front. Leaf nodes are initial distilled models; the final solutions are the root. The history of recombinations leading to different solutions varies widely in terms of complexity, with apparent motifs and symmetries. The ancestries show that the search is behaving as expected, in that the cost of the child usually lands between the costs of its parents (indicated by color). This property is also visualized in b (and c), where child costs (and cases) are plotted over all recombinations from all trials (k-NN regression, k = 100). d, From ancestries, one can compute the relative contribution of each expert model to the final RHEA Pareto front (App C.5). This contribution is remarkably consistent across the independent runs, indicating that the approach is reliable (mean and st.dev. shown). e, Although there is a correlation between the performance of teams of expert models and their contribution to the final front, there are some teams with unimpressive quantitative performance in their submissions who end up making outsized contributions through the evolutionary process. This result highlights the value of soliciting a broad diversity of expertise, even if some of it does not have immediately obvious practical utility. AI can play a role in realizing this latent potential.
> </details>



![](https://ai-paper-reviewer.com/hw76X5uWrc/figures_17_1.jpg)

> 🔼 This figure compares the performance of RHEA and a standard evolutionary algorithm (without expert knowledge) in a synthetic domain.  The left panel (a) shows that RHEA consistently and efficiently finds the complete optimal Pareto front, even as the problem's complexity increases (more available policy interventions).  The right panel (b) shows that the standard evolutionary algorithm struggles to find the complete optimal Pareto front, particularly as the complexity of the problem grows. The key takeaway is that diverse expert knowledge, leveraged by RHEA, is crucial for reliably discovering optimal policies.
> <details>
> <summary>read the caption</summary>
> Figure 6: Experimental results comparing RHEA vs. Evolution alone (i.e., without knowledge of gathered expert solutions) in the illustrative domain. Whiskers show 1.5×IQR; the middle bar is the median. a, RHEA exploits latent expert knowledge to reliably and efficiently discover the full optimal Pareto front, even as the number of available policy interventions n increases (there are 2n possible actions for each context; 100 trials each). b, Evolution alone does not reliably discover the front even with 10 available interventions, and its performance drops sharply as the number increases (100 trials each). Thus, diverse expert knowledge is key to discovering optimal policies.
> </details>



![](https://ai-paper-reviewer.com/hw76X5uWrc/figures_18_1.jpg)

> 🔼 This figure illustrates the results of applying the RHEA framework to a synthetic problem.  It demonstrates how RHEA, by recombining and refining solutions from diverse experts (generalist and specialists), finds a superior Pareto front compared to methods like Mixture-of-Experts or Weighted Ensembling.  The results highlight RHEA's ability to discover novel and better solutions than would be found by individual experts or evolutionary algorithms alone, emphasizing the value of diverse expertise and RHEA's novel approach to combining and refining it.
> <details>
> <summary>read the caption</summary>
> Figure 2: An Illustration of RHEA in a Synthetic Domain. The plots show the Pareto front of prescriptors discovered by RHEA vs. those of alternative prescriptor combination methods, highlighting the kinds of opportunities RHEA is able to exploit. The specialist expert prescriptors a and b and the generalist expert prescriptor c are useful but suboptimal on their own (purple •'s). RHEA recombines and innovates upon their internal structure and is able to discover the full optimal Pareto front (blue *'s). This front dominates that of Mixture-of-Experts (MoE; green ×'s), which can only mix expert behavior independently in each context. It also dominates that of Weighted Ensembling (yellow +'s), which can only choose a single combination of experts to apply everywhere. Evolution alone (without expert knowledge) also struggles in this domain due to the vast search space (App. Fig. 6), as do MORL methods (App. Fig. 7,8). Thus, RHEA unlocks the latent potential in expert solutions.
> </details>



![](https://ai-paper-reviewer.com/hw76X5uWrc/figures_18_2.jpg)

> 🔼 This figure quantitatively compares the performance of four different methods: Random, Evolved, Distilled, and RHEA. It shows that RHEA significantly outperforms other methods in terms of Pareto optimality and preference by decision-makers, highlighting its ability to improve upon both human-designed and AI-only solutions.
> <details>
> <summary>read the caption</summary>
> Figure 3: Quantitative comparison of solutions. a, Objective values for all solutions in the final population of a single representative run of each method. b, Pareto curves for these runs. Distilled provides improved tradeoffs over Random and Evolved (from random), and RHEA pushes the front out beyond Distilled. c, Overall Pareto front of the union of the solutions from these runs. The vast majority of these solutions are from RHEA. d, The distribution of actual stringencies implemented in the real world across all geos at the prescription start date, indicating which Pareto solutions real-world decision makers would likely select, i.e., which tradeoffs they prefer. e, Given this distribution, the proportion of the time the solution selected by a user would be from a particular method (the REM metric); almost all of them would be from RHEA. f, The same metric, but based on a uniform distribution of tradeoff preference (RUN) g, Domination rate (DR) w.r.t. Distilled, i.e. how much of the Distilled Pareto front is strictly dominated by another method's front. While Evolved (from scratch) sometimes discovers better solutions than those distilled from expert designs, RHEA improves ≈75% of them. h, Max reduction of cases (MCR) compared to Distilled across all stringency levels. i, Dominated hypervolume improvement (HVI) compared to Distilled. For each metric, RHEA substantially outperforms the alternatives, demonstrating that it creates improved solutions over human and AI design, and that those solutions would likely be preferred by human decision-makers. (Bars show mean and st.dev. See App. C.3 for technical details of each metric.)
> </details>



</details>






### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/hw76X5uWrc/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}