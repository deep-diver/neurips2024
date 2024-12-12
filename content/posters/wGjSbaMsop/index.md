---
title: "Algorithmic Collective Action in Recommender Systems: Promoting Songs by Reordering Playlists"
summary: "Fans can massively boost an artist's song exposure on music streaming platforms by strategically placing it in their playlists, achieving up to 40x more recommendations."
categories: []
tags: ["AI Applications", "Music", "🏢 University of Zurich",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} wGjSbaMsop {{< /keyword >}}
{{< keyword icon="writer" >}} Joachim Baumann et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=wGjSbaMsop" target="_blank" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/93164" target="_blank" >}}
↗ NeurIPS Homepage
{{< /button >}}{{< button href="https://huggingface.co/spaces/huggingface/paper-central?tab=tab-chat-with-paper&paper_id=wGjSbaMsop&paper_from=neurips" target="_blank" >}}
↗ Chat
{{< /button >}}



<audio controls>
    <source src="https://ai-paper-reviewer.com/wGjSbaMsop/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Music recommendation systems often suffer from popularity bias, disproportionately promoting popular artists and hindering the discovery of lesser-known artists.  This creates an unfair and imbalanced environment within the music industry.  Many artists have limited opportunities for gaining exposure. Existing approaches, such as data poisoning or shilling, are often adversarial and can negatively impact system performance. This paper explores a novel approach to promote visibility.

The research introduces two strategies for algorithmic collective action, allowing a group of fans to significantly increase an artist's song recommendations by strategically inserting songs into their playlists.  The strategies leverage discontinuities in recommendations and the long-tail nature of song distributions, achieving impressive results: even small collectives can achieve a disproportionate increase in recommendations. The approach is demonstrated using a real-world music recommender system, showing significant improvements while causing minimal disruption to the overall system performance and user experience. This provides a powerful method for users to exert influence on algorithmic outcomes.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Small groups of users can significantly amplify an underrepresented artist's song recommendations by strategically placing their songs in playlists. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The proposed strategies exploit statistical properties of recommendation algorithms and the long-tail nature of song distributions, offering a data-driven approach for achieving amplification. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} The strategies' externalities are minimal: overall recommendations remain largely unaffected, and gains are distributed across various artists. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers studying recommender systems, fairness, and algorithmic collective action.  It demonstrates a novel approach to address the popularity bias problem, offering insights into strategic user behavior and its impact on algorithmic outcomes.  The findings open new avenues for research into user-driven manipulation of algorithms and the design of more robust and equitable systems.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_1_1.jpg)

> This figure shows the results of a controlled experiment comparing the test-time recommendation frequency of a song promoted by a collective of users against the frequency of songs with similar training set occurrences. The x-axis represents the song's frequency in the training data, and the y-axis represents the song's frequency in the test data. The red dashed line represents an amplification of 1, meaning that the test frequency is equal to the training frequency. The purple dots represent songs not promoted by a collective, and the pink star represents a song promoted by a collective.  The figure demonstrates that, by strategically choosing the placement of a target song in playlists, a collective can achieve a significantly higher recommendation frequency than expected based on the training data frequency. This finding highlights the potential for algorithmic collective action to influence recommender systems.





![](https://ai-paper-reviewer.com/wGjSbaMsop/tables_17_1.jpg)

> This table presents the mean amplification and standard deviation for various baseline strategies that do not involve coordination among participants.  These strategies all involve inserting the target song (s*) at different positions within playlists: either at a fixed position (Insert@i) or at a random position within a specified range (Random@i-j), or randomly anywhere in the playlist (Random). The results are shown for different sizes of collectives (α), represented as the percentage of playlists controlled by the collective.  The DirLoF strategy, which is a coordinated collective action strategy, is included for comparison, highlighting the effectiveness of coordination.





### In-depth insights


#### Collective Action
The concept of 'collective action' in the context of this research paper centers on how a group of users, **acting in coordination**, can strategically influence a machine learning system's outcomes.  The focus is on leveraging the system's inherent biases and discontinuities, rather than through adversarial attacks.  The study demonstrates that even **small, well-organized groups** of users, controlling only a tiny fraction of the system's training data, can significantly amplify the visibility of an underrepresented song.  The key is in strategically choosing where to insert the song within their playlists to exploit statistical patterns in the recommendation model. This approach emphasizes **authenticity**, avoiding manipulations that could negatively impact user experience.  **Success is measured by comparing the increase in test-time recommendations of the target song relative to its representation in the training data.** The study highlights that carefully designed collective action strategies can be surprisingly effective, with minimal unintended consequences for other users or artists.

#### Algorithmic Lever
The core concept of "Algorithmic Lever" in this research paper revolves around how a collective of users can strategically manipulate a recommender system to promote underrepresented artists.  The authors propose two main strategies: **InClust**, which targets frequently occurring songs in playlists, and **DirLoF**, focusing on less frequent songs to exploit the long tail of song distributions.  Both leverage statistical properties of the system without needing knowledge of its internal workings.  **InClust** places the target song before a popular song to enhance its perceived association and boost recommendations.  **DirLoF**, conversely, strategically inserts the target song after a low-frequency song, capitalizing on the model's tendency to overrepresent certain patterns.  The choice of strategy, therefore, directly influences the effectiveness of collective action, demonstrating how **subtle manipulation of playlist data can disproportionately impact recommendations**, making it a powerful, yet non-adversarial, "lever" for influencing algorithmic outcomes.

#### Empirical Findings
The empirical findings section of this research paper would likely present strong evidence supporting the effectiveness of algorithmic collective action strategies in promoting less popular songs within music recommendation systems.  The results would likely show that even relatively small collectives of users can significantly amplify the visibility of a target song, achieving a disproportionate increase in recommendations compared to its initial presence in training data. **Key performance metrics** such as amplification would be reported, demonstrating the magnitude of this effect.  Crucially, the findings would likely demonstrate that this success is achieved while preserving the overall recommendation quality and fairness, minimizing negative externalities for other artists and users.  A key finding would likely highlight the impact of the chosen strategies on song placement, showing how carefully selecting the placement of the target song is crucial for maximizing its promotion.  **The analysis would likely differentiate between the effectiveness of various strategies**, potentially demonstrating that some approaches are superior to others in achieving this amplified visibility and comparing these strategies against baseline scenarios to show the unique impact of collective actions. **Additional investigation of the impact on various aspects of the system** (e.g., recommendation distribution, performance metrics, user experience) would strengthen the findings and illustrate the nuanced effects of this collective action.

#### Externalities & Bias
Analyzing the externalities and potential biases in algorithmic collective action within recommender systems reveals complex dynamics.  **Positive externalities** could include increased exposure for underrepresented artists, potentially enriching the listening experience and fostering a more diverse music landscape. However, **negative externalities** might arise if the actions of the collective disproportionately harm specific artists or genres, exacerbating existing inequalities.  **Bias amplification** is a crucial concern; if the underlying recommender system already exhibits popularity bias, collective action could unintentionally amplify it, potentially reducing the visibility of even more niche artists.  **Algorithmic fairness** is paramount; ensuring that collective action strategies do not unfairly disadvantage specific individuals or groups is vital.  A thorough investigation should assess the overall impact of these actions on the broader ecosystem and consider strategies to mitigate any biases or negative externalities, ideally promoting a more balanced and equitable platform.

#### Future Directions
Future research could explore the **generalizability** of these findings to other recommender systems and tasks.  It would be valuable to investigate how different model architectures or training data characteristics might affect the effectiveness of collective action strategies.  **Robustness** to adversarial attacks or manipulation by larger collectives also needs investigation.  The **ethical implications** of algorithmic collective action require careful consideration.  Research should examine potential misuse, develop mechanisms to prevent harmful applications, and explore frameworks for fair and transparent platform governance.  Finally, understanding the **long-term dynamics** of collective action is crucial, particularly its impact on diversity, competition, and the long-tail distribution of recommendations.  **Exploring different types of collective action**, beyond playlist manipulation, could also yield valuable insights.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_3_1.jpg)

> This figure demonstrates the imbalance in music recommendation systems. The left panel shows a Lorenz curve illustrating that 80% of recommendations go to only 10% of artists, highlighting a significant popularity bias.  The right panel displays a histogram of song frequencies in the Spotify Million Playlist Dataset, revealing a long tail distribution where nearly half of the tracks appear only once. This long tail characteristic further emphasizes the challenge of promoting less popular songs.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_5_1.jpg)

> This figure illustrates the two strategies used to insert a target song (s*) into a playlist.  The InClust strategy inserts s* before a high-frequency anchor song (s0) in the playlist. This targets clusters of similar song contexts. The DirLoF strategy inserts s* after a low-frequency anchor song (s0). This exploits the long-tail nature of song distributions. The figure also shows pseudocode for both strategies.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_6_1.jpg)

> This figure displays the amplification achieved by different collective action strategies (DirLoF, InClust, Random, AtTheEnd) as a function of the collective size (alpha).  The DirLoF strategy, which involves strategically placing the target song in playlists, shows a significant amplification, especially for small collectives. In contrast, the uncoordinated strategies (Random and AtTheEnd) are much less effective. For larger collectives, the InClust strategy outperforms DirLoF.  The amplification values consistently exceed 1, indicating that the strategies disproportionately boost the target song's recommendations at test time compared to its representation in the training data.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_7_1.jpg)

> This figure shows how the amplification achieved by the DirLoF strategy changes depending on the availability of song statistics.  The x-axis represents the size of the collective (α, on a logarithmic scale). The y-axis shows the amplification achieved as a percentage of the amplification achieved when full information on song statistics is available (full information).  The lines show the results for different levels of information: full information, 10% of training data, 1% of training data, and scraped stream counts.  The shaded area around the lines represent 95% confidence intervals. The figure demonstrates that even with limited information (e.g., only 1% of the training data), a considerable amount of amplification is still possible.  Furthermore, using scraped stream counts as a proxy for song frequencies proves to be a viable alternative, achieving a significant portion of the amplification observed with full information.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_8_1.jpg)

> This figure shows the impact of collective action strategies (hybrid and random) on the performance of a recommender system.  It compares the performance loss relative to a model trained on clean data. The solid lines represent the hybrid and random strategies, the dashed lines show a conservative adversarial baseline (where replacing a song negatively impacts recommendations), and the dotted lines illustrate an optimistic scenario where the promoted song is considered relevant.  The x-axis represents the fraction of manipulated playlists (α). The y-axis shows the performance loss for three metrics: NDCG, R-precision, and #C.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_9_1.jpg)

> This figure shows the impact of the collective action strategy on other songs besides the target song.  It plots the change in the number of recommendations (ΔR) for each song against the song's training set frequency.  Songs are grouped into bins based on training frequency, and the average change in recommendations for songs within each bin is shown with error bars (95% confidence intervals). A key observation is the relatively small impact on the recommendations of other songs, suggesting that the collective action strategy does not disproportionately harm other artists.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_14_1.jpg)

> This figure demonstrates the imbalance in the distribution of recommendations and song frequencies. The left panel shows a Lorenz curve illustrating that a small percentage (10%) of artists receive a large portion (80%) of recommendations, highlighting the popularity bias.  The right panel displays the distribution of song frequencies within Spotify playlists, revealing that many songs appear only once, indicating a long tail distribution.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_15_1.jpg)

> This figure visualizes the similarity scores between context embeddings and songs, comparing the scores when collective action (InClust strategy) is used versus when it is not. Three indirect anchor songs (s0, s1, s2), frequently occurring in the collectives' playlists, and the target song (s*) are evaluated against the three context clusters targeted by the InClust strategy.  The violin plots show the distribution of similarity scores for each song-context pair under different conditions. The dashed purple lines indicate the average similarity score required to be ranked among the top 50 most similar songs. The figure demonstrates the InClust strategy's effectiveness in increasing the similarity score between the target song and the targeted contexts.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_16_1.jpg)

> This figure displays the success rate of the InClust and Random strategies as a function of the collective size (α) and the number of indirect anchors used.  The InClust strategy, which strategically places the target song near frequently occurring songs, shows significantly higher success rates than the Random strategy, especially as the collective size and the number of anchors increase. This demonstrates that coordinated efforts in choosing the placement of the target song are highly effective in boosting its recommendations.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_16_2.jpg)

> This figure shows the estimated and true probabilities of targeted direct anchors for different levels of knowledge about song frequencies in the training data. The x-axis represents the targeted direct anchors, and the y-axis represents the probability. The figure demonstrates that with less information, the gap between the estimated and true probabilities widens, leading to less accurate selection of low-frequency songs. This directly affects the success of the DirLoF strategy, as it relies on accurate song frequency estimates.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_16_3.jpg)

> This figure compares the anchor song selection for three different strategies used to promote a song. The x-axis represents the true probability of a song's appearance in the training dataset. The y-axis shows the number of times a song is used as an anchor within playlists controlled by a collective. Each dot represents a song. The color of the dot indicates whether it is targeted or not. Red dots indicate anchor songs selected for the corresponding strategy. The three strategies are InClust, DirLoF, and Hybrid. InClust uses frequently occurring songs as anchors. DirLoF uses infrequently occurring songs as anchors. The hybrid strategy uses a combination of both.


![](https://ai-paper-reviewer.com/wGjSbaMsop/figures_19_1.jpg)

> This figure shows the distribution of precision scores for the top 50 recommendations in three scenarios: 1) when no collective action was taken, 2) when the InClust strategy was used, and 3) when a random strategy was used. The precision score measures the proportion of recommended songs that are relevant to the user. The distributions of precision scores for InClust and the baseline (no collective action) are very similar, indicating that the InClust strategy does not significantly affect the overall recommendation quality. The random strategy has slightly lower scores, suggesting that it is less effective in preserving the quality of recommendations.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/wGjSbaMsop/tables_17_2.jpg)
> This table presents the mean amplification and standard deviation achieved by the Random and DirLoF strategies under different hyperparameter configurations.  It shows the robustness of the DirLoF strategy across various hyperparameter settings, demonstrating that it consistently outperforms the Random strategy.

![](https://ai-paper-reviewer.com/wGjSbaMsop/tables_18_1.jpg)
> This table presents the mean amplification and standard deviation achieved by the DirLoF and Random strategies for different numbers of training epochs. The amplification is a measure of the effectiveness of the collective action strategy in increasing the number of test-time recommendations for a target song.  The table shows that the DirLoF strategy consistently outperforms the Random strategy, particularly when more training epochs are used. However, the amplification achieved by DirLoF peaks at 12 epochs and then declines slightly.

![](https://ai-paper-reviewer.com/wGjSbaMsop/tables_18_2.jpg)
> This table presents the results of an empirical evaluation of the impact of collective action strategies on song recommendations. It shows the mean and 95% confidence intervals for three key metrics: R0 (recommendations without collective action), AR (additional recommendations gained due to collective action), and AR as a percentage of R0.  The metrics are broken down for three categories of songs: direct anchors, indirect anchors, and other songs. The table also shows the number of songs in each category used in the five-fold cross-validation.

![](https://ai-paper-reviewer.com/wGjSbaMsop/tables_19_1.jpg)
> This table presents the average model performance metrics (NDCG, R-precision, and #C) under three different collective action strategies: None (no manipulation), Random (random song placement), and InClust (strategically placing the target song). The results show that the model's performance is robust across these strategies, indicating that the collective action methods do not significantly affect the overall recommendation quality.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/wGjSbaMsop/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}