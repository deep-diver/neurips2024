---
title: "Gradients of Functions of Large Matrices"
summary: "This research presents novel adjoint methods for efficiently differentiating Lanczos and Arnoldi iterations, unlocking accurate gradients for large-matrix functions in machine learning."
categories: []
tags: ["Machine Learning", "Deep Learning", "🏢 Technical University of Denmark",]
showSummary: true
date: 2024-09-26
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} RL4FXrGcTw {{< /keyword >}}
{{< keyword icon="writer" >}} Nicholas Krämer et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://openreview.net/forum?id=RL4FXrGcTw" target="_self" >}}
↗ OpenReview
{{< /button >}}
{{< button href="https://neurips.cc/virtual/2024/poster/95179" target="_self" >}}
↗ NeurIPS Homepage
{{< /button >}}


<audio controls>
    <source src="https://ai-paper-reviewer.com/RL4FXrGcTw/podcast.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>


### TL;DR


{{< lead >}}

Many machine learning models rely on evaluating functions of large matrices, often using Lanczos and Arnoldi iterations. However, differentiating these methods efficiently has been a challenge, hindering the development of more complex and powerful models.  Existing approaches often involve approximations, limiting accuracy and scalability.

This work introduces novel adjoint systems for Lanczos and Arnoldi iterations, overcoming previous limitations.  The proposed method is implemented in JAX and provides **exact gradients with linear time and memory complexity**. It significantly outperforms existing methods in various applications, demonstrating its efficiency and accuracy.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
{{< typeit speed=10 lifeLike=true >}} Adjoint methods enable efficient differentiation of Lanczos and Arnoldi iterations for large matrices. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=1000 lifeLike=true >}} The proposed approach yields exact gradients with linear time and memory complexity. {{< /typeit >}}
{{< /alert >}}

{{< alert "star" >}}
{{< typeit speed=10 startDelay=2000 lifeLike=true >}} Improved performance demonstrated in Gaussian process models, PDE solvers, and Bayesian neural networks. {{< /typeit >}}
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working with large-scale machine learning models that involve matrix operations.  It provides **efficient and accurate methods for differentiating functions of large matrices**, a common bottleneck in many applications.  This opens up new avenues for developing more sophisticated and scalable models across diverse fields.

------
#### Visual Insights



![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_1_1.jpg)

> This figure illustrates the workflow of the proposed method for computing gradients of functions of large matrices. It starts with a parameter θ that is used to compute a large matrix-vector product via Arnoldi or Lanczos iterations. These iterations are then used to obtain a small factorization, which is subsequently used to compute the function of the large matrix. Finally, the loss is calculated based on the function of the large matrix. The gradients are computed using the adjoint method, propagating the gradient backward through the entire process, making the matrix-free implementation of matrix decompositions differentiable.





![](https://ai-paper-reviewer.com/RL4FXrGcTw/tables_1_1.jpg)

> This table presents several machine learning applications that involve evaluating functions of matrices. Each row represents an application, specifying the function applied to the matrix, the type of matrix, the vector used in matrix-vector products, and the parameters that are optimized.  The applications span diverse areas such as solving partial differential equations (PDEs), Gaussian processes, invertible residual networks, Gaussian sampling, and Neural ODEs.





### In-depth insights


#### Matrix Calculus
Matrix calculus extends the rules of single-variable calculus to matrices, enabling the differentiation of functions involving matrices and vectors.  **This is crucial for optimizing machine learning models**, where many functions are defined in terms of matrix operations.  The paper's approach to matrix calculus is particularly notable because it leverages adjoint methods, avoiding the explicit computation of the Jacobian matrix, which would be computationally infeasible for large matrices.  **The adjoint method's efficiency stems from its ability to implicitly compute gradient information without forming the potentially immense Jacobian**.  The application to large-scale machine learning models—particularly those involving Gaussian processes, partial differential equations, and Bayesian neural networks—highlights the practical significance of their efficient approach to matrix calculus. **The derivation of adjoint systems for Lanczos and Arnoldi iterations is a key theoretical contribution**, enabling automatic differentiation for functions that are traditionally difficult to handle.  This allows for the development of matrix-free algorithms, where gradients are computed without explicitly forming the large matrices, making large-scale optimization tractable.

#### Adjoint Methods
Adjoint methods, in the context of this research paper, are presented as **efficient techniques for differentiating matrix functions** commonly used in machine learning models.  The core idea is to avoid explicitly computing the often massive Jacobian matrix, instead leveraging the adjoint system to calculate gradients implicitly.  This approach is particularly crucial when dealing with large matrices that are accessed only via matrix-vector products, as is common in various model types. **The paper focuses on deriving the adjoint systems for Lanczos and Arnoldi iterations**, which are widely used for matrix decompositions in situations where a full matrix factorization is computationally prohibitive.  This derivation enables the automatic differentiation of functions of large matrices, leading to efficient gradient computations and facilitating optimization of models with large numbers of parameters. The effectiveness of this approach is empirically demonstrated through case studies involving Gaussian processes, PDEs, and Bayesian neural networks, highlighting its practical advantages over conventional methods.

#### Lanczos Adjoints
The concept of "Lanczos Adjoints" revolves around efficiently computing gradients of functions involving large matrices, a common challenge in machine learning and scientific computing.  The Lanczos algorithm, a powerful technique for approximating matrix functions, is typically used in a matrix-free manner to avoid storing the full Jacobian matrix.  **The key innovation is the derivation of adjoint systems for the Lanczos iteration.** This allows the computation of gradients via reverse-mode automatic differentiation without ever explicitly forming or storing the large matrix, maintaining computational efficiency.  **The adjoint method avoids the inefficiency of direct backpropagation through the Lanczos algorithm** which would be prohibitively expensive for large matrices.  This approach enables the application of gradient-based optimization methods to problems previously intractable due to computational limitations. The presented adjoint systems provide a practical and computationally efficient mechanism for extending the applicability of Lanczos-based matrix computations to broader gradient-based optimization problems in machine learning and scientific computing.

#### GP & PDEs
The fusion of Gaussian Processes (GPs) and Partial Differential Equations (PDEs) presents a powerful paradigm for scientific machine learning. **GPs excel at modeling uncertainty and incorporating prior knowledge**, while **PDEs capture the underlying physical laws and relationships**.  By combining them, we can leverage the strengths of both to create models capable of handling complex, high-dimensional data with inherent uncertainty, typical of many scientific applications.  **A key challenge lies in efficient computation**, particularly when dealing with large datasets or complex PDEs.  The research likely explores novel methods for addressing this computational bottleneck, perhaps focusing on matrix-free techniques or leveraging the structure of the PDE to accelerate inference.  The approach might involve approximating the solution to the PDE and using this approximation to construct a GP model, or it could involve incorporating the PDE directly into the GP prior.  Another crucial aspect would be the **evaluation of model accuracy and uncertainty**, ensuring that the model's predictive power is reliable and that the uncertainty estimates accurately reflect the epistemic and aleatoric uncertainty in the data and the model.  Therefore, **a comprehensive comparison with traditional methods**, both for accuracy and efficiency, would be important to demonstrate the efficacy of this combined approach.  Overall, the study likely offers a compelling blend of theoretical and practical contributions, addressing a significant need in diverse fields such as materials science, fluid dynamics, and climate modeling.

#### BNN Calibration
Bayesian neural networks (BNNs) offer a principled approach to uncertainty quantification, but calibrating their predictive probabilities is crucial for reliable decision-making.  **BNN calibration focuses on aligning the network's reported confidence with its actual accuracy.**  Poor calibration leads to overconfident predictions, undermining the BNN's practical value.  Methods for BNN calibration often involve techniques to better approximate the posterior distribution of the network's weights, for example, through Laplace approximations. However, **exact computation of the posterior is often intractable for large BNNs**, necessitating approximations such as stochastic gradient Markov Chain Monte Carlo (MCMC) or variational inference.  **The choice of calibration method significantly impacts the BNN's predictive performance and computational cost.**  Furthermore, the effectiveness of calibration may depend on the network architecture, the dataset, and the hyperparameters. **Recent research emphasizes matrix-free methods for efficient gradient computations**, particularly relevant for large models where constructing and manipulating full Hessian or covariance matrices is prohibitive.  This allows for the calibration of models with a large number of parameters, otherwise computationally infeasible.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_2_1.jpg)

> This figure shows a schematic of the Lanczos/Arnoldi iteration.  A large matrix A(θ) is implicitly represented through matrix-vector products. The iteration generates an orthonormal matrix Q and a smaller Hessenberg matrix H such that A(θ)Q ≈ QH + r(ek). Here, ek is the k-th unit vector and r is a residual vector. This approximation is used for computing functions of large matrices in a matrix-free manner.


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_3_1.jpg)

> This figure compares the performance of backpropagation against the proposed adjoint method for computing gradients of functions of large sparse matrices using the Lanczos method.  The x-axis represents the Krylov subspace depth (K), and the y-axis represents the wall time (in seconds) for each method. It shows how backpropagation's time complexity increases significantly with K, making it impractical for larger K values, while the adjoint method maintains similar performance to the forward pass, making it more efficient for large-scale computations.


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_7_1.jpg)

> This figure compares the performance of three different methods (Arnoldi, Dopri5, and Tsit5) for solving a partial differential equation.  Panel A1 shows the forward error (RMSE) of each method as a function of the number of matrix-vector products. Panel A2 shows the gradient error for each method. Finally, panel B shows the training loss over time for each method. The Arnoldi method consistently outperforms the other two methods in terms of accuracy and training speed.


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_7_2.jpg)

> This figure compares the performance of three different solvers, Arnoldi, Dopri5, and Tsit5, in reconstructing a true coefficient field. Each subplot shows a contour plot of the reconstructed coefficient field using a specific solver. The color scale indicates the magnitude of the coefficient field, ranging from 0 to 0.014. The figure demonstrates that all three solvers accurately capture the overall shape and structure of the true coefficient field, although with varying degrees of accuracy and detail.


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_8_1.jpg)

> This figure compares the performance of Lanczos and diagonal approximation methods for optimizing the negative log-marginal likelihood of a Bayesian Visual Attention Network (VAN) during training. The y-axis represents the negative log-marginal likelihood (in millions), and the x-axis shows the number of epochs. The Lanczos method consistently achieves lower negative log-marginal likelihood values compared to the diagonal approximation method, indicating better calibration of the model parameters and ultimately, better performance. This demonstrates the effectiveness of the Lanczos iteration for approximating the log-determinant of the generalized Gauss-Newton matrix in the context of Bayesian neural networks.


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_16_1.jpg)

> The figure shows a comparison of the performance of three methods for computing gradients of matrix functions. The methods considered are backpropagation, the adjoint method, and the authors' new method. The x-axis represents the depth of the Krylov subspace used, and the y-axis represents wall-clock time in seconds. The figure shows that the authors' method is significantly faster than backpropagation, while maintaining the same linear runtime and memory complexity. The speedup provided by the new algorithm is due to the way gradients of matrix functions are computed. In standard backpropagation, gradients are obtained by backpropagating through the entire computation graph. The authors' method, on the other hand, computes gradients using the adjoint method, which is computationally more efficient and scalable. For small Krylov subspace depths, the computational time is approximately equal for all methods; however, as the Krylov subspace depth increases, backpropagation becomes significantly slower. 


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_23_1.jpg)

> The figure compares the performance of three different methods for computing gradients of a function of a sparse matrix, namely, backpropagation, the adjoint method (proposed by the authors), and a forward pass.  The x-axis represents the depth of the Krylov subspace used in the Lanczos iteration, and the y-axis represents the computation time in seconds. The results show that the adjoint method has linear time complexity, while backpropagation exhibits exponential growth in computation time as the Krylov subspace depth increases. The forward pass serves as a baseline.


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_24_1.jpg)

> This figure compares the runtime performance of different methods for computing matrix-vector products, a crucial operation in many machine learning algorithms involving large matrices.  It shows that for matrices with 10,000 or more rows/columns, the KeOps library significantly outperforms the custom JAX implementations described in the paper, despite the efforts to optimize those implementations using both `map` and `vmap` functions in JAX and varying the number of Krylov subspace iterations (K). This highlights that while the proposed method is efficient for smaller matrices, high-performance libraries like KeOps still offer significant advantages for truly large-scale problems.


![](https://ai-paper-reviewer.com/RL4FXrGcTw/figures_26_1.jpg)

> This figure shows three example pairs of input and output from the PDE dataset used in the paper. Each pair consists of a 2D spatial representation of the input and the corresponding output. The inputs represent initial conditions, while the outputs are obtained by solving the partial differential equation defined in the paper.  The visualization highlights the relationship between the input and the resulting output after applying the specified operations.


</details>




<details>
<summary>More on tables
</summary>


![](https://ai-paper-reviewer.com/RL4FXrGcTw/tables_5_1.jpg)
> This table shows the loss of accuracy when differentiating the Arnoldi iteration on Hilbert matrices of various sizes.  It compares three methods: adjoint without projection, adjoint with projection, and backpropagation. The results demonstrate that the adjoint method, especially with projection, maintains high accuracy, while backpropagation suffers from significant loss of accuracy as matrix size increases.

![](https://ai-paper-reviewer.com/RL4FXrGcTw/tables_6_1.jpg)
> This table compares the performance of the proposed method against GPyTorch on five different datasets for Gaussian process model selection.  The comparison includes RMSE, final training loss, and runtime per epoch.  The results show that the proposed method achieves similar RMSE and lower training loss but with significantly longer runtime, attributed to the differences in matrix-vector product backends.

![](https://ai-paper-reviewer.com/RL4FXrGcTw/tables_8_1.jpg)
> This table compares three different methods (Arnoldi, Dopri5, and Tsit5) for solving a physics-informed machine learning problem involving partial differential equations.  The methods are evaluated based on their test loss, parameter RMSE, and runtime per epoch.  The results show that while all three methods achieve comparable accuracy (as indicated by similar loss and RMSE values), the Arnoldi method using the authors' newly developed adjoint is significantly faster than the other two methods.

![](https://ai-paper-reviewer.com/RL4FXrGcTw/tables_9_1.jpg)
> This table compares the performance of the proposed method and GPyTorch on several datasets for Gaussian process model selection.  The RMSE, final training loss, and runtime per epoch are reported for both methods.  The results show comparable RMSEs, but the proposed method achieves lower training losses, although it is slower due to different matrix-vector product implementations.

![](https://ai-paper-reviewer.com/RL4FXrGcTw/tables_22_1.jpg)
> This table compares the performance of the proposed method and GPyTorch on several datasets for Gaussian process regression.  The metrics compared are RMSE, final training loss, and runtime per epoch. The table shows that both methods achieve similar RMSE, but the proposed method achieves lower training loss. The significant difference in runtime is attributed to the use of different matrix-vector product backends. 

![](https://ai-paper-reviewer.com/RL4FXrGcTw/tables_25_1.jpg)
> This table compares the performance of the proposed method against GPyTorch on five datasets for Gaussian process hyperparameter optimization.  The metrics considered are RMSE, final training loss, and runtime per epoch. The table shows that the proposed method achieves similar RMSE to GPyTorch, but with lower training losses, albeit at a significantly higher computational cost. This difference in speed is attributed to different matrix-vector product backends.

</details>




### Full paper

{{< gallery >}}
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/15.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/16.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/17.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/18.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/19.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="https://ai-paper-reviewer.com/RL4FXrGcTw/20.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}