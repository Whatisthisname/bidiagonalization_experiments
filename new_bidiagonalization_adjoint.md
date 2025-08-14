## Bidiagonalization returns:

$L$, $R$, $\{\alpha_i\}_{i=1}^k$, $\{\beta_i\}_{i=1}^{k-1}$, $res$, $c$


Let $B$ = `diag(alphas) + diag(betas, k=1)`

## To compute gradients:

Given: $\nabla_L, \nabla_R, \{\nabla_{\alpha_i}\}_{i=1}^k, \{\nabla_{\beta_i}\}_{i=1}^{k-1}, \nabla_{res}, \nabla_c$,


the VJP can be computed as
$$\begin{align*} 
\nabla_v &= -\kappa c
\\
\nabla_A &= \sum_{i=1}^k l_i \downarrow_i^\top + \uparrow_i r_i^\top
\end{align*}$$
with the following constraints on $\kappa$ and $\downarrow_i, \uparrow_i$:

$$\begin{align*} 
\downarrow_k &= \nabla_{res} + R \gamma
\\
r_1^\top \kappa &= c\nabla_c
\\
0 &= \nabla_L + A\Downarrow - \Uparrow B^\top + L(\Sigma + \Sigma^\top)
\\
0 &= \nabla_R + A^\top\Uparrow - \Downarrow B + R(\Omega + \Omega^\top) + \kappa e_1^\top + res \gamma^\top
\\ 
I_\geq \circ \Big(L^\top \Uparrow &= \text{diag}(\nabla_{\beta_{all}}, k=1)\Big)
\\
I_{\geq-1} \circ \Big(R^\top \Downarrow &= \text{diag}(\nabla_{\alpha_{all}})\Big)
\end{align*}$$