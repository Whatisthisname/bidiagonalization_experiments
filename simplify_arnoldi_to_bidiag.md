_Q.T @ [0, v] = Weave(R.T @ v, 0)
_Q.T @ [v, 0] = Weave(0, L.T @ v)

----

_nabla.res = [0, nabla.res]

eta = dH @ _e_K - Q.T @ _nabla.res
    = dH_k - Q.T @ _nabla.res

_Q.T @ _nabla.res = Weave(R.T @ nabla.res, 0)

_nabla.H_k = _e_{k-1} * a_k
eta = _e_{k-1} * a_k - Weave(R.T @ nabla.res, 0)

-------------------
lambda_k = _nabla.res + Q @ eta
         = _nabla.res + Q @ (dH_k - Q.T @ _nabla.res)
         = _nabla.res + Q @ dH_k - Q @ Q.T @ _nabla.res
         = _nabla.res + Q @ _e_{k-1} * a_k - Q @ Q.T @ _nabla.res
         = [0, nabla.res] + [0, r_k * a_k] - Q @ Q.T @ _nabla.res

Q @ Q.T = [[L @ L.T, 0], [0, R @ R.T]]
Q @ Q.T @ _nabla.res = [0, R @ R.T @ nabla.res]

         = [0, nabla.res] + [0, r_k * a_k] - [0, R @ R.T @ nabla.res]
         = [0, nabla.res + r_k * a_k - R @ R.T @ nabla.res]

-------------------

For even 1-indexed i:

rest := - e_i     * (alpha_{i/2} * grad_beta_{i/2 - 1} + beta_{i/2 - 1} * beta_{i/2})
       - e_{i-2} * (alpha_{i/2} * grad_alpha_{i/2})

diag(gamma) + triu (gamma.T) = triu[1, i](Q.T @ [grad_r_(i/2) + A@lambda_i, 0]) + rest
    = triu[1, i](Weave(0, L.T @ [grad_r_(i/2) + A@lambda_i, 0])) + rest
    = triu[1, i](Weave(0, L.T @ [grad_r_(i/2) + A@lambda_i, 0])) + rest

This is a vector that is zero for uneven indices