# Notes 07: a parity proof of the paper's sparsity theorem

Numerical verification: `exp16_parity_proof_sparsity_theorem.py`.

The paper proves its Theorem 4.1 (alternating block-zero structure of the
augmented adjoint variables: checkerboard Gamma, alternating gamma,
block-alternating Lambda columns, M-block lambda_0) by a multi-page
backward induction on sparsity patterns (its appendix "Sparsity proofs").
The entire theorem is a symmetry statement, and it has a half-page proof.

## 1. The involution

Let

    S = diag(I_N, -I_M)  on R^{N+M},      D = diag(s_1, ..., s_{2K}),
    s_i = (-1)^i.

The augmented data are parity-structured:

    S Aa S = -Aa        (Aa = [[0, A], [A^T, 0]] anticommutes with S)
    S va   = -va        (va = [0; v])

## 2. Primal parity (the patterns of Q, H, r)

Induction: q_1 = c va is S-odd (s_1 = -1). If q_1..q_i are S-pure with
signs s_1..s_i, then Aa q_i is S-pure with sign -s_i (anticommutation),
orthogonalization against S-pure vectors and normalization preserve
S-purity, so q_{i+1} is pure with s_{i+1} = -s_i. Hence

    S q_i = s_i q_i;    H_{ji} = q_j^T Aa q_i = 0 whenever s_j = s_i
    (same-parity inner products vanish: the factors lie in the +1 and -1
    eigenspaces of S respectively);    the residual r after 2K steps is
    S-odd (sign -s_{2K} = -1), i.e. r = [0; res].

This is exactly the alternating Q/H/r pattern of the paper's section 3
(and it holds EXACTLY in floating point as well -- the wrong-block entries
are never touched by the arithmetic: exp16 measures purity 0.0).

## 3. Equivariance of the adjoint system

The paper's Hessenberg adjoint system (its eqs. (adjoint 1)-(4)) for
multipliers (Lambda, Gamma, lambda_0, gamma, S_mult):

    (1) 0 = G_Q + Aa^T Lambda - Lambda H^T + lambda_0 e_1^T
            + Q tril(Gamma) + Q tril(Gamma)^T + r gamma^T
    (2) 0 = G_H - Q^T Lambda + I_<< o S_mult
    (3) 0 = G_r - Lambda e_2K + Q gamma
    (4) 0 = G_c - va^T lambda_0

The cotangents supplied through `extract` are parity-structured (this is
the content of the paper's eqs. for grad_Q, grad_H, grad_r patterns):

    S G_Q D = G_Q,      D G_H D = -G_H,      S G_r = -G_r.

**Lemma.** If (Lambda, Gamma, lambda_0, gamma, S_mult) solves (1)-(4),
then so does

    (Lambda', Gamma', lambda_0', gamma', S_mult')
      = (-S Lambda D,  D Gamma D,  -S lambda_0,  -D gamma,  -D S_mult D).

Proof: conjugate each equation. For (1), multiply by S on the left and D
on the right and use

    S (Aa^T Lambda) D = (S Aa S)(S Lambda D) = -Aa^T (S Lambda D),
    S (Lambda H^T) D  = (S Lambda D)(D H^T D) = -(S Lambda D) H^T
                        (D H D = -H by sec. 2),
    S lambda_0 e_1^T D = -(S lambda_0) e_1^T          (D e_1 = -e_1),
    S Q M D = Q (D M D) for any 2K x 2K M             (S Q D = Q),
    S r gamma^T D = -r (D gamma)^T                    (S r = -r),
    D tril(Gamma) D = tril(D Gamma D)                 (D diagonal).

Collecting signs, the conjugated (1) is (1) for the primed variables. For
(2): D(Q^T Lambda)D = Q^T (S Lambda D) and D G_H D = -G_H give (2) for the
primed variables after multiplying by -1. For (3): apply S and use
D e_2K = +e_2K. For (4): va^T (-S lambda_0) = +va^T lambda_0. QED

## 4. The sparsity theorem

The augmented Hessenberg constraint set is square and its adjoint
multiplier is unique (the paper's premise; equivalently the nonsingularity
of the dense system, cond ~ 1e2 in exp16). Uniqueness + the Lemma force
the solution to be a FIXED POINT of the transformation:

    Lambda = -S Lambda D    =>  S lambda_i = -s_i lambda_i
                                (lambda_i lives in the block OPPOSITE to q_i:
                                 odd i -> N-block, even i -> M-block)
    Gamma  = D Gamma D      =>  Gamma_{ji} = 0 whenever s_j s_i = -1
                                (the checkerboard)
    gamma  = -D gamma       =>  gamma_i = 0 for even i   (pattern *,0,*,0,..)
    lambda_0 = -S lambda_0  =>  lambda_0 = [0; *]        (M-block).

That is the paper's Theorem 4.1. QED

exp16 verifies at three sizes: all claimed zeros are EXACTLY 0.0 in the
dense float64 solve; the transformed multiplier satisfies the system with
residual ~1e-15 and equals the original to 0.0 (fixed point); the gradient
matches the study's stable recurrence to ~3e-15.

## 5. The pruning claim, closed

exp16 (C5) also verifies numerically what NOTES_03 sec. 5 argued
structurally: the unique augmented multiplier, pruned to its nonzero
blocks,

    Phi_n = lambda_{2n-1}|_N-part,   Xi_n = lambda_{2n}|_M-part,
    kappa = lambda_0|_M-part,        eta_n = gamma_{2n-1},

EQUALS the stable-gauge multiplier of the bidiagonal con_full system (the
one selected by the K^2 selection identities) to ~3e-15. So the chain

    augmented unique multiplier --prune--> stable gauge
    == [stationarity + selection] solution      (exp04)
    == paper's backward recurrence output       (exp03)

is now verified at every link.

## 6. Remarks for the paper

1. The parity proof replaces the backward-induction appendix and covers
   the primal patterns (sec. 2) and the adjoint patterns (sec. 4) with one
   argument. It also explains WHY the pattern is exact in floating point
   (the wrong-parity entries are never produced by the arithmetic), which
   the induction does not.
2. What parity does NOT give: lower-triangularity of Gamma (that is the
   paper's separate redefinition/convention step) and the future-support
   triangular patterns inside the nonzero bands (those do follow from the
   backward substitution order, as in the paper, or from NOTES_05).
3. The fixed-point argument needs uniqueness of the augmented multiplier
   -- the same premise the paper already relies on. If one only knows
   solvability, the argument still shows the solution SET is invariant, and
   the parity-symmetrized solution (z + z')/2 is a solution with the
   claimed zeros; combined with gradient invariance this suffices for all
   downstream uses.
