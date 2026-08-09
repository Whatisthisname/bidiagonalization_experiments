# A guide to gauge freedom, for readers who know linear algebra

Companion script: `guide_examples.py`. The script verifies every number in
this guide -- the worked examples, the float32 demonstrations, and the
solutions to all exercises.

**Prerequisites.** You need: matrices, the transpose, rank, the null
space, eigenvalues and eigenvectors (only in section 10), how to solve a
linear system, and dot products. That is all. Partial derivatives appear
only in one gentle form: "the gradient is the vector of partial
derivatives". Every function we differentiate in this guide is linear or
almost linear.

**How to read this guide.** This guide is written to be followed with pen
and paper. Every matrix is small enough to write down. Every computation
is shown, or is asked of you as an exercise. Each section ends with a
short recap that starts with "In short". The recaps repeat the main point
of the section in different words. This repetition is intentional. If a
section confuses you, read its recap, then read the section again.
Exercises appear at the ends of sections. They are tiny; each one needs
at most a few lines of writing. Full solutions are in section 15. Do the
exercises -- they are where the ideas become yours.

The guide itself is self-contained and generic: no knowledge of this
project is needed anywhere in sections 1-13. Section 14, at the end, maps
every concept onto the concrete study in this folder. Read it last, or
skip it.

**Notation.** Vectors are columns. $e_1$ is the first standard basis
vector $(1, 0, \dots, 0)^\top$. $W^\top$ is the transpose of $W$.
$\lvert v \rvert$ is the Euclidean norm. $\mathrm{null}(M)$ is the
null space of $M$: all vectors $\nu$ with $M\nu = 0$.
$\mathrm{range}(M)$ is the column space of $M$: all vectors of the
form $M z$.

## 1. The idea in one paragraph

Here is the whole idea. You want to compute some final quantity. Call
this final quantity the **observable**. Your computation does not produce
the observable directly. Instead, it first produces a helper quantity,
and then it extracts the observable from the helper quantity. Call the
helper quantity the **auxiliary quantity**. Now the key point: the
auxiliary quantity is **not uniquely determined**. Many different values
of the auxiliary quantity produce the exact same observable. This
freedom -- the freedom to change the auxiliary quantity without changing
the observable -- is called **gauge freedom**. A rule that picks one
specific value of the auxiliary quantity is called a **gauge choice**
(also: *gauge fixing*).

Everyday examples of the same pattern:

- Altitude. You measure altitude relative to an arbitrary zero, for
  example sea level. The zero is a choice. Only altitude *differences*
  are observable. The choice of zero is a gauge choice.
- Voltage. A voltage is defined up to an arbitrary ground. The ground is
  a gauge choice. Only voltage differences are observable.
- Arithmetic. You can write the number 5 as $a - b$. There is a
  one-parameter family of ways to do it: $(a, b) = (5 + t,\ t)$ for any
  $t$. The pair $(a, b)$ is the auxiliary quantity. The difference
  $a - b = 5$ is the observable. The parameter $t$ is the gauge freedom.

In each example, the observable is the same for every choice. We say the
observable is **gauge-invariant**. The auxiliary quantity is different
for every choice. We say the auxiliary quantity is **gauge-dependent**.

The word "gauge" comes from physics (section 11). But the mathematical
content is pure linear algebra, and this guide develops all of it from
scratch.

> **In short.** An observable is what you want. An auxiliary quantity is
> what you compute on the way. Gauge freedom means: many auxiliary values
> give the same observable. A gauge choice picks one of them. The
> observable is gauge-invariant. The auxiliary quantity is not.

**Exercise 1.** Write the number 7 as $a - b$. (a) Give the full family
of pairs $(a, b)$ that work. (b) Name the observable, the auxiliary
quantity, and the dimension of the gauge freedom. (c) Invent one gauge
choice (a rule that picks a single pair) and state which pair it picks.

## 2. The linear-algebra core: underdetermined consistent systems

Everything in this guide reduces to one linear-algebra fact. This section
states that fact. Read it slowly; the rest of the guide only applies it.

Consider a linear system

$$M \mu = b, \qquad M \in \mathbb{R}^{m \times n}, \quad n > \mathrm{rank}(M) = m'.$$

The matrix $M$ has more columns than its rank. So the system has more
unknowns than independent equations. Suppose the system is
**consistent**: the vector $b$ lies in $\mathrm{range}(M)$, so at
least one solution exists. Then the solution set is an **affine family**:

$$\mu = \mu_{\text{particular}} + \nu, \qquad \nu \in \mathrm{null}(M).$$

In words: take any one solution, and add any null-space vector of $M$.
The result is again a solution. All solutions arise this way. The family
has dimension $n - m'$ (the dimension of the null space of $M$).

One caution about the word "affine". The family is affine, not a
subspace. The *difference* of two solutions is in
$\mathrm{null}(M)$. But a solution itself is not in
$\mathrm{null}(M)$, unless $b = 0$. So: differences live in a
subspace; solutions live in a shifted copy of that subspace.

Now the second half of the fact. Suppose you do not report $\mu$ itself.
Instead you report a linear function of $\mu$:

$$g = W^\top \mu \qquad \text{for some fixed matrix } W.$$

When is $g$ the same for every solution of the family? Here is the chain
of equivalent conditions:

$$
\begin{aligned}
g \text{ is the same for every solution}
&\iff W^\top \nu = 0 \ \text{ for every } \nu \in \mathrm{null}(M) \\
&\iff \text{the columns of } W \text{ are orthogonal to } \mathrm{null}(M) \\
&\iff \text{the columns of } W \text{ lie in } \mathrm{range}(M^\top).
\end{aligned}
$$

If this condition holds, we call $g$ **gauge-invariant**. The
$(n - m')$-dimensional freedom in $\mu$ is the **gauge freedom**. Nothing
more is meant by the word "gauge" anywhere in this guide. To repeat:
gauge freedom = the null space of the system that defines the auxiliary
quantity; gauge invariance = the reported quantity is blind to that null
space.

> **In short.** A consistent system with more unknowns than rank has an
> affine family of solutions. The family is "one particular solution plus
> the null space". A linear report $g = W^\top \mu$ is the same for all
> solutions exactly when the columns of $W$ are orthogonal to the null
> space. That is all "gauge" means here.

**Exercise 2.** Let $M = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}$ (one
row, three columns) and $b = 2$, so the system is
$\mu_1 + \mu_2 + \mu_3 = 2$. (a) Give the solution family and its
dimension. (b) Show that the report $g = w^\top \mu$ with
$w = (1, 1, 1)^\top$ is gauge-invariant, and give its value. (c) Show
that the report with $w = (1, 2, 3)^\top$ is *not* gauge-invariant, by
exhibiting two solutions that give different values of $g$.

## 3. Where this appears: gradients through constraints

This section explains where the pattern of section 2 appears in
computational practice: the adjoint method for gradients.

The setup. A solver takes parameters $\theta$ and produces outputs $x$.
The outputs are characterized implicitly, by a list of constraint
equations:

$$\mathrm{con}(x, \theta) = 0 \qquad (\text{a vector of } d_O \text{ scalar equations}).$$

The meaning of this list: for each $\theta$, the solver output
$x(\theta)$ is exactly the $x$ that satisfies all the equations. The
constraints *define* the output. We want the gradient of a scalar loss
$\rho(x)$ with respect to $\theta$.

Write down the two Jacobian matrices. A Jacobian is a matrix of partial
derivatives. Both are evaluated at the solution:

$$J_x = \frac{\partial\, \mathrm{con}}{\partial x} \ \ (d_O \text{ rows},\ d_U \text{ columns}), \qquad
J_\theta = \frac{\partial\, \mathrm{con}}{\partial \theta} \ \ (d_O \text{ rows},\ P \text{ columns}),$$

where $d_U = \dim(x)$ and $P = \dim(\theta)$. Now one identity. The
equation $\mathrm{con}(x(\theta), \theta) = 0$ holds *identically in*
$\theta$: it is true for every $\theta$, not just one. So we may
differentiate both sides with respect to $\theta$. The chain rule gives,
with $S = dx/d\theta$ (a $d_U \times P$ matrix):

$$J_x S + J_\theta = 0. \tag{$*$}$$

Keep identity $(*)$ in mind. It is used twice below, and it is the entire
engine of section 5.

The gradient we want is, again by the chain rule,

$$\nabla_\theta \rho = S^\top \nabla_x \rho.$$

We could compute $S$ directly. That is expensive: one linear solve per
parameter, and there are $P$ parameters. The **adjoint method** avoids
computing $S$. It works in two steps. Step one: find any vector $\mu$
solving

$$J_x^\top \mu = -\nabla_x \rho. \tag{A}$$

Step two: read off the gradient with one matrix-vector product:

$$
\begin{aligned}
\nabla_\theta \rho &= S^\top \nabla_x \rho          && \text{(chain rule)} \\
                   &= -S^\top J_x^\top \mu          && \text{(substitute (A))} \\
                   &= -(J_x S)^\top \mu             && \text{(transpose of a product)} \\
                   &= J_\theta^\top \mu             && \text{(by } (*)\text{: } J_x S = -J_\theta\text{)}.
\end{aligned} \tag{G}
$$

One linear solve (A), one product (G), done. No $S$ anywhere.

**The word "multiplier", and how to read it.** The vector $\mu$ is
traditionally called a **Lagrange multiplier** (after Lagrange's method
for constrained optimization, where the same object appears). The name is
historical; do not look for extra meaning in the word itself. Whenever
this guide says "the multiplier $\mu$", read it as:

> **the vector of weights, one weight per constraint equation.**

The shape says the same thing: $\mu$ has $d_O$ entries, exactly one entry
$\mu_i$ for each constraint row $\mathrm{con}_i$. And the report (G)
says what the weights are *for*. Written as a sum over constraint rows,

$$\nabla_\theta \rho = J_\theta^\top \mu
= \sum_{i=1}^{d_O} \mu_i \cdot \big(\text{$\theta$-gradient of constraint } i\big).$$

Each constraint row knows how it depends on $\theta$ (its row of
$J_\theta$). The weight $\mu_i$ says how much of row $i$'s
$\theta$-dependence flows into the loss. System (A) is the rule that
determines the weights; report (G) spends them.

**The weights are sensitivities.** There is a second reading, and many
people find it the most concrete of all. Perturb constraint $i$: demand
$\mathrm{con}_i(x, \theta) = \epsilon_i$ instead of $0$. The solution $x$
moves, so the loss moves. Fact:

$$\frac{\partial \rho}{\partial \epsilon_i} = -\mu_i.$$

In words: the $i$-th weight is (minus) the sensitivity of the loss to a
small violation of the $i$-th constraint. A constraint whose violation
would barely change the loss gets a small weight. A constraint whose
violation would change the loss a lot gets a large weight. (This reading
is clean when the constraint rows are independent, so that each
$\epsilon_i$ can be turned on separately. Exercise 3 verifies it on a
small case.)

Now map the adjoint method onto section 1's vocabulary, because this is
the whole point. The multiplier $\mu$ (the weight vector) is the
auxiliary quantity. The gradient is the observable. System (A) is the
system "$M\mu = b$" of section 2, with $M = J_x^\top$ and
$b = -\nabla_x \rho$. The report (G) is the linear function
"$g = W^\top \mu$" of section 2, with $W = J_\theta$.

> **In short.** The adjoint method computes a gradient in two steps.
> Step one solves $J_x^\top \mu = -\nabla_x \rho$ for the multiplier
> $\mu$ -- a vector of weights, one per constraint row, where $\mu_i$ is
> (minus) the sensitivity of the loss to a violation of constraint $i$.
> Step two reports the gradient as the weighted sum
> $J_\theta^\top \mu$. The multiplier is the auxiliary quantity. The
> gradient is the observable. Section 2 applies verbatim with
> $M = J_x^\top$ and $W = J_\theta$.

**Exercise 3.** Take one parameter $\theta$, two outputs, two
constraints, and a loss:

$$c_1: x_1 - \theta = 0, \qquad c_2: x_2 - 2\theta = 0, \qquad
\rho(x) = x_1 + x_2.$$

(a) Here $J_x$ is the $2 \times 2$ identity. Solve system (A) for the
multiplier $\mu$. (b) Now verify the sensitivity reading by hand: replace
the constraints by $x_1 - \theta = \epsilon_1$ and
$x_2 - 2\theta = \epsilon_2$, solve for $x$ explicitly, substitute into
$\rho$, and compute $\partial\rho / \partial\epsilon_i$. Compare with
$-\mu_i$.

## 4. Redundant constraints and the transpose inversion

This section explains a counting subtlety that is easy to get wrong, so
we state it twice, from both directions.

Suppose the constraint list is **redundant**. That means: it has $d_O$
rows, but its rank is only $d_U < d_O$. Every row beyond a spanning set
is implied by the other rows. Write $r = d_O - d_U$ for the number of
redundant rows. Two facts are now true at the same time, and they sound
like opposites:

- The **primal** system $\mathrm{con}(x, \theta) = 0$ is a system for
  $x$. It looks overdetermined: it has more equations ($d_O$) than
  unknowns ($d_U$). But it is consistent: the solver output satisfies all
  $d_O$ equations. Redundant equations do not make a consistent system
  unsolvable. They are just repeated information.
- The **adjoint** system (A) is a system for the weight vector $\mu$. It
  is genuinely *underdetermined*. Count carefully: $J_x^\top$ is a
  $d_U \times d_O$ matrix. So system (A) has $d_U$ equations (one per
  output coordinate) and $d_O$ unknowns (one weight per constraint row).
  More unknowns than equations.

Here is the same statement again, as a slogan. **Constraint rows become
multiplier unknowns under transposition.** Each extra (redundant)
constraint row adds nothing to the primal system. But it adds one more
weight to the adjoint system. So "more constraints than outputs" flips
into "fewer equations than unknowns for the weights". The multiplier
solutions form an affine family (section 2) of dimension

$$d_O - \mathrm{rank}(J_x) = d_O - d_U = r.$$

That family is the gauge freedom of the multiplier.

Why is this easy to trip over? Because the words "overdetermined" and
"underdetermined" attach to the *same matrix*, and the correct word
depends on which side you look from. Looking at $J_x$ (primal side): tall
matrix, overdetermined-looking, consistent. Looking at $J_x^\top$
(adjoint side): wide matrix, underdetermined, affine solution family.
Same data, opposite adjectives.

> **In short.** Redundant constraints keep the primal solvable -- the
> extra rows are repeated information. But each constraint row is one
> weight in the multiplier. So redundancy makes the adjoint system
> underdetermined. The multiplier is not unique. It has a gauge family
> whose dimension $r = d_O - d_U$ is the number of redundant rows.

**Exercise 4.** Take the constraint Jacobian

$$J_x = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & -1 \end{bmatrix}.$$

(a) State $d_O$, $d_U$ (the rank), and the gauge dimension $r$. (b) Write
out the two equations of $J_x^\top \nu = 0$ and find a basis of
$\mathrm{null}(J_x^\top)$: two independent null vectors.

## 5. Why the gradient does not care: the invariance lemma

The multiplier is not unique. Does that make the gradient ambiguous? No.
This section proves it. The proof is three lines.

**Lemma.** If $\nu \in \mathrm{null}(J_x^\top)$, then
$J_\theta^\top \nu = 0$. Consequently, every solution of (A) yields the
same gradient through (G).

**Proof.** Transpose identity $(*)$: $S^\top J_x^\top + J_\theta^\top = 0$.
Apply both sides to $\nu$:
$S^\top (J_x^\top \nu) + J_\theta^\top \nu = 0$. The first term is
$S^\top 0 = 0$. Hence $J_\theta^\top \nu = 0$. QED

Read the lemma once more in the language of section 2. There, gauge
invariance required: the columns of $W$ lie in the orthogonal complement
of $\mathrm{null}(M)$. Here, $M = J_x^\top$ and $W = J_\theta$.
Identity $(*)$ says $J_\theta = -J_x S$. So every column of $J_\theta$ is
a combination of columns of $J_x$. So the columns of $J_\theta$ lie in
$\mathrm{range}(J_x)$. And $\mathrm{range}(J_x)$ is precisely
the orthogonal complement of $\mathrm{null}(J_x^\top)$. So the
invariance condition of section 2 holds automatically. It is not luck. It
is forced by identity $(*)$.

One caution, which will matter later. Identity $(*)$ holds because the
constraints hold *identically* along the solution path. If your
"constraints" are only approximately satisfied -- for example, at a
floating-point solution -- then invariance is only approximate. See
pitfall 4 in section 12.

So we have the central dichotomy of this guide. Say it twice:

- **The gradient is gauge-invariant.** Every member of the multiplier
  family gives the same gradient, exactly, in exact arithmetic.
- **The multiplier is gauge-dependent.** Different members of the family
  are genuinely different weight vectors. Sometimes very different.

Any member of the family is equally *correct*. The rest of this guide
explains why the choice of member still *matters*.

> **In short.** Null vectors of $J_x^\top$ are invisible to
> $J_\theta^\top$. That is forced by the identity $J_x S = -J_\theta$.
> Therefore all multipliers in the gauge family give the same gradient.
> The gradient is gauge-invariant; the multiplier is not. Correctness
> does not depend on the gauge choice. Numerical accuracy will (sections
> 7 and 8).

**Exercise 5.** Peek ahead to example A in section 6 and take its data:
$J_\theta = (-1, -2, 0)^\top$ and the null vector $\nu = (-2, 1, 1)^\top$
of $J_x^\top$. (a) Compute $J_\theta^\top \nu$ by hand. (b) State in one
sentence why the lemma forced the value you got, before you computed it.

## 6. Worked example A: a redundant constraint set, by hand

This example shows everything from sections 2-5 in the smallest possible
size. You can verify every line by hand. The companion script verifies it
too.

The setup: one parameter $\theta$, two outputs $x = (x_1, x_2)$, and
three constraints. The third constraint is redundant -- it follows from
the first two:

$$
\begin{aligned}
c_1 &: & x_1 - \theta &= 0 \\
c_2 &: & x_2 - 2\theta &= 0 \\
c_3 &: & 2 x_1 - x_2 &= 0 \qquad (\text{redundant: } 2 c_1 - c_2 \text{ implies it}).
\end{aligned}
$$

The solution is $x(\theta) = (\theta, 2\theta)$. Take the loss
$\rho(x) = x_1 + x_2$. Then $\nabla_x \rho = (1, 1)^\top$. The true
gradient is easy to compute directly, without any multipliers: along the
solution path, $\rho = \theta + 2\theta = 3\theta$, so

$$\frac{d\rho}{d\theta} = 3.$$

So the correct answer is $3$. Every method below must produce $3$.

The Jacobians (one row per constraint, in the order $c_1, c_2, c_3$; one
column of $J_x$ per output, one column of $J_\theta$ per parameter):

$$J_x = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 2 & -1 \end{bmatrix}, \qquad
J_\theta = \begin{bmatrix} -1 \\ -2 \\ 0 \end{bmatrix}.$$

Note the counts: $d_O = 3$ constraints, $d_U = 2$ outputs. So the gauge
dimension should be $r = d_O - d_U = 1$. Let us confirm that.

For pen and paper, first write down the transpose explicitly:

$$J_x^\top = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & -1 \end{bmatrix}.$$

The adjoint system (A) is $J_x^\top \mu = -(1, 1)^\top$. Each *row* of
$J_x^\top$ gives one equation. Written out, that is two equations in
three unknown weights:

$$
\begin{aligned}
\mu_1 \phantom{ + \mu_2 } + 2\mu_3 &= -1 \\
\mu_2 - \mu_3 &= -1.
\end{aligned}
$$

Two equations, three unknowns: underdetermined, as section 4 predicted.
Solve for $\mu_1$ and $\mu_2$ in terms of $\mu_3$. The general solution,
with free parameter $t := \mu_3$, is:

$$\mu(t) = (-1 - 2t,\ -1 + t,\ t).$$

Check the null-space structure. The vector $\nu = (-2, 1, 1)^\top$
satisfies $J_x^\top \nu = 0$; verify by hand:
$-2 + 0 + 2 = 0$ (first row) and $0 + 1 - 1 = 0$ (second row). And indeed
$\mu(t) = \mu(0) + t\nu$: one particular solution plus the null space,
exactly as section 2 said. The gauge freedom is one-dimensional, matching
$r = 3 - 2 = 1$.

Now the gradient, from formula (G), for an *arbitrary* $t$:

$$
\begin{aligned}
J_\theta^\top \mu(t) &= (-1)(-1 - 2t) + (-2)(-1 + t) + 0 \cdot t \\
                     &= 1 + 2t + 2 - 2t \\
                     &= 3.
\end{aligned}
$$

The $t$-terms cancel. The gradient is $3$ for every $t$. Invariant, as
the lemma of section 5 promised. The auxiliary quantity $\mu(t)$ moves;
the observable does not.

Pause on the weights reading. At $t = 0$ the weights are
$\mu = (-1, -1, 0)$: the loss is sensitive to violations of $c_1$ and
$c_2$, and the redundant row $c_3$ carries weight zero. At $t = 1$ the
weights are $(-3, 0, 1)$: now $c_2$ carries no weight, and its share of
the $\theta$-dependence has been rerouted through $c_1$ and $c_3$. The
gauge freedom is precisely this rerouting freedom: because the rows are
linearly dependent, the same total can be assembled with different
weights.

Here are three different gauge choices. All three are "correct":

| gauge rule | $t$ | $\mu$ |
|---|---|---|
| "drop the redundant row" | $0$ | $(-1,\ -1,\ 0)$ |
| minimal norm (least squares) | $-1/6$ | $(-2/3,\ -7/6,\ -1/6)$ |
| "pin $\mu_2 = 0$" | $1$ | $(-3,\ 0,\ 1)$ |

All three produce gradient $3$. They differ only in which member of the
family they select. (Exercise 7 asks you to derive the minimal-norm value
$t = -1/6$ yourself.)

> **In short.** Three constraints, two outputs, so a one-dimensional
> multiplier family $\mu(t)$. The gradient equals $3$ for every $t$ --
> the free parameter cancels in the assembly. Different gauge rules pick
> different $t$, that is, different but equally valid weightings of the
> dependent constraint rows. All of them are correct.

**Exercise 6.** Repeat example A from start to finish with a different
loss: $\rho(x) = x_2$, so $\nabla_x \rho = (0, 1)^\top$. (a) Predict the
correct gradient directly from $x(\theta) = (\theta, 2\theta)$. (b) Write
out the two equations of the adjoint system and give the multiplier
family $\mu(t)$. (c) Assemble the gradient $J_\theta^\top \mu(t)$ and
confirm it matches your prediction for every $t$.

**Exercise 7.** Derive the minimal-norm gauge value $t = -1/6$ of
example A. Hint: the minimal-norm member of an affine family
$\mu(t) = \mu(0) + t\nu$ is the one orthogonal to the null space, so
solve $\mu(t) \cdot \nu = 0$ for $t$.

## 7. Why the choice matters: cancellation, worked example B

Sections 5 and 6 showed that every gauge choice gives the correct
gradient *in exact arithmetic*. This section shows what changes in
floating-point arithmetic. The answer: the size of the multiplier decides
how many digits you lose. A big weight vector destroys the answer. A
small weight vector preserves it.

Take the same solution manifold as example A, but write the third
(redundant) constraint as the sum $c_1 + c_2$:

$$c_3' :\quad x_1 + x_2 - 3\theta = 0.$$

Then the last row of $J_x$ becomes $(1, 1)$, and the last entry of
$J_\theta$ becomes $-3$:

$$J_x = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
J_\theta = \begin{bmatrix} -1 \\ -2 \\ -3 \end{bmatrix}.$$

Write out the adjoint system (A), row by row of $J_x^\top$, with
$\nabla_x \rho = (1,1)^\top$ as before:

$$
\begin{aligned}
\mu_1 + \mu_3 &= -1 \\
\mu_2 + \mu_3 &= -1.
\end{aligned}
$$

With free parameter $t := \mu_3$, the multiplier family and the assembled
gradient are

$$\mu(t) = (-1 - t,\ -1 - t,\ t), \qquad
J_\theta^\top \mu(t) = (1 + t) + (2 + 2t) - 3t = 3 \quad \text{for every } t.$$

Again: gradient $3$, for every $t$. Now deliberately pick an *absurd*
gauge, $t = 10^8$:

$$\mu = (-100000001,\ -100000001,\ 100000000).$$

In exact arithmetic the gradient is still $3$: the assembly computes

$$(1 + 10^8) + (2 + 2 \cdot 10^8) - 3 \cdot 10^8 = 3,$$

and the enormous terms cancel. But now run the same assembly in float32.
Float32 carries about 7 decimal digits. The intermediate value
$1 + 10^8$ rounds to $10^8$: the "$+1$" is below the resolution of
numbers that large. The same happens to the "$+2$". After rounding, the
assembly computes $10^8 + 2 \cdot 10^8 - 3 \cdot 10^8 = 0$. The result is
**$0.0$ instead of $3$** (`guide_examples.py` shows this).

What happened? The answer ($3$) is small. The intermediate terms (about
$10^8$) are large. To produce a small answer from large terms, the
assembly must *cancel* the large terms against each other. Cancellation
eats digits: to cancel 8 orders of magnitude, you need at least 8 spare
digits of precision. Float32 does not have 8 spare digits. So the answer
is lost.

The general law. If the representation forces the multiplier norm
$\lvert\mu\rvert$ to be large while the answer norm $\lvert\nabla\rvert$
is small, then the assembly (G) must cancel
$\log_{10}(\lvert\mu\rvert / \lvert\nabla\rvert)$ decimal digits. The
achievable relative error in precision $\varepsilon$ is therefore about

$$\varepsilon \cdot \frac{\lvert\mu\rvert}{\lvert\nabla\rvert}.$$

Check the law on this example:
$1.2 \times 10^{-7} \cdot 10^8 / 3 \approx 4$. A relative error of order
one means: no correct digits at all. Total loss. That matches what we
observed ($0.0$ instead of $3$).

So here is the design rule, stated twice. **A gauge is numerically good
exactly to the extent that it keeps the multiplier small relative to the
answer.** Equivalently: among all correct weight vectors, prefer the
small ones, because the error floor scales with $\lvert\mu\rvert$. The
minimal-norm choice (least squares) is the mathematical optimum of this
criterion; cheap near-optimal choices exist too (section 9).

> **In short.** All gauges are correct; they are not all equally
> computable. Assembling a small answer from a large weight vector
> requires cancellation, and cancellation costs
> $\log_{10}(\lvert\mu\rvert/\lvert\nabla\rvert)$ digits. The error floor
> is $\varepsilon \cdot \lvert\mu\rvert / \lvert\nabla\rvert$. Keep the
> multiplier small.

**Exercise 8.** You work in float32 ($\varepsilon \approx 1.2 \times
10^{-7}$). Your gradient has size $\lvert\nabla\rvert = 3$, and you need
it with relative error at most $10^{-3}$ (about 3 correct digits). Using
the error-floor law, how large may $\lvert\mu\rvert$ be, at most?

## 8. Unique does not mean safe: formulation gauge, worked example C

Example B might look like an unforced error. Nobody would *choose*
$t = 10^8$. Just pick a sensible $t$, and the problem goes away --
right? There is a subtler trap, and it deserves its own toy example. The
trap: the huge multiplier can be *forced on you* by an innocent-looking
reformulation, with no visible parameter to choose.

Suppose you eliminate the redundancy entirely. Use a **square**
constraint set: exactly as many constraints as outputs. Then $J_x^\top$
is square and nonsingular, system (A) has a *unique* solution, and there
is no gauge family at all. Sounds safe. Here are two equivalent square
formulations of the same solution $x(\theta)$:

$$
\begin{aligned}
F_1 &: \quad \{\, x_1 - \theta = 0, \qquad\qquad\qquad\ \ x_2 - 2\theta = 0 \,\} \\
F_2 &: \quad \{\, x_1 - \theta = 0, \qquad 10^8 (x_1 - \theta) + (x_2 - 2\theta) = 0 \,\}.
\end{aligned}
$$

Look at $F_2$'s second row. It is $10^8$ times the first row, plus the
old second row. On the solution manifold, both rows of both formulations
are zero. So $F_1$ and $F_2$ describe exactly the same solution set. They
are equivalent formulations. But their unique multipliers differ wildly.
For $F_1$:

$$J_x = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \qquad
J_x^\top \mu = -(1,1)^\top \ \Rightarrow\ \mu = (-1,\ -1). \qquad \text{Small.}$$

For $F_2$:

$$J_x = \begin{bmatrix} 1 & 0 \\ 10^8 & 1 \end{bmatrix}, \qquad
J_x^\top \mu = -(1,1)^\top \ \Rightarrow\
\begin{cases} \mu_1 + 10^8 \mu_2 = -1 \\ \mu_2 = -1 \end{cases}
\ \Rightarrow\ \mu = (10^8 - 1,\ -1). \qquad \text{Huge.}$$

The weights reading explains the size. $F_2$'s second row is dominated by
$10^8$ copies of the first row. To recover the modest sensitivity of the
loss to the *first* physical relation, the weight on row 1 must cancel
$10^8$ times the weight on row 2. The formulation forces enormous
weights on rows whose actual information content is modest.

Both formulations produce gradient $3$ in exact arithmetic
(`guide_examples.py` checks both; for $F_2$,
$J_\theta = (-1,\ -(10^8 + 2))^\top$ and the assembly cancels to $3$).
But $F_2$'s assembly must again cancel 8 digits, and in float32 it again
returns **$0.0$**. Note what did *not* happen here: nobody picked a bad
free parameter. There is no free parameter. The multiplier of a square
system is forced. The bad choice was made earlier and invisibly -- in
*which equations were written down*.

Two morals, and they are the punchline of the whole guide:

1. Equivalent constraint sets are related by an invertible on-manifold
   mixing of their rows. Their multipliers are then related by the
   transpose-inverse of that mixing. So choosing the constraint
   *functions* is choosing the gauge -- even when the multiplier is
   unique. Call this **formulation gauge**.
2. Squareness and uniqueness buy you nothing numerically. What decides
   your numerical fate is $\lvert\mu\rvert / \lvert\nabla\rvert$, and
   nothing else.

> **In short.** A square constraint set has a unique multiplier, but
> "unique" does not mean "small". Rewriting the constraints (an
> equivalent formulation) silently transforms the unique weight vector,
> and can make it enormous. That is a gauge choice hidden in the
> formulation itself. The diagnostic is always the same number:
> $\lvert\mu\rvert / \lvert\nabla\rvert$.

**Exercise 9.** Generalize $F_2$: replace the factor $10^8$ by a constant
$c$, so the second constraint is $c\,(x_1 - \theta) + (x_2 - 2\theta) = 0$
and $J_\theta = (-1,\ -(c + 2))^\top$. (a) Find the unique multiplier
$\mu(c)$ by hand. (b) Assemble the gradient and check that it equals $3$
for every $c$. (c) Using the error-floor law with float32, roughly how
large may $c$ be before the computed gradient becomes useless (relative
error of order one)?

## 9. Gauge fixing done right

Sections 7 and 8 established the goal: keep the multiplier small. This
section describes how to fix the gauge well.

To make the multiplier unique, you append extra equations to system (A).
You need exactly as many extra equations as the gauge dimension ($r$ in
general, $1$ in example A). You must choose them so that the augmented
square system is nonsingular. Any such rule is a gauge fix. Good gauge
fixes have three properties:

1. **They keep the representative small** (sections 7-8). The
   minimal-norm choice is optimal, but it requires a least-squares solve.
   Pinning components of $\mu$ to known $O(1)$ values is nearly as good
   and much cheaper.
2. **They pin things you know in advance.** Suppose the extra equations
   say: "this linear functional of $\mu$ equals this number", and the
   number is known *before* the solve starts. Then you gain something
   beyond uniqueness. If the solve is iterative and rounding makes the
   pinned functional drift away from its known value, you can
   *re-enforce* it -- overwrite the drifting value with the known one --
   cheaply, at any point during the computation. This re-enforcement is
   called **reprojection**. Section 10 shows, in two dimensions, how
   dramatic the effect can be.
3. **They respect the solve order.** If the solve proceeds step by step,
   the quantities pinned at step $n$ should be exactly the components
   that the step-$n$ equations need.

> **In short.** A gauge fix appends one equation per gauge dimension. A
> good gauge fix keeps the multiplier small, pins values that are known
> in advance (so they can be re-enforced mid-solve -- that is
> reprojection), and pins them in the order the solve needs them.

## 10. Worked example D: reprojection, in two dimensions

This section demonstrates reprojection in the smallest case that can
show it: a two-dimensional iteration with one pinned quantity. Everything
is checkable by hand, and `guide_examples.py` runs the float32
experiment.

**The setup.** Many adjoint solves are *recurrences*: they build the
multiplier step by step, $w_0 \to w_1 \to w_2 \to \cdots$, instead of
solving one big system at once. Model this with the simplest possible
recurrence, repeated multiplication by a fixed matrix:

$$w_{n+1} = G w_n, \qquad
G = \begin{bmatrix} 3 & 1 \\ 0 & 1/3 \end{bmatrix}.$$

The matrix $G$ has eigenvalues $3$ and $1/3$ (read them off the
diagonal, since $G$ is triangular). The corresponding right eigenvectors
are

$$u = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \ (\text{eigenvalue } 3, \text{ growing}),
\qquad
v = \begin{pmatrix} 3 \\ -8 \end{pmatrix} \ (\text{eigenvalue } 1/3, \text{ shrinking}).$$

Exercise 10 asks you to verify these. Start the iteration on the
shrinking eigenvector: $w_0 = v$. Then the exact solution is

$$w_n = 3^{-n} v,$$

a sequence that shrinks by a factor of $3$ every step. That is the
answer a computer should reproduce.

**The pinned quantity.** Here is the one extra ingredient, and it is the
heart of the example. There is a *left* eigenvector for the eigenvalue
$3$: a row vector $\ell^\top$ with $\ell^\top G = 3 \ell^\top$. It is

$$\ell = \begin{pmatrix} 8 \\ 3 \end{pmatrix},$$

and it has the key property $\ell \cdot v = 24 - 24 = 0$ (a left
eigenvector is orthogonal to the right eigenvectors of the *other*
eigenvalues). Consequently the number $p_n := \ell \cdot w_n$ satisfies a
recurrence of its own:

$$p_{n+1} = \ell^\top G\, w_n = 3\, \ell^\top w_n = 3 p_n,$$

and it starts at $p_0 = \ell \cdot v = 0$. So in exact arithmetic,

$$\ell \cdot w_n = 0 \quad \text{for every } n.$$

This is a **pinned identity**: a linear functional of the running
quantity whose value we know in advance, at every step, before computing
anything. It plays exactly the role of the gauge-fixing rows of
section 9. And note what the functional *measures*: since
$\ell \cdot v = 0$ and $\ell \cdot u = 8 \ne 0$, the number
$\ell \cdot w$ is (8 times) the coefficient of the growing mode $u$
inside $w$. The pin says: "the growing mode is absent". The gauge fix
pins exactly the direction that the dynamics amplify.

**What float32 does without help.** Run the iteration in float32. Each
step commits a relative rounding error of order
$\varepsilon \approx 10^{-7}$, in an essentially arbitrary direction --
including the direction $u$. The dynamics then take over: the true
answer shrinks by $3$ per step, while the injected $u$-component *grows*
by $3$ per step. The relative error therefore grows by a factor of $9$
per step:

$$\text{relative error after } n \text{ steps} \ \approx\ \varepsilon \cdot 9^{\,n}.$$

It reaches order one -- total loss -- when $9^n \approx 1/\varepsilon$,
that is, after only $n \approx 7$ or $8$ steps (exercise 11). The
experiment confirms it: the naive float32 iteration has relative error
$7 \times 10^{-3}$ at step 7, $4.8$ at step 10 (dead), and $10^{10}$ at
step 20. The computed vector is soon a pure multiple of $u$, pointing in
a completely wrong direction and growing while the true answer shrinks.

**Reprojection.** Now use the pinned identity. After every multiplication
by $G$, re-enforce $\ell \cdot w = 0$ by removing the offending
component along $u$:

$$w \ \leftarrow\ w - \frac{\ell \cdot w}{\ell \cdot u}\, u
\ =\ w - \frac{\ell \cdot w}{8}\, u.$$

This correction is exact surgery (exercise 12): decompose
$w = a v + b u$; since $\ell \cdot v = 0$, the measured value is
$\ell \cdot w = 8 b$; the update subtracts exactly $b u$ and leaves the
$v$-part untouched. On exact data ($b = 0$) the correction does nothing
at all -- reprojection never changes the mathematics. On rounded data it
deletes precisely the component that the dynamics would amplify. The
cost is two dot products per step.

The float32 experiment with reprojection: relative error
$3 \times 10^{-7}$ at step 10, $1.3 \times 10^{-6}$ at step 40,
$2 \times 10^{-6}$ at step 60. The instability is gone. The error stays
at the precision floor for as long as you care to iterate.

**Why the pinned value had to be known in advance.** Suppose you knew
only that "some quantity $\ell \cdot w$ should be preserved", but its
target value had to be *computed* -- say, by carrying $p_n$ along in
float32 too. Then the drift of your computed target would match the
drift of $w$ itself, and the correction would remove nothing. The
correction works because the target ($0$, here; loss-supplied numbers in
general) comes from outside the unstable loop. This is exactly why
property 2 of section 9 insists that a good gauge fix pins values known
in advance.

> **In short.** An unstable recurrence amplifies rounding noise along its
> growing mode: error $\sim \varepsilon \cdot 9^n$ here, total loss in 8
> steps. A pinned identity -- a functional of the running quantity whose
> value is known beforehand -- lets you delete the amplified component at
> every step for the cost of a dot product. That is reprojection. It
> changes nothing in exact arithmetic and changes everything in float32.
> The best pins are aimed at the directions the dynamics amplify.

**Exercise 10.** Verify the eigen-structure of example D by hand:
(a) $G v = \tfrac{1}{3} v$ for $v = (3, -8)^\top$;
(b) $\ell^\top G = 3 \ell^\top$ for $\ell = (8, 3)^\top$;
(c) $\ell \cdot v = 0$ and $\ell \cdot u = 8$;
(d) from (b), derive the pinned-identity recurrence
$\ell \cdot w_{n+1} = 3 \, (\ell \cdot w_n)$.

**Exercise 11.** In float32 ($\varepsilon \approx 1.2 \times 10^{-7}$),
the naive iteration's relative error grows like $\varepsilon \cdot 9^n$.
(a) After how many steps does it reach order one? (b) A colleague
proposes to fix the problem by iterating in float64
($\varepsilon \approx 2.2 \times 10^{-16}$) instead. How many steps does
that buy before total loss? Does higher precision *solve* the problem or
only postpone it?

**Exercise 12.** Prove that the reprojection step of example D is exact
surgery. Write $w = a v + b u$ and show: (a) $\ell \cdot w = 8b$;
(b) the update $w \leftarrow w - \tfrac{\ell \cdot w}{8} u$ maps
$a v + b u$ to $a v$: the growing component is removed exactly and the
shrinking component is untouched; (c) if $w$ is exact ($b = 0$), the
update does nothing.

## 11. Why "gauge"? (the name, for context only)

This section explains the name. You do not need it to use the concept.

In electromagnetism, the observable fields $E$ and $B$ are computed from
auxiliary potentials $(\phi, A)$. The potentials can be shifted by any
function $\chi$ (namely $A \to A + \nabla\chi$,
$\phi \to \phi - \partial\chi/\partial t$) without changing $E$ and $B$.
Choosing $\chi$ is "choosing a gauge" (Coulomb gauge, Lorenz gauge, ...).
The term descends from a 19th-century use of *gauge* as "a standard of
measurement", as in railway track gauge.

The dictionary into our setting: the multiplier plays the role of the
potential. It is an auxiliary quantity, determined only up to a family.
The gradient plays the role of the fields $E$ and $B$: it is the
observable, and it is extracted from the auxiliary quantity in a
gauge-invariant way. The analogy is exact at the linear-algebra level of
section 2, and you never need any physics to use it.

> **In short.** Physics computes observable fields from non-unique
> potentials; we compute an observable gradient from non-unique
> multipliers. Same structure, hence the same word.

## 12. Common pitfalls

Each pitfall below is a sentence someone might reasonably believe, plus
the reason it is wrong. The first three are the most common.

1. **"More constraints than unknowns means no solution."** Split this by
   side. Primal side: the redundant system is still consistent (its rank
   is $d_U$), so it is exactly solvable; the extra rows are repeated
   information, not contradictions. Adjoint side: transposition turns the
   redundancy into underdetermination (section 4). Same matrix, opposite
   adjectives, both harmless for correctness.
2. **"The multiplier system is underdetermined, so the gradient is
   ambiguous."** No. The gradient map is constant on the whole solution
   family (section 5). The ambiguity is in the representation, never in
   the observable.
3. **"Make the constraint set square and the problem disappears."** The
   multiplier becomes unique -- but possibly enormous. Uniqueness is not
   smallness (section 8). Practical diagnostic: compute
   $\lvert\mu\rvert / \lvert\nabla\rvert$ at small sizes and watch how it
   grows with problem size.
4. **Invariance is on-manifold.** The lemma of section 5 uses
   $\mathrm{con}(x(\theta), \theta) = 0$ *identically*. At a
   floating-point solution, the constraints hold only to
   $\varepsilon_{\text{primal}}$. Then gauge moves change the gradient by
   $O(\varepsilon_{\text{primal}} \cdot \lvert\text{move}\rvert)$. Large
   gauge vectors therefore leak primal error into the gradient. This
   sharpens the error law to
   $(\varepsilon_{\text{arith}} + \varepsilon_{\text{primal}}) \cdot
   \lvert\mu\rvert / \lvert\nabla\rvert$.
5. **Gauge freedom is not a change of basis.** A basis change
   reparametrizes everything, invertibly. Gauge freedom moves you along a
   family of equally valid representatives of the *same* object, in the
   *same* coordinates. The two interact, but they are different
   operations.
6. **The free parameters need not look free.** A formulation can carry a
   catastrophic gauge choice with no visible parameter anywhere: the
   choice is hidden in *which equations were written down* (formulation
   gauge, section 8).
7. **Reprojection is not "fudging the numbers".** The reprojection
   correction is zero on exact data (exercise 12c). It never alters the
   mathematical iteration. It only removes noise components that the
   dynamics would amplify -- and it can do so only because the pinned
   values come from outside the unstable loop (section 10).

## 13. The whole guide in nine sentences

1. An observable is computed via an auxiliary quantity; when many
   auxiliary values give the same observable, that freedom is gauge
   freedom, and picking one value is a gauge choice.
2. In linear algebra, gauge freedom is the null space of the consistent
   underdetermined system that defines the auxiliary quantity.
3. In the adjoint method, the auxiliary quantity is the multiplier
   $\mu$ -- a vector of weights, one per constraint row, each weight
   being (minus) the sensitivity of the loss to a violation of that
   constraint -- and the observable is the gradient
   $J_\theta^\top \mu$.
4. Redundant constraints flip, under transposition, into an
   underdetermined system for the weights: one gauge dimension per
   redundant row.
5. The gradient is gauge-invariant anyway: null vectors of $J_x^\top$
   are annihilated by $J_\theta^\top$, by the identity
   $J_x S = -J_\theta$.
6. Numerically, only one number decides your fate:
   $\lvert\mu\rvert / \lvert\nabla\rvert$ -- the error floor is
   $\varepsilon \cdot \lvert\mu\rvert / \lvert\nabla\rvert$, because
   assembling a small answer from a large weight vector requires
   cancellation.
7. Even a square formulation with a unique multiplier can hide a
   catastrophic gauge choice in its formulation (example C).
8. A good gauge fix keeps the multiplier small and pins values that are
   known in advance, aimed at the directions the dynamics amplify.
9. Pinned-in-advance values can be re-enforced during an unstable
   iterative solve -- reprojection -- which deletes the amplified noise
   mode at the cost of a dot product, changes nothing in exact
   arithmetic, and turns exponential error growth into a flat precision
   floor (example D).

## 14. Dictionary into this study

This section maps the guide onto the concrete study in this folder. Skip
it if you only wanted the concept. All claims below are verified by the
numbered experiments.

The study differentiates Golub-Kahan bidiagonalization. The parameters
$\theta$ are the matrix $A$ and the starting vector $v$; the outputs $x$
are the factors $L, R, B$ and the terminal data $\mathit{res}, c$; the
constraints are the recurrence and orthonormality equations (NOTES_01
sec. 0).

| guide concept | in this study |
|---|---|
| observable | gradient ($\nabla_A \rho$, $\nabla_v \rho$) |
| auxiliary quantity | multipliers $\mu = (\Xi, \Phi, \kappa, \Sigma, \Omega, \eta)$ |
| defining system (A) | stationarity system, NOTES_01 sec. 0 |
| gauge dimension $r = d_O - d_U$ | $K^2$ (exp01) |
| $\mathrm{null}(J_x^\top)$ | verified dimension $K^2$ (exp01) |
| invariance lemma | NOTES_01 sec. 4 + exp01 numerical check |
| "drop redundant rows" gauge ($t = 0$ in example A) | reduced / appendix-B.2 gauge (NOTES_02) |
| pin-components gauge fix | the $K^2$ selection identities (NOTES_03) |
| reprojection (example D) | the paper's reprojection: re-enforcing the selection identities during the backward recurrence (NOTES_03 sec. 4, NOTES_05 remark 2) |
| formulation gauge (example C) | `con_red` vs `V3` vs `con_full`: equivalent sets, different unique multipliers (NOTES_04, NOTES_06) |
| the $\varepsilon \cdot \lvert\mu\rvert / \lvert\nabla\rvert$ law | NOTES_06 sec. 2, exp13 |
| the sharpened law of pitfall 4 | $(\varepsilon_{\text{arith}} + \varepsilon_{\text{primal}}) \cdot \lvert\mu\rvert/\lvert\nabla\rvert$, exp13 |
| $\lvert\mu\rvert$ growth rate | backward loop gain $\lVert A\rVert^2 / (\alpha \beta)$ per step, exp15 |
| minimal-norm representative | `mu_full(lstsq)` in exp01 (norm 31 vs reduced 212 at $K = 4$) |

Some of the correspondences, spelled out:

**Counts.** For problem sizes $N, M, K$ the study has
$d_U = NK + MK + M + 2K$ outputs and $d_O = d_U + K^2$ constraint rows,
so the gauge dimension is $r = K^2$. Concretely, at
$N = 7$, $M = 5$, $K = 4$: $d_U = 28 + 20 + 5 + 8 = 61$, $d_O = 77$,
gauge dimension $16$ -- exactly what exp01 measures, including the
null-space dimension of $J_x^\top$.

**Example A at scale.** The `con_full` multiplier family of the study is
example A with $K^2$ free parameters instead of one. The study's
"reduced gauge" is the $t = 0$ rule: set the weights of the redundant
orthogonality rows to zero (NOTES_01 sec. 4). The paper's stable gauge
is a specific "pin these components to these values" rule (NOTES_03).

**Example C at scale.** The appendix-B.2 "reduced" constraints of the
paper are a perfectly square, perfectly nonsingular formulation -- and
still catastrophic, exactly as in example C. Its row mixing, relative to
the well-behaved formulations, involves the inverse of a bidiagonal
recursion, so the analogue of the $10^8$ factor grows like
$10^{0.6 K}$ for random matrices and up to $10^{5 K}$ for Hilbert-type
matrices. That is why its exact multiplier reaches $10^{70}$ while the
gradient stays $O(100)$ (exp13, exp15, NOTES_06 sec. 2). And as in
example C, the solver is not at fault: the recurrence that computes that
multiplier is columnwise accurate to machine precision. The answer dies
in the assembly step, by the cancellation law of section 7.

**Example D at scale.** The paper's stable backward recurrence carries
multiplier columns whose inner products against the computed basis
vectors are pinned by the $K^2$ selection identities, with values known
in closed form from the loss cotangents -- known in advance, exactly as
example D requires. Reprojection overwrites the drifting inner products
with those known values at every backward step. The structural result of
the study (NOTES_03, exp04): the paper's boxed "stable adjoint system"
is literally

$$[\ \text{stationarity of the redundant constraint set}\ ]
\ +\ [\ K^2 \text{ gauge-fixing rows of exactly this kind}\ ],$$

and the paper's backward recurrence is the back-substitution of that
square system. The paper's stable method *is* a good gauge fix in the
precise sense of section 9, with reprojection as its example-D-style
enforcement.

## 15. Solutions to the exercises

All solutions are verified numerically by `guide_examples.py`
(function `exercises()`).

**Solution 1.** (a) The family is $(a, b) = (7 + t,\ t)$ for any real
$t$. Check: $a - b = (7 + t) - t = 7$ for every $t$. (b) The observable
is the number $7$ (the difference $a - b$). The auxiliary quantity is the
pair $(a, b)$. The gauge freedom is one-dimensional: one free parameter
$t$. (c) Any rule that determines $t$ works. Example: "require $b = 0$"
picks $(7, 0)$. Another example: "require $a = -b$" (minimal norm) picks
$(3.5, -3.5)$.

**Solution 2.** (a) A particular solution is $\mu = (2, 0, 0)$. The null
space of $M$ is the plane $\{\nu : \nu_1 + \nu_2 + \nu_3 = 0\}$, which
has dimension $2$. So the family is $(2, 0, 0) + \nu$ over that plane:
a two-dimensional affine family. (b) For $w = (1, 1, 1)^\top$, the
report is $g = \mu_1 + \mu_2 + \mu_3$. That is exactly the left-hand side
of the defining equation, so $g = 2$ for every solution. Invariant.
(Consistency check with section 2: $w$ is the single row of $M$, so $w$
lies in $\mathrm{range}(M^\top)$.) (c) For $w = (1, 2, 3)^\top$:
the solution $(2, 0, 0)$ gives $g = 2$; the solution $(0, 2, 0)$ gives
$g = 4$. Different values, so not invariant. (Indeed $w$ is not a
multiple of $(1,1,1)$, so $w \notin \mathrm{range}(M^\top)$.)

**Solution 3.** (a) With $J_x = I$, system (A) reads
$\mu = -\nabla_x \rho = (-1, -1)^\top$. (b) The perturbed constraints
solve to $x = (\theta + \epsilon_1,\ 2\theta + \epsilon_2)$. Substitute:
$\rho = x_1 + x_2 = 3\theta + \epsilon_1 + \epsilon_2$. Hence
$\partial\rho/\partial\epsilon_1 = \partial\rho/\partial\epsilon_2 = 1$,
and indeed $-\mu_i = 1$ for both $i$. The weights are (minus) the
sensitivities of the loss to constraint violations, as claimed in
section 3.

**Solution 4.** (a) $d_O = 4$ rows; the rank is $d_U = 2$ (the first two
rows already span $\mathbb{R}^2$); the gauge dimension is
$r = 4 - 2 = 2$. (b) $J_x^\top \nu = 0$ reads

$$\nu_1 + \nu_3 + 2\nu_4 = 0, \qquad \nu_2 + \nu_3 - \nu_4 = 0.$$

Choose the free variables $(\nu_3, \nu_4)$. Setting
$(\nu_3, \nu_4) = (1, 0)$ gives $\nu = (-1, -1, 1, 0)^\top$; setting
$(\nu_3, \nu_4) = (0, 1)$ gives $\nu = (-2, 1, 0, 1)^\top$. These two
vectors are independent and span $\mathrm{null}(J_x^\top)$.

**Solution 5.** (a)
$J_\theta^\top \nu = (-1)(-2) + (-2)(1) + (0)(1) = 2 - 2 + 0 = 0$.
(b) The lemma says every null vector of $J_x^\top$ is annihilated by
$J_\theta^\top$; since $\nu$ is such a null vector, the result had to be
$0$ before any arithmetic was done.

**Solution 6.** (a) Along the solution path, $\rho = x_2 = 2\theta$, so
the correct gradient is $2$. (b) The adjoint system is
$J_x^\top \mu = -(0, 1)^\top$, i.e.

$$\mu_1 + 2\mu_3 = 0, \qquad \mu_2 - \mu_3 = -1,$$

with general solution (free parameter $t := \mu_3$):

$$\mu(t) = (-2t,\ -1 + t,\ t).$$

(c) The assembly:

$$J_\theta^\top \mu(t) = (-1)(-2t) + (-2)(-1 + t) + 0 \cdot t
= 2t + 2 - 2t = 2.$$

Invariant in $t$, and equal to the prediction from (a).

**Solution 7.** Set the dot product of $\mu(t) = (-1 - 2t, -1 + t, t)$
with $\nu = (-2, 1, 1)$ to zero:

$$\mu(t) \cdot \nu = (-1 - 2t)(-2) + (-1 + t)(1) + t
= 2 + 4t - 1 + t + t = 1 + 6t.$$

Setting $1 + 6t = 0$ gives $t = -1/6$, and
$\mu(-1/6) = (-2/3,\ -7/6,\ -1/6)$, as in the table of section 6. (Why
orthogonality: the norm $\lvert \mu(0) + t\nu \rvert$ is minimal exactly
when the vector has no component along the null direction $\nu$.)

**Solution 8.** The law says the relative error is about
$\varepsilon \lvert\mu\rvert / \lvert\nabla\rvert$. Require
$\varepsilon \lvert\mu\rvert / \lvert\nabla\rvert \le 10^{-3}$, i.e.

$$\lvert\mu\rvert \le \frac{10^{-3} \cdot \lvert\nabla\rvert}{\varepsilon}
= \frac{10^{-3} \cdot 3}{1.2 \times 10^{-7}} = 2.5 \times 10^{4}.$$

So the multiplier must stay below about $25000$. A multiplier of size
$10^8$ (examples B and C) exceeds the budget by almost four orders of
magnitude.

**Solution 9.** (a) $J_x = \begin{bmatrix} 1 & 0 \\ c & 1 \end{bmatrix}$,
so $J_x^\top \mu = -(1, 1)^\top$ reads

$$\mu_1 + c\,\mu_2 = -1, \qquad \mu_2 = -1,$$

giving $\mu(c) = (c - 1,\ -1)$. (b) With
$J_\theta = (-1,\ -(c + 2))^\top$:

$$J_\theta^\top \mu = (-1)(c - 1) + (-(c + 2))(-1) = -c + 1 + c + 2 = 3$$

for every $c$. (c) $\lvert\mu\rvert \approx \lvert c \rvert$ for large
$c$, so the relative error is about
$\varepsilon \lvert c \rvert / 3$. It reaches order one when
$\lvert c \rvert \approx 3 / \varepsilon \approx 2.5 \times 10^{7}$.
Beyond that, the float32 gradient is useless -- consistent with the
observed total loss at $c = 10^8$ (example C), and with a harmless
computation at, say, $c = 100$.

**Solution 10.** (a)
$G v = (3 \cdot 3 + 1 \cdot (-8),\ \tfrac{1}{3} \cdot (-8))^\top
= (1, -\tfrac{8}{3})^\top = \tfrac{1}{3}(3, -8)^\top = \tfrac{1}{3} v$.
(b) $\ell^\top G = (8 \cdot 3 + 3 \cdot 0,\ 8 \cdot 1 + 3 \cdot \tfrac13)
= (24, 9) = 3\,(8, 3) = 3 \ell^\top$.
(c) $\ell \cdot v = 8 \cdot 3 + 3 \cdot (-8) = 24 - 24 = 0$;
$\ell \cdot u = 8 \cdot 1 + 3 \cdot 0 = 8$.
(d) $\ell \cdot w_{n+1} = \ell^\top (G w_n) = (\ell^\top G) w_n
= 3 \ell^\top w_n = 3\,(\ell \cdot w_n)$. Starting from
$\ell \cdot w_0 = \ell \cdot v = 0$, the pinned value is $0$ forever --
but any *nonzero* drift is tripled at every step, which is exactly why
it must be re-enforced.

**Solution 11.** (a) Total loss at
$\varepsilon \cdot 9^n \approx 1$, i.e.
$n \approx \dfrac{\log_{10}(1/\varepsilon)}{\log_{10} 9}
= \dfrac{6.9}{0.95} \approx 7.3$: between 7 and 8 steps, matching the
experiment (error $7 \times 10^{-3}$ at step 7, order one by step 10).
(b) In float64, $n \approx 15.7 / 0.95 \approx 16$ steps. Higher
precision only postpones the loss -- the error still grows by a factor
of $9$ per step. Reprojection, by contrast, removes the growth mechanism
itself; the error then stays at the precision floor at *any* depth.

**Solution 12.** Write $w = a v + b u$. (a) Then
$\ell \cdot w = a\,(\ell \cdot v) + b\,(\ell \cdot u) = a \cdot 0 + 8b
= 8b$. (b) The update subtracts
$\tfrac{\ell \cdot w}{8} u = \tfrac{8b}{8} u = b u$, so
$w \mapsto a v + b u - b u = a v$: the growing component is deleted
exactly, and the coefficient $a$ of the shrinking component is not
touched. (c) Exact data means $b = 0$; then the subtracted vector is
$0 \cdot u = 0$ and the update is the identity. Reprojection never
changes the mathematics -- it only removes noise.
