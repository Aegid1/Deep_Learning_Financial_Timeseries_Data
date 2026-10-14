# Experiment 35: Weighted Loss for Predicting Large Price Movements

This experiment contains the same architecture as in experiment 30. This experiment focuses on how the model and the
training need to be adjusted such that the proposed LSTM architecture has a higher precision outside of the center of
the data distribution, i.e. it should be **precise on large price movements** and in exchange **accepts lower precision on small movements**.

This document describes the theoretical derivation of the loss used to train the LSTM for predicting intraday log returns. 

The derivation follows the following steps in which the approach was developed:

1. [MSE as maximum likelihood estimation](#1-mse-as-maximum-likelihood-estimation)
2. [The implicit marginal distribution: Gaussian KDE](#2-the-implicit-marginal-distribution-gaussian-kde)
3. [First approach: Student-t kernel via reverse engineering (discarded)](#3-first-approach-student-t-kernel-via-reverse-engineering-discarded)
4. [Weighted MSE](#4-weighted-mse-theory)
5. [Second approach: exponential weighting (discarded)](#5-second-approach-exponential-weighting-discarded)
6. [Final approach: sigmoid tercile weighting](#6-final-approach-sigmoid-tercile-weighting)

---

## Notation

| Symbol | Meaning |
|---|---|
| $x_i$ | Input sequence (23 time steps of technical indicators) of sample $i$ |
| $y_i$ | Target: log return $y_i = \log(P_{t+1}/P_t)$ |
| $f_\theta(x)$ | Model prediction with parameters $\theta$ |
| $N$ | Number of training samples |
| $p(x, y)$ | True (data-generating) distribution |
| $w(y_i)$ | Weight of a sample as a function of its label $y_i$ |

---

## 1. MSE as maximum likelihood estimation

Assuming that the model describes the conditional distribution of the return as a Gaussian centred on the prediction 
with fixed variance $\sigma^2$:

$$
p_\theta(y \mid x) = \mathcal{N}\big(y;\ f_\theta(x),\ \sigma^2\big) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y - f_\theta(x))^2}{2\sigma^2}\right).
$$

Maximum likelihood estimation (MLE) maximizes the likelihood of the training data which is equivalent to minimizing
the negative log-likelihood:

$$
\begin{aligned}
\text{NLL}(\theta) &= -\sum_{i=1}^{N} \log p_\theta(y_i \mid x_i) \\
&= \sum_{i=1}^{N} \left[ \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2) \right] \\
&= \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - f_\theta(x_i))^2 + \underbrace{\frac{N}{2}\log(2\pi\sigma^2)}_{\text{independent of } \theta}.
\end{aligned}
$$

Because $\sigma^2$ is fixed the NLL and MSE have the same minimization objective:

$$
\arg\min_\theta \text{NLL}(\theta) = \arg\min_\theta \underbrace{\frac{1}{N}\sum_{i=1}^{N} (y_i - f_\theta(x_i))^2}_{\text{MSE}}.
$$

**By minimizing the MSE the model therefore implicitly learns the conditional distribution $p_\theta(y \mid x)$ as a Gaussian.**

## 2. The implicit marginal distribution: Gaussian KDE

Ding et al. (2019), *"Modeling Extreme Events in Time Series Prediction"*, found that the usage of an MSE for training a model
and marginalizing the implicitly learned conditional model over the empirical input distribution gives

$$
p_\theta(y) = \int p_\theta(y \mid x)\, \hat p(x)\, dx \approx \frac{1}{N} \sum_{i=1}^{N} \mathcal{N}\big(y;\ f_\theta(x_i),\ \sigma^2\big).
$$

This is a **kernel density estimator (KDE) with a Gaussian kernel** that centres around $f_\theta(x_i)$ and has fixed bandwidth $\sigma$.

A Gaussian kernel decays like $\exp(-y^2/2\sigma^2)$, i.e. faster than any power of $y$. 
Empirical stock returns, have heavy tails whose density decays more like a power law. Thus, the KDE with a gaussian kernel 
is a bad approach of modeling the fat-tailed distribution of the actual data.
---

## 3. First approach: Student-t kernel via reverse engineering (discarded)

### Idea

If the marginal distribution as a Gaussian KDE models the tails badly, replace the Gaussian kernel with a heavy-tailed kernel, 
for example with the **Student-t distribution**, and derive the corresponding loss as its negative log-likelihood 
(reverse engineering: desired distribution → loss).

### Derivation

$$
p_\theta(y \mid x) = t_\nu\big(y;\ \mu,\ \sigma\big) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\,\sigma} \left(1 + \frac{(y-\mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}, \qquad \mu = f_\theta(x).
$$

Negative log-likelihood:

$$
\text{NLL} = -\log\Gamma\!\left(\tfrac{\nu+1}{2}\right) + \log\Gamma\!\left(\tfrac{\nu}{2}\right) + \tfrac{1}{2}\log(\nu\pi) + \log\sigma + \frac{\nu+1}{2}\log\!\left(1 + \frac{(y-\mu)^2}{\nu\sigma^2}\right).
$$

With $\nu$ and $\sigma$ fixed, every term except the last is constant in $\theta$. With $r = y - f_\theta(x)$ the gradient is:

$$
\nabla_\theta \text{NLL} = -\frac{(\nu+1)\, (y - f_\theta(x))}{\nu\sigma^2 + (y - f_\theta(x))^2}\ \nabla_\theta f_\theta(x).
$$

### Behaviour of the gradient

The scalar factor $\alpha(r) = \frac{(\nu+1)\,(y - f_\theta(x))}{\nu\sigma^2 + (y - f_\theta(x))^2}$ behaves as follows:

| Regime | Approximation | Meaning |
|---|---|---|
| $\lvert y - f_\theta(x)\rvert \ll \sigma\sqrt{\nu}$ | $\alpha(r) \approx \frac{\nu+1}{\nu\sigma^2}\, (y - f_\theta(x))$ | similar to MSE |
| $\lvert y - f_\theta(x)\rvert \gg \sigma\sqrt{\nu}$ | $\alpha(r) \approx \frac{\nu+1}{y - f_\theta(x)} \to 0$ | large errors are **ignored** |

This shows that the proposed approach results in the opposite of the desired behavior. The outliers should not be ignored.

### Why the approach is conceptually wrong

The kernel in the marginal distribution is the **noise model** $p(y \mid x)$ around the prediction. 
A Student-t kernel does produce more heavy tails in the marginal distribution, but **through the noise, not through the predictions**:

$$
p_\theta(y) = \frac{1}{N}\sum_{i=1}^{N} t_\nu\big(y;\ \underbrace{f_\theta(x_i)}_{\approx\, 0},\ \sigma\big).
$$

In effect the model learns: *"Extreme price movements are heavy noise around a prediction close to 0."* 
The resulting decreased gradient for large residuals shows this exactly as large errors are treated as explainable by noise, so $f_\theta$ does not need to follow them.

**The goal, however, is the opposite!** In the tails the model should be precise, i.e. little noise and high weight where reality is extreme. 
Changing the shape of the kernel thus is not a good approach. What could be changed is the **importance of the samples**.

---

## 4. Weighted MSE

### Definition

$$
\mathcal{L}_w(\theta) = \frac{1}{N}\sum_{i=1}^{N} w(y_i)\,\big(y_i - f_\theta(x_i)\big)^2, \qquad w(y) > 0.
$$

The weight depends on the **true label** $y_i$ and not on the prediction. For an increasing weight by an increasing
value of the label, this can be interpreted as *"Become more precise where reality is becoming more extreme."*



### Implicit marginal distribution: weighted Gaussian KDE

If $p(y_i\vert{}x_i) \sim \mathcal{N}(f(x), \sigma_i^2)$ is chosen such that the noise is not fixed and different for 
every sample this reslts to:

$$\text{NLL} \propto \sum_{i=1}^N \left( \frac{(f(x_i) - y_i)^2}{\sigma_i^2} + \log(\sigma_i^2) \right)$$

And during optimization of the model we are only concerned about the gradient w.r.t to the model parameters and thus, 
the constant noise term can be ignored because its not dependent on the models parameters.
Thus, leading to $$\sum_{i=1}^N \frac{1}{\sigma_i^2} (f(x_i) - y_i)^2$$

The weighted MSE minimizes:

$$\frac{1}{N}\sum_{i=1}^N q_i (f(x_i) - y_i)^2 \propto \sum_{i=1}^N q_i (f(x_i) - y_i)^2, \qquad q_i = \sum_{i=1}^{N} \frac{w_i}{\sum_j w_j}$$


And therefore one can see that the weights $q_i$ are related to the per-sample noise $q_i \propto 1/\sigma_i^2$. 
One could interpret this as the model assumes that in the extreme areas of the distribution there is less noise and that 
the model needs to be more precise in these areas. This might make sense because small return movements can be considered 
as more random than strong return movements for which a specific highly influential event is responsible.

---

## 5. Second approach: exponential weighting (discarded)

### Approach

$$
w(y) = e^{a|y|}, \qquad a > 0.
$$

Larger price movements receive exponentially more weight.

### Analysis

**Weight ratios.** For 2-hour returns with a standard deviation of roughly 1%, 
the boundary between the middle and outer thirds lies at about $|y| \approx 0.4\%$. For $a = 20$:

| $\lvert y\rvert$ | $w(y) = e^{20\lvert y\rvert}$ |
|---|---|
| 0 | 1.00 |
| 0.004 (tercile boundary) | 1.08 |
| 0.02 | 1.49 |
| 0.05 | 2.72 |
| 0.10 | 7.39 |

The exponential function has no threshold. Between the center and the tercile boundary the weights differ by only about 8%.
The weight concentrates on the few extreme values.

**Dominance of individual samples.** The gradient is a weighted sum $\sum_i w_i\, (f(x_i) - y_i)\, \nabla_\theta f_\theta(x_i)$. 
Because extreme samples have both large weights and large residuals, they dominate the gradient disproportionately.

**Conclusion:** $e^{a|y|}$ mainly amplifies the rare extreme values. It misses the goal of predicting **the outer thirds as a whole** more precisely.

---

## 6. Final approach: sigmoid tercile weighting

### Requirements

1. Low weight in the middle third, high weight in both outer thirds.
2. Roughly constant weight **within** the outer thirds (a plateau, no explosion).
3. Per company, since the stocks have different volatilities.

### Step 1: rank the returns to retrieve empirical distribution function

For each company with sorted training labels $y_{(1)} \le \dots \le y_{(n)}$:

$$
u(y) = \frac{\#\{\,j : y_j \le y\,\}}{n} \in (0, 1].
$$
This thus provides the empirical density for a given return. An extreme outlier simply receives the highest rank and 
its magnitude no longer matters.

### Step 2: distance from the median

The distance of a return from the median is defined as

$$
d(y) = \left|\, u(y) - \tfrac{1}{2} \,\right| \in \left[0, \tfrac{1}{2}\right].
$$

The tercile boundaries $u = 1/3$ and $u = 2/3$ both correspond to $d = 1/6$:

$$
\text{middle third} \iff d < \tfrac{1}{6}, \qquad \text{outer thirds} \iff d > \tfrac{1}{6}.
$$

### Step 3: soft weight increase (sigmoid)

$$
\boxed{\ w(y) = 1 + (W - 1)\ \sigma\!\left(\frac{d(y) - 1/6}{\tau}\right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}\ }
$$

with

- $W$: maximal weight,
- $\tau > 0$: width of the transition (smoothing), the smaller the faster the weight increase

Properties:

- **Bounded:** $1 \le w(y) < W$; no sample can dominate the gradient.
- **Plateau:** for $d \gg 1/6$, $w \approx W$; the outer thirds are weighted uniformly.
