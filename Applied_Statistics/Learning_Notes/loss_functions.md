# Loss Functions

## Loss Function vs Cost Function
* Loss function is calculated for each record, focusing on comparing `y_true` and `y_pred`.
* Cost function is the metrics we use to measure the model performance on all the records, it's the aggregated results of loss function.

## Regression Loss Functions

### Mean Square Loss (L2 Loss)
* `L = power((y_true-y_pred), 2)`
* It's used in MSE (mean squared errors), `MSE = sum(power((y_true_i - y_pred_i), 2))/n`
* It gives more penality to larger errors.
* Square Value vs Absolute Value
  * The squared difference has nicer mathematical properties; it's continuously differentiable (nice when you want to minimize it), it's a sufficient statistic for the Gaussian distribution, and it's (a version of) the L2 norm which comes in handy for proving convergence and so on.
* Same as MSE, will be affected by outliers a lot.

### Absolute Error Loss (L1 Loss)
* `L = |y_true - y_pred|`
* It's used in mean absolute errors (MAE)
* Less affected by outliers

### Huber Loss
* It combines L1 loss and L2 loss. When the error is small, the loss value is quadratic, otherwise linear.
* If `|y_true - y_pred| <= δ` then `L = power((y_true-y_pred), 2)/2`
* Otherwise `L = δ*|y_true-p_pred| - δ*δ/2`
* More robust to outliers than MSE

## Binary Classification Loss Functions

### Binary Corss Entropy Loss (log-loss)
* The goal is to minimize log-loss
* Predicted value is probability p
  * `p` for class 1, `1-p` for class 0
  * p can be calculated through sigmoid function
* `L = -y*log(p) - (1-y)*log(1-p)`
  * y = 0 then L = -log(1-p)
  * y = 1 then L = -y*log(p)
  
 ### Hinge Loss
 * `L = max(0, 1-y*f(x))`
 * Majorly used in SVM with class labels -1 and 1.
 * Hinge Loss not only penalizes the wrong predictions but also the right predictions that has lower confident.

## Multi-Class Classification Loss Functions

### Multi-Class Cross Entropy Loss
* `Lij = -yi1 * log(pi1) - yi2 * log(pi2) - ... - yij * log(pij)`
  * `j` means class j
  * if ith element is in class j, `yij` is 1, otherwise 0
  * `pij = f(Xi)` is the probability of ith record for class j
* Softmax is used to find the probability pij
  * Softmax is implemented through a neural network layer just before the output layer. 
  * The Softmax layer must have the same number of nodes as the output layer.
* In Keras, specify `loss` param as `categorical_crossentropy` in `compile()`

### K-L Divergence (Kullback-Liebler Divergence)
* It's a measure of how a probability distribution differs from another distribution. A KL-divergence of zero indicates that the distributions are identical.
* The divergence function is NOT symmetric.
  * This is also why KL divergence cannot be used as a distance metric.
  * `Dkl(P||Q) != Dkl(Q||P)`
    * P, Q are 2 distributions
    * `Dkl(P||Q)` minimize forward KL, used in supervised learning
    * `Dkl(Q||P)` minimize backward KL,used in reinforcement learning
  * `Dkl(P||Q) = -P(x)*logQ(x) - P(x)*logP(x) = H(P, Q) - H(P, P)`
    * H(P, P) is the entropy of P
    * H(P, Q) is the corss entropy of P, Q
* In Keras, specify `loss` param as `kullback_leibler_divergence` in `compile()`

[Reference][1]

[1]:https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
