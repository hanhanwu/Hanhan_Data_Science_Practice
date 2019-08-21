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


