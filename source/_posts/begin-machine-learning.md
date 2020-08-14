---
title: Begin Machine Learning
categories: Machine Learning
date: 2019-03-30 22:42:55
---
Welcome to Machine Learning. Everything is imported from class lectures and backup for future study.
<br>

### Train_Test_Split:
```
Input: X, y, test_size, shuffle, random_state
Output: X_train, X_test, y_train, y_test
```
- (X, y): features and the target variable.
- (test_size): between 0 and 1 - how much to allocate to the test set; the rest goes to the train set. 
- (shuffle): if True, shuffle the dataset, otherwise not.
- (random_state): integer; if None, then results are random, otherwise fixed to a given seed. 
<br>

Machine Learning need split for **Train Set** and **Test Set**. We train the model from **Train Set** and use the model for unknowed **Test Set**.
<hr>


### MSE - Mean Square Error
- Mean of the square difference between the predicted value and true value.

$$ MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat y_i)^2 $$

{% codeblock lang:python %}
# function mse		- return Mean-Squared Error
def mse(y_predicted, y_true):
    return ((y_predicted - y_true)**2).mean()
{% endcodeblock %}
<br>

### RMSE - Root Mean Square Error
- Square root for Mean of the square difference between the predicted value and true value.

{% codeblock lang:python %}
# function rmse		- return Root Mean-Squared Error
def rmse(y_predicted, y_true):
    return np.sqrt(((y_predicted - y_true)**2).mean())
{% endcodeblock %}
<br>

### RSQ - R Square Score - $ R^2 $
- $ 0 \le R^2 \le 1 $ 

The goal is to make the $ R^2 $ approaching 1 to train the model perfect.

$$ R^2 = 1 - { mse \over value } $$

{% codeblock lang:python %}
# function rsq		- return R^2
def rsq(y_predicted, y_true):
    return 1 - ((y_predicted - y_true)**2).mean() / ((y_true - y_true.mean())**2).mean()
{% endcodeblock %}
<hr>

### Overfitting

Overfitting means our model fit the trainning set really well, maybe too well. $R^2$ train set way better than $R^2$ test set.

e.g. Train Set $R^2 = 0.97$ , Test Set $R^2 = 0.64$.

Solution: Reduce the complexity of our model from high frequency model to a lower degree polynomial model.
<br>

### Underfitting
Underfitting mean low $R^2$ on both train set and test set. $R^2$ train set and $R^2$ test set are low.

e.g. Train Set $R^2 = 0.64$, Test Set $R^2 = 0.62$.

Solution: We can try more complex model from low degree polynomial model to a higher degree polynomial model.