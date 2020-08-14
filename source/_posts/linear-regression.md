---
title: Linear Regression
categories: Machine Learning
date: 2019-04-23 02:26:55
---

### One dimensional Case
h is the output for prediction if give x-value, and y is output, h is predict value

$$ h=ax+b $$

#### Loss Function

$$ Loss = L(a, b) $$


The goal is to minize the loss function, the value a and b define the predict function. When the function achive the lowest value will be a good loss function.

$$ Loss = MSE = \frac{1}{N} \sum_{i=1}^N (y_i - h_i)^2 $$


Start from a = 0, b = 0, then update at each iteration. Using derivative to find which direction we need to change. $ \frac{\partial L}{\partial a} = 0$ and $ \frac{\partial L}{\partial b} = 0 $

Therefore, from Loss function we can get:
$$ L(a, b)= \frac{1}{N} \sum_{i=1}^N (y_i - a x_i - b)^2 $$

Giving learning rate: $ \alpha $ = learning rate 

$$ \frac{\partial L}{\partial a} = \frac{1}{N} \sum_{i=1}^N (h_i - y_i) * x_i $$

$$ \frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^N (h_i - y_i) $$

For a and b updating :

a = $ a - \frac{\partial L}{\partial a} * \alpha $

b = $ b - \frac{\partial L}{\partial b} * \alpha $
 
After all, when we find the minimum value for Loss function, then it will be the a & b for the linear regression.

#### Excel Example

|  x  |  y  | y_pred | y_pred - y | (y_pred-y) * x |
| :-: | :-: | :-: | :-: | :-: |
| [0] |[2.3]|  0  |-2.3 |  0  |
|[1.11]|[2.7]|  0  |-2.7 |-2.997|
|[2.8]|[7.1]|  0  |-7.1 |-19.88|
|**Average**| | |-4.0333|-7.62567|

|   |Initial|Example|Equation|
| :-: | :-: | :-: | :-: |
|**a**|0|0 - (0.1 * -7.62567)|a - [LR x Average((y_pred-y) * x)]|
|**b**|0|0 - (0.1 * -4.0333)|b - [LR x Average((y_pred_y)]|
|**Learning rate (LR)**|0.1|
|**Projected**|0|$\sqrt{(0)^2 + (0)^2}$|$\sqrt{(a)^2 + (b)^2}$|
|**Loss**|62.99|$\frac{(-2.3)^2 + (-2.7)^2 + (-7.1)^2}{3} $|$\frac{1}{N}\sum_{i=1}^N (y\\_pred -y)^2$|

<hr>

### Multi-Dimensional Case
In multi-dimensional case, the a will be replace by $w_i$, and the formular is same as the one dimensional case.

$$ h = \sum_{i=1}^N (w_i * x_i) + b $$


#### Normal Distribution

Input **X** change to **X_norm**
Input **Y** change to **y_norm**

For each feature of X input, We have to calculate **mean** and **STD** (Standard Deviation). And Using those two value to get **X_norm** and **y_norm**.

From real value change to normal distribution:

$$ X\\_NORM = \frac{(X - \overline X)}{STD} $$
$$ y\\_NORM = \frac{(y - \overline y)}{STD} $$

Becuase we using normal distribution to get **weight** therefore, when we get back y predict value we need using **mean** and **STD** to change back real value:

$$ y = y\\_Predict \cdot y\\_STD + \overline y $$

#### Batch
With batch **stochastic gradient descent**, we'd need fewer iterations and improve the **LOSS** value faster. 

Think of iteration as epoch - you pass over the entire dataset. So if you split into batches, you’d go over all batches and this will be one iteration.

{% codeblock lang:python %}
batch_size = 4
batches = int( X.shape[0] / self.batch_size )

for i in range(batches):
	start = i * batch_size
	end = start + batch_size
	X = X[start:end]
	y = y[start:end]
{% endcodeblock %}

