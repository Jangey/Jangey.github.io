---
title: Backpropagation
categories: Machine Learning
date: 2019-05-11 18:04:41
---

#### Loss Function
We use binary log loss (**cross entropy**). 
$$ Loss = \frac{1}{N} \sum_{i=1}^N -{(y_i\log(p_i) + (1 - y_i)\log(1 - p_i))} $$
Remember: Here the **log** is natual-log ($ln$), because the exponential $e$ should match $ln$.
<hr>

#### Forward
From the **Neural Network**, We using forward function to find the predict value and **Loss** value.

Predict Value: The value we using the input and weight to calculate the predict for output.

Loss Value: We using the **Predict Value** and the **Actual Value** into Loss Function to get the Loss Value for current prediction.
<hr>

#### Backward
We using the **Neural Network**, start from the **Loss** value backward, and use Derivative for each **Gate** go back to change the value on each weight.

$$ \frac{\partial y}{\partial x} = \frac{\partial y}{\partial y} \cdot \frac{\partial y}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$$ 
<hr>

#### Code Example
The input value is **Variable**, the weight value and learning rate we put into **Parameter**.

{% codeblock lang:python %}
# y = a * x1 + b
# y = (a * x1 + b - y1)^2
# learning_rate = 0.1, a=0, b=0

x1 = graph.Variable(0.5)
y1 = graph.Variable(1.7)
a = graph.Parameter(0, 0.1)
b = graph.Parameter(0, 0.1)
print(a.value, b.value)

loss = graph.Add(
        graph.Add(graph.Mul(x1, a), b),
        graph.Mul(y1, graph.Variable(-1)))

loss.forward() # -> your curretn loss
loss.backward(1) # -> cal backward to change weight, inital the '1' for input dy/dy
{% endcodeblock %}