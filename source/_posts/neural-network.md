---
title: Neural Network
categories: Machine Learning
date: 2019-04-14 21:06:39
---
### Boolean Funcitons

Artificial Neuron will have X for input and y for output, the output y from all of x value multiply the weight and plus the bias with bias weight. 
- $x$: x value, $w$: weight, 

 $$ y = \sum_{i=1}^N (x_i * w_i) + (bias)(bias_{weight}) $$

<center><img src="{% asset_path ArtificialNeuralNetwork.png %}"  width="50%"></center>
<hr>
<center><h4>OR</h4></center>

|  x1 |  x2 |  y  |
| :-: | :-: | :-: |
|  0  |  0  |  0  |
|  0  |  1  |  1  |
|  1  |  0  |  1  |
|  1  |  1  |  1  |

{% codeblock lang:python %}
class create_OR:

    def forward(self, X):
        x1 = X[0]
        x2 = X[1]
        x1_WEIGHT = 1
        x2_WEIGHT = 1
        CONSTANCE = 1
        CONSTANCE_WEIGHT = 0
        result = x1 * x1_WEIGHT + x2 * x2_WEIGHT + CONSTANCE * CONSTANCE_WEIGHT

        return {True: 1, False: 0} [result > 0]
{% endcodeblock %}

<center><h4>AND</h4></center>

|  x1 |  x2 |  y  |
| :-: | :-: | :-: |
|  0  |  0  |  0  |
|  0  |  1  |  0  |
|  1  |  0  |  0  |
|  1  |  1  |  1  |

{% codeblock lang:python %}
class create_AND:
    
    def forward(self, X):
        x1 = X[0]
        x2 = X[1]
        x1_WEIGHT = 1
        x2_WEIGHT = 1
        CONSTANCE = 1
        CONSTANCE_WEIGHT = -1
        result = x1 * x1_WEIGHT + x2 * x2_WEIGHT + CONSTANCE * CONSTANCE_WEIGHT

        return {True: 1, False: 0} [result > 0]
{% endcodeblock %}

<center><h4>NOT</h4></center>

|  x  |  y  |
| :-: | :-: |
|  0  |  1  |
|  1  |  0  |

{% codeblock lang:python %}
class create_NOT:

    def forward(self, X):
        result = X

        return {True: 1, False: 0} [result == 0]
{% endcodeblock %}

<center><h4>XNOR</h4></center>

|  x1 |  x2 |  y  |
| :-: | :-: | :-: |
|  0  |  0  |  1  |
|  0  |  1  |  0  |
|  1  |  0  |  0  |
|  1  |  1  |  1  |

{% codeblock lang:python %}
class create_XNOR:

    def forward(self, X):
        x1 = X[0]
        x2 = X[1]

        xAND = create_AND().forward([x1, x2])
        xNotAND = create_AND().forward([create_NOT().forward(x1), create_NOT().forward(x2)])
        xOR = create_OR().forward([xAND, xNotAND])

        return xOR
{% endcodeblock %}

<center><h4>XOR</h4></center>

|  x1 |  x2 |  y  |
| :-: | :-: | :-: |
|  0  |  0  |  0  |
|  0  |  1  |  1  |
|  1  |  0  |  1  |
|  1  |  1  |  0  |

{% codeblock lang:python %}
class create_XOR:

    def forward(self, X):
        XNOR = create_XNOR().forward(X)
        XOR = create_NOT().forward(XNOR)
        
        return XOR
{% endcodeblock %}