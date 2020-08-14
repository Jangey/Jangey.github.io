---
title: Decision Tree
categories: Machine Learning
date: 2019-04-06 14:15:50
---

### Decision Tree Regression

Using features data from DataFrame to train a regression tree model. The way to find the best split value is to find the best **MSE** in each feature type. When we found the **BEST MSE** value, the model will set it as the split value for current level. Separate the X and y into left subtree and right subtree, and keep going to train the model tree from subtree. 

More Info Using Scikit-Learn Built-in library: [Scikit-Learn Decision Tree Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

- max_depth: int or None, optional (default=None)
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

- min_samples_leaf: int, float, optional (default=1)
The minimum number of samples required to be at a leaf node.

#### Methods
- fit(self, X: pd.DataFrame, y: np.array)	
Build a decision tree regressor from the training set (X, y).

- predict(self, X: pd.DataFrame)    
Predict regression value for X.

- score(self, X: pd.DataFrame, y: np.array)
Returns the coefficient of determination $R^2$ of the prediction.

#### Graphic Example 
<img src="{% asset_path DecisionTreeRegressor.png %}"  width="100%">

<br><hr>

### Decision Tree Classification
Using features data from DataFrame to train a regression tree model. The way to find the best split value is to find the best average subtree of **GINI Impurity X Data Weight** in each feature type. When we found the **BEST (GINI Impurity X Data Weight)** value, the model will set it as the split value for current level. Separate the X and y into left subtree and right subtree, and keep going to train the model tree from subtree.

More Info Using Scikit-Learn Built-in library. [Scikit-Learn Decision Tree Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 

- GINI Impurity:
$$ GINI =1-\sum_{i=1}^{N}{p_{i}}^{2} $$

{% codeblock lang:python %}
def calc_gini(y, valueType):
    subvalue = [0] * len(valueType)

    for i in y:
        if i == valueType[i]:
            subvalue[i] += 1

    giniSum = 0
    for i in subvalue:
        giniSum += (i/sum(subvalue)) ** 2
    subgini = 1-giniSum

    return subgini
{% endcodeblock %}

- Data Weight: 
$$ Data Weight = {\sum_{i=1}^{N}{SubtreeNode}\over{TotalNode}} $$

{% codeblock lang:python %}
if left.any() and right.any():
	cur_giniWeight = self.calc_gini(y[left], self.valueType) * (len(y[left]) / len(y)) \
	+ self.calc_gini(y[right], self.valueType) * (len(y[right])/len(y))
{% endcodeblock %}

- max_depth: int or None, optional (default=None)
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

- min_samples_leaf: int, float, optional (default=1)
The minimum number of samples required to be at a leaf node.

#### Methods
- fit(self, X: pd.DataFrame, y: np.array)	
Build a decision tree classifier from the training set (X, y).

- predict(self, X: pd.DataFrame)    
Predict class value for X.

- score(self, X: pd.DataFrame, y: np.array)
Returns the mean accuracy on the given test data and labels.

#### Graphic Example 
<img src="{% asset_path DecisionTreeClassifier.png %}"  width="100%">