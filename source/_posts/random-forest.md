---
title: Random Forest
categories: Machine Learning
date: 2019-04-14 18:19:41
---

### Random Forest Regression

Random Forest Regression is based on [Decision Tree Regression](../decision-tree/). It will build 'n_estimators' of Decision Tree Regression, and using the average each decision tree predict value to improve the accuracy value, and control over-fitting. 

More Info Using Scikit-Learn Built-in library: [Scikit-Learn Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

- n_estimators : integer, optional (default=10)
The number of trees in the forest.

- sample_ratio: percentage of total size
For some big dataset, it may take long time to build for random forest, therefore we can take some sample of the data to same time for build the random forest.

#### Methods
- fit(self, X: pd.DataFrame, y: np.array)	
Build a forest of trees from the training set (X, y).

- predict(self, X: pd.DataFrame)    
Predict regression target for X.

- score(self, X: pd.DataFrame, y: np.array)
Returns the coefficient of determination $R^2$ of the prediction.

#### Graphic Example 
- X-axis: Number of Random Forest estimators
- y-axis: The accuracy for predict values.

<img src="{% asset_path RandomForestRegressor.png %}"  width="100%">

{% codeblock lang:python %}
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

class RandomForestRegressor():

	def __init__(self, n_estimators, sample_ratio, random_state=None):
		self.n_estimators = n_estimators
		self.sample_ratio = sample_ratio
		self.random_state = random_state

	def fit(self, X: pd.DataFrame, y: np.array):
		if self.random_state!=None:
			np.random.seed(self.random_state)

		n=self.n_estimators
		n_sample = X.shape[0]

		self.trees = []
		for i in range(n):
			tree = DecisionTreeRegressor()
			indices = np.random.randint(0, n_sample, int(self.sample_ratio*n_sample))
			_ = tree.fit(X.iloc[indices, :], y[indices])
			self.trees.append(tree)

	def predict(self, X: pd.DataFrame):
		predictList = []
		for t in self.trees:
			predictList.append(t.predict(X))
		predictList = np.array(predictList)
		predictSampled = predictList.mean(axis=0)
		return predictSampled


	def score(self, X: pd.DataFrame, y: np.array):
		return metrics.rsq(self.predict(X), y)
{% endcodeblock %}