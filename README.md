# my_open_the_iris
Open the iris!

A common mistake businesses make is to assume machine learning is magic, so it’s okay to skip thinking about what it means to do the task well.

Introduction
Time do to an end-to-end project in data science. which means:

Loading the dataset.
Summarizing the dataset.
Visualizing the dataset.
Evaluating some algorithms.
Making some predictions.
A must seen example about data science is the iris dataset. We will predict which class of iris plant does a plant belongs to based on its characteristics.

  
Iris versicolor - Iris setosa - Iris virginica

Where to get started?
Environment. We will use jupyter.

In Data science, the winning combo is pandas (and/or numpy), matplotlib, sklearn (and/or keras).
In this project we will use:

pandas to load the data
matplotlib to do the visualization
sklearn to do the prediction
Load dataset
url = "URL"
dataset = read_csv(url)
Summarizing the dataset
A - Printing dataset dimension

print(dataset.shape)
# should something like: (150, 5)
B - It is also always a good idea to actually eyeball your data.

print(dataset.head(20))
C - Statistical Summary
This includes the count, mean, the min and max values as well as some percentiles.

print(dataset.describe())
D - Class Distribution
Group by in order to see how our data are distributed.

print(dataset.groupby('class').size())
Visualization
After having a basic idea about our dataset. We need to extend it that with some visualizations.

For this dataset we will be focus on two types of plots:

Univariate plots to better understand each attribute.
Multivariate plots to better understand the relationships between attributes.
A - Univariate

from pandas import read_csv
from matplotlib import pyplot

dataset.hist()
pyplot.show()

It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.

B - Multivariate

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

scatter_matrix(dataset)
pyplot.show()

We can note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship. :-)

Building our code to evaluate some algorithms
it is time to create some models of the data and estimate their accuracy.

Here is what we are going to cover in this step:

Separate out a validation dataset.

array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
Experiment!
Build multiple different models from different algorithms.

# DecisionTree
model = DecisionTreeClassifier()

# GaussianNB
model = GaussianNB()

# KNeighbors
model = KNeighborsClassifier()

# LogisticRegression
model = LogisticRegression(solver='liblinear', multi_class='ovr')

# LinearDiscriminant
model = LinearDiscriminantAnalysis()

# SVM
model = SVC(gamma='auto')
How to run the model?

cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
Improving

Improving your data and your model is a iterative process, and you will have to loop through this process again and again.

Now it's time to do!

Technical specifications
You will create a end-to-end analysis on the dataset.

Part I Load data
Create a function load_dataset(), you will load the dataset and returns it.

Part II Summarizing the dataset
Summarizing the dataset:
Create a function summarize_dataset(), it will print (in this order):

its shape
its 10 first lines
its statistical summary
Its distribution
Part III
Create two functions print_plot_univariate() and print_plot_multivariate(). Each function will setup and show its corresponding plot.

Part IV
Create a function my_print_and_test_models(), it will (in this order)
DecisionTree, GaussianNB, KNeighbors, LogisticRegression, LinearDiscriminant and SVM

Remember to split your dataset in two: train and validation.

Following this format:

# print('%s: %f (%f)' % (model_name, cv_results.mean(), cv_results.std()))
DecisionTree: 0.927191 (0.043263)
GaussianNB: 0.928858 (0.052113)
KNeighbors: 0.937191 (0.056322)
LogisticRegression: 0.920897 (0.043263)
LinearDiscriminant: 0.923974 (0.040110)
SVM: 0.973972 (0.032083)
