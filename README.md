# Google-Search-Predictions-Time-Series-Analysis

Introduction
In this assignment we are going to implement a KNN algorithm also known as K-Nearest Neighbors.  It is one of the types of supervised learning algorithm also known as lazy learning algorithm due to its proficient learning behavior to a discriminative function from the training dataset or we can call it a non-parametric algorithm. This theory less functionality makes it unique as the data which is generated from any source doesn’t follow any specific assumptions. 

As we have labelled the dataset, we are going to supervise the machine learning algorithm, so extracted dataset will be divided into 2 parts i.e., training and test sets. This algorithm can be used as both regression and classification, in the assignment as the dataset is continuous not discrete so we are going to use KNN regression algorithm. 

The problem statement for the below work is obtained from ( https://drive.google.com/file/d/1YUP1oVuegqI6fIpU9b0uayi3XimEwlit/view?usp=sharing ), the dataset used for the problem and python code used for the code has been uploaded in the google folder (https://drive.google.com/drive/folders/1K-FfIcA2k223Rv-QoRnA2KbtVvi9fBNK?usp=sharing  ). As per the given problem statement we have built the model using a number of dimensional n = {2,3} over the 5 year of data taken from google trends distributed week wise. The computed k values are k = (1,3,5,7) where the distance metric with Euclidean distance and Manhattan Distance

In the following step we are importing all the required libraries based on requirement of the assignment
 

We have downloaded the dataset from google trends website for over the period of 5 years week wise. Our dataset consists of the number of times ‘Predictive Analytics’ was searched in the United States. The dataset consists of around 258 rows and 2 columns one is Week and second is number of counts. Here, we have converted the Weeks from date format to number of weeks counts in sequential numbers.
 
Here after we have dropped the Week column and stored it into a new variable called data_2d.
 

The nearest neighbors in any given problem are calculated based on the value of k i.e., number of neighbors. Whereas, in a general classified problem dataset the number of predictors n is plotted in n dimensional plan and for every entry of test dataset the data points are plotted in same n-dimensional space thereafter the distance is calculated for all the values of the dataset. In the form of an equation, we can represent the knn model as. For the given problem we have given n+1 as the dimensionality of the model.
xi+1 = m(xi, xi−1, ..., xi−n) 
Here, we have used the lag method to increase the dimensionality of the dataset. Firstly, we are started with n = 2.
 

Distance Metrics:
As we are aware that in knn algorithms the distance between the data points are calculated to do so there are a number of methods available. In our problem statement we are asked to calculate distance using Euclidean and Manhattan Distance. For this problem we have used a ‘sklearn’ library which states the knn classifier if we allocate the power parameter for default metric Minkowski as p = 1, the algorithm calculates the values using Manhattan distance. Whereas with p = 2 it will be calculated as Euclidean distance [1]
 


In the below steps we are going to repeat all the steps which we did for n= 2 dimensional data, here we are going to use n = 3 dimensional data.
 
 

After repeating the steps we are going to compare the results and display in the tabular form.
 
Conclusion:

1. In case of k=1 overfits the training data.
2. As k value is increasing the RMSE for the model is increasing.  
3. In conclusion as per the results we have observed that the following combination of parameters where n = 2, k = 3, distance metric = Manhattan results into rmse = 8.923423 with given data. Results given are as follows:  Real value = 46 and Predicted Value = 44.

 


References:

[1] sklearn.neighbors.KNeighborsRegressor¶
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html?highlight=kneighbors+regressor#sklearn.neighbors.KNeighborsRegressor



