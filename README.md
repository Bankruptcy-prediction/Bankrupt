# Bankrupt

# Abstract

Nowadays, one of the significant problems that we are facing is bankruptcy. However, observing the preceding data [Taiwan Economic journal for the years 1999 to 2009] of Bankrupt organizations frequently causes incorrect prediction and economic loss to the company. So to decrease the effect, we use machine learning concepts to predict any fraud or data imbalance.
It is imperative to choose the best Pre-processing method like correlation, SMOTE[Synthetic Minority Oversampling Technique], an algorithm to predict the Bankrupt. Therefore, we use suitable classification methods like Support Vector Machine [SVM], Logistic Regression, K- Nearest Neighbours, Random forest, Decision tree, and Gradient Boosting to analyse and predict the model's accuracy. We are trying to build a model which analyses the data.
By this, we can expect which model is best suitable among various models, incorrect analysis, and economic losses to the bank. This study indicates multiple models by using different pre- processing methods and show the graph of two pre-processing data with their accuracy score of the various model. Finally ensemble the resulting models to have a better prediction. Optimize the model and eventually run the model on new data to observe its performance.


# INTRODUCTION
Bankruptcy means , when an organization or person is unable to honour its financial obligations or make payment to its creditors, it files for bankruptcy. A petition is filed in the court for the same where all the outstanding debts of the company are measured and paid out if not in full from the company’s assets.
Evaluating the risk of organization is an imperative to the investors and creditors. They are large features that causes the economic loss and bankruptcies for this reason bankruptcy prediction has become a extensive area of research .Few cases bankruptcy has become business to suffer. It become a serious issue which effects globally and locally. Keeping this issue in mind ,we undertook the task of building different supervised-machine learning algorithms, along with a comparative analysis of each model, in order to identify those that are better suited for predicting economic bankruptcy
There are some existing model to predict and analyse the bankruptcy based on statistical modelling and machine learning algorithms that have been proposed. Recently some of the models have successfully made for classification ,regression [which is a part of supervised learning]
The purpose of bankruptcy prediction is to assess the financial health status and future perspectives of a company. Based on the all existing model , now we are trying to develop a model in such a way to increase the accuracy , so we choose a preprocessing technique called correlation , then we train the model and check the accuracy this process gives less performance so we develop another model by changing the data preprocessing technique called SMOTE , then we train the model and check the accuracy this process gives better performance in random forest then the before technique
In summary, the results presented by before model were reproduced. In addition, a random forest with significantly better performance than previously model was built. Finally, it was found that missing values, single valued feature , pre-processing techniques in the data set play an important role, they carry almost all information that is useful for the predictions.

# PROCESS FLOW

![image](https://user-images.githubusercontent.com/71624353/180659363-97f63c73-7866-4c0c-982b-cdaa98fe3658.png)



# LOADING DATASET AND PACKAGES:

Initially we select the dataset from the Kaggle of bankruptcy data from the Taiwan Economic Journal for the years 1999–2009 [https://www.kaggle.com/datasets/fedesoriano/company- bankruptcy-prediction]. We understood the dataset's features and how it can interrupt the output of the data. we started developing the project. We install all the packages that are needed for the development of the model.
DATA PRE-PROCESSING:

![image](https://user-images.githubusercontent.com/71624353/180659402-97c283c0-3e59-4930-a086-2b513dc188f1.png)

we perform the data pre-processing, on the data set to reduce the feature[i.e columns] . For the selection of features we chosen the SMOTE, CORRELATION techniques, check any null values
,duplicate rows , inconsistence of data.

# SPLITING AND TRAINING OF DATASET:

we split the data into two different datasets one for testing ,another for training. For training 70% of dataset and 30% of dataset. Perform the model training for the training dataset . we choose the different models[ Random Forest Classifier, Gradient Boosting Classifier, SVC , Robust Scaler, train_test_split, Random Forest Classifier, Gradient Boosting Classifier , Logistic Regression , KNeighborsClassifier] for training the dataset. We perform another pre-processing method called the correlation technique on the dataset. We train the model by using a different model and we test the model that how accurate the model is performing .
TESTING OF DATA SET :

We train the model; here, we are testing the model how accurate the model is performing. We compare the accuracy of different algorithms and choose the best-given accuracy model among all. We compare the accuracy of different algorithms and choose the best-given accuracy model among all.
# FLASK :

We are done with the model training and testing , next we are changing our code into application by using the flask. Flask is a small and lightweight Python web framework that provides useful tools and features that make creating web applications in Python easier. It gives developers flexibility and is a more accessible framework for new developers since you can build a web application quickly using only a single Python file. In the flask we have included the static web page to upload the file form the user and predict the result.
 
# XAMPP:

With the help of XAMPP we were able to serve our web pages with were embedded by our python application with the help of flask package on the Internet and create and manipulate database in MySQL by storing user data and login credentials to access the webpage and to validate the user information which were given during the registration.
1)	Installing the XAMPP application by following steps from the guide which is given by the XAMPP organization.
2)	Run the application

![image](https://user-images.githubusercontent.com/71624353/180659434-c1d76146-ce91-4804-a987-1571375132c4.png)

3)	After starting the application we will be getting to the home page of XAMPP where we have to click start the Apache and MySQL module for the connection of our web page and to store the user information in the MySQL data base.

![image](https://user-images.githubusercontent.com/71624353/180659452-9c61a274-f9ad-46f2-8a9a-2369acef6a01.png)


4)	After starting the modules, we must go to MyPHPAdmin home page with the help of the URL: http://localhost/phpmyadmin/ . Then there by clicking on database we can create a new database in the table form where we can link it to the html source code where we are taking the user information to access our webpage. So with the help of user information, we can validate and verify the user who is trying to access our webpage.
5)	Now we are ready to host our web page to which our python application has been embedded.


# ALGORITHMS




# RANDOM FOREST

How Random Forest Works-

Random forest is a supervised learning algorithm. The “forest” it builds is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.
The big advantage of random forest is that it can be used for classification and regression problems, which form the majority of current machine learning systems.

Let’s look at the random forest in classification since classification is sometimes considered the building block of machine learning. Below you can see what a random forest would look like with two	trees:


Random Forest Feature Importance-

Another excellent advantage of the random forest approach is that determining the relative value of each feature on the forecast is relatively simple. Sklearn has a fantastic tool for measuring the relevance of a feature by looking at how much the tree nodes that use it reduce impurity over the entire forest. After training, it calculates this score for each characteristic and adjusts the findings so that the total importance is equal to one.

The Working process can be explained in the below steps-

1)	Select random K data points from the training set.

2)	Build the decision trees associated with the selected data points (Subsets).

3)	Choose the number N for decision trees that you want to build.

4)	Repeat Step 1 & 2.

5)	For new data points, find the predictions of each decision tree, and assign the new data points to the category that wins the majority votes.

# DECISION TREE ALGORITHM


Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.

The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).



 
In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

Types of Decision Trees

Types of decision trees are based on the type of target variable we have. It can be of two types:

1.	Categorical Variable Decision Tree: Decision Tree which has a categorical target variable then it called a Categorical variable decision tree.
2.	Continuous Variable Decision Tree: Decision Tree has a continuous target variable then it is called Continuous Variable Decision Tree.

Important Terminology related to Decision Trees-

Root Node, Splitting, Decision Node, Leaf / Terminal Node, Pruning, Branch / Sub-Tree, Parent and Child Node.

How do Decision Trees work-

The decision of making strategic splits heavily affects a tree’s accuracy. The decision criteria are different for classification and regression trees.
Decision trees use multiple algorithms to decide to split a node into two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that the purity of the node increases with respect to the target variable. The decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.
Steps-

1)Begin the tree with the root node, says S, which contains the complete dataset.


 
3)Divide the S into subsets that contains possible values for the best attributes. 4)Generate the decision tree node, which contains the best attribute.
5)Recursively make new decision trees using the subsets of the dataset created in step -3. Continue this process until a stage is reached where you cannot further classify the nodes and called the final node as a leaf node.

# K-NEAREST NEIGHBORS

K-nearest neighbours (KNN) is a type of supervised learning algorithm used for both regression and classification. KNN tries to predict the correct class for the test data by calculating the distance between the test data and all the training points. Then select the K number of points which is closet to the test data. The KNN algorithm calculates the probability of the test data belonging to the classes of ‘K’ training data and class holds the highest probability will be selected. In the case of regression, the value is the mean of the ‘K’ selected training points.

How does K-NN work-

The K-NN working can be explained on the basis of the below algorithm:

1)	Select the number K of the neighbours

2)	Calculate the Euclidean distance of K number of neighbours

3)	Take the K nearest neighbours as per the calculated Euclidean distance.

4)	Among these k neighbours, count the number of the data points in each category.

5)	Assign the new data points to that category for which the number of the neighbour is maximum.

6)	Our model is ready.
 
![image](https://user-images.githubusercontent.com/71624353/180660100-be0edc74-16cd-4cf5-bf31-2d2d4a95b71e.pn

# LOGISTIC REGRESSION


Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. The nature of target or dependent variable is dichotomous, which means there would be only two possible classes.
In simple words, the dependent variable is binary in nature having data coded as either 1 (stands for success/yes) or 0 (stands for failure/no).
Mathematically, a logistic regression model predicts P(Y=1) as a function of X. It is one of the simplest ML algorithms that can be used for various classification problems such as spam detection, Diabetes prediction, cancer detection etc.
Types-Binary, Multinomial, Ordinal

Steps in Logistic Regression: To implement the Logistic Regression using Python, we will use the same steps as we have done in previous topics of Regression. Below are the steps:
 
Data Pre-processing step
Fitting Logistic Regression to the Training set Predicting the test result
Test accuracy of the result(Creation of Confusion matrix) Visualizing the test set result.


# SVM (LINEAR KERNEL)

Linear Kernel is used when the data is Linearly separable, that is, it can be separated using a single Line. It's one of the most often utilised kernels. It's typically employed when a data set has a lot of features. Text Classification is an example of a feature with a lot of features, because each alphabet is a new feature. As a result, Linear Kernel is frequently used in Text Classification.

 ![image](https://user-images.githubusercontent.com/71624353/180660118-c13069e6-1855-45de-975f-543c1aec19be.png)

In the above image, there are two set of features “Blue” features and the “Yellow” Features. Since these can be easily separated or in other words, they are linearly separable, so the Linear Kernel can be used here.

Advantages of using Linear Kernel:
1.	Training a SVM with a Linear Kernel is Faster than with any other Kernel.

2.	When training a SVM with a Linear Kernel, only the optimisation of the C

Regularisation parameter is required. On the other hand, when training with other kernels, there is a need to optimise the γ parameter which means that performing a grid search will usually take more time.



SAMPLE OUTPUT AND FUTURE SCOPE
1.	We develop the model for the user to predict the bankruptcy analyze.
2.	Initially to use the application, We create a login page where if the user has an existing account, he can directly access the application.

![image](https://user-images.githubusercontent.com/71624353/180660532-83e92f80-7dce-410d-9a3b-689edc9a40d2.png)


3.	If the user is new but wants to use the application, he needs to click on Register Here now, fill in the basic details asked, and click on Register to get registered.


 ![image](https://user-images.githubusercontent.com/71624353/180660572-88217c34-5dc2-48c7-bb7b-a3e2dee5e567.png)


4.	Now all the user information has been saved in the server-side database and the user gets access to the web application by entering his E-mail id and password, which he provided during the registration.
5.	After the user logins, the page will atomically redirect to the application home page, where there is a short introduction on bankruptcy next, the user gets an option to upload the data in the CSV file format which he wants to predict and can click on upload.
![image](https://user-images.githubusercontent.com/71624353/180660630-44939045-2442-405d-92bd-ba0af76bc15e.png)


6.	Just wait for two minutes to train and test the data set after uploading.
7.	Finally, the user gets the prediction over the different models and pre-processes for the dataset he has provided.
 
 ![image](https://user-images.githubusercontent.com/71624353/180660654-8f49183c-2344-45dc-8864-3e8085fde5ab.png)
![image](https://user-images.githubusercontent.com/71624353/180660742-db7cd28b-178b-4800-9361-e9fa6cdbd77e.png)

Representations:
Count plot-
![image](https://user-images.githubusercontent.com/71624353/180660772-c8090e53-1b21-44af-8c4a-1d55cc67a8be.png)

HeatMap-

![image](https://user-images.githubusercontent.com/71624353/180660781-b294d463-cf35-4142-998e-dccb591cb5a2.png)\

![image](https://user-images.githubusercontent.com/71624353/180660792-62e77629-fe8b-4b3b-93a8-4e23b3348e7f.png)

Future Scope:
Since Bankruptcy prediction is an important problem in finance, since successful predictions would allow stakeholders to take early actions to limit their economic losses. In recent years many studies have explored the application of machine learning models to bankruptcy prediction with financial ratios as predictors.
In future work, we can consider developing the bankrupt prediction system for an individual user. We can do this by acquiring the user's financial details with the help of government-issued financial proof such as a pan card. Once the user enters his government-issued financial proof details, the system gets all the user's financial information, such as bank savings, property details, loans, etc. Using this information our system uses machine learning algorithms to analyze the data and efficiently classify whether the user will go bankrupt in the near future.If so the system can automatically send an e-mail alert to the user along with some of the prevention measures that are required to avoid bankrupt
