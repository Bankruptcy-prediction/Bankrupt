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

