import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.metrics import  accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from flask import * 
app = Flask(__name__)  

 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  


@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        a='C:\\Users\bujji\OneDrive\Desktop\Project\\'+str(f.filename)
        data=pd.read_csv(a)

        counts=data.nunique()
        for i,v in enumerate(counts):
            if(v==1):
                to_del=i 
        count=0
        insg=""
        for col_name in data.columns:
            if(count==to_del):
                insg=col_name
                break
            count+=1
        x=data.drop(insg,axis=1)

        def value0_1(df):
            q = []
            for colu in df.columns:
                if (df[colu].max()<=1) & (df[colu].min() >= 0):
                    q.append(colu)
            return(q)
        k=x.drop(['Bankrupt?'],axis=1)
        fraction= value0_1(k)
        non_fraction= x.drop(['Bankrupt?'],axis=1).columns.difference(fraction)

        p = preprocessing.normalize(x[non_fraction])
        norm_data = pd.DataFrame(p, columns=x[non_fraction].columns)

        data_scale = pd.concat([x.drop(non_fraction,axis=1),norm_data],axis = 1)

        X = data_scale.drop('Bankrupt?', axis = 1)
        y = data_scale['Bankrupt?']

        sm = SMOTE(random_state=123)
        X_sm , y_sm = sm.fit_resample(X,y)

        reduced_models={"Logistic Regression": LogisticRegression(),"K-Nearest Neighbors": KNeighborsClassifier(),"Decision Tree": DecisionTreeClassifier(),"SVM(Linear Kernel)": LinearSVC(),"SVM(RBF Kernel)": SVC(),"Neural Network": MLPClassifier(),"Random Forest": RandomForestClassifier(),"Gradient Boosting": GradientBoostingClassifier()}

        x_train, x_test, Y_train, Y_test = train_test_split(X_sm,y_sm,test_size = .2,random_state = 124)
        answer = RandomForestClassifier(n_estimators=100, random_state = 777)
        answer.fit(x_train, Y_train)
        Y_pred = answer.predict(x_test)
        ans1=[]
        s=""
        for name, model in reduced_models.items():
            model.fit(x_train, Y_train)
            ans1.append(name)
            s=s+str(name)+" trained."+"<br/>"

        resu= []
        ss=""
        for name, model in reduced_models.items():
            result = model.score(x_test, Y_test)
            resu.append(result)
            ss=ss+str(name)+": {:.2f}%".format(result * 100)+"<br/>"
        
        sns.countplot(x=data['Bankrupt?'])

        ins = []
        ins1 = []
        for ans in x.iloc[:,:].columns:
            if (x[ans].corr(x['Bankrupt?']) > 0.00000000001):
                ins.append(ans)
            ins1.append(ans)
        res = x.copy()
        plt.figure(figsize=(20,20))
        res = res[ins].corr()
        # plot the heatmap
        sns.heatmap(res,xticklabels=res.columns,yticklabels=res.columns,cmap='RdYlGn')

        def preprocess_inputs(df):
            df = df.copy()
            y = df['Bankrupt?']
            X = df.drop('Bankrupt?', axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
            return X_train, X_test, y_train, y_test

        X_train, X_test, y_train, y_test=preprocess_inputs(data)

        for i in X_test.columns:
            if(i not in ins):
                X_test=X_test.drop(i,axis=1)

        for i in X_train.columns:
            if(i not in ins):
                X_train=X_train.drop(i,axis=1)

        ans=[]
        sss=""
        for name, model in reduced_models.items():
            model.fit(X_train, y_train)
            ans.append(name)
            sss=sss+name+" trained"+"<br/>"

        reduced_results = []
        ssss=""
        for name, model in reduced_models.items():
            result = model.score(X_test, y_test)
            reduced_results.append(result)
            ssss=ssss+name+": {:.2f}%".format(result * 100)+"<br/>"

        maxi=max(reduced_results)
        k=reduced_results.index(maxi)

        plt.figure(figsize= (20,20))
        sns.barplot(x = ans,y = reduced_results,palette='pastel')
        plt.title("Plotting the Model Accuracies", fontsize=16, fontweight="bold")


        plt.figure(figsize = (15,10))   
        plt.plot(ans1               ,reduced_results,'g*',ans1, resu, 'ro')
        plt.title("Comparing the Accuracy of model using two different preprocesss",fontsize = 20)
        plt.show()
        

        maxi=max(reduced_results)
        y1=reduced_results.index(maxi)
        yy=ans1[y1]
        maxi1=max(resu)
        k1=resu.index(maxi1)
        kk=ans1[k1]
        s1=""
        if(maxi>maxi1):
            s1=s1+"We have used two methods called SMOTE and correlation for preprocessing of data ,here using correlation have the best performance for preprocessing than the SMOTE with an accuracy of"+" "+str(maxi)+" "+"model"+" "+str(yy)
        else:
            s1=s1+"We have used two methods called SMOTE and correlation for preprocessing of data ,here using SMOTE have the best performance for preprocessing than the correlation with an accuracy of"+" "+str(maxi1)+" "+"model"+" "+str(kk)
        
        
        #render_template("success.html", name =plt.show())
        
        return "Shape of the dataset: "+str(data.shape)+"<br/>"+"<br/>"+"Name of the features:"+"<br/>"+str(data.columns)+"<br/>"+"<br/>"+"Deleted columns number and name after pre-processing of the data: "+str(to_del)+" "+"("+str(insg)+")"+"<br/>"+"<br/>"+"Shape of dataset after pre-processing:  "+str(x.shape)+"<br/>"+"<br/>"+"Number of columns having duplicates: "+str(x.duplicated().sum())+"<br/>"+"<br/>"+"Data pre-processing by using SMOTE"+"<br/>"+"<br/>"+"Model traning"+"<br/>"+s+"<br/>"+"Model's accuracy:"+"<br/>"+ss+"<br/>"+"Count Plot of bankrupt "+"<br/>"+render_template("success.html", name =plt.show())+"<br/>"+"<br/>"+"Data Pre-Processing by using corelation"+"<br/>"+"<br/>"+"Reduced columns after implementing the correlation(Pre-Processing): "+"<br/>"+str(list(ins))+"<br/>"+"<br/>"+"Heatmap representation of reduced columns: "+"<br/>"+render_template("success.html", name1 =plt.show())+"<br/>"+"<br/>"+"Model traning"+"<br/>"+sss+"<br/>"+"<br/>"+"Model's accuracy:"+"<br/>"+ssss+"<br/>"+"<br/>"+render_template("success.html", name3 =plt.show())+"<br/>"+"<br/>"+s1

if __name__ == '__main__':  
    app.run(debug = True)