import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn . preprocessing import StandardScaler
from sklearn . naive_bayes import BernoulliNB
from sklearn . neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn . linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectKBest,SelectFromModel,SequentialFeatureSelector,RFE, VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score,matthews_corrcoef,roc_curve
from tabulate import tabulate
import warnings


############## Data Preprocessing ##############

df = pd.read_csv('URL Dataset.csv')
df.drop(columns = 'HttpsInHostname', inplace = True)

count = df.isna().sum()

if(sum(count) == 0):
    print("No null values in the dataset")
    df.to_csv("Preprocessed_Dataset.csv")
else:
    print("processing is needed")
    print("New dataset is \n")
    df = df.dropna(how='any',axis=0)
    

print(df.columns)



############ Checking Data Ratio to identify if its balanced or not ##############   

if((len(df.loc[df['CLASS_LABEL']== 0])) == (len(df.loc[df['CLASS_LABEL']== 1]))):
    print("dataset is balanced")
else:
    print("dataset is imbalanced")

 
fig, ax = plt.subplots(1,2, figsize=(10, 4))
g1 = sns.countplot(df.CLASS_LABEL,ax=ax[0],palette="bright");
g1.set_title("Count of legitimate and suspicious URL")
g1.set_xlabel("CLASS_LABEL")
g1.set_ylabel("Count")
g2 = plt.pie(df["CLASS_LABEL"].value_counts().values,explode=[0,0],labels=df.CLASS_LABEL.value_counts().index, autopct='%1.1f%%',colors=['Teal','Hotpink'])
plt.show()
    
df.to_csv('PreProcessedDataset.csv', header=True)

X = df.drop('CLASS_LABEL', axis=1)
Y = df['CLASS_LABEL']

correlation = df.corr(method = 'pearson')
print(correlation)

plt.figure(figsize=(50, 50))
ax = sns.heatmap(correlation, vmin=-1, vmax=1,annot=True, center=0,cmap='BrBG',square=True,annot_kws={'size': 10})
ax.set_xticklabels(ax.get_xticklabels(),fontdict={'fontsize':24},rotation=45,horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(),fontdict={'fontsize':24},verticalalignment='top')
ax.set_title('Correlation Heatmap', fontdict={'fontsize':24}, pad=12)
plt.show()
############## 5 classification algorithms ##############

####### Logistic Regression 
def URL_LogReg(X_train ,X_test , Y_train , Y_test, cv, lab):
    
    log_reg_classifier = LogisticRegression ()
    Grid = GridSearchCV(log_reg_classifier,{'penalty':['l2']}, cv=cv)
    Grid.fit (X_train , Y_train)
    prediction = Grid.predict (X_test)
    accuracy = np. mean(prediction == Y_test)
    rocaucscore = roc_auc_score(Y_test, Grid.predict_proba(X_test)[:, 1])
    lr_probs = Grid.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    label = "ROC Curve for Log Reg with " + lab
    plt.title(label)
    plt.show()

    print("Accuracy in prediction for Log Regression with",lab,accuracy)
    return(accuracy,prediction,Y_test,rocaucscore)


####### Random Forest 
def URL_RandomForest(X_train ,X_test , Y_train , Y_test, cv, lab):
    
    model = RandomForestClassifier()
    model = RandomForestClassifier( n_estimators =10 , max_depth =5,criterion ='entropy')
    Grid = GridSearchCV(model, {'criterion' :['entropy']},cv=cv)
    Grid.fit (X_train , Y_train)
    prediction = Grid.predict (X_test)
    accuracy = np. mean(prediction == Y_test)
    rocaucscore = roc_auc_score(Y_test, Grid.predict_proba(X_test)[:, 1])
    lr_probs = Grid.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    plt.plot(lr_fpr, lr_tpr, marker='.')
    label = "ROC Curve for Random Forest with " + lab
    plt.title(label)
    plt.show()
    print("Accuracy of Random forest with",lab, accuracy)    
    return(accuracy,prediction,Y_test,rocaucscore)


####### KNN 
def URL_KNN(X_train ,X_test , Y_train , Y_test, cv, lab):

    knn_classifier = KNeighborsClassifier(n_neighbors = 3)
    Grid = GridSearchCV(knn_classifier,{'weights':['uniform']}, cv=cv)
    Grid.fit (X_train , Y_train)
    prediction = Grid.predict (X_test)
    accuracy = np. mean(prediction == Y_test)
    rocaucscore = roc_auc_score(Y_test, Grid.predict_proba(X_test)[:, 1])
    lr_probs = Grid.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    plt.plot(lr_fpr, lr_tpr, marker='.', label='KNN')
    label = "ROC Curve for KNN with " + lab
    plt.title(label)
    plt.show()
    print("Accuracy of  KNN with", lab, accuracy)
    return(accuracy,prediction,Y_test,rocaucscore)

####### Naive Bayesian
def URL_NaiveBayesian(X_train ,X_test , Y_train , Y_test, cv, lab):
    
    NB_classifier = BernoulliNB()
    Grid = GridSearchCV(NB_classifier,{'fit_prior':['True']}, cv=cv)
    Grid.fit (X_train , Y_train)
    prediction = Grid.predict (X_test)
    accuracy = np. mean(prediction == Y_test)
    rocaucscore = roc_auc_score(Y_test, Grid.predict_proba(X_test)[:, 1])
    lr_probs = Grid.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    plt.plot(lr_fpr, lr_tpr, marker='.', label='NaiveBayesian')
    label = "ROC Curve for Naive Bayesian with " + lab
    plt.title(label)
    plt.show()
    print("Accuracy in prediction for Naive Bayesian with",lab, accuracy)
    
    return(accuracy,prediction,Y_test,rocaucscore)

####### Adaboost
def URL_adaboost(X_train ,X_test , Y_train , Y_test, cv, lab):
    
    adaboost = AdaBoostClassifier(n_estimators=100, base_estimator= None,learning_rate=1, random_state = 1)
    Grid = GridSearchCV(adaboost,{'algorithm':['SAMME']}, cv=cv)
    Grid.fit (X_train , Y_train)
    prediction = Grid.predict (X_test)
    accuracy = np. mean(prediction == Y_test)
    rocaucscore = roc_auc_score(Y_test, Grid.predict_proba(X_test)[:, 1])
    lr_probs = Grid.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Adaboost')
    label = "ROC Curve for Adaboost with " + lab
    plt.title(label)
    plt.show()
    print("Accuracy in prediction for Adaboost with",lab, accuracy)
    
    return(accuracy,prediction,Y_test,rocaucscore)


############## 5 Feature Selection ##############

#### Select K Best
def selectKbest(X_train,Y_train):
    
    warnings.simplefilter('ignore')
    kbest = SelectKBest(k=10)
    warnings.simplefilter('ignore')
    kbest.fit(X_train, Y_train)
    warnings.simplefilter('ignore')
    kbest_features = X_train.columns[kbest.get_support()]
    
    return(kbest_features)

#### Variance Threshold
def variancethreshold(X_train,Y_train):
    
    warnings.simplefilter('ignore')
    selector = VarianceThreshold(1)
    warnings.simplefilter('ignore')
    selector.fit(X)
    
    return(X.columns[selector.get_support()])

#### Recursive feature selection
def recursivefeatureselection(X_train,Y_train):
    
    warnings.simplefilter('ignore')
    rfe_selector = RFE(estimator=LogisticRegression(),n_features_to_select = 10, step = 1)
    warnings.simplefilter('ignore')
    rfe_selector.fit(X, Y)
    
    return(X.columns[rfe_selector.get_support()])

#### Selectfrommodel
def selectfrommodel(X_train,Y_train):
    
    warnings.simplefilter('ignore')
    sfm_selector = SelectFromModel(estimator=LogisticRegression())
    warnings.simplefilter('ignore')
    sfm_selector.fit(X, Y)
    
    return(X.columns[sfm_selector.get_support()])

#### Sequentialfeatureselection
def sequentialfeatureselection(X_train,Y_train):
    
    warnings.simplefilter('ignore')
    sfs_selector = SequentialFeatureSelector(estimator=LinearRegression(), n_features_to_select = 10, cv =5, direction ='backward')
    warnings.simplefilter('ignore')
    sfs_selector.fit(X, Y)
    
    return(X.columns[sfs_selector.get_support()])

result = []

######## Performance measures.
def confusion_matrix(prediction, Y_test,rocaucscore,acc,classifier):
    
    # res = []
    TP,TN,FP,FN = 0,0,0,0

    for i in range (0,len(Y_test)):
        if((Y_test.values[i]== 0) and (prediction[i])==0):
            TP=TP+1
        elif((Y_test.values[i]== 1) and (prediction[i])==1):
            TN= TN+1
        elif(((Y_test.values[i]== 0) and (prediction[i])==1)):
            FP=FP+1
        else:
            FN=FN+1
    
    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    mcc = matthews_corrcoef(prediction, Y_test)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1_score =  2/(1/Precision + 1/Recall)
    rocaucscore = rocaucscore
    accuracy = acc
    
    res = [classifier,FPR,TPR,Precision,Recall,F1_score,mcc,rocaucscore,accuracy]
    result.append(res)
    
    if(classifier == 'Adaboost with Sequential Feature Selection' ):
        return(result)
        


######################### Splitting Datasets , K fold cross validation

def splitfn(df):
    
    accuracy = []
    accuracybase = []
    
    X = df.drop('CLASS_LABEL', axis=1)
    Y = df['CLASS_LABEL']
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=None)
    X_train ,X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.34 ,stratify=Y, random_state =42)
    
    a=selectKbest(X_train,Y_train)
    b=variancethreshold(X_train,Y_train)
    c=recursivefeatureselection(X_train,Y_train)
    d=selectfrommodel(X_train,Y_train)
    e=sequentialfeatureselection(X_train,Y_train)
    
    
    print("The best features using SelectKbest are \n",a)
    print("The best features using variancethreshold are \n",b)
    print("The best features using recursivefeatureselection are \n",c)
    print("The best features using selectfrommodel are \n",d)
    print("The best features using sequentialfeatureselection are \n",e)
    
    
    acc1,prediction1,Y_test1,rocaucscore1 = URL_LogReg(X_train ,X_test , Y_train , Y_test, cv,"all attributes")
    confusion_matrix(prediction1, Y_test1,rocaucscore1,acc1,'Log regression with all attributes')
    accuracybase.append(acc1)
    FullTrainset = X_train.reset_index(drop=True)
    FullTestset = X_test.reset_index(drop=True)
    FullTrainlabels = Y_train.reset_index(drop=True)
    FullTestlabels = Y_test.reset_index(drop=True)
    FullTrainset.to_csv('TrainSetwithallfeatures.csv', header=True)
    FullTestset.to_csv('TestSetwithallfeatures.csv', header=True)
    FullTrainlabels.to_csv('TrainSetwithlabels.csv', header=True)
    FullTestlabels.to_csv('TestSetwithlabels.csv', header=True)
    
    X_trainres1 = X_train[a].reset_index(drop=True)
    X_testres1 = X_test[a].reset_index(drop=True)
    X_trainres1.to_csv('TrainSetwithSelectKbest.csv', header=True)
    X_testres1.to_csv('TestSetwithSelectKbest.csv', header=True)
    acc2,prediction2,Y_test2,rocaucscore2 = URL_LogReg(X_trainres1 ,X_testres1 , Y_train , Y_test, cv,"Select K best")
    confusion_matrix(prediction2, Y_test2,rocaucscore2,acc2,'Log regression with Select K best')
    accuracy.append(acc2)

    X_trainres2 = X_train[b].reset_index(drop=True)
    X_testres2 = X_test[b].reset_index(drop=True)
    X_trainres2.to_csv('TrainSetwithvariancethreshold.csv', header=True)
    X_testres2.to_csv('TestSetwithvariancethreshold.csv', header=True)
    acc3,prediction3,Y_test3,rocaucscore3= URL_LogReg(X_trainres2 ,X_testres2 , Y_train , Y_test, cv,"variance threshold")
    confusion_matrix(prediction3, Y_test3,rocaucscore3,acc3,'Log regression with variance threshold')
    accuracy.append(acc3)
    
    X_trainres3 = X_train[c].reset_index(drop=True)
    X_testres3= X_test[c].reset_index(drop=True)
    X_trainres3.to_csv('TrainSetwithrecursivefeatureselection.csv', header=True)
    X_testres3.to_csv('TestSetwithrecursivefeatureselection.csv', header=True)
    acc4,prediction4,Y_test4,rocaucscore4 = URL_LogReg(X_trainres3 ,X_testres3 , Y_train , Y_test, cv,"RecursiveFeatureSelection")
    confusion_matrix(prediction4, Y_test4,rocaucscore4,acc4,'Log regression with Recursive Feature Selection')
    accuracy.append(acc4)
    
    X_trainres4 = X_train[d].reset_index(drop=True)
    X_testres4= X_test[d].reset_index(drop=True)
    X_trainres4.to_csv('TrainSetwithselectfrommodel.csv', header=True)
    X_testres4.to_csv('TestSetwithselectfrommodel.csv', header=True)
    acc5,prediction5,Y_test5,rocaucscore5 = URL_LogReg(X_trainres4 ,X_testres4 , Y_train , Y_test, cv,"Select from model")
    confusion_matrix(prediction5, Y_test5,rocaucscore5,acc5,'Log regression with Select from model')
    accuracy.append(acc5)
    
    X_trainres5 = X_train[e].reset_index(drop=True)
    X_testres5= X_test[e].reset_index(drop=True)
    X_trainres5.to_csv('TrainSetwithsequentialfeatureselection.csv', header=True)
    X_testres5.to_csv('TestSetwithsequentialfeatureselection.csv', header=True)
    acc6,prediction6,Y_test6,rocaucscore6 = URL_LogReg(X_trainres5 ,X_testres5 , Y_train , Y_test, cv,"Sequential Feature Selection")
    confusion_matrix(prediction6, Y_test6,rocaucscore6,acc6,'Log regression with Sequential Feature Selection')
    accuracy.append(acc6)

    acc7,prediction7,Y_test7,rocaucscore7 = URL_RandomForest(X_train ,X_test , Y_train , Y_test, cv,"all attributes")
    confusion_matrix(prediction7, Y_test7,rocaucscore7,acc7,'RandomForest with all attributes')
    accuracybase.append(acc7)    
    X_trainres12 = X_train[a]
    X_testres12 = X_test[a]
    acc8,prediction8,Y_test8,rocaucscore8 = URL_RandomForest(X_trainres12 ,X_testres12 , Y_train , Y_test, cv,"Select K best")
    confusion_matrix(prediction8, Y_test8,rocaucscore8,acc8,'RandomForest with Select K best')
    accuracy.append(acc8)
    X_trainres22 = X_train[b]
    X_testres22 = X_test[b]
    acc9,prediction9,Y_test9,rocaucscore9 = URL_RandomForest(X_trainres22 ,X_testres22 , Y_train , Y_test, cv,"variance threshold")
    confusion_matrix(prediction9, Y_test9,rocaucscore9,acc9,'RandomForest with variance threshold')
    accuracy.append(acc9)
    
    X_trainres32 = X_train[c]
    X_testres32= X_test[c]
    acc10,prediction10,Y_test10,rocaucscore10 = URL_RandomForest(X_trainres32 ,X_testres32 , Y_train , Y_test, cv,"RecursiveFeatureSelection")
    confusion_matrix(prediction10, Y_test10,rocaucscore10,acc10,'RandomForest with Recursive Feature Selection')
    accuracy.append(acc10)
    
    X_trainres42 = X_train[d]
    X_testres42= X_test[d]
    acc11,prediction11,Y_test11,rocaucscore11 = URL_RandomForest(X_trainres42 ,X_testres42, Y_train , Y_test, cv,"Select from model")
    confusion_matrix(prediction11, Y_test11,rocaucscore11,acc11,'RandomForest with Select from model')
    accuracy.append(acc11)
    
    X_trainres52 = X_train[e]
    X_testres52= X_test[e]
    acc12,prediction12,Y_test12,rocaucscore12 = URL_RandomForest(X_trainres52 ,X_testres52 , Y_train , Y_test, cv,"Sequential feature Selection")
    confusion_matrix(prediction12, Y_test12,rocaucscore12,acc12,'RandomForest with Sequential Feature Selection')
    accuracy.append(acc12)
    
    
    acc13,prediction13,Y_test13,rocaucscore13 = URL_KNN(X_train ,X_test , Y_train , Y_test, cv,"all attributes")
    confusion_matrix(prediction13, Y_test13,rocaucscore13,acc13,'KNN with all attributes')
    accuracybase.append(acc13)    
    X_trainres13 = X_train[a]
    X_testres13 = X_test[a]
    acc14,prediction14,Y_test14,rocaucscore14 = URL_KNN(X_trainres13 ,X_testres13 , Y_train , Y_test, cv,"Select K best")
    confusion_matrix(prediction14, Y_test14,rocaucscore14,acc14,'KNN with Select K best')
    accuracy.append(acc14)
    
    X_trainres23 = X_train[b]
    X_testres23= X_test[b]
    acc15,prediction15,Y_test15,rocaucscore15 = URL_KNN(X_trainres23 ,X_testres23 , Y_train , Y_test, cv,"variance threshold")
    confusion_matrix(prediction15, Y_test15,rocaucscore15,acc15,'KNN with variance threshold')
    accuracy.append(acc15)
    
    X_trainres33 = X_train[c]
    X_testres33= X_test[c]
    acc16,prediction16,Y_test16,rocaucscore16 =URL_KNN(X_trainres33 ,X_testres33 , Y_train , Y_test, cv,"RecursiveFeatureSelection")
    confusion_matrix(prediction16, Y_test16,rocaucscore16,acc16,'KNN with Recursive Feature Selection')
    accuracy.append(acc16)
    
    X_trainres43 = X_train[d]
    X_testres43 = X_test[d]
    acc17,prediction17,Y_test17,rocaucscore17 = URL_KNN(X_trainres43 ,X_testres43, Y_train , Y_test, cv,"Select from model")
    confusion_matrix(prediction17, Y_test17,rocaucscore17,acc17,'KNN with Select from model')
    accuracy.append(acc17)
    
    X_trainres53 = X_train[e]
    X_testres53= X_test[e]
    acc18,prediction18,Y_test18,rocaucscore18 = URL_KNN(X_trainres53 ,X_testres53 , Y_train , Y_test, cv, "Sequential feature Selection")
    confusion_matrix(prediction18, Y_test18,rocaucscore18,acc18,'KNN with Sequential Feature Selection')
    accuracy.append(acc18)
    
    
    acc19,prediction19,Y_test19,rocaucscore19 = URL_NaiveBayesian(X_train ,X_test , Y_train , Y_test, cv,"all attributes")
    confusion_matrix(prediction19, Y_test19,rocaucscore19,acc19,'NaiveBayesian with all attributes')
    accuracybase.append(acc19)    
    X_trainres14 = X_train[a]
    X_testres14= X_test[a]
    acc20,prediction20,Y_test20,rocaucscore20 = URL_NaiveBayesian(X_trainres14 ,X_testres14 , Y_train , Y_test, cv,"Select K best")
    confusion_matrix(prediction20, Y_test20,rocaucscore20,acc20,'NaiveBayesian with Select K best')
    accuracy.append(acc20)
    
    X_trainres24 = X_train[b]
    X_testres24 = X_test[b]
    acc21,prediction21,Y_test21,rocaucscore21 = URL_NaiveBayesian(X_trainres24 ,X_testres24 , Y_train , Y_test, cv,"variance threshold")
    confusion_matrix(prediction21, Y_test21,rocaucscore21,acc21,'NaiveBayesian with variance threshold')
    accuracy.append(acc21)
    

    X_trainres34 = X_train[c]
    X_testres34= X_test[c]
    acc22,prediction22,Y_test22,rocaucscore22 =URL_NaiveBayesian(X_trainres34 ,X_testres34 , Y_train , Y_test, cv,"Recursive Feature Selection")
    confusion_matrix(prediction22, Y_test22,rocaucscore22,acc22,'NaiveBayesian with Recursive Feature Selection')
    accuracy.append(acc22)
    
    X_trainres44 = X_train[d]
    X_testres44= X_test[d]
    acc23,prediction23,Y_test23,rocaucscore23 = URL_NaiveBayesian(X_trainres44 ,X_testres44, Y_train , Y_test, cv,"Select from model")
    confusion_matrix(prediction23, Y_test23,rocaucscore23,acc23,'NaiveBayesian with Select from model')
    accuracy.append(acc23)
    
    X_trainres54 = X_train[e]
    X_testres54= X_test[e]
    acc24,prediction24,Y_test24,rocaucscore24 = URL_NaiveBayesian(X_trainres54 ,X_testres54 , Y_train , Y_test, cv,"Sequential Feature Selection")
    confusion_matrix(prediction24, Y_test24,rocaucscore24,acc24,'NaiveBayesian with Sequential Feature Selection')
    accuracy.append(acc24)
    
    
    acc25,prediction25,Y_test25,rocaucscore25 = URL_adaboost(X_train ,X_test , Y_train , Y_test, cv,"all attributes")
    confusion_matrix(prediction25, Y_test25,rocaucscore25,acc25,'Adaboost with all attributes')
    accuracybase.append(acc25)    
    X_trainres15 = X_train[a]
    X_testres15= X_test[a]
    acc26,prediction26,Y_test26,rocaucscore26 = URL_adaboost(X_trainres15 ,X_testres15 , Y_train , Y_test, cv,"Select K best")
    confusion_matrix(prediction26, Y_test26,rocaucscore26,acc26,'Adaboost with Select K best')
    accuracy.append(acc26)
    
    X_trainres25 = X_train[b]
    X_testres25 = X_test[b]
    acc27,prediction27,Y_test27,rocaucscore27 = URL_adaboost(X_trainres25 ,X_testres25 , Y_train , Y_test, cv,"variance threshold")
    confusion_matrix(prediction27, Y_test27,rocaucscore27,acc27,'Adaboost with variance threshold')
    accuracy.append(acc27)
    
    X_trainres35 = X_train[c]
    X_testres35= X_test[c]
    acc28,prediction28,Y_test28,rocaucscore28 = URL_adaboost(X_trainres35 ,X_testres35 , Y_train , Y_test, cv,"Recursive Feature Selection")
    confusion_matrix(prediction28, Y_test28,rocaucscore28,acc28,'Adaboost with Recursive Feature Selection')
    accuracy.append(acc28)
    

    X_trainres45 = X_train[d]
    X_testres45= X_test[d]
    acc29,prediction29,Y_test29,rocaucscore29 = URL_adaboost(X_trainres45 ,X_testres45, Y_train , Y_test, cv,"Select from model")
    confusion_matrix(prediction29, Y_test29,rocaucscore29,acc29,'Adaboost with Select from model')
    accuracy.append(acc29)
    
    X_trainres55 = X_train[e]
    X_testres55= X_test[e]
    acc30,prediction30,Y_test30,rocaucscore30 = URL_adaboost(X_trainres55 ,X_testres55 , Y_train , Y_test, cv,"Sequential Feature Selection")
    result = confusion_matrix(prediction30, Y_test30,rocaucscore30,acc30,'Adaboost with Sequential Feature Selection')
    accuracy.append(acc30)
    
    return(accuracy,accuracybase,result)
    

accuracy, accuracybase,result  = splitfn(df)

accuracyallmodels = accuracy + accuracybase

print("\n")


maxidx = (accuracy.index(max(accuracy)))

listval = [ 'Log regression with Select K best' , 'Log regression with variance threshold','Log regression with RFE', 'Log regression with Select from model', 'Log regression with SFS',
            'RF with Select K best' , 'RF with variance threshold','RF with RFE', 'RF with Select from model', 'RF with SFS',
            'KNN with Select K best' , 'KNN with variance threshold','KNN with RFE', 'KNN with Select from model', 'KNN with SFS',
            'NB with Select K best' , 'NB with variance threshold','NB with RFE', 'NB with Select from model', 'NB with SFS',
            'Adaboost with Select K best' , 'Adaboost with variance threshold','Adaboost with RFE', 'Adaboost with Select from model', 'Adaboost with SFS']

listbase = ['Log regression with all data','RF with all data','KNN with all data','NB with all data','Adaboost with all data']

print("The highest accuracy is of",listval[maxidx],"model")



basemodelidx = 0

if(maxidx < 5 and maxidx >= 0):
    basemodelidx = 0
elif(maxidx < 10 and maxidx >= 5):
    basemodelidx = 1
elif(maxidx < 15 and maxidx >= 10):
    basemodelidx = 2
elif(maxidx < 20 and maxidx >= 15):
    basemodelidx = 3
elif(maxidx < 25 and maxidx >= 20):
    basemodelidx = 4
    

if(accuracy[maxidx] > accuracybase[basemodelidx]):
    print("The accuracy of model",listval[maxidx],"is more than its base model", listbase[basemodelidx])
elif(accuracy[maxidx] < accuracybase[basemodelidx]):
    print("The accuracy of model",listval[maxidx],"is less than its base model", listbase[basemodelidx])
else:
    print("The accuracy of model",listval[maxidx],"is equal to its base model", listbase[basemodelidx])


print(tabulate(result, headers=['Classifier','FP Rate','TP Rate','Precision','Recall','F_Measure','MCC','ROC_Area','Accuracy'],tablefmt='fancy_grid'))



dfresult = pd.DataFrame(result,columns=['Classifier','FP Rate','TP Rate','Precision','Recall','F_Measure','MCC','ROC_Area','Accuracy'])

dfresult.to_csv('Performancereult.csv', header=True)

