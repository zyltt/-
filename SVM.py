import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def read_data(path):
    DF = pd.read_csv(path)
    return DF

def splitdata(DF):
    X = DF.drop('species',axis=1)
    y = DF['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    X_train2, X_dev, y_train2, y_dev = train_test_split(X_train, y_train, test_size = 0.15)
    return X_train2, y_train2, X_dev, y_dev,X_test, y_test

def model(X_train2, y_train2, X_dev, y_dev,X_test, y_test):
    svc_model = SVC(C=1.0, kernel='linear',class_weight='balanced')
    svc_model.fit(X_train2,y_train2)
    Accuracy = svc_model.score (X_train2,y_train2)
    predictions = svc_model.predict(X_dev)
    return Accuracy, predictions

if __name__=="__main__":
    PATH = "C:\\Users\\zou\\Desktop\\ML\\svm_data\\iris.csv"
    
    IRIS = read_data(PATH)
    X_train2, y_train2, X_dev, y_dev,X_test, y_test = splitdata(IRIS)
    clf = SVC(gamma='auto', C = 1)
    clf.fit(X_train2,y_train2)
    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2, cv = 5, scoring='accuracy')
    grid.fit(X_train2,y_train2)
    grid_predictions = grid.predict(X_dev)
    reg_predictions = clf.predict(X_dev)
    print("精确度得分为：")
    print(accuracy_score(y_dev,reg_predictions ))
    print("分类报告为：")
    print(classification_report(y_dev,reg_predictions ))
    print("所选择的最好的变量集为：")
    print()
    print(grid.best_params_)
    print()
    print("其精确度得分为：")
    print(accuracy_score(y_dev,grid_predictions))
    print("其grid得分为：")
    print()
    print(classification_report(y_dev,grid_predictions))
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    y_true, y_pred = y_test, grid.predict(X_test)
    print("其在测试集的精确度为：")
    print(accuracy_score(y_true,y_pred))
    print("相关信息表格：")
    print(classification_report(y_true, y_pred))