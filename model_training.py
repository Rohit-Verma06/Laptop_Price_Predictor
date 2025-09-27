import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import numpy as np
url="https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/refs/heads/main/laptop_data.csv"
headers={"User-Agent":"Mozilla"}
req=requests.get(url,headers=headers)
data=StringIO(req.text)
data=pd.read_csv(data)
data=data.drop(columns=["Unnamed: 0"],axis=1)
data["Ram"]=data["Ram"].str.replace("GB",repl="").astype(int)
data["Weight"]=data["Weight"].str.replace("kg","").astype(float)
# sns.barplot(x=data["Company"],y=data["Price"])
# plt.xticks(rotation="vertical")
# data["Inches"].plot(kind='kde')
# plt.show()

data["Touch_Screen"]=data["ScreenResolution"].apply(lambda x :1 if "Touchscreen" in x else 0)
data["IPS_Panel"]=data["ScreenResolution"].apply(lambda x :1 if "IPS" in x else 0)
# sns.barplot(x=data["IPS_Panel"],y=data["Price"])
# plt.show()
# print(data["ScreenResolution"].sample(20))
temp=data["ScreenResolution"].str.split("x")
data["x_resolution"]=temp.apply(lambda x : x[0][len(x[0])- 4 : len(x[0])]).astype(int)
data["y_resolution"]=temp.apply(lambda x : x[1]).astype(int)
# sns.barplot(x=data["y_resolution"],y=data["Price"])
# plt.show()
data["PPI"]=((np.sqrt(data["x_resolution"]**2+data["y_resolution"]**2))/data["Inches"]).astype(float)
data=data.drop(columns=["x_resolution","y_resolution","Inches","ScreenResolution"],axis=1)
temp2=data["Cpu"].apply(lambda x : x.split()[0:3])


# data["i3"]=temp2.apply(lambda x : 1 if x[2]=="i3" else 0)
# data["i5"]=temp2.apply(lambda x : 1 if x[2]=="i5" else 0)
# data["i7"]=temp2.apply(lambda x : 1 if x[2]=="i7" else 0)
# data["AMD"]=temp2.apply(lambda x : 1 if x[0]=="AMD" else 0)
# data["Other_Intel"]=temp2.apply(lambda x : 1 if (x[2]!="i3" and x[2]!="i5" and x[2]!="i7" and x[0]!="AMD") else 0)


def transform(list):
    str= " ".join(list)
    if(str=="Intel Core i3"):
        return "Intel Core i3"
    elif(str=="Intel Core i5"):
        return "Intel Core i5"
    elif(str=="Intel Core i7"):
        return "Intel Core i7"
    elif(list[0]=="AMD"):
        return "AMD"
    else :
        return "Others_Intel"




data["Cpu_brand"]=temp2.apply(lambda x: transform(x))

data=data.drop(columns=["Cpu"],axis=1)

data['Memory'] = data['Memory'].astype(str).replace('\.0', '', regex=True)
data["Memory"] = data["Memory"].str.replace('GB', '')
data["Memory"] = data["Memory"].str.replace('TB', '000')

new=data["Memory"].str.split("+",n=1,expand=True)
# print(new.sample(50))

data["first"]= new[0]
data["first"]=data["first"].str.strip()   # removes extra whitespace

data["second"]= new[1]
data["second"]=data["second"].fillna("0") 
data["second"]=data["second"].str.strip()

data["first_SSD"]=data["first"].apply(lambda x : 1 if "SSD" in x else 0)
data["first_HDD"]=data["first"].apply(lambda x : 1 if "HDD" in x else 0)
data["first_Hybrid"]=data["first"].apply(lambda x : 1 if "Hybrid" in x else 0)
data["first_flashstorage"]=data["first"].apply(lambda x : 1 if "Flash Storage" in x else 0)

data["first"]=data["first"].str.replace(r'\D', '', regex=True)

data["second_SSD"]=data["second"].apply(lambda x : 1 if "SSD" in x else 0)
data["second_HDD"]=data["second"].apply(lambda x : 1 if "HDD" in x else 0)
data["second_Hybrid"]=data["second"].apply(lambda x : 1 if "Hybrid" in x else 0)
data["second_flashstorage"]=data["second"].apply(lambda x : 1 if "Flash Storage" in x else 0)

data["second"]=data["second"].str.replace(r"\D",'',regex=True)

data["first"] = pd.to_numeric(data["first"], errors='coerce').fillna(0)
data["second"] = pd.to_numeric(data["second"], errors='coerce').fillna(0)

data["SSD"]=data["first"]*data["first_SSD"]+ data["second"]*data["second_SSD"]
data["HDD"]=data["first"]*data["first_HDD"] + data["second"]*data["second_HDD"]
data["Hybrid"]=data["first"]*data["first_Hybrid"]+ data["second"]*data["second_Hybrid"]
data["FlashStorage"]=data["first"]*data["first_flashstorage"] + data["second"]*data["second_flashstorage"]

data=data.drop(columns=['first', 'second', 'first_HDD', "first_SSD", 'first_Hybrid',
       'first_flashstorage', 'second_HDD', 'second_SSD', 'second_Hybrid',
       'second_flashstorage'], axis=1)

data=data.drop(columns=["Memory"],axis=1)
# print(data.head())
# print(data.corr(numeric_only=True)["Price"])
data=data.drop(columns=["FlashStorage","Hybrid"],axis=1)
# print(data.corr(numeric_only=True)["Price"])
# print(data["Gpu"].value_counts())

data["Gpu_brand"]=data["Gpu"].apply(lambda x  : x.split()[0])

# print(data.shape)
data=data[data["Gpu_brand"]!="ARM"]  # only 1 for ARM
# print(data.shape)
data=data.drop(columns=["Gpu"],axis=1)
# print(data.shape)
# print(data["OpSys"].value_counts())
def categorize(x):
    if(x=="Windows 7" or x=="Windows 10" or x=="Windows 10 S"):
        return "Windows"
    elif(x=="macOS" or x=="Mac OS X"):
        return "MacOs"
    else:
        return "Others/No Os/Linux"
    
data["Operating_Sys"]=data["OpSys"].apply(lambda x : categorize(x))
data=data.drop(columns=["OpSys"],axis=1)
# data["Price"].plot(kind="kde")
# plt.show()
X=data.drop(columns=["Price"],axis=1)
y=np.log(data["Price"])   # log transformation for kind of getting a normalized shape 


# import sklearn
# from sklearn.model_selection import train_test_split

# Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.15,random_state=2)

# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import r2_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# from sklearn.linear_model import LinearRegression,Ridge,Lasso
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
# from xgboost import XGBRegressor


# step1=ColumnTransformer(transformers=[("col_tnf",OneHotEncoder(handle_unknown="ignore",sparse_output=False,drop="first"),[0,1,7,10,11])],remainder="passthrough")
# step2=XGBRegressor(n_estimators=946,learning_rate=0.0521929165991926,max_depth=6,subsample= 0.835112564266041,colsample_bytree= 0.5284213835567312,gamma=0.002737671295410027)

# pipe=Pipeline([("Step 1",step1),("Step 2",step2)])
# pipe.fit(Xtrain,ytrain)   # 0.8907031545887689.

# ypred=pipe.predict(Xtest)
# print(r2_score(ytest,ypred))

# from sklearn.metrics import r2_score
# lin_reg=LinearRegression()
# ridge=Ridge()
# knn=KNeighborsRegressor()
# dec_tree=DecisionTreeRegressor()
# svr=SVR()
# random_forest=RandomForestRegressor()
# Gb_reg=GradientBoostingRegressor()
# ada_boost=AdaBoostRegressor()
# xgb=XGBRegressor()
# from sklearn.model_selection import cross_val_score


# import optuna 
# def objective(trial):

#     classifier_name = trial.suggest_categorical("classifier", [
#         "lin_reg", "ridge", "knn", "dec_tree", "svr", 
#         "random_forest", "gb_reg", "ada_boost", "xgb"
#     ])


#     if classifier_name == "lin_reg":
#         model = LinearRegression()

#     elif classifier_name == "ridge":
#         alpha = trial.suggest_float("alpha_ridge", 1e-4, 1e2, log=True)
#         max_iter = trial.suggest_int("max_iter_ridge", 100, 1000)
#         model = Ridge(alpha=alpha, max_iter=max_iter)

#     elif classifier_name == "knn":
#         n_neighbors = trial.suggest_int("n_neighbors", 2, 20)
#         weights = trial.suggest_categorical("weights", ["uniform", "distance"])
#         algorithm = trial.suggest_categorical("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
#         model = KNeighborsRegressor(
#             n_neighbors=n_neighbors, 
#             weights=weights, 
#             algorithm=algorithm, 
#         )

#     elif classifier_name == "dec_tree":
#         max_depth = trial.suggest_int("max_depth_dt", 3, 30)
#         min_samples_split = trial.suggest_int("min_samples_split_dt", 2, 20)
#         min_samples_leaf = trial.suggest_int("min_samples_leaf_dt", 1, 20)
#         criterion = trial.suggest_categorical("criterion_dt", ["squared_error", "friedman_mse", "absolute_error"])
#         model = DecisionTreeRegressor(
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             criterion=criterion,
#             random_state=42
#         )

#     elif classifier_name == "svr":
#         C = trial.suggest_float("C_svr", 1e-4, 1e4, log=True)
#         kernel = trial.suggest_categorical("kernel_svr", ['linear', 'poly', 'rbf', 'sigmoid'])
#         gamma = trial.suggest_categorical("gamma_svr", ['scale', 'auto'])
#         model = SVR(C=C, kernel=kernel, gamma=gamma)
        
#     elif classifier_name == "random_forest":
#         n_estimators = trial.suggest_int("n_estimators_rf", 100, 1000)
#         max_depth = trial.suggest_int("max_depth_rf", 5, 50)
#         min_samples_split = trial.suggest_int("min_samples_split_rf", 2, 20)
#         min_samples_leaf = trial.suggest_int("min_samples_leaf_rf", 1, 20)
#         max_features = trial.suggest_categorical("max_features_rf", ['sqrt', 'log2', None])
#         model = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             random_state=42
#         )

#     elif classifier_name == "gb_reg":
#         n_estimators = trial.suggest_int("n_estimators_gb", 100, 1000)
#         learning_rate = trial.suggest_float("learning_rate_gb", 1e-3, 0.5, log=True)
#         max_depth = trial.suggest_int("max_depth_gb", 3, 10)
#         subsample = trial.suggest_float("subsample_gb", 0.5, 1.0)
#         model = GradientBoostingRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             max_depth=max_depth,
#             subsample=subsample,
#             random_state=42
#         )

#     elif classifier_name == "ada_boost":
#         n_estimators = trial.suggest_int("n_estimators_ada", 50, 500)
#         learning_rate = trial.suggest_float("learning_rate_ada", 1e-3, 1.0, log=True)
#         loss = trial.suggest_categorical("loss_ada", ['linear', 'square', 'exponential'])
#         model = AdaBoostRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             loss=loss,
#             random_state=42
#         )

#     elif classifier_name == "xgb":
#         n_estimators = trial.suggest_int("n_estimators_xgb", 100, 1000)
#         learning_rate = trial.suggest_float("learning_rate_xgb", 1e-3, 0.5, log=True)
#         max_depth = trial.suggest_int("max_depth_xgb", 3, 10)
#         subsample = trial.suggest_float("subsample_xgb", 0.5, 1.0)
#         colsample_bytree = trial.suggest_float("colsample_bytree_xgb", 0.5, 1.0)
#         gamma = trial.suggest_float("gamma_xgb", 0, 5)
#         model = XGBRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             max_depth=max_depth,
#             subsample=subsample,
#             colsample_bytree=colsample_bytree,
#             gamma=gamma,
#             random_state=42
#         )
    
#     step1=ColumnTransformer(transformers=[("col_tnf",OneHotEncoder(handle_unknown="ignore",sparse_output=False,drop="first"),[0,1,7,10,11])],remainder="passthrough")
#     pipe=Pipeline([("Step 1",step1),("Step 2",model)])
#     score=cross_val_score(pipe,Xtrain,ytrain,scoring="r2",cv=5,n_jobs=-1).mean()
#     return score 

    
# study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler())
# study.optimize(objective,n_trials=100)

# print("The best score is  : ",study.best_value)
# print("The best model is : ",study.best_params)


# import pickle 
# pickle.dump(data,open("data.pkl","wb"))
# pickle.dump(pipe,open("pipe.pkl","wb"))
