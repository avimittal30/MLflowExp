import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, required=False, default=0.01)
parser.add_argument("--max_depth", type=int, required=False, default=6)
args = parser.parse_args()


def eval_metrics(actual, pred):
    acc=accuracy_score(actual,pred)
    precision=precision_score(y_test,pred)    
    recall=recall_score(y_test,pred)
    return acc, precision,recall

if __name__=="__main__":
    df=pd.read_csv('C:\\MLOps\\MLflow_demo\\Churn_Modelling.csv', header="infer" )
    df.drop(["Surname"], axis=1, inplace =True)
    df['Gender']=df['Gender'].map({"Male":1, "Female":0})
    df=pd.get_dummies(df,"Geography",drop_first=True)
    X_train,X_test, y_train, y_test=train_test_split(df.drop(['Exited'], axis=1), df['Exited'],test_size=0.2, random_state=20)
    mlflow.set_tracking_uri(uri="")
    print(f"The tracking URI is {mlflow.get_tracking_uri()}")
    
    # Tags are at experiment level and not at run level. They will help you identify different runs.
    # exp_id=mlflow.create_experiment(name="exp_create_exp", tags={"version":'v1', "priority":'p1'})
    exp=mlflow.set_experiment(experiment_name="exp_create_exp")
    # create_experiment object returns an exp_id unlike set_experiment that returns an object

# in start_run, you can also specify run_id and overwrite wthe run with the artefacts of new run
    mlflow.start_run(experiment_id=exp.experiment_id)
    xgb_obj=xgb.XGBClassifier(learning_rate=args.learning_rate,max_depth=args.max_depth)
    xgb_obj.fit(X_train,y_train)
    (accuracy, precision, recall)= eval_metrics(y_test, xgb_obj.predict(X_test))
    print(f"Precision:{precision}")
    print(f"Recall:{recall}")
    print(f"Accuracy:{accuracy}")
    
    # mlflow.log_param("learning_rate", args.learning_rate)
    # mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_params({"learning_rate": args.learning_rate,"max_depth": args.max_depth })

    mlflow.log_metrics({"precision": precision, "recall": recall, "accuracy": accuracy })
    mlflow.sklearn.log_model(xgb_obj, "mymodel")
    
    run=mlflow.active_run()
    print(f"The run id is :{run.info.run_id}")
    print(f"The run name is :{run.info.run_name}")
    mlflow.end_run()
# location of MLruns folder can be set using set_tracking_uri()
# It can also be a remote path like "https://my-tracking-server:5000"        
#  This argument is also available in create_experiment function with the name artifact_location         
