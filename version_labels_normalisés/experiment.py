import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
from fonctions import *
import numpy as np

def create_experiment(experiment_name, run_metrics, model, model_name, artifact_dir,
                      confusion_matrix_path=None, roc_auc_plot_path=None, 
                      classification_report_plot_path=None, feature_importance_path=None,
                      run_params=None, mode='Standardscaler'):
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    # Define the name of our run
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    name = experiment_name + '_' + mode + '_' + dt_string
    
    with mlflow.start_run(run_name=experiment_name + '_' + mode):
        
        if run_params is not None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        mlflow.sklearn.log_model(model, "model")
        
        if confusion_matrix_path is not None:
            mlflow.log_artifact(confusion_matrix_path, artifact_dir)
            
        if classification_report_plot_path is not None:
            mlflow.log_artifact(classification_report_plot_path, artifact_dir)
            
        if roc_auc_plot_path is not None:
            mlflow.log_artifact(roc_auc_plot_path, artifact_dir)
        
        if feature_importance_path is not None:
            mlflow.log_artifact(feature_importance_path, artifact_dir)
        
        mlflow.set_tag("tag1", experiment_name)
        
    print('Run - %s is logged to Experiment - %s' % (name + '_' + mode, experiment_name))

df = pd.read_csv('df_clean.csv')

X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
class_counts = np.bincount(y_train)
n_classes = len(class_counts)
class_weights = {i: sum(class_counts)/class_counts[i] for i in range(n_classes)}

models = [
    (LogisticRegression(max_iter=500), 'logistic_regression', {'C': [0.1, 1.0, 10], 'solver': ['lbfgs', 'liblinear']}),
    (RandomForestClassifier(), 'random_forest', {'n_estimators': [255, 260], 'max_depth': [13, 14], 'class_weight': ['balanced', class_weights]}),
    (VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=500)),
        ('rf', RandomForestClassifier())
    ]), 'voting_classifier', {'voting': ['soft', 'hard'], 'lr__C': [0.1, 1.0, 10], 'lr__solver': ['lbfgs', 'liblinear'], 'rf__n_estimators': [255, 260], 'rf__max_depth': [13, 14], 'rf__class_weight': ['balanced', class_weights]})
]

artifact_base_dir = 'output_doc'
os.makedirs(artifact_base_dir, exist_ok=True)

for model, model_name, param_grid in models:
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring='precision', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    run_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    
    artifact_dir = os.path.join(artifact_base_dir, model_name)
    os.makedirs(artifact_dir, exist_ok=True)
    
    cm_path = os.path.join(artifact_dir, 'confusion_matrix.png')
    cr_path = os.path.join(artifact_dir, 'classification_report.txt')
    roc_auc_path = os.path.join(artifact_dir, 'roc_auc_plot.png') if y_proba is not None else None
    
    create_confusion_matrix_plot(y_pred, y_test, cm_path)
    create_classification_report(y_test, y_pred, cr_path)
    if roc_auc_path:
        create_roc_auc_plot(y_test, y_proba, roc_auc_path)
    
    create_experiment(
        experiment_name=model_name,
        run_metrics=run_metrics,
        model=best_model,
        model_name=model_name,
        artifact_dir=artifact_dir,
        confusion_matrix_path=cm_path,
        roc_auc_plot_path=roc_auc_path,
        classification_report_plot_path=cr_path,
        run_params=grid_search.best_params_,
        mode='Standardscaler'
    )
    print(f"{model_name} training and logging complete.")
