import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc




def create_roc_auc_plot(y_test, y_proba, roc_auc_plot_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(roc_auc_plot_path)
    plt.close()

def create_confusion_matrix_plot(y_pred, y_test, cm_path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(cm_path)
    plt.close()


def create_classification_report(y_test, y_pred, cr_path):
    report = classification_report(y_test, y_pred)
    with open(cr_path, 'w') as f:
        f.write(report)

def feature_importance_rf(model,x_col,model_name):
    
    importance_data = pd.DataFrame()
    importance_data['features'] = x_col
    importance_data['importance']= model.feature_importances_

    sns.barplot(data=importance_data[:15], y='features',x='importance')
    plt.savefig(f'output_doc/feature_importance_{model_name}.png')
    plt.show()
