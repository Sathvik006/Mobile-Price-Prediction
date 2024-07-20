from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f'{model_name} Classification Report:\n', classification_report(y_test, y_pred))
    print(f'{model_name} Accuracy Score: ', accuracy_score(y_test, y_pred))
    sns.heatmap(cm, annot=True, fmt='g', cmap='BuPu')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return accuracy_score(y_test, y_pred)
