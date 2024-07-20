from data_preprocessing import load_data, preprocess_data, split_data, scale_data
from model_training import train_logistic_regression, train_knn, train_svm, train_decision_tree, train_random_forest
from model_evaluation import evaluate_model

# Load and preprocess data
file_path = './mobile_price_range_data.csv'
data = load_data(file_path)
x, y = preprocess_data(data)
x_train, x_test, y_train, y_test = split_data(x, y)
x_train, x_test = scale_data(x_train, x_test)

# Train models
logistic_regression_model = train_logistic_regression(x_train, y_train)
knn_model = train_knn(x_train, y_train)
svm_model = train_svm(x_train, y_train)
decision_tree_model = train_decision_tree(x_train, y_train)
random_forest_model = train_random_forest(x_train, y_train)

# Evaluate models
logistic_regression_accuracy = evaluate_model(logistic_regression_model, x_test, y_test, 'Logistic Regression')
knn_accuracy = evaluate_model(knn_model, x_test, y_test, 'KNN')
svm_accuracy = evaluate_model(svm_model, x_test, y_test, 'SVM')
decision_tree_accuracy = evaluate_model(decision_tree_model, x_test, y_test, 'Decision Tree')
random_forest_accuracy = evaluate_model(random_forest_model, x_test, y_test, 'Random Forest')

# Report best model
best_accuracy = max(logistic_regression_accuracy, knn_accuracy, svm_accuracy, decision_tree_accuracy, random_forest_accuracy)
print(f'The best model accuracy is: {best_accuracy}')
