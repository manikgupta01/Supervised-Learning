import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve, validation_curve
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

RANDOM_STATE = 21

# load the dataset
def load_dataset(full_path):
    wine = pd.read_csv(full_path, delimiter=";")

    wine['quality'] = wine['quality'].replace([3,4,5,6,7,8,9],['Low','Low','Medium','Medium','High','High','High'])
    X = wine.iloc[:, :-1].values
    y = wine.iloc[:, -1].values

    X = preprocessing.scale(X)

    y = LabelEncoder().fit_transform(y)

    return X, y


def classification_metrics(Y_pred, Y_true):
    accuracy_lr = accuracy_score(Y_true, Y_pred)
    return accuracy_lr


def display_metrics(classifierName,Y_pred,Y_true):
    print("______________________________________________")
    print(("Classifier: "+classifierName))
    acc = classification_metrics(Y_pred,Y_true)
    print(("Accuracy: "+str(acc)))
    print("______________________________________________")
    print("")
    return str(acc)

def plot_validation_curve(estimator, title, X, y, param_name, param_range,
                          ylim=None, cv=None, n_jobs=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    train_scores, test_scores = validation_curve(estimator, X, y,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=cv, scoring="accuracy",
                                                 n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training Accuracy",
         color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation Accuracy",
         color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig(str(title) + ".png")


def plot_validation_log_curve(estimator, title, X, y, param_name, param_range,
                          ylim=None, cv=None):

    train_scores, test_scores = validation_curve(estimator, X, y,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=cv, scoring="accuracy")
    plt.figure()
    plt.semilogx(param_range, np.mean(train_scores, axis=1), label="Training Accuracy")
    plt.semilogx(param_range, np.mean(test_scores, axis=1), label="Cross-validation Accuracy")
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(str(title) + ".png")

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(str(title) + ".png")

def plot_learning_curves_NN(train_losses, train_accuracies, valid_accuracies):
    # Reference : https://matplotlib.org/tutorials/introductory/pyplot.html
    # Reference: https://stackoverflow.com/questions/4805048/how-to-get-different-colored-lines-for-different-plots-in-a-single-figure
    # Reference: https://github.com/ast0414/CSE6250BDH-LAB-DL/blob/master/3_RNN.ipynb
    plt.figure()
    plt.grid()
    plt.plot(np.arange(len(train_losses)), train_losses, label='Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Loss Curve')
    plt.legend(loc="best")
    plt.savefig('MLP_Loss_Curve1.png')

    plt.figure()
    plt.grid()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.title('Accuracy Curve')
    plt.legend(loc="best")
    plt.savefig('MLP_Accuracy_Curve1.png')

def evaluate(estimator, X, y, cv, n_jobs=1):
    metrics = []
    metrics.append(('Accuracy', 'accuracy'))

    for label, name in metrics:
        results = cross_val_score(estimator, X, y, cv=cv, scoring=name, n_jobs=n_jobs)

    return np.mean(results)

def main():

    full_path = 'data/winequality-white.csv'
    X, y = load_dataset(full_path)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)

    accuracy = np.zeros(6)
    train_time = np.zeros(6)
    query_time = np.zeros(6)

    # Decision Tree
    title_vc = "Validation Curves1 (Decision Tree)"
    title_lc = "Learning Curves1 (Decision Tree)"

    # Check which criterion to use - Gini or Entropy
    decisionTreeEnt = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=RANDOM_STATE)
    decisionTreeEnt.fit(X_train, Y_train)
    Y_pred = decisionTreeEnt.predict(X_test)
    temp = display_metrics("Decision Tree 1st Dataset - Entropy",Y_pred,Y_test)

    decisionTreeGini = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=RANDOM_STATE)
    decisionTreeGini.fit(X_train, Y_train)
    Y_pred = decisionTreeGini.predict(X_test)
    temp = display_metrics("Decision Tree 1st Dataset - Gini",Y_pred,Y_test)

    # Check the validation curve using max_depth parameter for Pruning and avoid overfitting.
    # The accuracy for the Entropy is better hence it is choosen as split criteria
    param_name = 'max_depth'
    param_range = np.arange(1, 23)
    plot_validation_curve(decisionTreeEnt, title_vc,
                       X_train, Y_train, param_name, param_range, ylim=(0.0, 1.01), cv=cv,
                       n_jobs=4)

    decisionTree = DecisionTreeClassifier(criterion='entropy', max_depth=18, random_state=RANDOM_STATE)
    t_bef = time.time()
    decisionTree.fit(X_train, Y_train)
    t_aft = time.time()
    train_time[0] = t_aft - t_bef

    t_bef = time.time()
    Y_pred = decisionTree.predict(X_test)
    t_aft = time.time()
    dt_accuracy = display_metrics("Decision Tree 1st Dataset - After tuning",Y_pred,Y_test)
    query_time[0] = t_aft - t_bef
    accuracy[0] = dt_accuracy

    plot_learning_curve(decisionTree, title_lc, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=4)


    # MLP
    title_vc = "Validation Curves1 (MLP-NN)"
    title_lc = "Learning Curves1 (MLP-NN)"

    mlpClass = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=RANDOM_STATE, max_iter=5000)
    mlpClass.fit(X_train, Y_train)
    Y_pred = mlpClass.predict(X_test)
    temp = display_metrics("MLP NN 1st Dataset - Before tuning",Y_pred,Y_test)

    param_name = 'alpha'
    param_range = np.logspace(-3, 3, 7)
    plot_validation_log_curve(mlpClass, title_vc,
                       X_train, Y_train, param_name, param_range, ylim=(0.0, 1.01), cv=cv)

    mlpClass_lc = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=RANDOM_STATE, max_iter=5000, alpha=0.1)
    t_bef = time.time()
    mlpClass_lc.fit(X_train, Y_train)
    t_aft = time.time()
    train_time[1] = t_aft - t_bef

    plot_learning_curve(mlpClass_lc, title_lc, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=4)

    mlpClass_lc = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=RANDOM_STATE, max_iter=5000, warm_start=True, alpha=0.1)
    num_epochs = 1000
    train_losses, train_accuracies, valid_accuracies = np.empty(num_epochs), np.empty(num_epochs), np.empty(num_epochs)
    # Split training set into training and validation
    X_train_NN, X_val, Y_train_NN, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=RANDOM_STATE)
    for i in range(num_epochs):
        mlpClass_lc.fit(X_train_NN, Y_train_NN)
        train_losses[i] = mlpClass_lc.loss_
        train_accuracies[i] = accuracy_score(Y_train_NN, mlpClass_lc.predict(X_train_NN))
        valid_accuracies[i] = accuracy_score(Y_val, mlpClass_lc.predict(X_val))

    t_bef = time.time()
    Y_pred = mlpClass_lc.predict(X_test)
    t_aft = time.time()
    mlp_accuracy = display_metrics("MLP NN 1st Dataset - After tuning", Y_pred,Y_test)
    query_time[1] = t_aft - t_bef
    accuracy[1] = mlp_accuracy

    plot_learning_curves_NN(train_losses, train_accuracies, valid_accuracies)

    # Boosting
    title_vc = "Validation Curves1 (Adaboost)"
    title_lc = "Learning Curves1 (Adaboost)"

    decisionTree_boost = DecisionTreeClassifier(max_depth=18)
    boostedClass = AdaBoostClassifier(base_estimator=decisionTree_boost, random_state=RANDOM_STATE)
    boostedClass.fit(X_train, Y_train)
    Y_pred = boostedClass.predict(X_test)
    temp = display_metrics("Adaboost 1st Dataset - Before tuning",Y_pred,Y_test)

    # Tried with multiple num_learners = 500, 1000, 2000, Decision tree already pruned for max_depth
    num_learners = 1000
    learning_rate = 1
    boostedClass_lc = AdaBoostClassifier(base_estimator=decisionTree_boost,
            n_estimators=num_learners, learning_rate=learning_rate, random_state=RANDOM_STATE)

    t_bef = time.time()
    boostedClass_lc.fit(X_train, Y_train)
    t_aft = time.time()
    train_time[2] = t_aft - t_bef

    t_bef = time.time()
    Y_pred = boostedClass_lc.predict(X_test)
    t_aft = time.time()
    ada_accuracy = display_metrics("Adaboost 1st Dataset - After tuning",Y_pred,Y_test)
    query_time[2] = t_aft - t_bef
    accuracy[2] = ada_accuracy

    plot_learning_curve(boostedClass_lc, title_lc, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=4)


    #SVM - Linear
    title_vc = "Validation Curves1 (SVM - Linear)"
    title_lc = "Learning Curves1 (SVM - Linear)"

    svcClassLinear = SVC(kernel='linear', random_state=RANDOM_STATE)
    svcClassLinear.fit(X_train, Y_train)
    Y_pred = svcClassLinear.predict(X_test)
    temp = display_metrics("SVM (Linear Kernel) 1st Dataset - Before tuning",Y_pred,Y_test)

    # param_name = 'C'
    # param_range = np.logspace(-3, 3, 7)
    # plot_validation_log_curve(svcClassLinear, title_vc,
    #                    X_train, Y_train, param_name, param_range, ylim=(0.0, 1.01), cv=cv)

    svcClassLinear_lc = SVC(C=1.2, kernel='linear', random_state=RANDOM_STATE)
    t_bef = time.time()
    svcClassLinear_lc.fit(X_train, Y_train)
    t_aft = time.time()
    train_time[3] = t_aft - t_bef

    t_bef = time.time()
    Y_pred = svcClassLinear_lc.predict(X_test)
    t_aft = time.time()
    svc_linear_accuracy = display_metrics("SVM (Linear Kernel) 1st Dataset - After tuning",Y_pred,Y_test)
    query_time[3] = t_aft - t_bef
    accuracy[3] = svc_linear_accuracy

    plot_learning_curve(svcClassLinear_lc, title_lc, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=4)


    #SVM - RBF
    title_vc = "Validation Curves1 (SVM - RBF)"
    title_lc = "Learning Curves1 (SVM - RBF)"

    svcClassRBF = SVC(kernel='rbf', random_state=RANDOM_STATE)
    svcClassRBF.fit(X_train, Y_train)
    Y_pred = svcClassRBF.predict(X_test)
    temp = display_metrics("SVM (RBF) 1st Dataset - Before tuning",Y_pred,Y_test)

    # param_name = 'C'
    # param_range = np.logspace(-3, 3, 7)
    # plot_validation_log_curve(svcClassRBF, title_vc,
    #                    X_train, Y_train, param_name, param_range, ylim=(0.0, 1.01), cv=cv)


    svcClassRBF_lc = SVC(C=1.2, kernel='rbf', random_state=RANDOM_STATE)
    t_bef = time.time()
    svcClassRBF_lc.fit(X_train, Y_train)
    t_aft = time.time()
    train_time[4] = t_aft - t_bef

    t_bef = time.time()
    Y_pred = svcClassRBF_lc.predict(X_test)
    t_aft = time.time()
    svc_rbf_accuracy = display_metrics("SVM (RBF) 1st Dataset - After tuning",Y_pred,Y_test)
    query_time[4] = t_aft - t_bef
    accuracy[4] = svc_rbf_accuracy

    plot_learning_curve(svcClassRBF_lc, title_lc, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=4)


    # KNN
    title_vc = "Validation Curves1 (KNN)"
    title_lc = "Learning Curves1 (KNN)"

    knnClass = KNeighborsClassifier()
    knnClass.fit(X_train, Y_train)
    Y_pred = knnClass.predict(X_test)
    temp = display_metrics("KNN 1st Dataset - Before tuning",Y_pred,Y_test)

    param_name = 'n_neighbors'
    param_range = np.arange(1, 11, 2)
    plot_validation_curve(knnClass, title_vc,
                        X_train, Y_train, param_name, param_range, ylim=(0.0, 1.01), cv=cv,
                        n_jobs=4)


    knnClass_lc = KNeighborsClassifier(n_neighbors=5)
    t_bef = time.time()
    knnClass_lc.fit(X_train, Y_train)
    t_aft = time.time()
    train_time[5] = t_aft - t_bef

    t_bef = time.time()
    Y_pred = knnClass_lc.predict(X_test)
    t_aft = time.time()
    knn_accuracy = display_metrics("KNN 1st Dataset - After tuning",Y_pred,Y_test)
    query_time[5] = t_aft - t_bef
    accuracy[5] = knn_accuracy

    plot_learning_curve(knnClass_lc, title_lc, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=4)

    classifier = ['Decision Tree', 'MLP NN', 'Adaboost', 'SVC-Linear', 'SVC-RFB', 'KNN']
    np_classifier = np.array(classifier)

    # Accuracy score of different classifiers
    plt.figure()
    plt.barh(np_classifier, accuracy, align = 'center')
    plt.title('Classifier Accuracy')
    plt.ylabel('Classifier Name')
    plt.xlabel('Accuracy')
    plt.savefig('Classifiers_Accuracy1.png', bbox_inches = "tight")

    # Train time different classifiers
    plt.figure()
    plt.barh(np_classifier, train_time, align = 'center')
    plt.title('Classifier Train Time')
    plt.ylabel('Classifier Name')
    plt.xlabel('Time (seconds)')
    plt.savefig('Classifiers_Traintime1.png', bbox_inches = "tight")

    # Query time different classifiers
    plt.figure()
    plt.barh(np_classifier, query_time, align = 'center')
    plt.title('Classifier Query Time')
    plt.ylabel('Classifier Name')
    plt.xlabel('Time (seconds)')
    plt.savefig('Classifiers_Querytime1.png', bbox_inches = "tight")

if __name__ == "__main__":
    main()
