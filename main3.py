import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import seaborn as sns

class Classification:

    def __init__(self, clf_opt='rf', impute_opt='mean'):
        self.clf_opt = clf_opt
        self.impute_opt = impute_opt

    def standard_scaling(self, df):
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(data=scaler.fit_transform(df), columns=df.columns)
        return X_scaled

    def mean_imputer(self,X):
        mean_imputer = SimpleImputer(strategy='mean')
        X_mean_imp = pd.DataFrame(
            data=mean_imputer.fit_transform(X),
            columns=X.columns   ) 
        return X_mean_imp
    
    def mode_imputer(self,X):
        mode_imputer = SimpleImputer(strategy='most_frequent')
        X_mode_imp = pd.DataFrame(
            data=mode_imputer.fit_transform(X),
            columns=X.columns   ) 
        return X_mode_imp
    
    def median_imputer(self,X):
        median_imputer = SimpleImputer(strategy='median')
        X_median_imp = pd.DataFrame(
            data=median_imputer.fit_transform(X),
            columns=X.columns   ) 
        return X_median_imp
    
    def knn_imputer(self,X):
        knn_imputer = KNNImputer(n_neighbors=2)
        X_knn_imp = pd.DataFrame(
            data=knn_imputer.fit_transform(X),
            columns=X.columns) 
        return X_knn_imp

    def oversampling(self, X, y):
        smote = ADASYN(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def impute_data(self, X):
        imputer = self.get_imputer()
        return pd.DataFrame(data=imputer.fit_transform(X), columns=X.columns)

    def get_imputer(self):
        imputation_methods = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'knn': KNNImputer(n_neighbors=3),
            'mode': SimpleImputer(strategy='most_frequent')
        }

        selected_imputer = imputation_methods.get(self.impute_opt)
        if selected_imputer is None:
            raise ValueError('Invalid imputation option')
        return selected_imputer

    def Pipeline(self):
        if self.clf_opt == 'rf':
            print('\nRandom Forest Classifier\n')
            clf = RandomForestClassifier(random_state=42)
            param_grid = {
                'clf__criterion': ['gini', 'entropy'],
                'clf__n_estimators': [10, 20, 50, 60],
                'clf__max_depth': [None, 10, 20, 30],
                'clf__min_samples_split': [1, 2, 3, 4],
                'clf__min_samples_leaf': [1, 2, 3, 4],
            }
        elif self.clf_opt == 'dt':
            print('\nDecision Tree Classifier\n')
            clf = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'clf__criterion': ['gini', 'entropy'],
                'clf__max_features': ['auto', 'sqrt', 'log2'],
                'clf__max_depth': [10, 40, 45, 60, 70, 80],
                'clf__ccp_alpha': [0.009, 0.01, 0.05, 0.1, 0.2],
            }
            #linear support vector classifier
        elif self.clf_opt == 'lsv':
            print('\nLinear Support Vector Classifier\n')
            clf = svm.LinearSVC(random_state=42)
            param_grid = {
                'clf__C': [0.1, 0.5, 5, 10],
                'clf__loss': ['hinge', 'squared_hinge'],
                'clf__penalty': ['l1', 'l2'],
                'clf__max_iter': [1000, 2000, 3000],
            }
               
        elif self.clf_opt == 'svm':
            print('\nSVM Classifier\n')
            clf = svm.SVC(random_state=42)
            param_grid = {
                'clf__C': [0.1, 0.5, 5, 10],
                'clf__gamma': [0.1, 0.5, 1, 10],
                'clf__kernel': ['rbf', 'poly', 'sigmoid'],
                'clf__class_weight': [None, 'balanced'],
            }
            

        elif self.clf_opt == 'ab':
            print('\nAdaBoost Classifier\n')
            be1 = RandomForestClassifier(max_depth=50, n_estimators=100)
            be2 = LogisticRegression(solver='liblinear', class_weight='balanced')
            be3 = DecisionTreeClassifier(max_depth=50)
            clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100)
            param_grid = {
                'clf__base_estimator': [be1, be2, be3],
                'clf__n_estimators': [50, 100, 150],
                'clf__learning_rate': [0.01, 0.1, 1],
                'clf__random_state': [0, 10]
            }
        elif self.clf_opt == 'lr':
            print('\nLogistic Regression Classifier\n')
            clf = LogisticRegression()
            param_grid = {
                'clf__penalty': ['l1', 'l2', 'elasticnet'],
                'clf__C': [0.001, 0.01, 0.1, 1, 10],
                'clf__fit_intercept': [True, False],
                'clf__max_iter': [100, 200, 300],
                'clf__solver': ['liblinear',  'saga'],
                'clf__class_weight': [None, 'balanced'],  
            }

        else:
            raise ValueError('Invalid classifier option')

        imputer = self.get_imputer()
        pipeline = Pipeline([
            ('imputer', imputer),
            ('clf', clf)
        ])
        return pipeline, imputer, param_grid

    def analysis(self, X, y):
        missing_values_per_column = X.isnull().sum()
        missing_values_per_column = missing_values_per_column[missing_values_per_column > 0]
        total = missing_values_per_column[missing_values_per_column > 0].sum()
        y_counts = y.value_counts()
        corr_mat= X.corr()

        print("## Shape of the Data ##")
        print()
        print("Shape of Training Data", X.shape)
        print("Shape of Training_Data_Targets", y.shape)
        print()
        print("## Number of Missing Values in Data ##")
        print(missing_values_per_column)
        print()
        print("TOTAL:", total)
        print()
        print("## Values in Each Class ##")
        print(y_counts)
        # Plot the correlation matrix heatmap
        plt.figure(figsize=(6, 6))
        sns.heatmap(np.array(corr_mat), annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix Heatmap")
        plt.savefig("correlation_matrix_heatmap.jpeg")
  

    def classification(self):
        X = pd.read_csv("training_data.csv")
        Y = pd.read_csv("training_data_targets.csv", header=None)
        self.analysis(X, Y)

        X_imputed = self.impute_data(X)
        X_scaled = self.standard_scaling(X_imputed)
       #Feature selection By correlation matrix
        corr_matrix = X_scaled.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
        X_scaled_selected = X_scaled.drop(columns=to_drop)
        print("## Shape of the Data after Feature Selection ##")
        print()
        print("Shape of Training Data", X_scaled_selected.shape)
        # selector = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=40)
        # X_scaled_selected = selector.fit_transform(X_scaled, Y.values.ravel())

        # Split the data into training and testing sets after feature selection
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_selected, Y, test_size=0.2, random_state=42, stratify=Y)
        # Use oversampling on the training set only
        X_resampled, y_resampled = self.oversampling(X_train, y_train)
        pipeline, _, param_grid = self.Pipeline()
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1_macro', cv=10)
        grid_search.fit(X_resampled, y_resampled.values.ravel())

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_model.predict(X_test)
        f1_score_final = f1_score(y_test, y_pred,average='macro')
        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred,average='macro')
        # recall = recall_score(y_test, y_pred,average='macro')
        clas_report=classification_report(y_test, y_pred)
        # conf_matrix = confusion_matrix(y_test, y_pred)

        print("\nMetrics for the Best Model:")
        print("Best Hyperparameters:", best_params)
        print(f"F1-Score: {f1_score_final:.2f}")
        print()
        print(clas_report)
        # print(f"Accuracy: {accuracy:.2f}")
        # print(f"Precision: {precision:.2f}")
        # print(f"Recall: {recall:.2f}")
        # print("Confusion Matrix:")
        # print(conf_matrix)

