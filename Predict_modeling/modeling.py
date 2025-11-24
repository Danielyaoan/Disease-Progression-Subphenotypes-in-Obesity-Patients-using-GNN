import shap

from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import joblib

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
model_type = 'GraphSAGE'
model_name = model_type+'_cluster'
data = pd.read_csv('')

train_indices = pd.read_csv("")["index"].tolist()
test_indices = pd.read_csv("")["index"].tolist()

output_path = '' + model_type + '/'
print(data.shape)
def compute_macro_roc(y_true, y_pred_prob):
    """
    Computes macro-average ROC curve for multi-class data.

    Parameters:
    - y_true: Ground truth (true) labels.
    - y_pred_prob: Prediction probabilities.

    Returns:
    - fpr_macro: Macro-average false positive rate.
    - tpr_macro: Macro-average true positive rate.
    - auroc_macro: Area under the macro-average ROC curve.
    """
    n_classes = len(np.unique(y_true))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    auroc_macro = auc(fpr_macro, tpr_macro)

    return fpr_macro, tpr_macro, auroc_macro
    

features = data.iloc[:, :2068].copy()
target = data[model_name]
target_label = model_name


X_train, X_test, y_train, y_test = features.iloc[train_indices, :], features.iloc[test_indices, :], target[train_indices] , target[test_indices]

# over sampling
ros = RandomOverSampler(random_state=42)
X_train_overampled, y_train_oversampled = ros.fit_resample(X_train, y_train)

# Check the distribution of the classes after resampling
print("Class distribution after over resampling:", dict(zip(*np.unique(y_train_oversampled, return_counts=True))))

# under sampling
rus = RandomUnderSampler(random_state=42)
X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)

# Check the distribution of the classes after resampling
print("Class distribution after under resampling:", dict(zip(*np.unique(y_train_undersampled, return_counts=True))))

imp = SimpleImputer( strategy='mean')

#lrc = linear_model.LogisticRegression()
lrc = linear_model.LogisticRegression(multi_class='multinomial')

#xgb = XGBClassifier()
number_of_classes = len(np.unique(y_test))
xgb = XGBClassifier(objective='multi:softprob', num_class=number_of_classes)

model_1 = Pipeline([
    ('imputation', imp),
    ('classifier', lrc)
])

model_2 = Pipeline([
    ('imputation', imp),
    ('classifier', xgb)
])


# Define the search space for Logistic Regression
model_1_parameters = {
    "classifier__C": Real(1e-5, 10, prior='log-uniform'),
    "classifier__solver": Categorical(["newton-cg", "lbfgs", "sag", "saga"]),
    "classifier__penalty": Categorical(["none",  "l2"])#"l1", "elasticnet"
}

# Define the search space for XGBoost
model_2_parameters = {
    "classifier__max_depth": Integer(3, 9),
    "classifier__min_child_weight": Real(0.1, 1),
    "classifier__gamma": Real(0.01, 10, prior='log-uniform'),
    "classifier__subsample": Real(0.5, 0.75),
    "classifier__reg_alpha": Real(0.01, 10, prior='log-uniform'),
    "classifier__reg_lambda": Real(0.01, 10, prior='log-uniform'),
    "classifier__n_estimators": Integer(50, 200)
}

# Create and fit the BayesSearchCV models
model_3 = BayesSearchCV(model_1, model_1_parameters, n_iter=50, scoring='roc_auc_ovr', cv=3, n_jobs=12, verbose=10)
model_4 = BayesSearchCV(model_2, model_2_parameters, n_iter=50, scoring='roc_auc_ovr', cv=3, n_jobs=12, verbose=10)

model_3.fit(X_train, y_train)
model_4.fit(X_train, y_train)

# Compute ROC curves and classification reports for models 5 and 6
model_3_predict_prob = model_3.predict_proba(X_test)
model_4_predict_prob = model_4.predict_proba(X_test)

#save model 
joblib.dump(model_3.best_estimator_, output_path + 'model_1.joblib')
joblib.dump(model_4.best_estimator_, output_path + 'model_2.joblib')


# Assuming you have already defined your models and pipelines (like model_1 and model_2)
# Create and fit the BayesSearchCV models
model_7 = BayesSearchCV(model_1, model_1_parameters, n_iter=50, scoring='roc_auc_ovr', cv=3, n_jobs=12, verbose=10)
model_8 = BayesSearchCV(model_2, model_2_parameters, n_iter=50, scoring='roc_auc_ovr', cv=3, n_jobs=12, verbose=10)

# Train the models on the undersampled data
model_7.fit(X_train_undersampled, y_train_undersampled)
model_8.fit(X_train_undersampled, y_train_undersampled)

# Compute ROC curves and classification reports for models 9 and 10
model_7_predict_prob = model_7.predict_proba(X_test)
model_8_predict_prob = model_8.predict_proba(X_test)

#save model 
joblib.dump(model_7.best_estimator_, output_path + 'model_3.joblib')
joblib.dump(model_8.best_estimator_, output_path + 'model_4.joblib')


# Assuming you have already defined your models and pipelines (like model_1 and model_2)
# Create and fit the BayesSearchCV models
model_5 = BayesSearchCV(model_1, model_1_parameters, n_iter=50, scoring='roc_auc_ovr', cv=3, n_jobs=12, verbose=10)
model_6 = BayesSearchCV(model_2, model_2_parameters, n_iter=50, scoring='roc_auc_ovr', cv=3, n_jobs=12, verbose=10)

# Train the models on the resampled data
model_5.fit(X_train_overampled, y_train_oversampled)
model_6.fit(X_train_overampled, y_train_oversampled)

# Compute ROC curves and classification reports for models 7 and 8
model_5_predict_prob = model_5.predict_proba(X_test)
model_6_predict_prob = model_6.predict_proba(X_test)

#save model 
joblib.dump(model_5.best_estimator_, output_path + 'model_5.joblib')
joblib.dump(model_6.best_estimator_, output_path + 'model_6.joblib')






# Use the function for model_1 and model_2
model_3_fpr_macro, model_3_tpr_macro, model_3_auroc_macro = compute_macro_roc(y_test, model_3_predict_prob)
model_4_fpr_macro, model_4_tpr_macro, model_4_auroc_macro = compute_macro_roc(y_test, model_4_predict_prob)

model_5_fpr_macro, model_5_tpr_macro, model_5_auroc_macro = compute_macro_roc(y_test, model_5_predict_prob)
model_6_fpr_macro, model_6_tpr_macro, model_6_auroc_macro = compute_macro_roc(y_test, model_6_predict_prob)


model_7_fpr_macro, model_7_tpr_macro, model_7_auroc_macro = compute_macro_roc(y_test, model_7_predict_prob)
model_8_fpr_macro, model_8_tpr_macro, model_8_auroc_macro = compute_macro_roc(y_test, model_8_predict_prob)




model_3_auroc_macro = round(model_3_auroc_macro, 2)
model_4_auroc_macro = round(model_4_auroc_macro, 2)
model_5_auroc_macro = round(model_5_auroc_macro, 2)
model_6_auroc_macro = round(model_6_auroc_macro, 2)
model_7_auroc_macro = round(model_7_auroc_macro, 2)
model_8_auroc_macro = round(model_8_auroc_macro, 2)


# Now, you can plot the macro-average ROC curves:
plt.figure(figsize=(30,20))
ax = plt.subplot(1,3,1)
ax.plot([0, 1], [0, 1], 'k--*')

ax.plot(model_3_fpr_macro, model_3_tpr_macro, label='Logistic Regression (Bayes) -> ' + str(round(model_3_auroc_macro, 2)))
ax.plot(model_4_fpr_macro, model_4_tpr_macro, label='XGBoost (Bayes) -> ' + str(round(model_4_auroc_macro, 2)))
ax.plot(model_5_fpr_macro, model_5_tpr_macro, label='Logistic Regression (Oversampled + Bayes) -> ' + str(round(model_5_auroc_macro, 2)))
ax.plot(model_6_fpr_macro, model_6_tpr_macro, label='XGBoost (Oversampled + Bayes) -> ' + str(round(model_6_auroc_macro, 2)))
ax.plot(model_7_fpr_macro, model_7_tpr_macro, label='Logistic Regression (Undersampled + Bayes) -> ' + str(round(model_7_auroc_macro, 2)))
ax.plot(model_8_fpr_macro, model_8_tpr_macro, label='XGBoost (Undersampled + Bayes) -> ' + str(round(model_8_auroc_macro, 2)))
ax.legend()
plt.savefig(output_path +'macro_roc_curves.png')
ax.set_aspect(1)

print("model 1 summary: \n", classification_report(y_test, model_3.predict(X_test)))
print("model 2 summary: \n", classification_report(y_test, model_4.predict(X_test)))
print("model 3 summary: \n", classification_report(y_test, model_5.predict(X_test)))
print("model 4 summary: \n", classification_report(y_test, model_6.predict(X_test)))
print("model 5 summary: \n", classification_report(y_test, model_7.predict(X_test)))
print("model 6 summary: \n", classification_report(y_test, model_8.predict(X_test)))






