import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline  
from imblearn.over_sampling import SMOTE

# Fetch dataset from UCI ML repo 
vertebral_column = fetch_ucirepo(id=212)
X = vertebral_column.data.features
y = vertebral_column.data.targets


# Label encode class variables 
le = LabelEncoder()
y_encoded = le.fit_transform(y['class'])

# Split data to train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Build Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Scale the predictors 
    ('smote', SMOTE(random_state=42)),     # Apply SMOTE to handle class imbalance 
    ('classifier', VotingClassifier(       # Choose Voting Classifer ML Model that combines following: SV, RF and LR models based on optimal parameters detected when running in notebook. 
        estimators=[
            ('svm', SVC(C=100, gamma=0.01, kernel='rbf')),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('lr', LogisticRegression(solver='lbfgs', max_iter=500))
            ],
            voting='hard'
    ))
])

# Fit the trained model 
pipeline.fit(X_train, y_train)

# Predict with test model 
y_pred = pipeline.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred),4))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
joblib.dump(pipeline, 'vertebral_voting_hard_svm-rf-logreg_smote_v1.pkl')
joblib.dump(le, 'label_encoder.pkl')
