from all_imports import *

# Preprocessing data
def data_preprocess(df):

# Altering data due to missing values:
# ca, 611, (~66%) | slope, 309, (~33%) | thal, 486, (~53%) | trestbps, chol, thalch, oldpeak, exang : under 10%
    for col in ['ca', 'id']:
        if col in df.columns: #id has no meaning in it whatsoever,
            df = df.drop(columns=[col]) # too many missing ca vals

    # defining categorical / numerical columns
    categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numerical_cols = df.drop(columns=['num'] + categorical_cols, errors="ignore").columns
    # 'num' is target, so not included in features. target : 0(no disease) / 1,2,3,4 (disease, with severity levels)

    #  Fill missing values
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())  # median, dataset size intact.

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])  # mode, replace missing categorical vals with the most frequent category.

    # One-hot encoding categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # separating features and the target
    X = df.drop(columns=['num']) #cols except target
    
    """
       y = df['num']
       
       changing target to binary classification becase the dataset is both imbalanced and small size
    """
 
    y=df['num'].apply(lambda x: 1 if x > 0 else 0) #only predicting disease presence
    
    
    # 6. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = data_preprocess(df)

#Summaries
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Target distribution:\n", y_train.value_counts(normalize=True))

# Model training and evaluation. Model: RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # ->with 100 trees, balanced class weight

rf_model.fit(X_train, y_train) # training
y_pred = rf_model.predict(X_test) # predicting

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.show()


# impactful features
importants = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:\n", importants.head(10))



#joblib to save model
joblib.dump(rf_model,'heart_disease_model.pkl' )
#column names
joblib.dump(X_train.columns,'model_columns.pkl' )
print("Model and columns saved.")