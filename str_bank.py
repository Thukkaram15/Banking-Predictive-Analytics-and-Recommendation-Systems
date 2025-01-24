import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import time

#Step-1:Generate synthetic loan data for testing purposes
def generate_loan_data(n=1000):
    np.random.seed(42)
    data = {
        "customer_id": [f"cust_{i}" for i in range(n)],
        "age": np.random.randint(18, 70, n),
        "income": np.random.randint(20000, 200000, n),
        "credit_score": np.random.randint(300, 850, n),
        "loan_amount": np.random.randint(1000, 50000, n),
        "interest_rate": np.round(np.random.uniform(2.0, 15.0, n), 2),
        "loan_term": np.random.choice([12, 24, 36, 48, 60], n),
        "repayment_status": np.random.choice([0, 1], n, p=[0.8, 0.2])  # 0: No Default, 1: Default
    }
    return pd.DataFrame(data)

loan_data = generate_loan_data()

#Step-2:Preprocessing and Feature Extraction for Loan Default Prediction
def preprocess_data(loan_data):
    X = loan_data.drop(columns=["customer_id", "repayment_status"])
    y = loan_data["repayment_status"]

    #One-hot encoding categorical features (if any)
    X = pd.get_dummies(X, drop_first=True)
    
    #Normalize the data (important for some models like Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

#Step-3:Exploratory Data Analysis (EDA)
def perform_eda(loan_data):
    st.subheader("Exploratory Data Analysis (EDA)")

    #Display basic statistics of the dataset
    st.write("### Data Summary:") 
    st.write(loan_data.describe())

    #Distribution of the target variable: repayment_status
    st.write("### Distribution of Repayment Status (Defaults):")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='repayment_status', data=loan_data, ax=ax, palette="Set2")
    ax.set_title("Count of Defaults vs Non-Defaults")
    ax.set_xticklabels(["No Default", "Default"])
    st.pyplot(fig)

    #Distribution of numeric features (age, income, credit score, loan amount)
    st.write("### Distribution of Numeric Features:")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(loan_data['age'], kde=True, ax=axes[0, 0]).set(title="Age Distribution")
    sns.histplot(loan_data['income'], kde=True, ax=axes[0, 1]).set(title="Income Distribution")
    sns.histplot(loan_data['credit_score'], kde=True, ax=axes[1, 0]).set(title="Credit Score Distribution")
    sns.histplot(loan_data['loan_amount'], kde=True, ax=axes[1, 1]).set(title="Loan Amount Distribution")
    st.pyplot(fig)

    #Correlation heatmap for numeric features
    st.write("### Correlation Heatmap:")
    loan_data_numeric = loan_data.select_dtypes(include=[np.number])  # Selecting numeric columns
    corr = loan_data_numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    #Boxplots to visualize defaults based on age, credit score, loan amount, and income
    st.write("### Defaults vs Features (Age, Credit Score, Loan Amount, Income):")
    for feature in ['age', 'credit_score', 'loan_amount', 'income']:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='repayment_status', y=feature, data=loan_data, ax=ax, palette="Set2")
        ax.set_title(f"Defaults vs {feature.capitalize()}")
        ax.set_xticklabels(["No Default", "Default"])
        st.pyplot(fig)

#Step-4:Loan Default Prediction with Hyperparameter Tuning and SMOTE
def loan_default_prediction(X_train, X_test, y_train, y_test):
    st.subheader("Loan Default Prediction - Model Training & Tuning")

    #Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    #Define hyperparameters for tuning
    logreg_params = {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['liblinear', 'saga'], 'max_iter': [100, 200, 500], 'class_weight': ['balanced', None]}
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10], 'class_weight': ['balanced', None]}
    gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0]}

    #Collect results for models
    results = {}

    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            params = logreg_params
        elif model_name == "Random Forest":
            params = rf_params
        else:
            params = gb_params

        #Run model without SMOTE
        accuracy_before, precision_before, recall_before, f1_before, roc_auc_before, run_time_before, model_before, y_pred_before = run_model(
            model, X_train, X_test, y_train, y_test, params=None, smote=False
        )

        #Run model with SMOTE
        accuracy_after, precision_after, recall_after, f1_after, roc_auc_after, run_time_after, model_after, y_pred_after = run_model(
            model, X_train, X_test, y_train, y_test, params=params, smote=True
        )

        #Store results
        results[model_name] = {
            "Before SMOTE": (accuracy_before, precision_before, recall_before, f1_before, roc_auc_before, run_time_before),
            "After SMOTE": (accuracy_after, precision_after, recall_after, f1_after, roc_auc_after, run_time_after)
        }

        #Display the results for each model
        st.write(f"### {model_name}")
        st.write(f"**Before SMOTE**: Accuracy: {accuracy_before:.2f}, Precision: {precision_before:.2f}, Recall: {recall_before:.2f}, F1: {f1_before:.2f}, ROC AUC: {roc_auc_before:.2f}")
        st.write(f"**After SMOTE**: Accuracy: {accuracy_after:.2f}, Precision: {precision_after:.2f}, Recall: {recall_after:.2f}, F1: {f1_after:.2f}, ROC AUC: {roc_auc_after:.2f}")
        st.write(f"Training time (before SMOTE): {run_time_before:.2f} seconds, Training time (after SMOTE): {run_time_after:.2f} seconds")

    #Display model recommendation based on performance (e.g., based on ROC AUC)
    best_model_name = max(results, key=lambda model: results[model]["After SMOTE"][4])  # Highest ROC AUC
    st.subheader("Model Recommendation")
    st.write(f"Based on the ROC AUC scores after applying SMOTE, we recommend using the **{best_model_name}** for loan default prediction.")

    return results

#Function to evaluate model performance
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc

#Function to run models and evaluate performance
def run_model(model, X_train, X_test, y_train, y_test, params=None, smote=False):
    #Apply SMOTE if needed
    if smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    #Hyperparameter tuning if parameters are passed
    if params is not None:
        model = RandomizedSearchCV(model, param_distributions=params, n_iter=10, cv=3, verbose=1, random_state=42)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    run_time = time.time() - start_time

    #Make predictions
    y_pred = model.predict(X_test)

    #Evaluate performance
    accuracy, precision, recall, f1, roc_auc = evaluate_model(y_test, y_pred)

    return accuracy, precision, recall, f1, roc_auc, run_time, model, y_pred

#Step-5:Customer Segmentation (Unsupervised Learning - KMeans)
def customer_segmentation(loan_data):
    st.subheader("Customer Segmentation")

    #Selecting features for segmentation
    features = ['age', 'income', 'credit_score', 'loan_amount']
    X = loan_data[features]

    #Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    loan_data['Cluster'] = kmeans.fit_predict(X_scaled)

    #Display the clusters
    st.write("### Customer Segments (Clusters):")
    st.write(loan_data[['customer_id', 'Cluster']].head())

    #Visualization of clusters
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    st.write("### PCA Visualization of Customer Segments:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=loan_data['Cluster'], palette='Set1', ax=ax)
    ax.set_title("Customer Segments Visualization")
    st.pyplot(fig)

#Step-6:Recommendation Engine
def product_recommendations(loan_data):
    st.subheader("Product Recommendations")

    #Generate a dummy customer-product interaction matrix for demonstration
    interaction_data = {
        "customer_id": np.random.choice(loan_data["customer_id"], 100),
        "product_id": np.random.choice([f"prod_{i}" for i in range(10)], 100),
        "interaction_type": np.random.choice(["viewed", "purchased"], 100),
    }
    interaction_df = pd.DataFrame(interaction_data)

    #Example of Collaborative Filtering using CountVectorizer (could be expanded with real data)
    st.write("### Example Recommendations (Collaborative Filtering)")

    #Display interaction matrix for example
    st.write(interaction_df.head())

#Step-7:Streamlit User Interface
def main():
    st.title("Banking Analytics")

    st.sidebar.header("Select Options:")
    st.sidebar.write("Choose from the following functionalities:")

    menu = ["EDA", "Loan Default Prediction", "Customer Segmentation", "Product Recommendations"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "EDA":
        perform_eda(loan_data)

    elif choice == "Loan Default Prediction":
        X, y = preprocess_data(loan_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        loan_default_prediction(X_train, X_test, y_train, y_test)

    elif choice == "Customer Segmentation":
        customer_segmentation(loan_data)

    elif choice == "Product Recommendations":
        product_recommendations(loan_data)

if __name__ == "__main__":
    main()
