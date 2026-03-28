import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prep_data(features_path, labels_path):
    """Loads feature matrix (e.g., pangenome matrix) and labels (e.g., MIC phenotypes)."""
    X = pd.read_csv(features_path, index_col=0)
    y = pd.read_csv(labels_path, index_col=0).squeeze()
    
    # Stratify ensures balanced classes in train/test splits
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_random_forest(X_train, y_train):
    """Trains a Random Forest classifier."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test, model_name="Random_Forest"):
    """Evaluates the model and plots a confusion matrix."""
    predictions = model.predict(X_test)
    
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{model_name.replace('_', ' ')} Confusion Matrix")
    plt.ylabel('True Phenotype')
    plt.xlabel('Predicted Phenotype')
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_cm.png", dpi=300)
    plt.close()

def run_ml_pipeline(features_path, labels_path):
    """Executes the full exploratory ML workflow."""
    print("Loading data and splitting into train/test sets...")
    X_train, X_test, y_train, y_test = load_and_prep_data(features_path, labels_path)
    
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("Evaluating model on test set...")
    evaluate_model(rf_model, X_test, y_test)
    
    print("Running 5-fold cross-validation on training data...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return rf_model

if __name__ == "__main__":
    print("Exploratory ML module loaded.")
    # rf_model = run_ml_pipeline("roary_gene_presence_absence.csv", "mic_phenotypes.csv")

