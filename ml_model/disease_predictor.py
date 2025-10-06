# =============================================================================
# Step 1: Import Necessary Libraries
# =============================================================================
import pandas as pd
import numpy as np
import joblib # For saving and loading the model
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Import the models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("Libraries imported successfully.")

# =============================================================================
# Step 2: Load and Prepare the Data
# =============================================================================

# Load the training dataset
try:
    df = pd.read_csv('Training.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Make sure 'Training.csv' is in the same directory as the script.")
    exit()

# Drop the unnecessary 'Unnamed: 133' column if it exists
if 'Unnamed: 133' in df.columns:
    df = df.drop('Unnamed: 133', axis=1)
    print("Dropped 'Unnamed: 133' column.")

# Separate features (X) and target (y)
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Encode the target variable 'prognosis' into numerical labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the label encoder for later use
joblib.dump(le, 'label_encoder.joblib')

# =============================================================================
# Step 3: Create a More Realistic Train-Test Split
# =============================================================================
# We split the main dataset into 80% for training and 20% for testing.
# This provides a much more robust evaluation than the original Testing.csv.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nData split into training and testing sets.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# =============================================================================
# Step 4: Model Training and Comparative Evaluation
# =============================================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

print("\n--- Model Training and Evaluation on New Test Set ---")
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the new test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[model_name] = {'accuracy': accuracy, 'f1_score': f1}

    print(f"\nModel: {model_name}")
    print(f"  -> Accuracy: {accuracy:.4f}")
    print(f"  -> F1-Score: {f1:.4f}")

# =============================================================================
# Step 5: Select and Save the Best Model
# =============================================================================

# Select the best model based on F1-score
best_model_name = max(results, key=lambda k: results[k]['f1_score'])
best_model = models[best_model_name]
print(f"\nBest performing model is: {best_model_name}")

# Save the trained model to a file
joblib.dump(best_model, 'best_model.joblib')
print(f"Best model ('{best_model_name}') saved to 'best_model.joblib'")


# Optional: Visualize the confusion matrix for the best model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix for {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# plt.show() # Uncomment to display the plot

# =============================================================================
# Step 6: Create a Prediction Function Using the Saved Model
# =============================================================================

def predict_disease(symptom_string):
    """
    Takes a comma-separated string of symptoms, loads the saved model,
    and predicts the disease.
    """
    try:
        # Load the saved model and label encoder
        model = joblib.load('best_model.joblib')
        encoder = joblib.load('label_encoder.joblib')
    except FileNotFoundError:
        return "Error: Model or encoder file not found. Please train the model first."

    # Get the list of all possible symptoms from the training data columns
    all_symptoms = X.columns.tolist()
    input_vector = np.zeros(len(all_symptoms))

    # Clean the user's input symptoms
    user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptom_string.split(',')]

    # Set the corresponding indices to 1
    for symptom in user_symptoms:
        symptom = symptom.replace('__', '_') # Fix for extra underscores
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            input_vector[index] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not recognized.")

    # Create a DataFrame for prediction with correct feature names
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    # Predict the disease code
    predicted_code = model.predict(input_df)[0]

    # Decode the prediction back to the original disease name
    predicted_disease = encoder.inverse_transform([predicted_code])[0]

    return predicted_disease

# =============================================================================
# Step 7: Interactive Testing Loop
# =============================================================================
if __name__ == '__main__':
    print("\n\n--- Interactive Disease Predictor ---")
    print("Enter symptoms separated by commas (e.g., itching,skin_rash,chills).")
    print("Type 'exit' to quit.")

    while True:
        # Get input from the user
        symptom_input = input("\nEnter your symptoms: ")

        # Check if the user wants to exit
        if symptom_input.lower() == 'exit':
            print("Exiting the predictor. Stay healthy!")
            break

        # Ensure the user has entered something
        if not symptom_input:
            print("Please enter at least one symptom.")
            continue

        # Get the prediction
        predicted_disease = predict_disease(symptom_input)
        print(f"  -> Predicted Disease: {predicted_disease}")

