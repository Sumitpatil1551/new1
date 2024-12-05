import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import time
from queue import Queue
import threading
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from flask import Flask, render_template, jsonify, request, abort
import logging
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Set the template folder to be in the same directory as the script
template_dir = os.path.join(script_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Global variables
data = None
X = None
y = None
X_train_balanced = None
y_train_balanced = None
X_test = None
y_test = None
selected_features = None
selector = None
classical_svm = None
total_transactions = 0
fraud_detected = 0
simulator = None

# Helper functions
def plot_confusion_matrix(cm, name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    train_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    n_classes = len(np.unique(y))
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, name)
    
    return {
        'Name': name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Confusion Matrix': cm,
        'Train Time': train_time
    }

def load_transaction_csv(filename):
    df = pd.read_csv(filename)
    return df[selected_features].to_dict('records')

class TransactionSimulator:
    def __init__(self, transactions, interval=1):
        self.transactions = transactions
        self.interval = interval
        self.queue = Queue()
        self.thread = None
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run(self):
        for transaction in self.transactions:
            if not self.running:
                break
            self.queue.put(transaction)
            time.sleep(self.interval)

    def get_transaction(self):
        return self.queue.get() if not self.queue.empty() else None

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')



# Update the manual_transaction function
@app.route('/api/manual_transaction', methods=['POST'])
def manual_transaction():
    global total_transactions, fraud_detected
    
    try:
        transaction = request.json
        
        # Validate input
        required_fields = [
            'is_first_loan', 
            'fully_repaid_previous_loans', 
            'currently_repaying_other_loans',
            'total_credit_card_limit', 
            'avg_percentage_credit_card_limit_used_last_year',
            'saving_amount', 
            'checking_amount', 
            'is_employed', 
            'yearly_salary', 
            'age', 
            'dependent_number'
        ]
        
        for field in required_fields:
            if field not in transaction:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate data types and ranges
        try:
            transaction['is_first_loan'] = int(transaction['is_first_loan'])
            transaction['fully_repaid_previous_loans'] = int(transaction['fully_repaid_previous_loans'])
            transaction['currently_repaying_other_loans'] = int(transaction['currently_repaying_other_loans'])
            transaction['is_employed'] = int(transaction['is_employed'])
            
            if not all(0 <= transaction[field] <= 1 for field in ['is_first_loan', 'fully_repaid_previous_loans', 'currently_repaying_other_loans', 'is_employed']):
                raise ValueError("Boolean fields must be 0 or 1")
            
            transaction['total_credit_card_limit'] = float(transaction['total_credit_card_limit'])
            transaction['avg_percentage_credit_card_limit_used_last_year'] = float(transaction['avg_percentage_credit_card_limit_used_last_year'])
            transaction['saving_amount'] = float(transaction['saving_amount'])
            transaction['checking_amount'] = float(transaction['checking_amount'])
            transaction['yearly_salary'] = float(transaction['yearly_salary'])
            
            if any(value < 0 for value in [transaction['total_credit_card_limit'], transaction['saving_amount'], transaction['checking_amount'], transaction['yearly_salary']]):
                raise ValueError("Financial amounts cannot be negative")
            
            if not 0 <= transaction['avg_percentage_credit_card_limit_used_last_year'] <= 100:
                raise ValueError("Average percentage must be between 0 and 100")
            
            transaction['age'] = int(transaction['age'])
            if not 18 <= transaction['age'] <= 120:
                raise ValueError("Age must be between 18 and 120")
            
            transaction['dependent_number'] = int(transaction['dependent_number'])
            if transaction['dependent_number'] < 0:
                raise ValueError("Number of dependents cannot be negative")
            
        except ValueError as ve:
            return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
        
        # Create features in the correct order for prediction
        features = [transaction[feature] for feature in selected_features]

        # Create a DataFrame for prediction
        features_df = pd.DataFrame([features], columns=selected_features)
        
        # Make prediction using the pipeline
        prediction = pipeline.predict(features_df)
        prediction_proba = pipeline.predict_proba(features_df)[0, 1]

        # Update transaction counters
        total_transactions += 1
        is_fraud = bool(prediction[0])
        if is_fraud:
            fraud_detected += 1

        # Return the response
        return jsonify({
            'transaction': transaction,
            'is_fraud': is_fraud,
            'fraud_probability': float(prediction_proba),
            'total_transactions': total_transactions,
            'fraud_detected': fraud_detected,
            'fraud_percentage': (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
        })
    except Exception as e:
        logging.error(f"Error in manual_transaction: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Update the update_dashboard function
@app.route('/api/update', methods=['GET'])
def update_dashboard():
    global total_transactions, fraud_detected
    
    try:
        new_transaction = simulator.get_transaction()
        
        if new_transaction is None:
            return jsonify({'error': 'No more transactions'}), 404

        # The transaction is already in the correct format for prediction
        features_for_prediction = pd.DataFrame([new_transaction])
        
        # Make prediction using the pipeline
        prediction = pipeline.predict(features_for_prediction)
        prediction_proba = pipeline.predict_proba(features_for_prediction)[0, 1]

        total_transactions += 1
        is_fraud = bool(prediction[0])
        if is_fraud:
            fraud_detected += 1

        return jsonify({
            'transaction': new_transaction,
            'is_fraud': is_fraud,
            'fraud_probability': float(prediction_proba),
            'total_transactions': total_transactions,
            'fraud_detected': fraud_detected,
            'fraud_percentage': (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
        })
    except Exception as e:
        logging.error(f"Error in update_dashboard: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500
    
@app.route('/api/stats', methods=['GET'])
def get_stats():
    global total_transactions, fraud_detected
    
    try:
        return jsonify({
            'total_transactions': total_transactions,
            'fraud_detected': fraud_detected,
            'fraud_percentage': (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0,
            'non_fraud_percentage': 100 - (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
        })
    except Exception as e:
        logging.error(f"Error in get_stats: {str(e)}")
        abort(500)

# Update the load_model function
def load_model():
    global pipeline, selected_features
    try:
        pipeline = joblib.load('fraud_detection_pipeline.joblib')
        selected_features = joblib.load('selected_features.joblib')
        logging.info("Model pipeline loaded successfully")
        print("Selected features:", selected_features)
    except FileNotFoundError:
        logging.error("Model files not found. Please run the script with 'train' argument first.")
        sys.exit(1)

# Update the main function
def main():
    global data, X, y, X_train_balanced, y_train_balanced, X_test, y_test, selected_features, pipeline, simulator

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        data = pd.read_csv('modified_loan_fraud_dataset.csv')

        features = ['is_first_loan', 'fully_repaid_previous_loans', 'currently_repaying_other_loans',
                   'total_credit_card_limit', 'avg_percentage_credit_card_limit_used_last_year',
                   'saving_amount', 'checking_amount', 'is_employed', 'yearly_salary', 'age', 'dependent_number']

        X = data[features]
        y = data.iloc[:, -1]

        # Handle missing values before splitting the data
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

        # Split the data using imputed features
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.33, random_state=42)

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Convert to DataFrame and ensure no NaN values
        X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=features)
        
        # Verify no NaN values
        assert not X_train_balanced_df.isna().any().any(), "NaN values found in training data"



        # Feature selection with LASSO
        logging.info("Performing feature selection...")
        lasso = Lasso(alpha=0.1)
        selector = SelectFromModel(lasso, prefit=False)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]

        # Split the data and handle class imbalance
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.33, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Convert to DataFrame with selected feature names
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=selected_features)
        X_test = pd.DataFrame(X_test, columns=selected_features)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(multi_class='ovr', class_weight='balanced', max_iter=1000),
            'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100),
            'Classical SVM': SVC(kernel='rbf', probability=True, class_weight='balanced'),
        }

        # Existing analysis code
        logging.info("Training and evaluating models...")
        with ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(evaluate_model, name, model, X_train_balanced, y_train_balanced, X_test, y_test): name for name, model in models.items()}
            results = {}
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    results[model_name] = future.result()
                except Exception as exc:
                    logging.error(f'{model_name} generated an exception: {exc}')

        # Print results
        for name, result in results.items():
            logging.info(f"\n{name} Results:")
            for metric, value in result.items():
                if metric != 'Confusion Matrix':
                    logging.info(f"{metric}: {value}")
            logging.info("Confusion Matrix:")
            logging.info(result['Confusion Matrix'])
            logging.info(f"Confusion matrix saved as confusion_matrix_{name.lower().replace(' ', '_')}.png")

        # Hyperparameter tuning for Random Forest
        logging.info("\nPerforming hyperparameter tuning for Random Forest...")
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        start_time = time.time()
        random_search.fit(X_train_balanced, y_train_balanced)
        end_time = time.time()

        logging.info(f"Time taken for RandomizedSearchCV: {end_time - start_time:.2f} seconds")
        logging.info(f"Best Random Forest Parameters: {random_search.best_params_}")
        logging.info(f"Best Random Forest Accuracy: {random_search.best_score_}")

        # Additional investigation
        logging.info("\nPerforming additional investigation...")

        # Cross-validation
        cv_scores = cross_val_score(random_search.best_estimator_, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
        logging.info(f"Cross-validation scores: {cv_scores}")
        logging.info(f"Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

        # Feature importance
        perm_importance = permutation_importance(random_search.best_estimator_, X_test, y_test, n_repeats=10, random_state=42)
        feature_importance = perm_importance.importances_mean
        sorted_idx = feature_importance.argsort()
        logging.info("\nTop 5 most important features:")
        for idx in sorted_idx[-5:]:
            logging.info(f"{selected_features[idx]}: {feature_importance[idx]:.4f}")

        # Class balance
        logging.info(f"\nClass balance in training set:")
        logging.info(y_train.value_counts(normalize=True))

        # Evaluate on test set
        y_pred = random_search.best_estimator_.predict(X_test)
        y_pred_proba = random_search.best_estimator_.predict_proba(X_test)

        logging.info("\nClassification Report on Test Set:")
        logging.info(classification_report(y_test, y_pred))

        # Calculate ROC AUC for multi-class
        n_classes = len(np.unique(y))
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        if n_classes == 2:
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
        logging.info(f"\nROC AUC Score: {auc_score:.4f}")

        # Visualizations
        logging.info("\nGenerating visualizations...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Feature Importance
        ax1.barh(range(len(feature_importance)), feature_importance[sorted_idx])
        ax1.set_yticks(range(len(feature_importance)))
        ax1.set_yticklabels(np.array(selected_features)[sorted_idx])
        ax1.set_title('Feature Importance')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix')
        ax2.set_ylabel('Actual Label')
        ax2.set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig('fraud_detection_results.png')
        plt.close()

        # Fraud detection results
        fraud_count = np.sum(y_pred == 1)
        logging.info(f"\nNumber of fraud loans detected: {fraud_count}")
        logging.info(f"Total number of loans in the test set: {len(y_test)}")
        logging.info(f"Percentage of loans detected as fraud: {fraud_count / len(y_test) * 100:.2f}%")

        actual_fraud_count = np.sum(y_test == 1)
        logging.info(f"Actual number of fraud loans in the test set: {actual_fraud_count}")
        logging.info(f"Actual percentage of fraud loans: {actual_fraud_count / len(y_test) * 100:.2f}%")

        logging.info("\nFraud detection analysis complete. Results visualization saved as 'fraud_detection_results.png'.")

        # Train and save classical SVM for real-time predictions
        logging.info("Training classical SVM for real-time predictions...")
        classical_svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
        classical_svm.fit(X_train_balanced, y_train_balanced)
        joblib.dump(classical_svm, 'classical_svm.joblib')
        joblib.dump(selector, 'feature_selector.joblib')
        joblib.dump(selected_features, 'selected_features.joblib')
        logging.info("Classical SVM trained and saved.")

# Create a pipeline with scaling, feature selection, and SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectFromModel(Lasso(alpha=0.1))),
            ('classifier', SVC(kernel='rbf', probability=True, class_weight='balanced'))
        ])

        # Fit the pipeline using the original feature set
        X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=features)
        pipeline.fit(X_train_balanced_df, y_train_balanced)

        # Get selected features (fix for the IndexError)
        feature_selector = pipeline.named_steps['selector']
        feature_mask = feature_selector.get_support()
        selected_features = np.array(features)[feature_mask]

        # Print debug information
        print("Original features:", features)
        print("Number of features selected:", sum(feature_mask))
        print("Selected features:", selected_features)

        # Save the entire pipeline and selected features
        joblib.dump(pipeline, 'fraud_detection_pipeline.joblib')
        joblib.dump(selected_features, 'selected_features.joblib')
        logging.info("Model pipeline trained and saved.")
        # Additional debug verification
        logging.info(f"Original number of features: {len(features)}")
        logging.info(f"Number of selected features: {len(selected_features)}")
        logging.info(f"Selected features: {selected_features}")

    else:
        load_model()
        # Load transaction data for simulation
        try:
            transaction_data = load_transaction_csv('realtime_transactions.csv')
            simulator = TransactionSimulator(transaction_data, interval=1)
            logging.info("Successfully loaded realtime transactions data")
        except Exception as e:
            logging.error(f"Error loading realtime transactions: {str(e)}")
            sys.exit(1)

if __name__ == '__main__':
    main()
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        logging.info("Training complete. Confusion matrices saved as PNG files.")
        logging.info("Run the script without arguments to start the web server.")
    else:
        logging.info("Starting web server for real-time fraud detection...")
        print("Current working directory:", os.getcwd())
        print("Template folder path:", os.path.abspath('templates'))
        simulator.start()  # Start the transaction simulator
        app.run(debug=True, use_reloader=False)
        simulator.stop()  # Stop the simulator when the app stops
        