import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import os
import tldextract
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tqdm import tqdm

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

def extract_features(urls):
    features = {}
    
    # basics features
    features['url_length'] = [len(url) for url in urls]
    features['url_dot_count'] = [url.count('.') for url in urls]
    features['url_slash_count'] = [url.count('/') for url in urls]
    
    # advanced features
    domain_name_length = []
    path_length = []
    tld_type = []
    subdomain_count = []
    is_ip = []
    special_char_count = []
    digit_count = []
    
    for url in urls:
        url = url.lower()
        while url.endswith('/') and len(url) > 1:
            url = url[:-1]
            
        try:
            parsed = urlparse(url)
            extracted = tldextract.extract(url)
            
            domain_name_length.append(len(extracted.domain) if extracted.domain else 0)
            
            path_length.append(len(parsed.path) if parsed.path else 0)
            
            # TLD type (TLD separates doamins, subdomains, etc,)
            tld = extracted.suffix
            if not tld:
                tld_type.append(0)  # No TLD
            elif tld in ['com', 'org', 'net', 'edu', 'gov', 'mil']:
                tld_type.append(1)  # Common TLDs
            else:
                tld_type.append(2)  # Other TLDs
            
            subdomains = extracted.subdomain.split('.') if extracted.subdomain else []
            if subdomains and subdomains[0]:
                subdomain_count.append(len(subdomains))
            else:
                subdomain_count.append(0)
            
            is_ip.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', extracted.domain) else 0)
            
            special_chars = re.findall(r'[-_~%&=+#@!$*()\[\]{}|;:,<>?]', url)
            special_char_count.append(len(special_chars))
            
            digits = re.findall(r'\d', extracted.domain + parsed.path)
            digit_count.append(len(digits))
            
        except Exception as e:
            print(f"Error processing URL: {url} - {str(e)}")
            domain_name_length.append(0)
            path_length.append(0)
            tld_type.append(0)
            subdomain_count.append(0)
            is_ip.append(0)
            special_char_count.append(0)
            digit_count.append(0)
    
    # adding advanced features to the dictionaies
    features['domain_name_length'] = domain_name_length
    features['path_length'] = path_length
    features['tld_type'] = tld_type
    features['subdomain_count'] = subdomain_count
    features['is_ip'] = is_ip
    features['special_char_count'] = special_char_count
    features['digit_count'] = digit_count
    
    df_features = pd.DataFrame(features)
    
    return df_features

# the data is loaded here
def load_data(file_paths):
    all_data = []
    
    for file_path in file_paths:
        try:
            print(f"Loading dataset: {file_path}")
            df = pd.read_csv(file_path)
            
            required_cols = ['url', 'type']
            if all(col in df.columns for col in required_cols):
                df = df[required_cols]
                df = df[df['type'].isin(['good', 'bad'])]
                df = df.dropna(subset=['url']) #removes rows with missing urls
                df['url'] = df['url'].astype(str) #convert urls as string
                all_data.append(df)
            else:
                print(f"Required columns not found in {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not all_data:
        raise ValueError("No valid data loaded from provided files")
    
    # combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total combined instances: {combined_data.shape[0]}")
    
    # good=0, bad=1
    label_encoder = LabelEncoder()
    combined_data['label'] = label_encoder.fit_transform(combined_data['type'])
    
    
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return combined_data, label_encoder

# MAIN TRAINING MODEL
def train_model(dataset_paths, epochs=50, save_interval=10):
    data, label_encoder = load_data(dataset_paths)
    
    print("Extracting features from URLs...")
    features = extract_features(data['url'].values)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    print("Selecting best features...")
    selector = SelectKBest(f_classif, k=min(10, X_scaled.shape[1]))  # select top 10 features 
    X_selected = selector.fit_transform(X_scaled, data['label'])
    
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = features.columns[selected_feature_indices].tolist()
    print(f"Selected features: {selected_feature_names}")
    
    # split data into 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, data['label'], test_size=0.2, random_state=42, stratify=data['label']
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    
    epoch_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # TRAINING
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test) #evaluates the trained model
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        epoch_metrics['accuracy'].append(accuracy)
        epoch_metrics['precision'].append(precision)
        epoch_metrics['recall'].append(recall)
        epoch_metrics['f1'].append(f1)
        
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if epoch % save_interval == 0:
            intermediate_model_path = f"models/url_classifier_epoch_{epoch}.pkl"
            with open(intermediate_model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved intermediate model to {intermediate_model_path}")
    
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for metric, values in epoch_metrics.items():
        plt.plot(range(1, epochs + 1), values, marker='o', linestyle='-', label=metric)
    plt.title('Model Performance Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/training_metrics.png')
    plt.close()
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(selected_feature_names)), importances[indices], align='center')
    plt.xticks(range(len(selected_feature_names)), [selected_feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    # SAVING MODEL
    print("\nSaving final model and preprocessing components...")
    with open('models/url_classifier_final.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/feature_selector.pkl', 'wb') as f:
        pickle.dump(selector, f)
    
    with open('models/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_feature_names, f)
    
    print("Training completed successfully!")
    return model, scaler, selector, label_encoder, selected_feature_names

if __name__ == "__main__":
    dataset_paths = ['dataset_1.csv', 
                     'dataset_2.csv', 
                     'dataset_3.csv', 
                     'dataset_4.csv', 
                     'dataset_5.csv', 
                     'dataset_6.csv', 
                     'dataset_7.csv']
    
    # train model
    model, scaler, selector, label_encoder, selected_features = train_model(
        dataset_paths, 
        epochs=20,  
        save_interval=5 
    )
    
    print("\nModel training and evaluation completed.")
    print(f"Final model and components saved in 'models/' directory")
    print(f"Visualizations saved in 'outputs/' directory")