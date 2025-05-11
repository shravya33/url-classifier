import pandas as pd
import numpy as np
import pickle
import re
import tldextract
from urllib.parse import urlparse
import argparse
import sys
from colorama import Fore, Style, init

init() #for colorama

# loading the model and its components
def load_model_components():
    try:
        with open('models/url_classifier_final.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/feature_selector.pkl', 'rb') as f:
            feature_selector = pickle.load(f)
        
        with open('models/selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        
        return model, label_encoder, scaler, feature_selector, selected_features
 
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the training script first to generate model components.")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

def extract_features_from_url(url, selected_features):
    features = {}
    url = url.lower()
    
    while url.endswith('/') and len(url) > 1:
        url = url[:-1]
    
    # basic features
    features['url_length'] = len(url)
    features['url_dot_count'] = url.count('.')
    features['url_slash_count'] = url.count('/')
    
    # advanced features
    try:
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        
        features['domain_name_length'] = len(extracted.domain) if extracted.domain else 0
        
        features['path_length'] = len(parsed.path) if parsed.path else 0
        
        tld = extracted.suffix
        if not tld:
            features['tld_type'] = 0  # No TLD
        elif tld in ['com', 'org', 'net', 'edu', 'gov', 'mil']:
            features['tld_type'] = 1  # Common TLDs
        else:
            features['tld_type'] = 2  # Other TLDs
        
        subdomains = extracted.subdomain.split('.') if extracted.subdomain else []
        if subdomains and subdomains[0]:
            features['subdomain_count'] = len(subdomains)
        else:
            features['subdomain_count'] = 0
        
        features['is_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', extracted.domain) else 0
        
        special_chars = re.findall(r'[-_~%&=+#@!$*()\[\]{}|;:,<>?]', url)
        features['special_char_count'] = len(special_chars)
        
        digits = re.findall(r'\d', extracted.domain + parsed.path)
        features['digit_count'] = len(digits)
        
    except Exception as e:
        print(f"Error processing URL: {url} - {str(e)}")
        # Set default values for features that couldn't be extracted
        features['domain_name_length'] = 0
        features['path_length'] = 0
        features['tld_type'] = 0
        features['subdomain_count'] = 0
        features['is_ip'] = 0
        features['special_char_count'] = 0
        features['digit_count'] = 0
    
    df_all_features = pd.DataFrame([features])
    
    for feature in selected_features:
        if feature not in df_all_features.columns:
            df_all_features[feature] = 0
    
    df_selected_features = df_all_features[selected_features]
    
    return df_selected_features

def determine_risk_level(probability):
    if probability < 0.3:
        return "Low", Fore.GREEN
    elif probability < 0.7:
        return "Medium", Fore.YELLOW
    else:
        return "High", Fore.RED

def classify_url(url, model, label_encoder, scaler, feature_selector, selected_features):
    features_df = extract_features_from_url(url, selected_features)
    scaled_features = scaler.transform(features_df)
    selected_features = feature_selector.transform(scaled_features)
    prediction = model.predict(selected_features)[0]
    probabilities = model.predict_proba(selected_features)[0]
    
    class_name = label_encoder.inverse_transform([prediction])[0]
    
    class_probability = probabilities[prediction]
    
    risk_level, color = determine_risk_level(probabilities[1])  # bad=1
    
    return {
        'url': url,
        'prediction': class_name,
        'confidence': class_probability,
        'risk_level': risk_level,
        'risk_color': color,
        'probabilities': {
            label_encoder.inverse_transform([i])[0]: prob 
            for i, prob in enumerate(probabilities)
        }
    }

def classify_urls_from_file(file_path, model, label_encoder, scaler, feature_selector, selected_features):
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        else:
            print("Error: File must be TXT format")
            return []
        
        results = []
        for url in urls:
            result = classify_url(url, model, label_encoder, scaler, feature_selector, selected_features)
            results.append(result)
        
        return results
    
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []

def print_results(results):
    for result in results:
        print("\n" + "="*80)
        print(f"URL: {result['url']}")
        print(f"Risk Level: {result['risk_color']}{result['risk_level']}{Style.RESET_ALL}")
    
    print("\n" + "="*80)
    print(f"Total URLs analyzed: {len(results)}")
    print(f"Results: {sum(1 for r in results if r['prediction'] == 'good')} bad, "
          f"{sum(1 for r in results if r['prediction'] == 'bad')} good")

def main():
    parser = argparse.ArgumentParser(description='URL Classification System')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-u', '--url', help='To classify a single URL')
    group.add_argument('-f', '--file', help='To classify URLs contained in a TXT file (with one URL per line)')
    parser.add_argument('-o', '--output', help='Output results to a TXT file')
    args = parser.parse_args()
    
    # Load model and components
    print("Loading model components...")
    model, label_encoder, scaler, feature_selector, selected_features = load_model_components()
    print("Model components loaded successfully!")
    
    results = []
    
    if args.url:
        print(f"\nClassifying URL: {args.url}")
        result = classify_url(args.url, model, label_encoder, scaler, feature_selector, selected_features)
        results.append(result)
    elif args.file:
        print(f"\nClassifying URLs from file: {args.file}")
        results = classify_urls_from_file(args.file, model, label_encoder, scaler, feature_selector, selected_features)
    
    print_results(results)
    
    if args.output and results:
        try:
            results_df = pd.DataFrame([
                {
                    'url': r['url'],
                    'prediction': r['prediction'],
                    'confidence': r['confidence'],
                    'risk_level': r['risk_level'],
                    'probability_good': r['probabilities'].get('good', 0),
                    'probability_bad': r['probabilities'].get('bad', 0)
                }
                for r in results
            ])
            
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    main()