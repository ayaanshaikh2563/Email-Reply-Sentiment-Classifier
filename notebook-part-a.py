# ============================================================
# SVARAAI REPLY CLASSIFICATION PIPELINE - PART A
# ============================================================
# This script performs the following steps:
# 1. Imports required libraries and suppresses warnings.
# 2. Loads and preprocesses a real reply classification dataset.
# 3. Splits the dataset into training and testing sets.
# 4. Trains baseline models: Logistic Regression and LightGBM.
# 5. Fine-tunes a DistilBERT model for sequence classification.
# 6. Compares performance of all models using Accuracy and F1-Score.
# 7. Saves the trained baseline models and TF-IDF vectorizer for deployment.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# ============================================================
# Function: load_and_preprocess_real_dataset
# - Loads CSV dataset from the specified file path.
# - Normalizes labels and maps them to numerical values.
# - Cleans text data by converting to lowercase and stripping whitespace.
# - Drops rows with missing labels.
# - Returns cleaned DataFrame ready for model training.
# ============================================================
def load_and_preprocess_real_dataset():
    file_path = r"C:\Users\Ayaan Shaikh\OneDrive\Documents\Desktop\Ayaan Docs\SvaraAI_Assignment\Models\data\reply_classification_dataset_1.csv"
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")

    # Normalize and map labels
    df['label'] = df['label'].str.lower().str.strip()
    label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label_encoded'] = df['label'].map(label_mapping)

    print(f"\nLabel distribution after normalization:\n{df['label_encoded'].value_counts()}")

    # Preprocess text
    df['reply'] = df['reply'].astype(str).str.lower().str.strip()
    df = df.rename(columns={'reply': 'text'})
    df = df.dropna(subset=['label_encoded'])

    print(f"\nCleaned dataset shape: {df.shape}")
    return df

# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("SVARAAI REPLY CLASSIFICATION PIPELINE - PART A")
    print("="*60)

    # ========================================================
    # Step 1: Load and preprocess dataset
    # ========================================================
    print("\n1. Loading and preprocessing data...")
    df = load_and_preprocess_real_dataset()
    X = df['text']
    y = df['label_encoded']

    # ========================================================
    # Step 2: Split dataset into train and test sets
    # ========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # ========================================================
    # Step 3: Train Logistic Regression baseline model
    # - Uses TF-IDF vectorization of text (max 1000 features, unigrams+bigrams)
    # ========================================================
    print("\n2. Training Logistic Regression baseline...")
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)

    lr_pred = lr_model.predict(X_test_tfidf)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average='weighted')

    print("Logistic Regression Results:")
    print(f"  Accuracy: {lr_accuracy:.4f}")
    print(f"  F1 Score: {lr_f1:.4f}")

    # ========================================================
    # Step 4: Train LightGBM baseline model
    # - Multiclass classification with 3 classes
    # ========================================================
    print("\n3. Training LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        verbose=-1
    )
    lgb_model.fit(X_train_tfidf, y_train)

    lgb_pred = lgb_model.predict(X_test_tfidf)
    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    lgb_f1 = f1_score(y_test, lgb_pred, average='weighted')

    print("LightGBM Results:")
    print(f"  Accuracy: {lgb_accuracy:.4f}")
    print(f"  F1 Score: {lgb_f1:.4f}")

    # ========================================================
    # Step 5: Fine-tune DistilBERT
    # - Uses Hugging Face Transformers and Datasets
    # - Tokenizes text and trains a sequence classification model
    # - Computes Accuracy and F1 on test set
    # ========================================================
    print("\n4. Fine-tuning DistilBERT (this may take several minutes)...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from datasets import Dataset

        train_dataset = Dataset.from_dict({
            'text': X_train.tolist(),
            'labels': y_train.tolist()
        })
        test_dataset = Dataset.from_dict({
            'text': X_test.tolist(),
            'labels': y_test.tolist()
        })

        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
            label2id={'negative': 0, 'neutral': 1, 'positive': 2}
        )

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='f1'
        )

        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
            return {'accuracy': acc, 'f1': f1}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()

        print(f"DistilBERT Evaluation Results:")
        print(eval_results)

        trainer.save_model('./Models/best_model')
        tokenizer.save_pretrained('./Models/best_model')
        print("Saved DistilBERT fine-tuned model and tokenizer to './Models/best_model'")

    except Exception as e:
        print(f"Error during DistilBERT fine-tuning: {e}")
        eval_results = {'eval_accuracy': np.nan, 'eval_f1': np.nan}

    # ========================================================
    # Step 6: Compare and summarize all model performances
    # ========================================================
    import pandas as pd

    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'LightGBM', 'DistilBERT'],
        'Accuracy': [lr_accuracy, lgb_accuracy, eval_results.get('eval_accuracy', np.nan)],
        'F1_Score': [lr_f1, lgb_f1, eval_results.get('eval_f1', np.nan)]
    })

    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(results_df)

    best_idx = results_df['F1_Score'].idxmax()
    print(f"\nBest performing model: {results_df.loc[best_idx, 'Model']}")
    print(f"Best F1 Score: {results_df.loc[best_idx, 'F1_Score']:.4f}")

    # ========================================================
    # Step 7: Save baseline models and vectorizer for deployment
    # ========================================================
    import joblib
    joblib.dump(lr_model, 'Models/lr_model.pkl')
    joblib.dump(lgb_model, 'Models/lgb_model.pkl')
    joblib.dump(tfidf, 'Models/tfidf_vectorizer.pkl')

    print("\nSaved baseline models and vectorizer for deployment.")
    print("="*60)
    print("PART A COMPLETE! ✅")
