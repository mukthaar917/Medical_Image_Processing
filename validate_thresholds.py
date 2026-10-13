import os
import glob
import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

# --- Path Configuration ---
DATASET_PATH = r"D:\Medical_Image_Processing\data\dataset"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

# Automatically locate the CSV file inside DATASET_PATH
csv_files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found inside {DATASET_PATH}. Please check your Kaggle extract.")
CSV_PATH = csv_files[0]
print(f"Using CSV: {CSV_PATH}")

# Look for image folder or fallback to root
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
if not os.path.exists(IMAGE_DIR):
    IMAGE_DIR = DATASET_PATH
print(f"Using Image Directory: {IMAGE_DIR}")

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
    "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
]

def load_dataset():
    """Loads the ChestXray8 dataframe and creates binary labels."""
    print("Loading dataset metadata...")
    df = pd.read_csv(CSV_PATH)
    
    # Identify finding label column
    label_col = 'Finding Labels' if 'Finding Labels' in df.columns else df.columns[1]
    
    # Create binary columns for each pathology
    for path in PATHOLOGIES:
        df[path] = df[label_col].astype(str).apply(lambda x: 1 if path in x else 0)
        
    return df

def calculate_optimal_threshold(y_true, y_pred_probs):
    """Calculates the optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], roc_auc_score(y_true, y_pred_probs)

def run_validation():
    df = load_dataset()
    
    # Identify image index column
    img_col = 'Image Index' if 'Image Index' in df.columns else df.columns[0]
    
    # Validate on a subset to start; adjust as needed
    eval_df = df.head(1000) 
    
    print("Loading BiomedCLIP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")
    
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = model.to(device)
    model.eval()

    texts = [f"Chest X-ray showing {path}" for path in PATHOLOGIES]
    text_tokens = tokenizer(texts).to(device)
    
    predictions = {path: [] for path in PATHOLOGIES}
    ground_truths = {path: [] for path in PATHOLOGIES}
    
    print("Extracting features and generating predictions...")
    with torch.no_grad():
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
            filename = str(row[img_col])
            img_path = os.path.join(IMAGE_DIR, filename)
            
            # Recursive check if image is not in default folder
            if not os.path.exists(img_path):
                img_path = os.path.join(DATASET_PATH, filename)
                if not os.path.exists(img_path):
                    continue
                    
            try:
                image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
                image_features, text_features_batch, logit_scale = model(image, text_tokens)
                logits = (logit_scale * image_features @ text_features_batch.T).softmax(dim=-1)
                
                raw_scores = logits[0].cpu().tolist()
                
                for i, path in enumerate(PATHOLOGIES):
                    predictions[path].append(raw_scores[i])
                    ground_truths[path].append(row[path])
                    
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                
    print("\n--- Validation Results & Calibrated Thresholds ---")
    optimal_thresholds = {}
    
    for path in PATHOLOGIES:
        y_true = np.array(ground_truths[path])
        y_pred = np.array(predictions[path])
        
        if len(np.unique(y_true)) > 1:
            best_thresh, auc = calculate_optimal_threshold(y_true, y_pred)
            optimal_thresholds[path] = round(float(best_thresh), 4)
            print(f"{path.ljust(20)} | AUC: {auc:.3f} | Threshold: {best_thresh:.4f}")
        else:
            print(f"{path.ljust(20)} | AUC: N/A (Insufficient class distribution in subset)")

    print("\nPaste these calibrated thresholds into your run.py:")
    print("self.thresholds =", optimal_thresholds)

if __name__ == "__main__":
    run_validation()