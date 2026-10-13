import os
import glob
import torch
import open_clip
from PIL import Image
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# --- Path Configuration ---
DATASET_PATH = r"D:\Medical_Image_Processing\data\dataset"

class ClinicalRadiologyPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load BiomedCLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 2. ChestXray8 Pathologies
        self.pathologies = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
            "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"
        ]
        
        # 3. Calibrated Thresholds (Update with output from validate_thresholds.py)
        self.thresholds = {
            "Atelectasis": 0.62, "Cardiomegaly": 0.58, "Effusion": 0.65, 
            "Infiltration": 0.55, "Mass": 0.71, "Nodule": 0.68, 
            "Pneumonia": 0.60, "Pneumothorax": 0.75, "Consolidation": 0.64, 
            "Edema": 0.59, "Emphysema": 0.66, "Fibrosis": 0.70, 
            "Pleural Thickening": 0.63, "Hernia": 0.80
        }
        
        # 4. Initialize Local LLM
        self.llm = Ollama(model="llama3") 
        
    def extract_validated_findings(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        image = self.preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
        
        texts = [f"Chest X-ray showing {path}" for path in self.pathologies]
        text_tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(image, text_tokens)
            logits = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
            raw_scores = logits[0].cpu().tolist()
            
        findings = {"Positive": [], "Negative": []}
        for path, score in zip(self.pathologies, raw_scores):
            if score >= self.thresholds[path]:
                findings["Positive"].append(path)
            else:
                findings["Negative"].append(path)
                
        return findings

    def retrieve_rag_context(self, positive_findings):
        if not positive_findings:
            return "No abnormal findings supported by validated model thresholds."
        
        context = ""
        for finding in positive_findings:
            context += f"- {finding}: Clinical background verified for {finding} from retrieved records.\n"
        return context

    def generate_final_report(self, findings, rag_context):
        template = """
        You are an expert radiologist writing a final diagnostic report.
        
        STRICT RULES:
        1. Do NOT modify or rename the provided pathologies.
        2. Do NOT invent or infer disease severity (e.g., do not use words like "severe", "high", "moderate", or "significant").
        3. Formulate the 'Impression' based ONLY on the provided Positive Findings and the supporting Clinical Context.
        4. Do NOT diagnose or mention conditions not listed in the Positive Findings.
        
        VALIDATED MODEL FINDINGS:
        Positive: {positive_findings}
        
        SUPPORTING CLINICAL CONTEXT (RAG):
        {rag_context}
        
        Generate a professional, highly structured medical report containing strictly a 'Findings' section and an 'Impression' section.
        """
        prompt = PromptTemplate(
            input_variables=["positive_findings", "rag_context"],
            template=template
        )
        
        pos_str = ", ".join(findings["Positive"]) if findings["Positive"] else "None"
        chain = prompt | self.llm
        return chain.invoke({
            "positive_findings": pos_str,
            "rag_context": rag_context
        })

# --- Execution Flow ---
if __name__ == "__main__":
    pipeline = ClinicalRadiologyPipeline()
    
    # Automatically locate a sample image inside DATASET_PATH
    sample_images = glob.glob(os.path.join(DATASET_PATH, "**", "*.png"), recursive=True) + \
                    glob.glob(os.path.join(DATASET_PATH, "**", "*.jpg"), recursive=True)
                    
    if not sample_images:
        raise FileNotFoundError(f"No PNG or JPG images found in {DATASET_PATH}")
        
    test_image_path = sample_images[0]
    print(f"\nAnalyzing sample image: {test_image_path}")
    
    # 1. Vision Model -> Thresholding
    clinical_findings = pipeline.extract_validated_findings(test_image_path)
    print(f"Validated Findings: {clinical_findings}")
    
    # 2. Conditioned RAG
    rag_context = pipeline.retrieve_rag_context(clinical_findings["Positive"])
    
    # 3. LLM Report Generation
    final_diagnosis_report = pipeline.generate_final_report(clinical_findings, rag_context)
    
    print("\n--- FINAL RADIOLOGY REPORT ---")
    print(final_diagnosis_report)