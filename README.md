🩺 MedScan AI — Chest X-Ray Disease Detection System

AI-powered chest X-ray analysis that detects 14 diseases using DenseNet121, with Grad-CAM explainability, MC Dropout uncertainty quantification, and downloadable PDF clinical reports.


----- Live Demo -----

🔗 https://medscan-ai-project.streamlit.app/

>> Overview
MedScan AI is a deep learning-based web application that analyzes chest X-rays and detects the presence of 14 different thoracic diseases. Built with PyTorch and Streamlit, it provides not just predictions but also visual explanations (Grad-CAM heatmaps), uncertainty estimates (MC Dropout), and professional PDF reports — making it suitable for research and educational use.

>> Features
FeatureDescription- DenseNet121Transfer learning on NIH ChestX-ray14 (112,120 images), Grad-CAM HeatmapVisual explanation of where disease was detected, MC DropoutUncertainty quantification over 20 forward passes📄 PDF ReportDownloadable clinical report with findings & specialist recommendations📈 Scan HistoryTracks all past scans with risk levels and confidence scores, Model ComparisonCompare DenseNet121, ResNet50, EfficientNet-B4 side by side 🎯 14 Disease ClassesMulti-label classification with sigmoid activation

>> Detectable Diseases

>>Atelectasis
>> Cardiomegaly
>> Effusion
>> Infiltration
>> Mass
>> Nodule
>> Pneumonia
>> Pneumothorax
>> Consolidation
>> Edema
>> Emphysema
>> Fibrosis
>>Pleural Thickening
>> Hernia

>> Architecture
Input X-Ray (224×224×3)
        ↓
DenseNet121 (ImageNet pretrained)
        ↓
Dropout(0.5) + Linear(1024 → 14)
        ↓
Sigmoid Activation (Multi-label)
        ↓
14 Disease Probabilities
Training Details
ParameterValueDatasetNIH ChestX-ray14Total Images112,120Train / Val / Test70% / 15% / 15%Phase 1Classifier only — 5 epochs, LR 1e-3Phase 2DenseBlock4 fine-tune — 10 epochs, LR 1e-4LossBCEWithLogitsLoss + class weightsOptimizerAdamW

>> Getting Started
1. Clone the repository
bashgit clone https://github.com/YOUR_USERNAME/medscan-ai.git
cd medscan-ai
2. Install dependencies
bashpip install -r requirements.txt
3. Add model weights
Place your trained model file in the root directory:
medscan_phase2_best.pt

Note: If no .pt file is found, the model will run with random weights (for UI testing only).

4. Run the app
bashstreamlit run app.py
Open your browser at http://localhost:8501
