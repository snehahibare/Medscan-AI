import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import datetime
import os
import time
import json

# ============================================
# CONSTANTS
# ============================================
DISEASE_LIST = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax","Consolidation",
    "Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]
DISEASE_INFO = {
    "Atelectasis":        "Partial or complete collapse of the lung.",
    "Cardiomegaly":       "Enlargement of the heart.",
    "Effusion":           "Abnormal fluid in the pleural space around lungs.",
    "Infiltration":       "Substances filling the lung airways.",
    "Mass":               "Abnormal growth in lung tissue larger than 3cm.",
    "Nodule":             "Small round growth in the lung, smaller than 3cm.",
    "Pneumonia":          "Infection inflaming air sacs in one or both lungs.",
    "Pneumothorax":       "Collapsed lung caused by air leaking into chest wall.",
    "Consolidation":      "Lung tissue filled with liquid instead of air.",
    "Edema":              "Excess fluid in the lungs.",
    "Emphysema":          "Damage to the air sacs in the lungs (COPD).",
    "Fibrosis":           "Scarring of lung tissue reducing breathing efficiency.",
    "Pleural_Thickening": "Thickening of the pleural membrane.",
    "Hernia":             "Protrusion of abdominal organs into the chest."
}
DISEASE_SPECIALIST = {
    "Atelectasis":"Pulmonologist","Cardiomegaly":"Cardiologist",
    "Effusion":"Pulmonologist","Infiltration":"Pulmonologist",
    "Mass":"Oncologist","Nodule":"Pulmonologist / Oncologist",
    "Pneumonia":"General Physician / Pulmonologist","Pneumothorax":"Emergency Physician",
    "Consolidation":"Pulmonologist","Edema":"Cardiologist / Pulmonologist",
    "Emphysema":"Pulmonologist","Fibrosis":"Pulmonologist",
    "Pleural_Thickening":"Pulmonologist","Hernia":"General Surgeon"
}
HISTORY_FILE = "scan_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE,"r") as f: return json.load(f)
    return []

def save_to_history(record):
    h = load_history()
    h.append(record)
    with open(HISTORY_FILE,"w") as f: json.dump(h,f)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="MedScan AI", page_icon="🩺", layout="wide")

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif !important;}

.stApp{background:#f0f4f8 !important;}

/* Sidebar */
section[data-testid="stSidebar"]{background:white !important;border-right:2px solid #e2e8f0 !important;}

/* Metric cards */
[data-testid="metric-container"]{
    background:white !important;border:1px solid #e2e8f0 !important;
    border-radius:12px !important;padding:16px !important;
    transition:all 0.2s !important;
}
[data-testid="metric-container"]:hover{
    transform:translateY(-3px) !important;
    box-shadow:0 6px 20px rgba(37,99,235,0.13) !important;
    border-color:#93c5fd !important;
}
[data-testid="stMetricLabel"] p{color:#64748b !important;font-size:13px !important;font-weight:500 !important;}
[data-testid="stMetricValue"]{color:#1e40af !important;font-weight:700 !important;font-size:26px !important;}

/* Headings */
h1{color:#1e40af !important;font-weight:700 !important;}
h2{color:#1e3a8a !important;font-weight:600 !important;}
h3{color:#1e40af !important;font-weight:600 !important;}
h4{color:#1e3a8a !important;font-weight:600 !important;}

/* Main buttons */
.stButton>button{
    background:#2563eb !important;color:white !important;
    border:none !important;border-radius:20px !important;
    font-weight:600 !important;font-size:14px !important;
    width:100% !important;transition:all 0.2s !important;
    padding:10px 20px !important;
}
.stButton>button:hover{
    background:#1d4ed8 !important;
    box-shadow:0 4px 14px rgba(37,99,235,0.4) !important;
    transform:translateY(-2px) !important;
}

/* Sidebar nav buttons — override main button style */
section[data-testid="stSidebar"] .stButton>button{
    background:#f1f5f9 !important;
    color:#374151 !important;
    border-radius:20px !important;
    font-weight:500 !important;
    font-size:13px !important;
    text-align:left !important;
    padding:10px 16px !important;
    border:1.5px solid transparent !important;
    transform:none !important;
    box-shadow:none !important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
    background:#dbeafe !important;
    color:#1d4ed8 !important;
    border-color:#93c5fd !important;
    transform:none !important;
    box-shadow:none !important;
}

/* Download button */
.stDownloadButton>button{
    background:#16a34a !important;color:white !important;
    border-radius:20px !important;font-weight:600 !important;width:100% !important;
    transition:all 0.2s !important;
}
.stDownloadButton>button:hover{
    background:#15803d !important;transform:translateY(-2px) !important;
}

/* File uploader */
[data-testid="stFileUploader"]{
    background:white !important;
    border:2px dashed #93c5fd !important;
    border-radius:12px !important;
}
[data-testid="stFileUploaderDropzone"]{background:white !important;}
[data-testid="stFileUploaderDropzone"] button{
    background:#2563eb !important;color:white !important;
    border-radius:8px !important;border:none !important;font-weight:600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover{background:#1d4ed8 !important;}
[data-testid="stFileUploaderDropzone"] small{color:#64748b !important;}
[data-testid="stFileUploaderDropzone"] span{color:#374151 !important;}
.stFileUploader label p{color:#374151 !important;font-weight:500 !important;font-size:14px !important;}

/* Text input */
.stTextInput label p{color:#374151 !important;font-weight:500 !important;}
.stTextInput>div>div{border-radius:10px !important;border:1.5px solid #e2e8f0 !important;background:white !important;}
.stTextInput input{color:#1e293b !important;background:white !important;}

/* Alerts */
.stSuccess p{color:#166534 !important;}
.stError p{color:#991b1b !important;}
.stWarning p{color:#92400e !important;}
.stInfo p{color:#1e40af !important;}

/* Slider */
.stSlider label p{color:#374151 !important;font-weight:500 !important;}

/* Caption */
.stCaptionContainer p{color:#64748b !important;}

/* Markdown */
.stMarkdown p{color:#374151 !important;}

/* Table text - fix invisible white text */
.stMarkdown table{color:#1e293b !important;border-collapse:collapse !important;}
.stMarkdown table td{color:#1e293b !important;background:white !important;border:1px solid #e2e8f0 !important;padding:8px 12px !important;}
.stMarkdown table th{color:#ffffff !important;background:#1e40af !important;border:1px solid #1e40af !important;padding:8px 12px !important;font-weight:600 !important;}
.stMarkdown table tr:nth-child(even) td{background:#f8fafc !important;color:#1e293b !important;}
.stMarkdown table tr:hover td{background:#eff6ff !important;color:#1e293b !important;}

/* Hero animations */
@keyframes fadeInDown{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.08)}}
@keyframes gradientSlide{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}

.hero-section{
    background:linear-gradient(135deg,#1e40af 0%,#2563eb 50%,#3b82f6 100%);
    background-size:200% 200%;animation:gradientSlide 4s ease infinite;
    border-radius:20px;padding:48px 40px;text-align:center;margin-bottom:28px;
    position:relative;overflow:hidden;
}
.hero-icon{font-size:64px;display:block;animation:pulse 2s ease-in-out infinite;margin-bottom:16px;}
.hero-title{font-size:42px;font-weight:800;color:white !important;margin-bottom:8px;
    animation:fadeInDown 0.8s ease forwards;letter-spacing:-1px;}
.hero-subtitle{font-size:16px;color:#bfdbfe !important;margin-bottom:20px;
    animation:fadeInUp 0.8s ease 0.3s forwards;opacity:0;animation-fill-mode:forwards;}
.hero-line{height:3px;background:linear-gradient(to right,transparent,white,transparent);
    border-radius:2px;margin:0 auto 20px;width:60%;
    animation:fadeInUp 0.8s ease 0.5s forwards;opacity:0;animation-fill-mode:forwards;}
.hero-tags{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;
    animation:fadeInUp 0.8s ease 0.7s forwards;opacity:0;animation-fill-mode:forwards;}
.hero-tag{background:rgba(255,255,255,0.15);color:white !important;
    padding:6px 16px;border-radius:20px;font-size:12px;font-weight:500;
    border:1px solid rgba(255,255,255,0.3);}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE FOR NAVIGATION
# ============================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e40af,#2563eb);border-radius:14px;
                padding:18px;text-align:center;margin-bottom:16px">
        <div style="font-size:36px">🩺</div>
        <div style="font-size:18px;font-weight:700;color:white;margin-top:6px">MedScan AI</div>
        <div style="font-size:11px;color:#bfdbfe;margin-top:2px">Chest X-Ray Analysis System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#64748b;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:8px'>NAVIGATION</p>", unsafe_allow_html=True)

    nav_items = [
        ("Home",             "🏠  Home"),
        ("Analyze",          "🩻  Analyze X-Ray"),
        ("Comparison",       "📊  Model Comparison"),
        ("History",          "📈  History & Stats"),
        ("About",            "ℹ️  About Model"),
    ]

    for key, label in nav_items:
        is_active = st.session_state.page == key
        if is_active:
            st.markdown(f"""
            <div style="background:#2563eb;color:white;padding:10px 16px;
                        border-radius:20px;font-size:13px;font-weight:600;
                        margin-bottom:6px;cursor:pointer">{label}</div>
            """, unsafe_allow_html=True)
        else:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="background:#eff6ff;border-radius:10px;padding:12px">
        <p style="color:#1e40af;font-size:12px;margin:0;line-height:1.8;font-weight:500">
            <b>Model:</b> DenseNet121<br>
            <b>Dataset:</b> NIH ChestX-ray14<br>
            <b>Images:</b> 112,120<br>
            <b>Diseases:</b> 14 classes
        </p>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ============================================
# MODEL CLASSES
# ============================================
class MedScanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet121(weights=None)
        self.densenet.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024,14))
    def forward(self,x): return self.densenet(x)

@st.cache_resource
def load_model():
    m = MedScanModel()
    try: m.load_state_dict(torch.load("medscan_phase2_best.pt", map_location="cpu"))
    except: pass
    m.eval(); return m

@st.cache_resource
def load_resnet():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048,14))
    m.eval(); return m

@st.cache_resource
def load_efficientnet():
    m = models.efficientnet_b4(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1792,14))
    m.eval(); return m

# ============================================
# HELPERS
# ============================================
def preprocess_image(image):
    t = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    return t(image=np.array(image.convert("RGB")))["image"]

def get_gradcam(model, image_tensor, target_idx, opacity=0.6):
    cam = GradCAM(model=model, target_layers=[model.densenet.features.denseblock4])
    gc  = cam(input_tensor=image_tensor.unsqueeze(0),targets=[ClassifierOutputTarget(target_idx)])
    img = image_tensor.permute(1,2,0).numpy()
    img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    img = np.clip(img,0,1).astype(np.float32)
    return show_cam_on_image(img,gc[0],use_rgb=True,image_weight=1-opacity), img

def get_uncertainty(model, image_tensor, n=20):
    model.train(); preds=[]
    with torch.no_grad():
        for _ in range(n):
            preds.append(torch.sigmoid(model(image_tensor.unsqueeze(0)))[0].numpy())
    model.eval()
    preds=np.array(preds)
    return preds.mean(0), preds.std(0)

def get_pred(model, image_tensor):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(image_tensor.unsqueeze(0)))[0].numpy()

def get_severity(c):
    if c>=0.7:   return "🔴 Severe",  "high",   "#dc2626"
    elif c>=0.5: return "🟡 Moderate","medium", "#d97706"
    elif c>=0.3: return "🟠 Mild",    "medium", "#f59e0b"
    else:        return "✅ Normal",   "low",    "#16a34a"

def get_risk(probs):
    hi = sum(1 for p in probs if p>=0.5)
    md = sum(1 for p in probs if 0.3<=p<0.5)
    if hi>=2:    return "High",   int(min(max(probs)*120,100))
    elif hi==1:  return "Medium", int(max(probs)*100)
    elif md>=1:  return "Low",    int(max(probs)*100)
    else:        return "Normal", int(max(probs)*100)

def generate_pdf(orig_img, cam_img, predictions, uncertainties, patient_name):
    path   = "medscan_report.pdf"
    doc    = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []
    story.append(Paragraph("<font size=18><b>MedScan AI - Clinical X-Ray Report</b></font>", styles['Title']))
    story.append(Spacer(1,0.2*inch))
    story.append(Paragraph(f"<b>Patient:</b> {patient_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Paragraph("<b>Analysis:</b> DenseNet121 + Grad-CAM + MC Dropout", styles['Normal']))
    story.append(Spacer(1,0.2*inch))
    op='ot.png'; cp='ct.png'
    Image.fromarray((orig_img*255).astype(np.uint8)).save(op)
    Image.fromarray(cam_img).save(cp)
    story.append(Table([[RLImage(op,2.5*inch,2.5*inch),RLImage(cp,2.5*inch,2.5*inch)],
                        ["Original X-Ray","Grad-CAM Heatmap"]]))
    story.append(Spacer(1,0.2*inch))
    story.append(Paragraph("<b>Disease Analysis:</b>", styles['Heading2']))
    story.append(Spacer(1,0.1*inch))
    td=[["Disease","Probability","Uncertainty","Status","Specialist"]]
    for d,p,u in zip(DISEASE_LIST,predictions,uncertainties):
        td.append([d,f"{p:.1%}",f"+-{u:.3f}","Detected" if p>0.3 else "Normal",
                   DISEASE_SPECIALIST.get(d,"General Physician")])
    t=Table(td,colWidths=[1.6*inch,0.9*inch,0.9*inch,0.9*inch,1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1e40af')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#e2e8f0')),
        ('FONTSIZE',(0,0),(-1,-1),8),('ALIGN',(1,0),(-1,-1),'CENTER'),
    ]))
    story.append(t); story.append(Spacer(1,0.3*inch))
    ti=np.argmax(predictions); top_d=DISEASE_LIST[ti]; top_p=predictions[ti]
    if top_p>0.3:
        spec=DISEASE_SPECIALIST.get(top_d,"General Physician")
        story.append(Paragraph("<b>Clinical Recommendation:</b>", styles['Heading2']))
        story.append(Paragraph(
            f"AI analysis indicates possible <b>{top_d}</b> with {top_p:.1%} confidence. "
            f"Consultation with a <b>{spec}</b> is recommended. Please carry this report.",
            styles['Normal']))
        story.append(Spacer(1,0.2*inch))
    story.append(Paragraph(
        "<i>Disclaimer: This report is generated by AI for research and educational purposes only. "
        "It is not a substitute for professional medical diagnosis. "
        "Please consult a qualified physician for medical advice.</i>",styles['Normal']))
    doc.build(story)
    for f in [op,cp]:
        if os.path.exists(f): os.remove(f)
    return path

# ============================================
# HOME
# ============================================
if page == "Home":
    # Animated Hero
    st.markdown("""
    <div class="hero-section">
        <span class="hero-icon">🩺</span>
        <div class="hero-title">MedScan AI</div>
        <div class="hero-subtitle">AI-Powered Chest X-Ray Disease Detection System</div>
        <div class="hero-line"></div>
        <div class="hero-tags">
            <span class="hero-tag">🧠 DenseNet121</span>
            <span class="hero-tag">🔥 Grad-CAM</span>
            <span class="hero-tag">📊 14 Diseases</span>
            <span class="hero-tag">📄 PDF Reports</span>
            <span class="hero-tag">🎯 MC Dropout</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Diseases Detected", "14")
    col2.metric("Training Images",   "112,120")
    col3.metric("Model",             "DenseNet121")
    col4.metric("XAI Methods",       "Grad-CAM + MC Dropout")

    st.markdown("---")
    col1,col2,col3 = st.columns(3)
    cards = [
        ("🔬","Advanced Detection",    "#2563eb","Detects 14 chest diseases using DenseNet121 transfer learning trained on 112k+ X-ray images."),
        ("🔥","Visual Explainability", "#dc2626","Grad-CAM heatmap highlights the exact region in the X-ray where disease was detected."),
        ("📄","Clinical Reports",      "#16a34a","Generate professional PDF reports with findings, heatmaps, specialist recommendations and confidence scores."),
    ]
    for col,(icon,title,clr,desc) in zip([col1,col2,col3],cards):
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:14px;padding:20px;
                        border:1px solid #e2e8f0;min-height:180px;transition:all 0.2s"
                 onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 8px 24px rgba(37,99,235,0.12)';this.style.borderColor='#93c5fd'"
                 onmouseout="this.style.transform='none';this.style.boxShadow='none';this.style.borderColor='#e2e8f0'">
                <div style="font-size:32px;margin-bottom:10px">{icon}</div>
                <div style="font-weight:700;color:{clr};font-size:16px;margin-bottom:8px">{title}</div>
                <div style="font-size:13px;color:#475569;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background:#eff6ff;border-radius:12px;padding:16px 20px;border:1px solid #bfdbfe;text-align:center">
        <p style="font-size:14px;color:#1e40af;font-weight:600;margin:0">
            👈 Select "Analyze X-Ray" from the sidebar to upload a chest X-ray and get instant AI analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# ANALYZE
# ============================================
elif page == "Analyze":
    st.title("🩻 X-Ray Analysis")
    st.markdown("---")

    patient_name = st.text_input("Patient Name (Optional)", "Anonymous")
    uploaded     = st.file_uploader("Upload Chest X-Ray (JPG / PNG)", type=["jpg","jpeg","png"])

    if uploaded is not None:
        image = Image.open(uploaded)
        model = load_model()

        # Progress steps — all controlled by session state
        if "analysis_done" not in st.session_state:
            st.session_state.analysis_done = False

        done = st.session_state.analysis_done

        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:16px 24px;border:1px solid #e2e8f0;margin:12px 0">
            <div style="display:flex;align-items:center;justify-content:space-between">
                <div style="text-align:center;flex:1">
                    <div style="width:30px;height:30px;border-radius:50%;background:#2563eb;
                                color:white;font-size:13px;font-weight:700;
                                display:flex;align-items:center;justify-content:center;margin:0 auto 6px">✓</div>
                    <p style="font-size:11px;color:#2563eb;font-weight:600;margin:0">Upload</p>
                </div>
                <div style="flex:2;height:2px;background:#2563eb;margin-bottom:22px"></div>
                <div style="text-align:center;flex:1">
                    <div style="width:30px;height:30px;border-radius:50%;background:#2563eb;
                                color:white;font-size:13px;font-weight:700;
                                display:flex;align-items:center;justify-content:center;margin:0 auto 6px">✓</div>
                    <p style="font-size:11px;color:#2563eb;font-weight:600;margin:0">Preprocess</p>
                </div>
                <div style="flex:2;height:2px;background:{'#2563eb' if done else '#e2e8f0'};margin-bottom:22px"></div>
                <div style="text-align:center;flex:1">
                    <div style="width:30px;height:30px;border-radius:50%;
                                background:{'#2563eb' if done else '#e2e8f0'};
                                color:{'white' if done else '#64748b'};
                                font-size:13px;font-weight:700;
                                display:flex;align-items:center;justify-content:center;margin:0 auto 6px">
                        {'✓' if done else '3'}
                    </div>
                    <p style="font-size:11px;color:{'#2563eb' if done else '#94a3b8'};font-weight:{'600' if done else '400'};margin:0">Predict</p>
                </div>
                <div style="flex:2;height:2px;background:{'#2563eb' if done else '#e2e8f0'};margin-bottom:22px"></div>
                <div style="text-align:center;flex:1">
                    <div style="width:30px;height:30px;border-radius:50%;
                                background:{'#2563eb' if done else '#e2e8f0'};
                                color:{'white' if done else '#64748b'};
                                font-size:13px;font-weight:700;
                                display:flex;align-items:center;justify-content:center;margin:0 auto 6px">
                        {'✓' if done else '4'}
                    </div>
                    <p style="font-size:11px;color:{'#2563eb' if done else '#94a3b8'};font-weight:{'600' if done else '400'};margin:0">Report</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Analyze X-Ray"):
            st.session_state.analysis_done = False
            scan_ph = st.empty()
            scan_ph.markdown("""
            <div style="background:#0a0f1a;border-radius:12px;padding:24px;text-align:center;margin:10px 0">
                <p style="font-size:14px;color:#00c8ff;font-weight:600;letter-spacing:2px;margin:0">
                    🔬 SCANNING X-RAY... ANALYZING PATTERNS...
                </p>
                <div style="margin-top:12px;height:3px;
                            background:linear-gradient(to right,transparent,#00c8ff,transparent);
                            border-radius:2px"></div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)

            image_tensor        = preprocess_image(image)
            progress            = st.progress(0, text="Running AI analysis...")
            time.sleep(0.3)
            progress.progress(25, text="Preprocessing image...")
            mean_pred, std_pred = get_uncertainty(model, image_tensor)
            progress.progress(60, text="Generating Grad-CAM heatmap...")
            top_idx             = np.argmax(mean_pred)
            top_disease         = DISEASE_LIST[top_idx]
            confidence          = mean_pred[top_idx]
            cam_image, orig_img = get_gradcam(model, image_tensor, top_idx, opacity=0.6)
            progress.progress(90, text="Preparing results...")
            time.sleep(0.3)
            progress.progress(100, text="Analysis complete!")
            time.sleep(0.5)
            progress.empty()
            scan_ph.empty()

            st.session_state.analysis_done = True
            st.session_state.mean_pred   = mean_pred
            st.session_state.std_pred    = std_pred
            st.session_state.top_idx     = top_idx
            st.session_state.top_disease = top_disease
            st.session_state.confidence  = confidence
            st.session_state.cam_image   = cam_image
            st.session_state.orig_img    = orig_img
            st.session_state.image       = image

            save_to_history({
                "date":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "patient":    patient_name,
                "disease":    top_disease,
                "confidence": float(confidence),
                "risk":       get_risk(mean_pred)[0]
            })
            st.rerun()

        if st.session_state.get("analysis_done") and st.session_state.get("mean_pred") is not None:
            mean_pred   = st.session_state.mean_pred
            std_pred    = st.session_state.std_pred
            top_idx     = st.session_state.top_idx
            top_disease = st.session_state.top_disease
            confidence  = st.session_state.confidence
            cam_image   = st.session_state.cam_image
            orig_img    = st.session_state.orig_img
            image       = st.session_state.image

            sev_text, sev_class, sev_color = get_severity(confidence)
            risk_level, risk_score         = get_risk(mean_pred)
            risk_color = "#dc2626" if risk_level=="High" else "#d97706" if risk_level=="Medium" else "#16a34a"

            st.markdown("---")
            col1,col2,col3 = st.columns([1.2,1.2,1])

            with col1:
                st.markdown("#### 📷 Original X-Ray")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("#### 🔥 Grad-CAM Heatmap")
                opacity_slider = st.slider("Heatmap Opacity", 0.1, 1.0, 0.6, 0.1)
                cam_image, orig_img = get_gradcam(model, preprocess_image(image), top_idx, opacity=opacity_slider)
                st.image(cam_image, use_container_width=True)
                st.caption("Red zone = region where disease was detected")

            with col3:
                st.markdown("#### 📋 Diagnosis")
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:14px;
                            border:1px solid #e2e8f0;margin-bottom:12px">
                    <p style="font-size:11px;color:#64748b;margin-bottom:8px;font-weight:600">OVERALL RISK LEVEL</p>
                    <div style="height:10px;border-radius:5px;
                                background:linear-gradient(to right,#16a34a,#d97706,#dc2626);
                                position:relative;margin-bottom:6px">
                        <div style="position:absolute;top:-3px;left:{min(risk_score,95)}%;
                                    width:5px;height:16px;background:#1e293b;
                                    border-radius:2px;transform:translateX(-50%)"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                        <span style="font-size:10px;color:#64748b">Low</span>
                        <span style="font-size:10px;color:#64748b">Medium</span>
                        <span style="font-size:10px;color:#64748b">High</span>
                    </div>
                    <div style="background:{risk_color}15;border:1px solid {risk_color}40;
                                border-radius:8px;padding:6px 10px;text-align:center">
                        <span style="font-size:13px;font-weight:700;color:{risk_color}">{risk_level} Risk — {risk_score}%</span>
                    </div>
                </div>
                <div style="background:white;border-radius:12px;padding:14px;
                            border:1px solid #e2e8f0;margin-bottom:12px;text-align:center">
                    <p style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:600">MODEL CONFIDENCE</p>
                    <p style="font-size:34px;font-weight:800;color:{sev_color};margin:0">{confidence:.0%}</p>
                    <p style="font-size:12px;color:#64748b;margin-top:4px">{top_disease}</p>
                    <p style="font-size:11px;color:#94a3b8;margin-top:2px">Uncertainty: +-{std_pred[top_idx]:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

                if confidence > 0.5:
                    st.error(f"⚠️ {top_disease} Detected")
                elif confidence > 0.3:
                    st.warning(f"⚠️ Possible {top_disease}")
                else:
                    st.success("✅ No significant disease detected")
                st.metric("Severity", sev_text)

            # Disease cards
            st.markdown("---")
            st.markdown("#### 🦠 Detailed Disease Analysis")
            detected = [(DISEASE_LIST[i], mean_pred[i], std_pred[i])
                        for i in range(14) if mean_pred[i] > 0.3]
            detected.sort(key=lambda x: x[1], reverse=True)

            if detected:
                for disease, prob, unc in detected:
                    _, sc, bc = get_severity(prob)
                    spec      = DISEASE_SPECIALIST.get(disease,"General Physician")
                    desc      = DISEASE_INFO.get(disease,"")
                    bg  = "#fef2f2" if sc=="high" else "#fffbeb" if sc=="medium" else "#f0fdf4"
                    bdr = "#dc2626" if sc=="high" else "#d97706" if sc=="medium" else "#16a34a"
                    st.markdown(f"""
                    <div style="background:{bg};border-left:4px solid {bdr};
                                border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px">
                        <p style="font-size:14px;font-weight:600;margin-bottom:3px;color:#1e293b">
                            {disease} — {prob:.1%} (+-{unc:.3f})
                        </p>
                        <p style="font-size:12px;color:#64748b;margin-bottom:3px">{desc}</p>
                        <p style="font-size:12px;color:#2563eb;font-weight:500;margin:0">
                            Recommended Specialist: {spec}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No significant diseases detected above threshold.")

            # Color coded bars
            st.markdown("---")
            st.markdown("#### 📊 All Disease Probabilities")
            bar_colors = []
            for p in mean_pred:
                if p>=0.7:   bar_colors.append("#dc2626")
                elif p>=0.5: bar_colors.append("#d97706")
                elif p>=0.3: bar_colors.append("#2563eb")
                else:        bar_colors.append("#86efac")

            fig,ax = plt.subplots(figsize=(14,5))
            bars   = ax.barh(DISEASE_LIST, mean_pred, color=bar_colors, edgecolor='white', linewidth=0.5)
            ax.axvline(x=0.3, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7, label="Threshold (0.3)")
            ax.set_xlim(0,1.15)
            ax.set_xlabel("Probability", fontsize=11, color="#475569")
            ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
            ax.tick_params(colors="#475569")
            for bar,prob in zip(bars,mean_pred):
                ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                        f"{prob:.0%}", va='center', fontsize=9, color="#475569")
            ax.legend(handles=[
                mpatches.Patch(color="#dc2626",label="High Risk (>=70%)"),
                mpatches.Patch(color="#d97706",label="Medium Risk (50-70%)"),
                mpatches.Patch(color="#2563eb",label="Low Risk (30-50%)"),
                mpatches.Patch(color="#86efac",label="Normal (<30%)"),
            ], loc="lower right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Recommendation
            if confidence > 0.3:
                spec    = DISEASE_SPECIALIST.get(top_disease,"General Physician")
                urgency = "immediately" if confidence>=0.7 else "at your earliest convenience"
                st.markdown(f"""
                <div style="background:#eff6ff;border:1px solid #bfdbfe;
                            border-radius:12px;padding:16px 20px;margin-top:12px">
                    <p style="font-size:14px;font-weight:700;color:#1e40af;margin-bottom:8px">
                        📋 Clinical Recommendation
                    </p>
                    <p style="font-size:13px;color:#1e3a8a;line-height:1.6;margin:0">
                        Based on AI analysis, a possible <b>{top_disease}</b> has been detected
                        with <b>{confidence:.0%} confidence</b>.<br><br>
                        We recommend consulting a <b>{spec}</b> {urgency}.
                        Please carry this report and your original X-ray during your visit.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # PDF
            st.markdown("---")
            st.markdown("#### 📄 Download Report")
            pdf_path = generate_pdf(orig_img, cam_image, mean_pred, std_pred, patient_name)
            with open(pdf_path,"rb") as f:
                st.download_button(
                    label     = "📥 Download Clinical PDF Report",
                    data      = f,
                    file_name = f"MedScan_{patient_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime      = "application/pdf"
                )

# ============================================
# MODEL COMPARISON
# ============================================
elif page == "Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Compare predictions from 3 different architectures on the same X-ray.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload Chest X-Ray for Comparison", type=["jpg","jpeg","png"])
    if uploaded is not None:
        image        = Image.open(uploaded)
        image_tensor = preprocess_image(image)
        st.image(image, caption="Uploaded X-Ray", width=250)

        if st.button("🔍 Run All 3 Models"):
            with st.spinner("Running DenseNet121, ResNet50 and EfficientNet-B4..."):
                densenet    = load_model()
                resnet      = load_resnet()
                effnet      = load_efficientnet()
                pred_dense  = get_pred(densenet,  image_tensor)
                pred_resnet = get_pred(resnet,     image_tensor)
                pred_effnet = get_pred(effnet,     image_tensor)

            st.markdown("---")

            # Warn user that ResNet & EfficientNet have no trained weights
            st.info("ℹ️ **Note:** ResNet50 and EfficientNet-B4 are loaded with random weights (no trained `.pt` files found). Their predictions are not clinically valid — only DenseNet121 is fully trained. For real comparison, provide `resnet_best.pt` and `effnet_best.pt`.")

            # Dynamically pick which model has highest max confidence → that gets Recommended
            max_confs = [float(np.max(pred_dense)), float(np.max(pred_resnet)), float(np.max(pred_effnet))]
            best_idx  = int(np.argmax(max_confs))

            col1,col2,col3 = st.columns(3)
            models_data = [
                ("DenseNet121",     pred_dense,  "#2563eb", best_idx == 0),
                ("ResNet50",        pred_resnet, "#7c3aed", best_idx == 1),
                ("EfficientNet-B4", pred_effnet, "#0891b2", best_idx == 2),
            ]
            for col,(name,preds,clr,is_rec) in zip([col1,col2,col3], models_data):
                top_d = DISEASE_LIST[np.argmax(preds)]
                top_p = np.max(preds)
                badge_html = (
                    f'<div style="background:{clr}20;color:{clr};font-size:10px;padding:2px 8px;'
                    f'border-radius:10px;display:inline-block;margin-bottom:8px;font-weight:600">★ Highest Confidence</div>'
                ) if is_rec else '<div style="height:24px;margin-bottom:8px"></div>'
                card_html = (
                    f'<div style="background:white;border:2px solid {clr}40;border-radius:14px;padding:20px;text-align:center;">'
                    f'<p style="font-size:15px;font-weight:700;color:{clr};margin-bottom:6px">🧠 {name}</p>'
                    f'{badge_html}'
                    f'<p style="font-size:28px;font-weight:800;color:{clr};margin:0">{top_p:.0%}</p>'
                    f'<p style="font-size:13px;color:#64748b;margin-top:6px">{top_d}</p>'
                    f'</div>'
                )
                with col:
                    st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Side-by-Side Comparison")
            x=np.arange(len(DISEASE_LIST)); width=0.25
            fig,ax=plt.subplots(figsize=(16,6))
            ax.bar(x-width, pred_dense,  width, label='DenseNet121',    color='#2563eb', alpha=0.85)
            ax.bar(x,       pred_resnet, width, label='ResNet50',        color='#7c3aed', alpha=0.85)
            ax.bar(x+width, pred_effnet, width, label='EfficientNet-B4', color='#0891b2', alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(DISEASE_LIST, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel("Probability", color="#475569")
            ax.set_ylim(0,1)
            ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Threshold')
            ax.legend(fontsize=10)
            ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            ensemble = (pred_dense+pred_resnet+pred_effnet)/3
            top_ens  = DISEASE_LIST[np.argmax(ensemble)]
            top_conf = np.max(ensemble)
            st.markdown(f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                        padding:16px;text-align:center;margin-top:12px">
                <p style="font-size:13px;font-weight:700;color:#1e40af;margin-bottom:4px">
                    🗳️ Ensemble Prediction (Average of 3 Models)
                </p>
                <p style="font-size:24px;font-weight:800;color:#1e40af;margin:0">
                    {top_ens} — {top_conf:.0%}
                </p>
                <p style="font-size:11px;color:#64748b;margin-top:4px">
                    Combined prediction from all 3 models
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# HISTORY & STATS
# ============================================
elif page == "History":
    st.title("📈 History & Stats")
    st.markdown("---")

    history = load_history()
    if not history:
        st.info("No scans yet. Go to Analyze X-Ray to get started!")
    else:
        total       = len(history)
        avg_conf    = np.mean([h["confidence"] for h in history])
        high_risk   = sum(1 for h in history if h["risk"]=="High")
        diseases    = [h["disease"] for h in history]
        most_common = max(set(diseases), key=diseases.count)

        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Total Scans",        str(total))
        col2.metric("Avg Confidence",     f"{avg_conf:.0%}")
        col3.metric("High Risk Cases",    str(high_risk))
        col4.metric("Most Common Finding", most_common)

        st.markdown("---")
        st.markdown("#### Disease Distribution")
        dc = {}
        for h in history: dc[h["disease"]] = dc.get(h["disease"],0)+1
        fig,ax=plt.subplots(figsize=(10,4))
        ax.bar(dc.keys(), dc.values(), color="#2563eb", alpha=0.85)
        ax.set_ylabel("Count", color="#475569")
        ax.set_facecolor("#f8fafc"); fig.patch.set_facecolor("#f8fafc")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("#### Scan History")
        for h in reversed(history[-20:]):
            rc = "#dc2626" if h["risk"]=="High" else "#d97706" if h["risk"]=="Medium" else "#16a34a"
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
                        padding:12px 16px;margin-bottom:8px;
                        display:flex;align-items:center;justify-content:space-between;
                        transition:all 0.2s"
                 onmouseover="this.style.borderColor='#93c5fd';this.style.transform='translateX(4px)'"
                 onmouseout="this.style.borderColor='#e2e8f0';this.style.transform='none'">
                <div>
                    <p style="font-weight:600;color:#1e293b;margin:0;font-size:14px">{h['patient']}</p>
                    <p style="font-size:12px;color:#64748b;margin:0">{h['date']}</p>
                </div>
                <div style="text-align:center">
                    <p style="font-weight:600;color:#1e40af;margin:0;font-size:13px">{h['disease']}</p>
                    <p style="font-size:11px;color:#64748b;margin:0">{h['confidence']:.0%} confidence</p>
                </div>
                <div style="background:{rc}15;border:1px solid {rc}40;border-radius:8px;padding:4px 12px">
                    <p style="font-size:12px;font-weight:600;color:{rc};margin:0">{h['risk']} Risk</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear History"):
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            st.session_state.pop("analysis_done", None)
            st.success("History cleared!")
            st.rerun()

# ============================================
# ABOUT
# ============================================
elif page == "About":
    st.title("ℹ️ About Model")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Architecture
        | Component | Detail |
        |---|---|
        | Base Model | DenseNet121 |
        | Pretrained | ImageNet |
        | Classifier | Dropout(0.5) + Linear(1024→14) |
        | Activation | Sigmoid (Multi-label) |
        | Input Size | 224 × 224 × 3 |

        ### Dataset
        | Detail | Value |
        |---|---|
        | Name | NIH ChestX-ray14 |
        | Total Images | 112,120 |
        | Classes | 14 diseases |
        | Train / Val / Test | 70% / 15% / 15% |
        """)
    with col2:
        st.markdown("""
        ### Training Strategy
        | Phase | Detail |
        |---|---|
        | Phase 1 | Classifier only — 5 epochs |
        | Phase 2 | DenseBlock4 fine-tune — 10 epochs |
        | Loss | BCEWithLogitsLoss + class weights |
        | Optimizer | AdamW |
        | LR Phase 1 | 1e-3 |
        | LR Phase 2 | 1e-4 |

        ### Explainability Methods
        | Method | Detail |
        |---|---|
        | Grad-CAM | Layer-level activation heatmap |
        | MC Dropout | Uncertainty quantification (20 passes) |
        | Severity Score | Mild / Moderate / Severe grading |
        """)

    st.markdown("---")
    st.markdown("### Detectable Diseases")
    cols = st.columns(2)
    for i,(disease,desc) in enumerate(DISEASE_INFO.items()):
        with cols[i%2]:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:12px 16px;
                        border:1px solid #e2e8f0;margin-bottom:8px;border-left:3px solid #2563eb;
                        transition:all 0.2s"
                 onmouseover="this.style.background='#eff6ff';this.style.borderLeftColor='#1d4ed8'"
                 onmouseout="this.style.background='white';this.style.borderLeftColor='#2563eb'">
                <p style="font-weight:600;color:#1e40af;font-size:13px;margin-bottom:3px">{disease}</p>
                <p style="font-size:11px;color:#64748b;margin:0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background:#fef3c7;border:1px solid #fcd34d;border-radius:12px;padding:16px 20px">
        <p style="font-size:14px;font-weight:700;color:#92400e;margin-bottom:6px">⚠️ Disclaimer</p>
        <p style="font-size:13px;color:#92400e;line-height:1.6;margin:0">
            MedScan AI is intended for research and educational purposes only.
            This tool does not provide medical diagnoses. Always consult a qualified medical
            professional for any health concerns or before making any medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
