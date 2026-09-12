import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import base64

st.set_page_config(
    page_title="🍎 Apple Counter",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- STYLE ----------
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1400px;}

    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e63946, #f77f00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #6c757d;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }

    .scroll-container {
        display: flex;
        overflow-x: auto;
        gap: 20px;
        padding-bottom: 15px;
    }
    
    .scroll-container img {
        max-height: 62vh;
        width: auto;
        object-fit: contain;
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.15);
    }

    .scroll-container::-webkit-scrollbar { height: 10px; }
    .scroll-container::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 5px; }
    .scroll-container::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 5px; }
    .scroll-container::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }

    .count-card {
        background: linear-gradient(135deg, #e63946, #f77f00);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 14px rgba(230,57,70,0.3);
        margin-bottom: 15px;
    }
    .count-number { font-size: 3rem; font-weight: 800; line-height: 1; }
    .count-label { font-size: 0.9rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #495057;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    section[data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# ---------- MODELE ----------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------- FONCTION UTILE ----------
def image_to_base64(img):
    buffered = BytesIO()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    conf_threshold = st.slider("Seuil de confiance", 0.1, 0.9, 0.25, 0.05)

    st.markdown("### 📷 Source de l'image")
    source = st.radio("", ["📁 Upload", "🔗 URL"], label_visibility="collapsed")

    image = None
    if source == "📁 Upload":
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        url = st.text_input("", placeholder="https://exemple.com/pommier.jpg", label_visibility="collapsed")
        if url:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.markdown("---")
    st.caption("Modèle : YOLOv8s · mAP50 = 0.90")

# ---------- HEADER ----------
st.markdown('<p class="app-title">🍎 Apple Counter</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Détection et comptage de pommes sur arbre — YOLOv8</p>', unsafe_allow_html=True)

# ---------- CONTENU PRINCIPAL ----------
if image is not None:
    with st.spinner("Détection en cours..."):
        results = model.predict(source=np.array(image), conf=conf_threshold, verbose=False)[0]

    n_apples = len(results.boxes)
    annotated = results.plot()[..., ::-1]

    col_img, col_info = st.columns([3, 1])

    with col_img:
        orig_b64 = image_to_base64(image)
        annot_b64 = image_to_base64(annotated)
        
        st.markdown(f"""
        <div class="scroll-container">
            <div style="text-align: center; min-width: 45%;">
                <p style="color: #6c757d; margin-bottom: 5px; font-weight: bold;">Image Originale</p>
                <img src="data:image/jpeg;base64,{orig_b64}" alt="Image originale">
            </div>
            <div style="text-align: center; min-width: 45%;">
                <p style="color: #e63946; margin-bottom: 5px; font-weight: bold;">Résultat Détecté</p>
                <img src="data:image/jpeg;base64,{annot_b64}" alt="Image détectée">
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class="count-card">
            <div class="count-number">{n_apples}</div>
            <div class="count-label">Pommes détectées</div>
        </div>
        """, unsafe_allow_html=True)

        if len(results.boxes.conf) > 0:
            avg_conf = float(results.boxes.conf.mean())
            st.metric("Confiance moyenne", f"{avg_conf:.0%}")
            
            # Bouton de téléchargement de l'image
            buf = BytesIO()
            Image.fromarray(annotated).save(buf, format="JPEG")
            st.download_button(
                label="📥 Télécharger le résultat",
                data=buf.getvalue(),
                file_name="pommes_detectees.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

    # ---------- NOUVELLE SECTION: ANALYSE DETAILLEE ----------
    confidences = results.boxes.conf.cpu().numpy()
    
    if len(confidences) > 0:
        st.markdown('<div class="section-title">📊 Analyse des détections</div>', unsafe_allow_html=True)
        col_chart, col_data = st.columns([2, 1])
        
        with col_chart:
            # Création d'un histogramme des scores de confiance
            hist, bins = np.histogram(confidences, bins=10, range=(0.0, 1.0))
            hist_df = pd.DataFrame({
                "Nombre de pommes": hist,
                "Plage de confiance": [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]
            }).set_index("Plage de confiance")
            
            st.caption("Distribution de la confiance")
            st.bar_chart(hist_df, height=250)
            
        with col_data:
            # Affichage du tableau détaillé
            df_details = pd.DataFrame({
                "Pomme N°": range(1, len(confidences) + 1),
                "Score de confiance": [f"{c:.1%}" for c in confidences]
            })
            st.caption("Données brutes")
            st.dataframe(df_details, use_container_width=True, height=250, hide_index=True)

else:
    st.info("👈 Choisis une image dans la barre latérale pour commencer.")
    st.markdown("### 🍎 Exemples d'images à tester")
    st.caption("Copie l'une de ces URLs et colle-la dans le champ **🔗 URL** de la barre latérale :")

    examples = [
        {
            "title": "Exemple 1 : Pommier chargé",
            "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQa03jgrypAbpT0wMnpwYswderel3mKJ8Up_eod3y42jQ&s=10"
        },
        {
            "title": "Exemple 2 : Pommes en gros plan",
            "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpWubGL3e3dO0i2mKvFq1mgr0aUoTza-SiH7JDcmvQ1g&s=10"
        },
        {
            "title": "Exemple 3 : Branche avec feuillage",
            "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT6uIYVgse0IKEesIy3RSa0Y_0mtF1bcBW6N3Sq2pYUKw&s=10"
        }
    ]

    cols = st.columns(3)
    for idx, (col, ex) in enumerate(zip(cols, examples), 1):
        with col:
            st.markdown(f"**{ex['title']}**")
            st.image(ex["url"], use_container_width=True)
            st.code(ex["url"], language="text")