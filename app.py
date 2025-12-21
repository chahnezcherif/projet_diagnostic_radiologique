# app.py – Système d’Aide au Diagnostic Radiologique par Intelligence Artificielle
# Interface explicable avec Grad-CAM++
# Projet réalisé sous la direction du Pr. Hela LTIFI – Décembre 2025

import streamlit as st
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
st.set_page_config(page_title="Aide au Diagnostic Radiologique", layout="centered")

# Titre simple et professionnel
st.title("Système Intelligent d’Aide au Diagnostic Radiologique")
st.caption(
    "Outil d’aide à la décision clinique basé sur l’analyse automatisée "
    "des radiographies thoraciques"
)

# Chargement du modèle
@st.cache_resource
def load_model():
    model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=15)
    model_path = Path(r"C:\Users\Asus\Diagnostic-Radiologique\models\best_convnext_chestxray14.pth")
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

model = load_model()
target_layer = model.stages[-1].blocks[-1].norm

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'Normal'
]

# Classe Grad-CAM++
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self.hooks = []
        self.hooks.append(target_layer.register_forward_hook(self.save_activation))
        self.hooks.append(target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def generate(self, x, class_idx=None):
        self.model.eval()
        logit = self.model(x)
        if class_idx is None:
            class_idx = logit.argmax(dim=1).item()
        score = logit[:, class_idx]
        self.model.zero_grad()
        score.backward()
        gradients = self.gradients[-1].cpu().data.numpy().squeeze()
        activations = self.activations[-1].cpu().data.numpy().squeeze()
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            if w > 0:
                cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Interface – directe et courte
st.markdown("### Module d’Analyse des Radiographies Thoraciques")

uploaded_file = st.file_uploader(
    "Importer une radiographie thoracique du patient",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiographie thoracique du patient", use_container_width=True)  # corrigé : use_container_width au lieu de use_column_width

    # Prétraitement
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_cv, (224, 224))
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Prédiction
    with torch.no_grad():
        logit = model(img_tensor)
        probas = torch.sigmoid(logit).squeeze().numpy()

    # Résultats avec couleurs claires et professionnelles
    st.markdown("### Résultats de l’Analyse Diagnostique")

    sorted_indices = np.argsort(probas)[::-1]
    for idx in sorted_indices:
        name = CLASSES[idx]
        prob = probas[idx]
        if prob > 0.1:
            if prob > 0.5:
                st.markdown(f"<p style='background-color:#ffebee;padding:10px;border-radius:5px;'><strong>{name}</strong> détectée – Probabilité : {prob:.3f}</p>", unsafe_allow_html=True)
            elif prob > 0.3:
                st.markdown(f"<p style='background-color:#fff3e0;padding:10px;border-radius:5px;'><strong>{name}</strong> possible – Probabilité : {prob:.3f}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='background-color:#e8f5e8;padding:10px;border-radius:5px;'>{name} – Probabilité : {prob:.3f}</p>", unsafe_allow_html=True)

    # Explicabilité
    st.markdown("### Analyse des Régions d’Intérêt Diagnostique")

    st.markdown("""
Cette visualisation met en évidence les régions anatomiques ayant contribué
de manière significative à l’analyse du modèle.
Elle permet d’appuyer l’interprétation clinique et de renforcer la
compréhension de la décision assistée par intelligence artificielle.
""")

    if st.button("Afficher les Régions d’Intérêt Diagnostique"):
        with st.spinner("Génération de la carte d’activation en cours..."):
            gradcam = GradCAMPlusPlus(model, target_layer)
            cam, pred_idx = gradcam.generate(img_tensor)
            gradcam.remove_hooks()

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_PLASMA)
            heatmap = np.float32(heatmap) / 255
            superimposed = heatmap * 0.4 + img_resized / 255.0
            superimposed = np.clip(superimposed, 0, 1)

            fig, ax = plt.subplots(1, 2, figsize=(18, 9))
            ax[0].imshow(image)
            ax[0].set_title("Radiographie Thoracique Originale", fontsize=16)
            ax[0].axis("off")
            ax[1].imshow(superimposed)
            ax[1].set_title("Visualisation des Régions d’Intérêt Diagnostique", fontsize=16)  # corrigé : parenthèse fermante ajoutée
            ax[1].axis("off")
            st.pyplot(fig)

        st.success("L’analyse visuelle a été générée avec succès.")
        st.info(
            "Note clinique : Les zones mises en évidence correspondent aux régions "
            "anatomiques ayant contribué de façon significative à l’analyse automatisée. "
            "Ces informations doivent être interprétées par un professionnel de santé."
        )

st.warning(
    "Avertissement : Cet outil constitue une aide à la décision clinique "
    "et ne remplace en aucun cas l’interprétation d’un radiologue qualifié."
)