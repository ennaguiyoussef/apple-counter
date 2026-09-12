# 🍎 Apple Counter — Comptage de pommes sur arbre (YOLOv8)

Détection et comptage automatique de pommes sur des photos de pommiers en verger, à partir d'un modèle **YOLOv8** entraîné sur mesure, servi via une application **Streamlit**.

🔗 **Démo en ligne** : [apple-comptage.streamlit.app](https://apple-comptage.streamlit.app/)

![Statut](https://img.shields.io/badge/status-fonctionnel-brightgreen)
![Modèle](https://img.shields.io/badge/model-YOLOv8s-blue)
![mAP50](https://img.shields.io/badge/mAP50-0.90-success)

---

## 📌 Contexte

Ce projet a été développé dans le cadre d'une préparation à un stage en **AI Engineering / Data Science**, en environ une semaine (sprint intensif). L'objectif : construire un pipeline complet — de la donnée brute jusqu'à une application déployée — pour estimer automatiquement le nombre de pommes sur un arbre à partir d'une simple photo, un cas d'usage utile pour l'estimation de rendement agricole.

## 🎯 Fonctionnalités

- Upload d'une image **ou** saisie d'une URL directe
- Détection des pommes avec bounding boxes et score de confiance
- Comptage automatique du nombre de pommes détectées
- Seuil de confiance ajustable en temps réel
- Interface responsive, pensée pour tenir sur un seul écran sans scroll

## 🗂️ Dataset

| | |
|---|---|
| **Source** | [Roboflow — apple_positivenegativa](https://universe.roboflow.com/appledataset-t37pk/apple_positivenegativa) |
| **Contenu** | Photos de pommiers en verger, forte densité de fruits, occlusions par le feuillage |
| **Classes** | 1 seule classe : `apple` |
| **Format** | YOLOv8 (bounding boxes normalisées) |
| **Split** | 449 images train / 96 val / 96 test |
| **Licence** | CC BY 4.0 |

## 🧠 Modèle & entraînement

Deux itérations ont été entraînées et comparées, sur GPU (Tesla T4, Google Colab) :

| | Baseline | Modèle final |
|---|---|---|
| Architecture | YOLOv8n | **YOLOv8s** |
| Epochs | 20 | 80 |
| Augmentation | Par défaut | Renforcée (rotation, échelle, HSV) |
| mAP50 (val) | 0.81 | **0.87** |
| mAP50 (test) | — | **0.90** |
| mAP50-95 (test) | — | **0.56** |
| Precision | 0.86 | **0.90** |
| Recall | 0.71 | **0.83** |

### Précision du comptage
Sur le split test (96 images, ~37 pommes/image en moyenne, contexte dense avec occlusions) :
- **Écart absolu moyen : 3.8 pommes** (~10 % d'erreur relative)
- Biais moyen : +0.6 pomme (légère tendance à sur-compter)

## 🛠️ Stack technique

- **Détection** : [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- **Entraînement** : Google Colab (GPU T4)
- **Dataset** : Roboflow
- **Interface** : Streamlit
- **Langage** : Python

## 🚀 Lancer le projet en local

```bash
git clone <url-du-repo>
cd apple-counter

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

streamlit run app.py
```

Assure-toi que `best.pt` (les poids du modèle entraîné) se trouve à la racine du projet, à côté de `app.py`.

## 📁 Structure du projet

```
apple-counter/
├── app.py                                    # Application Streamlit
├── best.pt                                    # Poids du modèle YOLOv8s entraîné
├── Detection_and_Comptage_des_Pommes.ipynb     # Notebook d'entraînement (Colab)
├── requirements.txt                            # Dépendances pour l'app Streamlit
├── packages.txt                                 # Dépendances système (Streamlit Cloud)
└── README.md
```

## 🏋️ Réentraîner le modèle

L'entraînement a été réalisé entièrement sur **Google Colab** (GPU T4), pas en local — `best.pt` fourni dans ce repo est directement utilisable par `app.py` sans rien réentraîner.

Si tu veux réentraîner ou ajuster le modèle toi-même :
1. Ouvre [Google Colab](https://colab.research.google.com/)
2. `Fichier` → `Importer un notebook` → sélectionne `Detection_and_Comptage_des_Pommes.ipynb` depuis ce repo
3. Active un runtime GPU (`Exécution` → `Modifier le type d'exécution` → `T4 GPU`)
4. Exécute les cellules dans l'ordre — le notebook télécharge le dataset (Roboflow), entraîne le modèle et sauvegarde les poids

Le notebook n'est pas pensé pour tourner en local (CPU trop lent pour ce volume d'epochs).

## 📈 Pistes d'amélioration

- Augmenter le volume de données d'entraînement (plus d'images en conditions variées)
- Ajouter une estimation du rendement (poids/volume) en plus du comptage
- Exporter le modèle en ONNX pour accélérer l'inférence
- Gérer le comptage sur vidéo (flux caméra en continu, avec tracking pour éviter les doubles comptages)

## 👤 Auteur

**Ennagui Youssef** — Étudiant en double Master IA/Data Science (Maroc–France)
Projet réalisé à des fins de portfolio, dans une optique de recherche de stage en AI Engineering / Data Science.