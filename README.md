# ⚽ FIFA World Cup 2026 | Projet Data FrenchTeam

## 🎯 Objectifs
- Prédire le résultat de chaque match (V/N/D) et le nombre de buts.
- Prédire le vainqueur du tournoi, top buteur, passeur et MVP.

## 👥 Équipe — FrenchTeam
| Membre   | Rôle Scrum    | Rôle Tech       |
|----------|---------------|-----------------|
| Aurélien | Product Owner | Power BI        |
| Cédric   | Scrum Master  | Coordination    |
| Ernest   | Dev Team      | Streamlit Dev   |
| Romain   | Dev Team      | Data Eng. + ML  |

## 🚀 Installation & Lancement

### 1. Récupérer le projet
```bash
git clone https://github.com/romainlafforgue-alt/WorldCup2026.git
cd WorldCup2026
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application Streamlit
```bash
streamlit run streamlit/apply.py
```
L'app s'ouvre automatiquement sur `http://localhost:8501`

### Mettre à jour depuis GitHub (si tu as déjà cloné)
```bash
git pull
```

### 4. Pousser ses modifications sur GitHub
```bash
# Vérifier les fichiers modifiés
git status

# Ajouter tous les fichiers modifiés
git add .

# Créer un commit avec un message descriptif
git commit -m "feat: description de tes modifications"

# Envoyer sur GitHub
git push origin main
```

> 💡 Si c'est ton premier push ou que Git demande tes identifiants :
> ```bash
> git config --global user.name "TonPrénom"
> git config --global user.email "ton@email.com"
> ```
> Puis relance `git push origin main`.

---

## 🗂️ Structure du projet
```
WorldCup2026/
├── data/            ← Placer ici les CSV Kaggle
├── notebooks/       ← Notebooks Jupyter
│   └── modele_niveau1.ipynb
├── models/          ← Modèles ML sauvegardés (.pkl)
├── streamlit/       ← Application Streamlit
├── powerbi/         ← Fichiers Power BI
└── docs/            ← Documentation
```

## 📦 Datasets à placer dans data/
1. `results.csv`      → kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
2. `fifa_ranking.csv` → kaggle.com/datasets/cashncarry/fifaworldranking
3. `train.csv`        → kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-prediction-system
4. `test.csv`         → même source que train.csv

## 📅 Planning — 7 Sprints
| Sprint | Dates          | Objectif               |
|--------|----------------|------------------------|
| S1     | 4 → 8 juin     | Collecte & Setup       |
| S2     | 9 → 13 juin    | Nettoyage & EDA        |
| S3     | 16 → 20 juin   | ML pt.1 — Vainqueur    |
| S4     | 23 → 27 juin   | ML pt.2 — Joueurs      |
| S5     | 30j → 4 juil.  | Streamlit App          |
| S6     | 7 → 11 juil.   | Power BI & Tests       |
| S7     | 14 → 23 juil.  | Finitions & Rendu      |

## 🛠️ Stack
Python · Pandas · Scikit-learn · Streamlit · Power BI · GitHub
