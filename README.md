# ⚽ FIFA World Cup 2026 | Projet Data FrenchTeam

## 🎯 Objectifs
- Prédire le résultat de chaque match (V/N/D) et le nombre de buts
- Prédire le vainqueur du tournoi, top buteur, passeur et MVP

## 👥 Équipe — FrenchTeam
| Membre   | Rôle Scrum    | Rôle Tech       |
|----------|---------------|-----------------|
| Aurélien | Product Owner | Power BI        |
| Cédric   | Scrum Master  | Coordination    |
| Ernest   | Dev Team      | Streamlit Dev   |
| Romain   | Dev Team      | Data Eng. + ML  |

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
