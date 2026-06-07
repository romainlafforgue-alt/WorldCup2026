import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).parent.parent

# ── Constantes du modele V19 ───────────────────────────────────────
DATE_TOURNOI = pd.Timestamp('2026-06-11')

# Noms d'affichage dans les groupes → noms du modele (historique FIFA)
NOM_NORMALISE = {
    'USA': 'United States',
    'Korea Republic': 'South Korea',
    'Czechia': 'Czech Republic',
    'IR Iran': 'Iran',
    "Cote d'Ivoire": 'Ivory Coast',
    'Bosnia': 'Bosnia and Herzegovina',
    'Congo DR': 'DR Congo',
}

PAYS_HOTES_MODELE = ['United States', 'Canada', 'Mexico']

# Points FIFA officiels avril 2026
POINTS_FIFA = {
    'France': 1877.32, 'Spain': 1876.40, 'Argentina': 1874.81,
    'England': 1825.97, 'Portugal': 1763.83, 'Brazil': 1761.16,
    'Netherlands': 1757.87, 'Morocco': 1755.87, 'Belgium': 1734.71,
    'Germany': 1730.37, 'Croatia': 1717.07, 'Colombia': 1693.09,
    'Senegal': 1688.99, 'Mexico': 1681.03, 'United States': 1673.13,
    'Uruguay': 1673.07, 'Japan': 1660.43, 'Switzerland': 1649.40,
    'Ecuador': 1619.20, 'Turkey': 1614.55, 'Sweden': 1598.30,
    'Norway': 1590.12, 'Austria': 1578.44, 'South Korea': 1566.23,
    'Tunisia': 1542.18, 'Algeria': 1538.90, 'Ghana': 1521.44,
    'Egypt': 1518.77, 'Saudi Arabia': 1512.33, 'Iran': 1512.00,
    'Australia': 1508.66, 'Iraq': 1489.21, 'Czech Republic': 1488.00,
    'Scotland': 1487.55, 'Paraguay': 1481.33, 'Ivory Coast': 1479.88,
    'Cape Verde': 1450.00, 'South Africa': 1421.34, 'Canada': 1418.77,
    'Qatar': 1398.22, 'Panama': 1392.11,
    'Bosnia and Herzegovina': 1388.44, 'Jordan': 1342.18,
    'Uzbekistan': 1338.90, 'DR Congo': 1321.44,
    'New Zealand': 1298.77, 'Haiti': 1245.33, 'Curacao': 1198.22,
}
_MOYENNE_FIFA = sum(POINTS_FIFA.values()) / len(POINTS_FIFA)

# Score qualite joueurs 0-100 (valeur marchande + stars mondiales)
QUALITE_JOUEURS = {
    'France': 95, 'England': 90, 'Brazil': 92, 'Spain': 85,
    'Germany': 82, 'Argentina': 88, 'Portugal': 78, 'Netherlands': 75,
    'Belgium': 72, 'Japan': 65, 'Norway': 60, 'Colombia': 62,
    'Croatia': 58, 'Uruguay': 58, 'Morocco': 55, 'Senegal': 55,
    'Mexico': 55, 'United States': 52, 'Switzerland': 50, 'Turkey': 50,
    'Austria': 47, 'Canada': 45, 'South Korea': 45, 'Sweden': 42,
    'Ivory Coast': 38, 'Ecuador': 38, 'Egypt': 38, 'Algeria': 35,
    'Ghana': 35, 'Czech Republic': 35, 'Scotland': 30, 'Tunisia': 30,
    'Paraguay': 30, 'Bosnia and Herzegovina': 30, 'Saudi Arabia': 28,
    'Australia': 25, 'Iran': 25, 'DR Congo': 25, 'Iraq': 22,
    'Jordan': 20, 'South Africa': 20, 'Uzbekistan': 20,
    'Qatar': 18, 'Cape Verde': 15, 'Panama': 15,
    'Haiti': 10, 'New Zealand': 10, 'Curacao': 10,
}
_MOYENNE_QUALITE = sum(QUALITE_JOUEURS.values()) / len(QUALITE_JOUEURS)

# Bonus pour les champions recents (gestion pression grands matchs)
BONUS_CHAMPION = {
    'Argentina': 0.10, 'France': 0.06, 'Germany': 0.04,
    'Spain': 0.03, 'Brazil': 0.02,
}

# Variantes de noms dans l'historique FIFA (pour le calcul d'experience)
_VARIANTES_NOMS = {
    'United States': ['United States', 'USA'],
    'South Korea':   ['South Korea', 'Korea Republic'],
    'Iran':          ['Iran', 'IR Iran'],
    'Czech Republic':['Czech Republic', 'Czechia'],
    'Ivory Coast':   ["Ivory Coast", "Cote d'Ivoire", "Côte d'Ivoire"],
}

_exp_cdm   = {}  # rempli au demarrage depuis results.csv
_forme_cache = {}

def _nom_modele(equipe):
    return NOM_NORMALISE.get(equipe, equipe)

def _get_pts_fifa(nom_modele):
    return POINTS_FIFA.get(nom_modele, _MOYENNE_FIFA)

def _get_qualite(nom_modele):
    return QUALITE_JOUEURS.get(nom_modele, _MOYENNE_QUALITE)

def _get_coefficient(nom_modele):
    noms = _VARIANTES_NOMS.get(nom_modele, [nom_modele])
    nb = max([_exp_cdm.get(n, 0) for n in noms])
    return 0.50 + min(nb / 40, 1.0) * 0.50

def _calculer_forme(equipe, df_off, n=5):
    domicile  = df_off[(df_off['home_team'] == equipe) & (df_off['date'] < DATE_TOURNOI)].tail(n)
    exterieur = df_off[(df_off['away_team'] == equipe) & (df_off['date'] < DATE_TOURNOI)].tail(n)
    matchs    = pd.concat([domicile, exterieur]).sort_values('date').tail(n)
    if len(matchs) == 0:
        return 0.5
    points_total = 0
    poids_total  = 0
    for _, match in matchs.iterrows():
        if match['home_team'] == equipe:
            adversaire = match['away_team']
            victoire   = match['home_score'] > match['away_score']
            nul        = match['home_score'] == match['away_score']
        else:
            adversaire = match['home_team']
            victoire   = match['away_score'] > match['home_score']
            nul        = match['home_score'] == match['away_score']
        poids = _get_pts_fifa(adversaire) / 1500
        if victoire:  points_total += 3 * poids
        elif nul:     points_total += 1 * poids
        poids_total += 3 * poids
    forme_brute = points_total / poids_total if poids_total > 0 else 0.5
    coeff = _get_coefficient(equipe)
    bonus = BONUS_CHAMPION.get(equipe, 0)
    return min(0.5 + (forme_brute - 0.5) * coeff + bonus, 1.0)

def _get_forme(nom_modele, df_off):
    if nom_modele not in _forme_cache:
        _forme_cache[nom_modele] = _calculer_forme(nom_modele, df_off)
    return _forme_cache[nom_modele]

st.set_page_config(
    page_title="⚽ FrenchTeam - FIFA 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS & style ────────────────────────────────────────────────────
def img_html(b64, width="100%", radius="4px"):
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};border-radius:{radius};display:block;margin:auto">'

FIFA_LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAVwAAAIUCAIAAADQUGHYAAABCGlDQ1BJQ0MgUHJvZmlsZQAAeJxjYGA8wQAELAYMDLl5JUVB7k4KEZFRCuwPGBiBEAwSk4sLGHADoKpv1yBqL+viUYcLcKakFicD6Q9ArFIEtBxopAiQLZIOYWuA2EkQtg2IXV5SUAJkB4DYRSFBzkB2CpCtkY7ETkJiJxcUgdT3ANk2uTmlyQh3M/Ck5oUGA2kOIJZhKGYIYnBncAL5H6IkfxEDg8VXBgbmCQixpJkMDNtbGRgkbiHEVBYwMPC3MDBsO48QQ4RJQWJRIliIBYiZ0tIYGD4tZ2DgjWRgEL7AwMAVDQsIHG5TALvNnSEfCNMZchhSgSKeDHkMyQx6QJYRgwGDIYMZAKbWPz9HbOBQAAEAAElEQVR4nOz..."

GROUPES_B64 = ""

st.markdown("""<style>
    .stApp { background:#0f111a !important; }
    h1,h2,h3,h4 { color:#FFFFFF !important; border-bottom:none; padding-bottom:4px; }
    p, li, label { color:#E0E0E0 !important; }
    .stMetric label { color:#aaa !important; }
    .stMetric [data-testid="metric-container"] > div:last-child { color:#34D399 !important; }
    .stButton > button { background:#1a3a6b !important; color:white !important; border-radius:8px !important; font-weight:bold !important; border:none !important; }
    .stButton > button:hover { background:#254f96 !important; }
    .stSelectbox label { color:#aaa !important; font-weight:bold !important; }
    .stDataFrame { border-radius:8px !important; }
    .stInfo    { background:rgba(0,48,135,0.3) !important; border-left:4px solid #4FC3F7 !important; color:#FFFFFF !important; }
    [data-testid="stSidebar"] { background:#0a0c14 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color:#FFFFFF !important; font-weight:bold !important; font-size:14px !important; padding:5px 0 !important; }
    ::-webkit-scrollbar { width:6px; }
    ::-webkit-scrollbar-track { background:#0f111a; }
    ::-webkit-scrollbar-thumb { background:#2a3a5c; border-radius:4px; }
</style>""", unsafe_allow_html=True)

# ── Chargement modèle ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    modele = joblib.load(ROOT / "models/modele_niveau1_final.pkl")
    enc_y  = joblib.load(ROOT / "models/encodeur_cible.pkl")
    df     = pd.read_csv(ROOT / "data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df     = df.dropna(subset=['home_score', 'away_score'])
    df_off = df[df['tournament'] != 'Friendly'].copy()
    # Calcul dynamique de l'experience CdM depuis l'historique complet
    cdm  = df[df['tournament'] == 'FIFA World Cup']
    h    = cdm.groupby('home_team').size().rename('n')
    a    = cdm.groupby('away_team').size().rename('n')
    exp  = pd.concat([h, a]).groupby(level=0).sum().to_dict()
    return modele, enc_y, df_off, exp

modele, enc_y, df_officiel, _exp_load = load_model()
_exp_cdm.update(_exp_load)

# ── Codes ISO drapeaux ────────────────────────────────────────────
FLAGS_ISO = {
    'Argentina':'ar','France':'fr','Belgium':'be','Brazil':'br',
    'England':'gb-eng','Portugal':'pt','Netherlands':'nl','Spain':'es',
    'Croatia':'hr','Germany':'de','USA':'us','Mexico':'mx',
    'Canada':'ca','Morocco':'ma','Senegal':'sn','Japan':'jp',
    'Korea Republic':'kr','Australia':'au','IR Iran':'ir',
    'Saudi Arabia':'sa','Ecuador':'ec','Uruguay':'uy',
    'Colombia':'co','Switzerland':'ch','Austria':'at',
    'Denmark':'dk','Serbia':'rs','Turkey':'tr','Norway':'no',
    'Scotland':'gb-sct','Czechia':'cz','Hungary':'hu',
    'Slovakia':'sk','Albania':'al','Bosnia':'ba',
    'Panama':'pa','Haiti':'ht','Egypt':'eg',
    'Algeria':'dz','Tunisia':'tn',"Cote d'Ivoire":'ci',
    'Ghana':'gh','South Africa':'za','Congo DR':'cd',
    'Jordan':'jo','New Zealand':'nz','Paraguay':'py',
    'Uzbekistan':'uz','Qatar':'qa','Curacao':'cw',
    'Iraq':'iq','Sweden':'se','Cape Verde':'cv',
}

def get_flag(team, size=24):
    iso = FLAGS_ISO.get(team, '')
    if iso:
        return f'<img src="https://flagcdn.com/w40/{iso}.png" width="{size}" style="vertical-align:middle;border-radius:3px">'
    return '🏳️'

# ── Groupes FIFA 2026 ─────────────────────────────────────────────
GROUPES = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Cote d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

equipes = sorted(set([t for teams in GROUPES.values() for t in teams]))

# Tableau ranking avec noms d'affichage (pour compatibilite avec le reste de l'app)
ranking = pd.DataFrame([
    {'team': t, 'fifa_points': POINTS_FIFA.get(_nom_modele(t), _MOYENNE_FIFA)}
    for t in equipes
])

def predire_match(home, away):
    home_m = _nom_modele(home)
    away_m = _nom_modele(away)

    def _features(a, b):
        fa = _get_forme(a, df_officiel)
        fb = _get_forme(b, df_officiel)
        pa = _get_pts_fifa(a)
        pb = _get_pts_fifa(b)
        ef = pa - pb
        qa = _get_qualite(a) / 100
        qb = _get_qualite(b) / 100
        eq = qa - qb
        h = 0.5 if a in PAYS_HOTES_MODELE else (-0.5 if b in PAYS_HOTES_MODELE else 0.0)
        return pd.DataFrame([{
            'ecart_fifa_brut':      ef,
            'ecart_fifa_normalise': ef / 100,
            'ecart_fifa_carre':     (ef / 100) ** 2 * np.sign(ef),
            'ecart_forme_ponderee': (fa * qa) - (fb * qb),
            'ecart_qualite':        eq,
            'ecart_qualite_double': eq * 2,
            'ecart_qualite_carre':  eq ** 2 * np.sign(eq),
            'avantage_hote':        h,
        }])

    classes = list(enc_y.classes_)
    idx_a = classes.index('A_gagne')
    idx_b = classes.index('B_gagne')
    idx_n = classes.index('Nul')

    # Prediction symetrique : moyenne AB et BA pour eviter le biais home/away
    pab = modele.predict_proba(_features(home_m, away_m))[0]
    pba = modele.predict_proba(_features(away_m, home_m))[0]

    p_v = (pab[idx_a] + pba[idx_b]) / 2
    p_d = (pab[idx_b] + pba[idx_a]) / 2
    p_n = (pab[idx_n] + pba[idx_n]) / 2

    total = p_v + p_d + p_n
    return {'V': p_v / total, 'N': p_n / total, 'D': p_d / total}

# ── Simulation Phase de Poule ──────────────────────────────────────
@st.cache_data(show_spinner="Simulation en cours...")
def simuler_phase_poule():
    """Simule tous les matchs et retourne les 2 premiers + 8 meilleurs 3es"""
    classements = {}
    
    for lettre, teams in GROUPES.items():
        stats = {t: {'pts':0,'j':0,'g':0,'n':0,'p':0,'bp':0,'bc':0} for t in teams}
        
        for home, away in combinations(teams, 2):
            try:
                p = predire_match(home, away)
                v, n, d = p.get('V',0), p.get('N',0), p.get('D',0)
                # Score simulé basé sur probabilités
                buts_home = round(1.5 * v + 0.8 * n + 0.3 * d)
                buts_away = round(0.3 * v + 0.8 * n + 1.5 * d)
                
                stats[home]['j'] += 1
                stats[away]['j'] += 1
                stats[home]['bp'] += buts_home
                stats[home]['bc'] += buts_away
                stats[away]['bp'] += buts_away
                stats[away]['bc'] += buts_home
                
                if v > n and v > d:  # Home gagne
                    stats[home]['pts'] += 3; stats[home]['g'] += 1
                    stats[away]['p'] += 1
                elif d > n and d > v:  # Away gagne
                    stats[away]['pts'] += 3; stats[away]['g'] += 1
                    stats[home]['p'] += 1
                else:  # Nul
                    stats[home]['pts'] += 1; stats[home]['n'] += 1
                    stats[away]['pts'] += 1; stats[away]['n'] += 1
            except:
                pass
        
        # Trier: pts > diff buts > buts marqués
        ranking_groupe = sorted(stats.items(),
            key=lambda x: (x[1]['pts'], x[1]['bp']-x[1]['bc'], x[1]['bp']),
            reverse=True)
        classements[lettre] = ranking_groupe
    
    return classements

def extraire_qualifies(classements):
    """Extrait les 32 qualifiés: 2 premiers par groupe + 8 meilleurs 3es"""
    premiers = []
    deuxiemes = []
    troisiemes = []
    
    for lettre in sorted(classements.keys()):
        rg = classements[lettre]
        if len(rg) >= 1:
            premiers.append((lettre, rg[0][0], rg[0][1]))
        if len(rg) >= 2:
            deuxiemes.append((lettre, rg[1][0], rg[1][1]))
        if len(rg) >= 3:
            troisiemes.append((lettre, rg[2][0], rg[2][1]))
    
    # 8 meilleurs 3es (sur 12 groupes, on prend les 8 avec le plus de points)
    troisiemes_sorted = sorted(troisiemes,
        key=lambda x: (x[2]['pts'], x[2]['bp']-x[2]['bc']),
        reverse=True)
    meilleurs_3es = troisiemes_sorted[:8]
    
    return premiers, deuxiemes, meilleurs_3es

def simuler_match_elim(equipe1, equipe2):
    """Pour les elims: redistribution proportionnelle du nul (formule notebook)"""
    try:
        p = predire_match(equipe1, equipe2)
        v, n, d = p.get('V', 0), p.get('N', 0), p.get('D', 0)
        denom   = (v + d) if (v + d) > 0 else 1
        prob_e1 = v + n * v / denom
        prob_e2 = d + n * d / denom
        if prob_e1 >= prob_e2:
            return equipe1, prob_e1, prob_e2
        else:
            return equipe2, prob_e2, prob_e1
    except Exception:
        pts1 = ranking[ranking['team'] == equipe1]['fifa_points'].values
        pts2 = ranking[ranking['team'] == equipe2]['fifa_points'].values
        p1 = float(pts1[0]) if len(pts1) > 0 else 1500
        p2 = float(pts2[0]) if len(pts2) > 0 else 1500
        return (equipe1, 0.6, 0.4) if p1 >= p2 else (equipe2, 0.6, 0.4)

def simuler_tournoi_complet():
    """Simule le tournoi complet et retourne tous les resultats."""
    classements = simuler_phase_poule()

    premiers_d, deuxiemes_d, troisiemes_l = {}, {}, []
    for lettre in sorted(classements.keys()):
        rg = classements[lettre]
        if len(rg) >= 1: premiers_d[lettre]  = rg[0][0]
        if len(rg) >= 2: deuxiemes_d[lettre] = rg[1][0]
        if len(rg) >= 3: troisiemes_l.append((rg[2][0], rg[2][1]['pts'], lettre))
    troisiemes_sorted = sorted(troisiemes_l, key=lambda x: x[1], reverse=True)

    slots_3es = _assigner_3es(troisiemes_sorted)
    def g3(s): return slots_3es.get(s, troisiemes_sorted[0][0])

    r32_paires = [
        (deuxiemes_d['A'], deuxiemes_d['B']),
        (premiers_d['E'],  g3('74')),
        (premiers_d['F'],  deuxiemes_d['C']),
        (premiers_d['C'],  deuxiemes_d['F']),
        (premiers_d['I'],  g3('77')),
        (deuxiemes_d['E'], deuxiemes_d['I']),
        (premiers_d['A'],  g3('79')),
        (premiers_d['L'],  g3('80')),
        (premiers_d['D'],  g3('81')),
        (premiers_d['G'],  g3('82')),
        (deuxiemes_d['K'], deuxiemes_d['L']),
        (premiers_d['H'],  deuxiemes_d['J']),
        (premiers_d['B'],  g3('85')),
        (premiers_d['J'],  deuxiemes_d['H']),
        (premiers_d['K'],  g3('87')),
        (deuxiemes_d['D'], deuxiemes_d['G']),
    ]
    vr32, matchs_r32 = _sim_paires(r32_paires)

    r16_d_paires = [(vr32[1],vr32[4]), (vr32[0],vr32[2]), (vr32[10],vr32[11]), (vr32[8],vr32[9])]
    r16_a_paires = [(vr32[3],vr32[5]), (vr32[6],vr32[7]), (vr32[13],vr32[15]), (vr32[12],vr32[14])]
    qf_d_eq, matchs_r16d = _sim_paires(r16_d_paires)
    qf_a_eq, matchs_r16a = _sim_paires(r16_a_paires)

    qf_d_paires = [(qf_d_eq[0],qf_d_eq[1]), (qf_d_eq[2],qf_d_eq[3])]
    qf_a_paires = [(qf_a_eq[0],qf_a_eq[1]), (qf_a_eq[2],qf_a_eq[3])]
    sf_d, matchs_qfd = _sim_paires(qf_d_paires)
    sf_a, matchs_qfa = _sim_paires(qf_a_paires)

    demi1_w, _, _ = simuler_match_elim(sf_d[0], sf_d[1])
    demi2_w, _, _ = simuler_match_elim(sf_a[0], sf_a[1])
    demi1_l = sf_d[1] if demi1_w == sf_d[0] else sf_d[0]
    demi2_l = sf_a[1] if demi2_w == sf_a[0] else sf_a[0]

    bronze_w, _, _ = simuler_match_elim(demi1_l, demi2_l)
    bronze_l = demi2_l if bronze_w == demi1_l else demi1_l

    champion, _, _ = simuler_match_elim(demi1_w, demi2_w)
    finaliste = demi2_w if champion == demi1_w else demi1_w

    qf_loosers  = [l for _,_,_,l,_,_ in matchs_qfd]  + [l for _,_,_,l,_,_ in matchs_qfa]
    r16_loosers = [l for _,_,_,l,_,_ in matchs_r16d] + [l for _,_,_,l,_,_ in matchs_r16a]
    r32_loosers = [l for _,_,_,l,_,_ in matchs_r32]

    return {
        'champion': champion, 'finaliste': finaliste,
        'bronze': bronze_w,   'bronze_l': bronze_l,
        'sf': (demi1_w, demi2_w, demi1_l, demi2_l),
        'sf_d': sf_d, 'sf_a': sf_a,
        'qf_loosers': qf_loosers, 'r16_loosers': r16_loosers,
        'r32_loosers': r32_loosers,
        'classements': classements,
        'premiers_d': premiers_d, 'deuxiemes_d': deuxiemes_d,
        'troisiemes_sorted': troisiemes_sorted,
        'matchs_r32': matchs_r32, 'matchs_r16d': matchs_r16d,
        'matchs_r16a': matchs_r16a, 'matchs_qfd': matchs_qfd,
        'matchs_qfa': matchs_qfa,
    }

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="height:4px;background:linear-gradient(90deg,#8a2020 33%,#FFFFFF 33%,#FFFFFF 66%,#006633 66%);margin-bottom:12px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:8px 0'>
        <h2 style='color:#FFD700 !important;font-family:Arial Black;margin:8px 0;font-size:1.2rem;border:none'>FRENCH TEAM</h2>
        <p style='color:#aaa !important;font-size:10px;margin:0'>FIFA World Cup 2026 · Prédictions Data</p>
    </div>
    <div style='height:3px;background:linear-gradient(90deg,#8a2020,#003087,#006633);margin:8px 0;border-radius:2px'></div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠 Accueil",
        "⚽ Prédire un match",
        "📊 Phase de poule",
        "🏆 Tableau final",
        "⭐ Favoris"
    ])

    st.markdown("""
    <div style='height:3px;background:linear-gradient(90deg,#8a2020,#003087,#006633);margin:10px 0;border-radius:2px'></div>
    <div style='text-align:center'>
        <p style='color:#8a2020 !important;font-size:11px;font-weight:bold;margin:4px 0'>🇺🇸 USA · 🇨🇦 Canada · 🇲🇽 Mexico</p>
        <p style='color:#aaa !important;font-size:10px;margin:2px 0'>11 juin → 19 juillet 2026</p>
        <p style='color:#34D399 !important;font-size:11px;font-weight:bold;margin:4px 0'>XGBoost · Accuracy 55.47%</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABLE OFFICIELLE FIFA — Attribution des 3es (source : Annexe C FIFA)
# frozenset(groupes_3es) → {slot: groupe_du_3e}
# ════════════════════════════════════════════════════════════════
_TABLE_3ES = {
    frozenset(['E','F','G','H','I','J','K','L']): {'74':'E','77':'I','79':'A','80':'L','81':'D','82':'G','85':'B','87':'K'},
    frozenset(['D','F','G','H','I','J','K','L']): {'74':'H','77':'I','79':'A','80':'L','81':'D','82':'G','85':'B','87':'K'},
    frozenset(['D','E','G','H','I','J','K','L']): {'74':'E','77':'I','79':'A','80':'L','81':'D','82':'H','85':'B','87':'K'},
    frozenset(['D','E','F','H','I','J','K','L']): {'74':'E','77':'I','79':'A','80':'L','81':'D','82':'H','85':'B','87':'K'},
    frozenset(['D','E','F','G','I','J','K','L']): {'74':'E','77':'I','79':'A','80':'L','81':'D','82':'G','85':'B','87':'K'},
    frozenset(['D','E','F','G','H','J','K','L']): {'74':'E','77':'J','79':'A','80':'L','81':'D','82':'H','85':'B','87':'K'},
    frozenset(['D','E','F','G','H','I','K','L']): {'74':'E','77':'I','79':'A','80':'L','81':'D','82':'H','85':'B','87':'K'},
    frozenset(['D','E','F','G','H','I','J','L']): {'74':'E','77':'I','79':'A','80':'L','81':'D','82':'H','85':'B','87':'J'},
    frozenset(['D','E','F','G','H','I','J','K']): {'74':'E','77':'I','79':'A','80':'K','81':'D','82':'H','85':'B','87':'J'},
    frozenset(['C','D','E','F','G','H','I','J']): {'74':'C','77':'I','79':'A','80':'J','81':'D','82':'H','85':'B','87':'E'},
    frozenset(['C','D','E','F','G','H','I','K']): {'74':'C','77':'I','79':'A','80':'K','81':'D','82':'H','85':'B','87':'E'},
    frozenset(['C','D','E','F','G','H','I','L']): {'74':'C','77':'I','79':'A','80':'L','81':'D','82':'H','85':'B','87':'E'},
    frozenset(['C','D','E','F','G','H','J','K']): {'74':'C','77':'J','79':'A','80':'K','81':'D','82':'H','85':'B','87':'E'},
    frozenset(['C','D','E','F','G','H','J','L']): {'74':'C','77':'J','79':'A','80':'L','81':'D','82':'H','85':'B','87':'E'},
    frozenset(['C','D','E','F','G','H','K','L']): {'74':'C','77':'K','79':'A','80':'L','81':'D','82':'H','85':'B','87':'E'},
    frozenset(['C','D','E','F','G','I','J','K']): {'74':'C','77':'I','79':'A','80':'K','81':'D','82':'G','85':'B','87':'J'},
    frozenset(['C','D','E','F','G','I','J','L']): {'74':'C','77':'I','79':'A','80':'L','81':'D','82':'G','85':'B','87':'J'},
    frozenset(['C','D','E','F','H','I','J','K']): {'74':'C','77':'I','79':'A','80':'K','81':'D','82':'H','85':'B','87':'J'},
    frozenset(['C','D','E','F','H','I','J','L']): {'74':'C','77':'I','79':'A','80':'L','81':'D','82':'H','85':'B','87':'J'},
    frozenset(['C','D','E','F','H','I','K','L']): {'74':'C','77':'I','79':'A','80':'L','81':'D','82':'H','85':'B','87':'K'},
    frozenset(['C','E','F','G','H','I','J','K']): {'74':'E','77':'I','79':'C','80':'K','81':'J','82':'H','85':'B','87':'G'},
}

_SLOTS_FALLBACK = [
    ('74',['A','B','C','D','F']), ('77',['C','D','F','G','H']),
    ('79',['C','E','F','H','I']), ('80',['E','H','I','J','K']),
    ('81',['B','E','F','I','J']), ('82',['A','E','H','I','J']),
    ('85',['E','F','G','I','J']), ('87',['D','E','I','J','L']),
]

def _assigner_3es(troisiemes_sorted):
    """Attribution des 3es selon la table officielle FIFA ou fallback."""
    grps_set   = frozenset(grp for _, _, grp in troisiemes_sorted[:8])
    eq_par_grp = {grp: eq for eq, _, grp in troisiemes_sorted[:8]}
    if grps_set in _TABLE_3ES:
        return {slot: eq_par_grp.get(grp, list(eq_par_grp.values())[0])
                for slot, grp in _TABLE_3ES[grps_set].items()}
    dispo = list(troisiemes_sorted[:8])
    slots = {}
    for slot, grps in _SLOTS_FALLBACK:
        for j, (eq, _, grp) in enumerate(dispo):
            if grp in grps:
                slots[slot] = eq; dispo.pop(j); break
        else:
            if dispo: slots[slot] = dispo.pop(0)[0]
    return slots

def _sim_paires(paires):
    """Simule une liste de paires, retourne (gagnants, matchs_info)."""
    gagnants, infos = [], []
    for e1, e2 in paires:
        w, pw, pl = simuler_match_elim(e1, e2)
        l = e2 if w == e1 else e1
        gagnants.append(w)
        infos.append((e1, e2, w, l, pw, pl))
    return gagnants, infos

# ════════════════════════════════════════════════════════════════
# ACCUEIL
# ════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    _, col_c, _ = st.columns([1,2,1])
    with col_c:
        st.markdown(f"""
        <div style='text-align:center;padding:20px 0'>
            <h1 style='font-size:2.5rem;color:#FFD700;font-family:Arial Black;margin:12px 0 4px;border:none'>
                ⚽ FRENCH TEAM
            </h1>
            <p style='color:#34D399 !important;font-size:15px;font-weight:bold;margin:0'>
                FIFA World Cup 2026 — Prédictions Data
            </p>
            <div style='background:linear-gradient(90deg,#8a2020,#003087,#006633);height:4px;border-radius:2px;margin:12px auto;width:60%'></div>
            <p style='color:#aaa !important;font-size:13px'>
                🇺🇸 United States &nbsp;·&nbsp; 🇨🇦 Canada &nbsp;·&nbsp; 🇲🇽 Mexico &nbsp;·&nbsp; 11 juin → 19 juillet 2026
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Accuracy Modèle", "55.47%", "+3.54%")
    c2.metric("📊 Matchs entraînement", "10 732", "depuis 2010")
    c3.metric("🏆 Équipes qualifiées", "48", "FIFA 2026")
    c4.metric("🤖 Algorithme", "XGBoost", "optimisé")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📅 Calendrier")
        st.markdown("""
        <div style='background:#112240;border-left:4px solid #8a2020;border-radius:6px;padding:15px'>
            <p style='color:#E0E0E0 !important'>🟢 <b style='color:#FFD700'>11 juin 2026</b> — Match ouverture (Mexico City)</p>
            <p style='color:#E0E0E0 !important'>📋 <b style='color:#4FC3F7'>11 juin → 4 juillet</b> — Phase de poule (12 groupes)</p>
            <p style='color:#E0E0E0 !important'>⚔️ <b style='color:#4FC3F7'>4 → 18 juillet</b> — Phases éliminatoires</p>
            <p style='color:#E0E0E0 !important'>🏆 <b style='color:#8a2020'>19 juillet 2026</b> — FINALE (New York / NJ)</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("### 🤖 Notre Modèle ML")
        st.markdown("""
        <div style='background:#112240;border-left:4px solid #006633;border-radius:6px;padding:15px'>
            <p style='color:#E0E0E0 !important'>✅ <b style='color:#34D399'>Niveau 1</b> — Victoire / Nul / Défaite par match</p>
            <p style='color:#E0E0E0 !important'>⚽ <b style='color:#4FC3F7'>Features</b> — FIFA pts, forme, buts, h2h</p>
            <p style='color:#E0E0E0 !important'>🌳 <b style='color:#4FC3F7'>Algorithme</b> — XGBoost optimisé</p>
            <p style='color:#E0E0E0 !important'>📈 <b style='color:#FFD700'>Accuracy</b> — 55.47% (+3.54% vs baseline)</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 👥 FrenchTeam — Wild Code School 2026")
    c1, c2, c3, c4 = st.columns(4)
    membres = [
        ("🧑‍✈️","Aurélien","Product Owner","Power BI","#8a2020"),
        ("🔧","Cédric","Scrum Master","Coordination","#4FC3F7"),
        ("💻","Ernest","Dev Team","Streamlit Dev","#34D399"),
        ("📊","Romain","Dev Team","Data Eng. + ML","#FFD700"),
    ]
    for col, (ico, nom, scrum, tech, color) in zip([c1,c2,c3,c4], membres):
        col.markdown(f"""
        <div style='background:#112240;border-top:4px solid {color};border-radius:6px;padding:15px;text-align:center'>
            <div style='font-size:32px'>{ico}</div>
            <h4 style='color:{color} !important;margin:8px 0;border:none'>{nom}</h4>
            <p style='color:#aaa !important;font-size:11px;margin:2px;font-weight:bold'>{scrum}</p>
            <p style='color:#777 !important;font-size:10px'>{tech}</p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PRÉDICTION
# ════════════════════════════════════════════════════════════════
elif page == "⚽ Prédire un match":
    st.markdown("<h1>⚽ Prédire un match</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,2])
    with col1:
        home = st.selectbox("🏠 Équipe domicile", equipes,
                            index=equipes.index("France") if "France" in equipes else 0)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#FFFFFF;font-size:1.4rem;font-weight:bold;padding:14px">VS</div>', unsafe_allow_html=True)
    with col3:
        away = st.selectbox("✈️ Équipe extérieure", equipes,
                            index=equipes.index("Argentina") if "Argentina" in equipes else 1)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 LANCER LA PRÉDICTION", type="primary"):
        if home == away:
            st.error("⚠️ Choisissez deux équipes différentes !")
        else:
            probas = predire_match(home, away)
            v, n, d = probas.get('V',0), probas.get('N',0), probas.get('D',0)
            fh = get_flag(home, 36)
            fa = get_flag(away, 36)

            st.divider()
            st.markdown(f"""
            <h2 style='text-align:center;color:#FFFFFF;border:none'>
                {fh} {home} &nbsp;<span style='color:#8a2020'>VS</span>&nbsp; {away} {fa}
            </h2>
            """, unsafe_allow_html=True)

            favori = home if v >= d and v >= n else (away if d > v and d >= n else "Nul")
            fav_color = "#27843a" if favori == home else ("#8a2020" if favori == away else "#888")
            st.markdown(f"""
            <div style='background:#0D1B2A;border-radius:12px;padding:20px;margin:15px 0'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
                <div style='display:flex;align-items:center;gap:8px'>{fh}<span style='color:#FFFFFF;font-size:16px;font-weight:bold'>{home}</span></div>
                <span style='color:#aaa;font-size:12px'>Probabilites</span>
                <div style='display:flex;align-items:center;gap:8px'><span style='color:#FFFFFF;font-size:16px;font-weight:bold'>{away}</span>{fa}</div>
              </div>
              <div style='display:flex;border-radius:8px;overflow:hidden;height:52px;margin-bottom:10px'>
                <div style='flex:{v:.4f};background:#27843a;display:flex;align-items:center;justify-content:center'>
                  <span style='color:white;font-weight:bold;font-size:16px'>{v:.1%}</span>
                </div>
                <div style='flex:{n:.4f};background:#888;display:flex;align-items:center;justify-content:center'>
                  <span style='color:white;font-weight:bold;font-size:12px'>Nul {n:.1%}</span>
                </div>
                <div style='flex:{d:.4f};background:#8a2020;display:flex;align-items:center;justify-content:center'>
                  <span style='color:white;font-weight:bold;font-size:16px'>{d:.1%}</span>
                </div>
              </div>
              <div style='display:flex;justify-content:space-between'>
                <span style='color:#27843a;font-size:12px;font-weight:bold'>Victoire {home}</span>
                <span style='color:#888;font-size:12px;font-weight:bold'>Match Nul</span>
                <span style='color:#8a2020;font-size:12px;font-weight:bold'>Victoire {away}</span>
              </div>
            </div>
            <div style='text-align:center;padding:8px;background:rgba({("39,132,58" if favori==home else ("138,32,32" if favori==away else "100,100,100"))},0.2);
                        border-radius:8px;border:1px solid {fav_color}'>
              <span style='color:{fav_color};font-weight:bold;font-size:15px'>
                {"Favori : " + favori if favori != "Nul" else "Match tres equilibre"}</span>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PHASE DE POULE
# ════════════════════════════════════════════════════════════════
elif page == "📊 Phase de poule":
    st.markdown("<h1>📊 Phase de poule — 12 Groupes · 48 Équipes</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa'>FIFA World Cup 2026 · XGBoost · Accuracy 55.47%</p>", unsafe_allow_html=True)

    groupe_sel = st.selectbox("🏟️ Sélectionner un groupe",
                               [f"Groupe {g}" for g in GROUPES.keys()])
    lettre = groupe_sel.split(" ")[1]
    teams  = GROUPES[lettre]

    st.markdown(f"### 🏟️ {groupe_sel} — {' · '.join(teams)}")

    # Cartes équipes
    cols = st.columns(len(teams))
    for col, team in zip(cols, teams):
        pts = ranking[ranking['team'] == team]['fifa_points'].values
        pts_val = f"{float(pts[0]):.0f}" if len(pts) > 0 else "N/A"
        col.markdown(f"""
        <div style='background:#112240;border-top:4px solid #8a2020;border-radius:6px;padding:10px;text-align:center'>
            <div style='margin:5px 0'>{get_flag(team, 40)}</div>
            <p style='color:#FFFFFF !important;font-weight:bold;margin:5px 0;font-size:12px'>{team}</p>
            <p style='color:#FFD700 !important;font-size:11px;font-weight:bold'>{pts_val} pts FIFA</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Matchs prédits
    st.markdown("### ⚔️ Matchs prédits")
    resultats = []
    stats_g = {t: {'pts':0,'j':0,'g':0,'n':0,'p':0,'bp':0,'bc':0} for t in teams}

    for home, away in combinations(teams, 2):
        try:
            probas = predire_match(home, away)
            v, n, d = probas.get('V',0), probas.get('N',0), probas.get('D',0)
            buts_home = round(1.5 * v + 0.8 * n + 0.3 * d)
            buts_away = round(0.3 * v + 0.8 * n + 1.5 * d)

            stats_g[home]['j'] += 1; stats_g[away]['j'] += 1
            stats_g[home]['bp'] += buts_home; stats_g[home]['bc'] += buts_away
            stats_g[away]['bp'] += buts_away; stats_g[away]['bc'] += buts_home

            if v > n and v > d:
                pred = f"✅ {home}"
                stats_g[home]['pts'] += 3; stats_g[home]['g'] += 1; stats_g[away]['p'] += 1
            elif d > n and d > v:
                pred = f"✅ {away}"
                stats_g[away]['pts'] += 3; stats_g[away]['g'] += 1; stats_g[home]['p'] += 1
            else:
                pred = "🤝 Nul"
                stats_g[home]['pts'] += 1; stats_g[home]['n'] += 1
                stats_g[away]['pts'] += 1; stats_g[away]['n'] += 1

            resultats.append({
                "Match":          f"{home} vs {away}",
                "% Victoire 1":   f"{v:.0%}",
                "% Nul":          f"{n:.0%}",
                "% Victoire 2":   f"{d:.0%}",
                "Score prédit":   f"{buts_home} - {buts_away}",
                "Prédiction":     pred
            })
        except:
            pass

    if resultats:
        st.dataframe(pd.DataFrame(resultats), use_container_width=True, hide_index=True)

    # Classement groupe
    st.markdown("### 📋 Classement simulé du groupe")
    rg_sorted = sorted(stats_g.items(),
        key=lambda x: (x[1]['pts'], x[1]['bp']-x[1]['bc'], x[1]['bp']),
        reverse=True)

    cl_rows = []
    for rank, (team, s) in enumerate(rg_sorted, 1):
        qualif = "🟢 Qualifié" if rank <= 2 else ("🟡 Possible" if rank == 3 else "🔴 Éliminé")
        cl_rows.append({
            "Pos": rank,
            "Équipe": team,
            "J": s['j'], "G": s['g'], "N": s['n'], "D": s['p'],
            "BP": s['bp'], "BC": s['bc'], "Diff": s['bp']-s['bc'],
            "Pts": s['pts'],
            "Statut": qualif
        })
    st.dataframe(pd.DataFrame(cl_rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TABLEAU FINAL — BRACKET OFFICIEL FIFA 2026
# ════════════════════════════════════════════════════════════════
elif page == "🏆 Tableau final":
    st.markdown("<h1>🏆 Tableau final — Bracket FIFA 2026</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#aaa'>Format officiel FIFA 2026 — Round of 32 → Finale.
    Attribution des 3es via la table officielle FIFA. Bracket Dallas / Atlanta.</p>
    """, unsafe_allow_html=True)

    with st.spinner("Simulation du tournoi..."):
        res = simuler_tournoi_complet()
    classements       = res['classements']
    premiers_d        = res['premiers_d']
    deuxiemes_d       = res['deuxiemes_d']
    troisiemes_sorted = res['troisiemes_sorted']

    # Afficher les qualifies
    st.divider()
    st.markdown("## ⚔️ Phases Eliminatoires — Format Officiel FIFA 2026")

    matchs_r32         = res['matchs_r32']
    matchs_r16d        = res['matchs_r16d']
    matchs_r16a        = res['matchs_r16a']
    matchs_qfd         = res['matchs_qfd']
    matchs_qfa         = res['matchs_qfa']
    sf_d               = res['sf_d']
    sf_a               = res['sf_a']
    demi1_w, demi2_w, demi1_l, demi2_l = res['sf']
    bronze_w           = res['bronze']
    champion           = res['champion']
    finaliste          = res['finaliste']

    # ── Helpers pour les cartes du bracket ──────────────────────────────
    def mc(e1, e2, w, bc="#4FC3F7"):
        c1 = "#FFD700" if w == e1 else "#444"
        c2 = "#FFD700" if w == e2 else "#444"
        chk1 = "<span style='color:#34D399;font-size:9px'>&#10003;</span>" if w == e1 else ""
        chk2 = "<span style='color:#34D399;font-size:9px'>&#10003;</span>" if w == e2 else ""
        f1, f2 = get_flag(e1, 14), get_flag(e2, 14)
        return (
            f"<div style='background:#0D1B2A;border:1px solid {bc};border-radius:4px;"
            f"padding:4px 6px;min-width:138px;max-width:152px;margin:1px 0'>"
            f"<div style='display:flex;align-items:center;gap:3px'>"
            f"{f1}<span style='color:{c1};font-size:9px;flex:1;overflow:hidden;white-space:nowrap'>{e1[:13]}</span>{chk1}"
            f"</div><div style='height:1px;background:#1a2a3a;margin:2px 0'></div>"
            f"<div style='display:flex;align-items:center;gap:3px'>"
            f"{f2}<span style='color:{c2};font-size:9px;flex:1;overflow:hidden;white-space:nowrap'>{e2[:13]}</span>{chk2}"
            f"</div></div>"
        )

    def mc_big(e1, e2, w, bc="#FFD700"):
        c1 = "#FFD700" if w == e1 else "#777"
        c2 = "#FFD700" if w == e2 else "#777"
        chk1 = "<span style='color:#34D399;font-weight:bold'>&#10003;</span>" if w == e1 else ""
        chk2 = "<span style='color:#34D399;font-weight:bold'>&#10003;</span>" if w == e2 else ""
        f1, f2 = get_flag(e1, 20), get_flag(e2, 20)
        return (
            f"<div style='background:#0D1B2A;border:2px solid {bc};border-radius:6px;"
            f"padding:8px 10px;min-width:165px;'>"
            f"<div style='display:flex;align-items:center;gap:5px'>"
            f"{f1}<span style='color:{c1};font-size:12px;font-weight:bold;flex:1'>{e1}</span>{chk1}"
            f"</div><div style='height:1px;background:#1a2a3a;margin:4px 0'></div>"
            f"<div style='display:flex;align-items:center;gap:5px'>"
            f"{f2}<span style='color:{c2};font-size:12px;font-weight:bold;flex:1'>{e2}</span>{chk2}"
            f"</div></div>"
        )

    # ── Construire le bracket HTML ───────────────────────────────────────
    r32d = "".join(mc(e1, e2, w, "#4FC3F7") for e1, e2, w, *_ in matchs_r32[:8])
    r16d = "".join(mc(e1, e2, w, "#FF9800") for e1, e2, w, *_ in matchs_r16d)
    qfd_h = "".join(mc(e1, e2, w, "#FFD700") for e1, e2, w, *_ in matchs_qfd)
    sfd_h = mc_big(sf_d[0], sf_d[1], demi1_w, "#8a2020")

    r32a = "".join(mc(e1, e2, w, "#4FC3F7") for e1, e2, w, *_ in matchs_r32[8:])
    r16a = "".join(mc(e1, e2, w, "#FF9800") for e1, e2, w, *_ in matchs_r16a)
    qfa_h = "".join(mc(e1, e2, w, "#FFD700") for e1, e2, w, *_ in matchs_qfa)
    sfa_h = mc_big(sf_a[0], sf_a[1], demi2_w, "#8a2020")

    fin_h = mc_big(demi1_w, demi2_w, champion, "#FFD700")
    bro_h = mc_big(demi1_l, demi2_l, bronze_w, "#CD7F32")

    bh = 480  # hauteur totale du bracket (px) — aligne automatiquement les rounds

    def col(content, justify="space-around"):
        return (f"<div style='display:flex;flex-direction:column;"
                f"justify-content:{justify};height:{bh}px;'>{content}</div>")

    labels = (
        "<div style='display:flex;gap:3px;margin-bottom:6px;padding:0 2px'>"
        "<div style='min-width:154px;text-align:center'><span style='color:#4FC3F7;font-size:10px;font-weight:bold'>R32 · DALLAS</span></div>"
        "<div style='min-width:154px;text-align:center'><span style='color:#FF9800;font-size:10px;font-weight:bold'>R16</span></div>"
        "<div style='min-width:154px;text-align:center'><span style='color:#FFD700;font-size:10px;font-weight:bold'>QF</span></div>"
        "<div style='min-width:175px;text-align:center'><span style='color:#8a2020;font-size:10px;font-weight:bold'>DEMI-FINALES</span></div>"
        "<div style='min-width:185px;text-align:center'><span style='color:#FFD700;font-size:11px;font-weight:bold'>FINALE · BRONZE</span></div>"
        "<div style='min-width:175px;text-align:center'><span style='color:#8a2020;font-size:10px;font-weight:bold'>DEMI-FINALES</span></div>"
        "<div style='min-width:154px;text-align:center'><span style='color:#FFD700;font-size:10px;font-weight:bold'>QF</span></div>"
        "<div style='min-width:154px;text-align:center'><span style='color:#FF9800;font-size:10px;font-weight:bold'>R16</span></div>"
        "<div style='min-width:154px;text-align:center'><span style='color:#4FC3F7;font-size:10px;font-weight:bold'>R32 · ATLANTA</span></div>"
        "</div>"
    )
    center = (
        f"<div style='display:flex;flex-direction:column;justify-content:center;"
        f"align-items:center;height:{bh}px;min-width:185px;padding:0 8px'>"
        f"<p style='color:#FFD700;font-size:11px;font-weight:bold;text-align:center;margin:0 0 5px'>FINALE</p>"
        f"{fin_h}"
        f"<p style='color:#CD7F32;font-size:10px;font-weight:bold;text-align:center;margin:22px 0 5px'>BRONZE</p>"
        f"{bro_h}"
        f"</div>"
    )
    bracket_html = (
        "<div style='overflow-x:auto;background:#071020;border-radius:10px;padding:15px'>"
        f"{labels}"
        f"<div style='display:flex;gap:3px;'>"
        f"{col(r32d)}{col(r16d)}{col(qfd_h)}{col(sfd_h,'center')}"
        f"{center}"
        f"{col(sfa_h,'center')}{col(qfa_h)}{col(r16a)}{col(r32a)}"
        "</div></div>"
    )
    st.markdown(bracket_html, unsafe_allow_html=True)

    # ── Podium final ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("## PODIUM FINAL — FIFA World Cup 2026")
    pc1, pc2, pc3 = st.columns(3)
    podium = [
        (pc1, "CHAMPION",   champion,  "#FFD700", "2.5rem"),
        (pc2, "Finaliste",  finaliste, "#C0C0C0", "1.8rem"),
        (pc3, "3eme place", bronze_w,  "#CD7F32", "1.5rem"),
    ]
    for col_p, titre, team, color, font_size in podium:
        col_p.markdown(f"""
        <div style='background:#112240;border:3px solid {color};border-radius:12px;
                    padding:20px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.5)'>
            <p style='color:{color};font-size:13px;font-weight:bold;margin:0'>{titre}</p>
            <div style='margin:12px 0'>{get_flag(team,56)}</div>
            <h3 style='color:{color};font-size:{font_size};margin:8px 0;border:none'>{team}</h3>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# FAVORIS
# ════════════════════════════════════════════════════════════════
elif page == "⭐ Favoris":
    st.markdown("<h1>Favoris — FIFA World Cup 2026</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa'>Classement predictif base sur la simulation complete du tournoi</p>", unsafe_allow_html=True)

    with st.spinner("Chargement simulation..."):
        res_fav = simuler_tournoi_complet()

    champion_f  = res_fav['champion']
    finaliste_f = res_fav['finaliste']
    bronze_f    = res_fav['bronze']
    bronze_l_f  = res_fav['bronze_l']
    sf1, sf2, sf3, sf4 = res_fav['sf']
    qf_l        = res_fav['qf_loosers']

    # Classement par performance dans le tournoi (top 12 affiches)
    tournoi_rank = [
        (champion_f,  "Champion du Monde",   "#FFD700", "1"),
        (finaliste_f, "Finaliste",            "#C0C0C0", "2"),
        (bronze_f,    "3eme place",           "#CD7F32", "3"),
        (bronze_l_f,  "4eme place",           "#4FC3F7", "4"),
    ] + [
        (t, "Demi-finaliste", "#3a7bd5", str(5 + i))
        for i, t in enumerate([x for x in (sf1, sf2, sf3, sf4) if x not in (champion_f, finaliste_f, bronze_f, bronze_l_f)])
    ] + [
        (t, "Quart de finaliste", "#2a4a7f", str(7 + i))
        for i, t in enumerate(qf_l[:4])
    ]

    cols = st.columns(4)
    for idx, (team, label, color, rang) in enumerate(tournoi_rank[:12]):
        cols[idx % 4].markdown(f"""
        <div style='background:#0D1B2A;border:1px solid #1e2d3d;border-top:3px solid {color};
                    border-radius:8px;padding:16px 10px;text-align:center;margin:5px 0'>
            <div style='display:inline-flex;align-items:center;justify-content:center;
                        background:{color}22;border:1px solid {color};border-radius:50%;
                        width:28px;height:28px;font-size:12px;font-weight:bold;
                        color:{color};margin-bottom:10px'>{rang}</div>
            <div style='margin:8px 0'>{get_flag(team, 44)}</div>
            <p style='color:#FFFFFF;font-size:13px;font-weight:bold;margin:6px 0'>{team}</p>
            <p style='color:{color};font-size:10px;font-weight:bold;margin:3px 0'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Comparer deux equipes")
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.selectbox("Equipe 1", equipes,
                           index=equipes.index("France") if "France" in equipes else 0)
    with c2:
        t2 = st.selectbox("Equipe 2", equipes,
                           index=equipes.index("Argentina") if "Argentina" in equipes else 1)

    if st.button("COMPARER CES DEUX EQUIPES", type="primary"):
        if t1 == t2:
            st.error("Choisissez deux equipes differentes !")
        else:
            probas = predire_match(t1, t2)
            v, n, d = probas.get('V',0), probas.get('N',0), probas.get('D',0)
            fh2, fa2 = get_flag(t1, 32), get_flag(t2, 32)
            favori2 = t1 if v >= d and v >= n else (t2 if d > v and d >= n else "Nul")
            fav_c2  = "#27843a" if favori2 == t1 else ("#8a2020" if favori2 == t2 else "#888")
            st.markdown(f"""
            <div style='background:#0D1B2A;border-radius:12px;padding:20px;margin:15px 0'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
                <div style='display:flex;align-items:center;gap:8px'>{fh2}<span style='color:#FFFFFF;font-size:16px;font-weight:bold'>{t1}</span></div>
                <span style='color:#555;font-size:12px'>vs</span>
                <div style='display:flex;align-items:center;gap:8px'><span style='color:#FFFFFF;font-size:16px;font-weight:bold'>{t2}</span>{fa2}</div>
              </div>
              <div style='display:flex;border-radius:8px;overflow:hidden;height:48px;margin-bottom:10px'>
                <div style='flex:{v:.4f};background:#27843a;display:flex;align-items:center;justify-content:center'>
                  <span style='color:white;font-weight:bold;font-size:15px'>{v:.1%}</span>
                </div>
                <div style='flex:{n:.4f};background:#555;display:flex;align-items:center;justify-content:center'>
                  <span style='color:white;font-weight:bold;font-size:11px'>Nul {n:.1%}</span>
                </div>
                <div style='flex:{d:.4f};background:#8a2020;display:flex;align-items:center;justify-content:center'>
                  <span style='color:white;font-weight:bold;font-size:15px'>{d:.1%}</span>
                </div>
              </div>
              <div style='text-align:center;padding:6px;background:rgba(0,0,0,0.3);border-radius:6px;border:1px solid {fav_c2}'>
                <span style='color:{fav_c2};font-weight:bold;font-size:14px'>
                  {"Favori : " + favori2 if favori2 != "Nul" else "Match equilibre"}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
