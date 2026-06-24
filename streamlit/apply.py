import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import textwrap
import base64

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
    'Ivory Coast': 40, 'Ecuador': 38, 'Egypt': 38, 'Algeria': 35,
    'Ghana': 35, 'Czech Republic': 35, 'Scotland': 33, 'Tunisia': 30,
    'Paraguay': 30, 'Bosnia and Herzegovina': 30, 'Saudi Arabia': 28,
    'Australia': 25, 'Iran': 25, 'DR Congo': 25, 'Iraq': 22,
    'Jordan': 20, 'South Africa': 20, 'Uzbekistan': 20,
    'Qatar': 18, 'Cape Verde': 15, 'Panama': 15,
    'New Zealand': 10, 'Haiti': 8, 'Curacao': 7,
}
_MOYENNE_QUALITE = sum(QUALITE_JOUEURS.values()) / len(QUALITE_JOUEURS)

# Coefficient d'experience CdM (poids du palmares dans la forme recente)
COEFFICIENT_EXPERIENCE = {
    'France': 1.00, 'Brazil': 1.00, 'Germany': 1.00, 'Spain': 1.00,
    'Argentina': 1.00, 'England': 1.00, 'Portugal': 1.00,
    'Netherlands': 1.00, 'Belgium': 1.00, 'Croatia': 1.00,
    'Uruguay': 1.00, 'Mexico': 1.00, 'United States': 1.00,
    'Canada': 0.95, 'Japan': 0.95, 'Morocco': 0.92, 'Senegal': 0.92,
    'South Korea': 0.92, 'Colombia': 0.90, 'Switzerland': 0.90,
    'Iran': 0.88, 'Saudi Arabia': 0.88, 'Ghana': 0.88, 'Ivory Coast': 0.88,
    'Ecuador': 0.85, 'Tunisia': 0.85, 'Algeria': 0.85, 'Austria': 0.85,
    'Sweden': 0.85, 'Turkey': 0.85, 'Egypt': 0.85, 'Paraguay': 0.85,
    'Czech Republic': 0.85, 'Norway': 0.50, 'Scotland': 0.52,
    'Australia': 0.48, 'Iraq': 0.70, 'Jordan': 0.60, 'Uzbekistan': 0.60,
    'Panama': 0.68, 'Curacao': 0.35, 'Cape Verde': 0.60,
    'New Zealand': 0.60, 'Haiti': 0.40, 'South Africa': 0.72,
    'Bosnia and Herzegovina': 0.75, 'DR Congo': 0.68, 'Qatar': 0.72,
}

# Zone de qualification -> credibilite de la confederation
ZONE_QUALIF = {
    'France':'UEFA','Germany':'UEFA','Spain':'UEFA','England':'UEFA',
    'Portugal':'UEFA','Netherlands':'UEFA','Belgium':'UEFA','Croatia':'UEFA',
    'Switzerland':'UEFA','Austria':'UEFA','Turkey':'UEFA','Scotland':'UEFA',
    'Norway':'UEFA','Sweden':'UEFA','Bosnia and Herzegovina':'UEFA','Czech Republic':'UEFA',
    'Argentina':'CONMEBOL','Brazil':'CONMEBOL','Colombia':'CONMEBOL',
    'Uruguay':'CONMEBOL','Ecuador':'CONMEBOL','Paraguay':'CONMEBOL',
    'Morocco':'CAF','Senegal':'CAF','Egypt':'CAF','Ivory Coast':'CAF',
    'Algeria':'CAF','Tunisia':'CAF','Ghana':'CAF','South Africa':'CAF',
    'DR Congo':'CAF','Cape Verde':'CAF',
    'United States':'CONCACAF','Mexico':'CONCACAF','Canada':'CONCACAF',
    'Panama':'CONCACAF','Haiti':'CONCACAF','Curacao':'CONCACAF',
    'Japan':'AFC','South Korea':'AFC','Iran':'AFC','Saudi Arabia':'AFC',
    'Australia':'AFC','Uzbekistan':'AFC','Iraq':'AFC','Jordan':'AFC','Qatar':'AFC',
    'New Zealand':'OFC',
}
CREDIBILITE_ZONE = {
    'UEFA':1.00,'CONMEBOL':0.95,'CAF':0.75,'AFC':0.70,'CONCACAF':0.50,'OFC':0.40,
}

# Bonus/plancher pour les grandes nations (gestion pression grands matchs)
_BONUS_FORME = {'Argentina':0.10,'France':0.08,'Brazil':0.10,'Germany':0.06,'Spain':0.05}
_PLANCHER_FORME = {'Brazil':0.75,'Germany':0.72,'Spain':0.80,'France':0.82,
                    'Argentina':0.85,'England':0.78,'Portugal':0.75}

_forme_cache = {}

def _nom_modele(equipe):
    return NOM_NORMALISE.get(equipe, equipe)

def _get_pts_fifa(nom_modele):
    return POINTS_FIFA.get(nom_modele, _MOYENNE_FIFA)

def _get_qualite(nom_modele):
    return QUALITE_JOUEURS.get(nom_modele, _MOYENNE_QUALITE)

def _get_coefficient(nom_modele):
    return COEFFICIENT_EXPERIENCE.get(nom_modele, 0.80)

def _get_credibilite(nom_modele):
    return CREDIBILITE_ZONE.get(ZONE_QUALIF.get(nom_modele, 'UEFA'), 0.75)

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
    coeff_final = _get_coefficient(equipe) * _get_credibilite(equipe)
    bonus = _BONUS_FORME.get(equipe, 0)
    forme = min(0.5 + (forme_brute - 0.5) * coeff_final + bonus, 1.0)
    return max(forme, _PLANCHER_FORME.get(equipe, 0.0))

def _get_forme(nom_modele, df_off):
    if nom_modele not in _forme_cache:
        _forme_cache[nom_modele] = _calculer_forme(nom_modele, df_off)
    return _forme_cache[nom_modele]

# ── Score de force composite — coeur du modele hybride v3 ──────────
# FORCE = 0.50 x qualite_squad + 0.35 x classement_FIFA + 0.15 x forme_recente
_FIFA_MIN,  _FIFA_MAX  = 1198,  1877
_QUAL_MIN,  _QUAL_MAX  = 7,     95
_FORME_MIN, _FORME_MAX = 0.25,  0.97
_POIDS_FORCE = {'qualite': 0.50, 'fifa': 0.35, 'forme': 0.15}

def _normaliser(val, vmin, vmax):
    return max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))

def _score_force(nom_modele, df_off):
    n_qual  = _normaliser(_get_qualite(nom_modele), _QUAL_MIN, _QUAL_MAX)
    n_fifa  = _normaliser(_get_pts_fifa(nom_modele), _FIFA_MIN, _FIFA_MAX)
    n_forme = _normaliser(_get_forme(nom_modele, df_off), _FORME_MIN, _FORME_MAX)
    return (_POIDS_FORCE['qualite'] * n_qual +
            _POIDS_FORCE['fifa']    * n_fifa +
            _POIDS_FORCE['forme']   * n_forme)

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

# ── Chargement modele hybride v3 ────────────────────────────────────
@st.cache_resource
def load_model():
    modele = joblib.load(ROOT / "models/modele_hybride.pkl")
    enc_y  = joblib.load(ROOT / "models/encodeur_cible_hybride.pkl")
    df     = pd.read_csv(ROOT / "data/results.csv")
    df['date'] = pd.to_datetime(df['date'])
    df     = df.dropna(subset=['home_score', 'away_score'])
    df_off = df[df['tournament'] != 'Friendly'].copy()
    return modele, enc_y, df_off

modele, enc_y, df_officiel = load_model()

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

def get_stars(team):

    points = POINTS_FIFA.get(
        _nom_modele(team),
        _MOYENNE_FIFA
    )

    if points >= 1800:
        return 5
    elif points >= 1700:
        return 4.5
    elif points >= 1600:
        return 4
    elif points >= 1500:
        return 3.5
    elif points >= 1400:
        return 3
    elif points >= 1300:
        return 2.5
    else:
        return 2

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

_CLASSES = list(enc_y.classes_)
_IDX_A = _CLASSES.index('A_gagne')
_IDX_B = _CLASSES.index('B_gagne')
_IDX_N = _CLASSES.index('Nul')
_ALPHA = 0.5  # compression des probas vers 1/3 (probas plus realistes face aux bookmakers)

def _corriger_coherence(p_a, p_n, p_b, force_a, force_b):
    """Corrige si le modele contredit le score de force."""
    ecart = force_a - force_b
    if abs(ecart) < 0.05:
        return p_a, p_n, p_b
    a_gagne = ecart > 0
    if not ((a_gagne and p_b > p_a) or (not a_gagne and p_a > p_b)):
        return p_a, p_n, p_b
    force_corr = min(0.35, abs(ecart) * 0.5)
    if a_gagne:
        pa2 = p_a + force_corr; pb2 = max(0.08, p_b - force_corr * 0.85); pn2 = max(0.10, p_n - force_corr * 0.15)
    else:
        pb2 = p_b + force_corr; pa2 = max(0.08, p_a - force_corr * 0.85); pn2 = max(0.10, p_n - force_corr * 0.15)
    t = pa2 + pn2 + pb2
    return pa2 / t, pn2 / t, pb2 / t

def predire_match(home, away):
    home_m = _nom_modele(home)
    away_m = _nom_modele(away)

    fa = _score_force(home_m, df_officiel)
    fb = _score_force(away_m, df_officiel)
    h  = 0.5 if home_m in PAYS_HOTES_MODELE else (-0.5 if away_m in PAYS_HOTES_MODELE else 0.0)
    ef_ab = fa - fb
    fm    = (fa + fb) / 2

    feat_ab = pd.DataFrame([{
        'ecart_force':       ef_ab,
        'ecart_force_carre': ef_ab ** 2 * np.sign(ef_ab),
        'abs_ecart_force':   abs(ef_ab),
        'force_moy':         fm,
        'avantage_hote':     h,
    }])
    feat_ba = pd.DataFrame([{
        'ecart_force':       -ef_ab,
        'ecart_force_carre': (-ef_ab) ** 2 * np.sign(-ef_ab),
        'abs_ecart_force':   abs(ef_ab),
        'force_moy':         fm,
        'avantage_hote':     -h,
    }])

    # Prediction symetrique : moyenne AB et BA pour eviter le biais home/away
    pab = modele.predict_proba(feat_ab)[0]
    pba = modele.predict_proba(feat_ba)[0]

    p_v = (pab[_IDX_A] + pba[_IDX_B]) / 2
    p_d = (pab[_IDX_B] + pba[_IDX_A]) / 2
    p_n = (pab[_IDX_N] + pba[_IDX_N]) / 2
    t = p_v + p_d + p_n
    p_v, p_d, p_n = p_v / t, p_d / t, p_n / t

    # Compression vers 1/3 (probas plus realistes, ne change pas le favori)
    p_v = 1/3 + _ALPHA * (p_v - 1/3)
    p_n = 1/3 + _ALPHA * (p_n - 1/3)
    p_d = 1/3 + _ALPHA * (p_d - 1/3)
    t = p_v + p_n + p_d
    p_v, p_n, p_d = p_v / t, p_n / t, p_d / t

    p_v, p_n, p_d = _corriger_coherence(p_v, p_n, p_d, fa, fb)
    return {'V': p_v, 'N': p_n, 'D': p_d}

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
    st.markdown('<div style="height:4px;background:linear-gradient(90deg,#002395 33%,#FFFFFF 33%,#FFFFFF 66%,#ED2939 66%);margin-bottom:12px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:8px 0'>
        <h2 style='color:#FFD700 !important;font-family:Arial Black;margin:8px 0;font-size:1.2rem;border:none'>FRENCH TEAM</h2>
        <p style='color:#aaa !important;font-size:10px;margin:0'>FIFA World Cup 2026 · Prédictions Data</p>
    </div>
    <div style='height:3px;background:linear-gradient(90deg,#8a2020,#003087,#006633);margin:8px 0;border-radius:2px'></div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠 Accueil",
        "⭐ Favoris",
        "👕 Équipes",
        "📡 Résultats en direct",
    ])

    st.markdown("""
    <div style='height:3px;background:linear-gradient(90deg,#8a2020,#003087,#006633);margin:10px 0;border-radius:2px'></div>
    <div style='text-align:center'>
        <p style='color:#8a2020 !important;font-size:11px;font-weight:bold;margin:4px 0'>🇺🇸 USA · 🇨🇦 Canada · 🇲🇽 Mexico</p>
        <p style='color:#aaa !important;font-size:10px;margin:2px 0'>11 juin → 19 juillet 2026</p>
        <p style='color:#34D399 !important;font-size:11px;font-weight:bold;margin:4px 0'>Modele Hybride v3 · Accuracy 53.94%</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABLE OFFICIELLE FIFA — Attribution des 3es (source : Annexe C FIFA +
# notebooks/datacup2026_bracket.ipynb)
# frozenset(groupes_3es) → {slot_1er: groupe_du_3e}
# ════════════════════════════════════════════════════════════════
_TABLE_3ES = {
    frozenset(['E','F','G','H','I','J','K','L']): {'1A':'E','1B':'J','1D':'I','1E':'F','1G':'H','1I':'G','1K':'L','1L':'K'},
    frozenset(['D','F','G','H','I','J','K','L']): {'1A':'H','1B':'G','1D':'I','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
    frozenset(['D','E','G','H','I','J','K','L']): {'1A':'E','1B':'J','1D':'I','1E':'D','1G':'H','1I':'G','1K':'L','1L':'K'},
    frozenset(['D','E','F','H','I','J','K','L']): {'1A':'E','1B':'J','1D':'I','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
    frozenset(['D','E','F','G','I','J','K','L']): {'1A':'E','1B':'G','1D':'I','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
    frozenset(['D','E','F','G','H','J','K','L']): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
    frozenset(['D','E','F','G','H','I','K','L']): {'1A':'E','1B':'G','1D':'I','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
    frozenset(['D','E','F','G','H','I','J','L']): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
    frozenset(['D','E','F','G','H','I','J','K']): {'1A':'E','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
    frozenset(['C','E','F','G','H','I','J','K']): {'1A':'E','1B':'G','1D':'J','1E':'C','1G':'H','1I':'F','1K':'I','1L':'K'},
    frozenset(['C','D','E','F','G','H','I','J']): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'E','1L':'I'},
    frozenset(['C','D','E','F','G','H','I','K']): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
    frozenset(['C','D','E','F','G','H','I','L']): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
    frozenset(['C','D','E','F','G','H','J','K']): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'E','1L':'K'},
    frozenset(['C','D','E','F','G','H','J','L']): {'1A':'C','1B':'G','1D':'J','1E':'D','1G':'H','1I':'F','1K':'L','1L':'E'},
    frozenset(['C','D','E','F','G','H','K','L']): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
    frozenset(['C','D','E','F','G','I','J','K']): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'J','1I':'F','1K':'I','1L':'K'},
    frozenset(['C','D','E','F','G','I','J','L']): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'J','1I':'F','1K':'L','1L':'I'},
    frozenset(['C','D','E','F','G','I','K','L']): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
    frozenset(['C','D','E','F','G','J','K','L']): {'1A':'C','1B':'G','1D':'E','1E':'D','1G':'J','1I':'F','1K':'L','1L':'K'},
    frozenset(['C','D','E','F','H','I','J','K']): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'H','1I':'F','1K':'I','1L':'K'},
    frozenset(['C','D','E','F','H','I','J','L']): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'H','1I':'F','1K':'L','1L':'I'},
    frozenset(['C','D','E','F','H','I','K','L']): {'1A':'C','1B':'E','1D':'I','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
    frozenset(['C','D','E','F','H','J','K','L']): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'H','1I':'F','1K':'L','1L':'K'},
    frozenset(['C','D','E','F','I','J','K','L']): {'1A':'C','1B':'J','1D':'E','1E':'D','1G':'I','1I':'F','1K':'L','1L':'K'},
}

_SLOT_VERS_MATCH = {
    '1E':'74','1I':'77','1A':'79','1L':'80',
    '1D':'81','1G':'82','1B':'85','1K':'87',
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
        return {_SLOT_VERS_MATCH[slot]: eq_par_grp.get(grp, list(eq_par_grp.values())[0])
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

def render_html(html_string):
    """Affiche du HTML en supprimant l'indentation parasite qui casse le rendu Markdown."""
    st.markdown(textwrap.dedent(html_string).strip(), unsafe_allow_html=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ════════════════════════════════════════════════════════════════
# ACCUEIL
# ════════════════════════════════════════════════════════════════

if page == "🏠 Accueil":

    # ── Header + équipe en petit dessous ─────────────────────────────
    _, col_c, _ = st.columns([0.2, 8, 0.2])
    with col_c:
        render_html("""
        <div style='
            text-align:center;
            padding:clamp(30px, 6vh, 80px) 0 16px;
            background:linear-gradient(135deg, #0d1b3e 0%, #1a1a2e 40%, #0f2027 100%);
            border-radius:12px;
            border:1px solid rgba(255,255,255,0.05);
        '>
            <h1 style='font-size:clamp(1.6rem, 3vw, 2.8rem);color:#FFD700;font-family:Arial Black;margin:12px 0 4px;border:none'>
                ⚽ FRENCH TEAM
            </h1>
            <p style='color:#34D399 !important;font-size:clamp(12px, 1.5vw, 16px);font-weight:bold;margin:0'>
                FIFA World Cup 2026 — Prédictions Data
            </p>
            <div style='background:linear-gradient(90deg,#8a2020,#003087,#006633);
                        height:4px;border-radius:2px;margin:10px auto;width:60%'></div>
            <p style='color:#aaa !important;font-size:clamp(11px, 1.2vw, 13px);margin:0 0 14px'>
                United States · Canada · Mexico · 11 juin → 19 juillet 2026
            </p>
            <div style='display:flex;justify-content:center;gap:18px;flex-wrap:wrap;padding:0 20px 8px'>
                <span style='color:#aaa;font-size:11px'>🧑‍✈️ <b style="color:#8a2020">Aurélien</b> · Product Owner</span>
                <span style='color:#aaa;font-size:11px'>🔧 <b style="color:#4FC3F7">Cédric</b> · Scrum Master</span>
                <span style='color:#aaa;font-size:11px'>💻 <b style="color:#34D399">Ernest</b> · Streamlit Dev</span>
                <span style='color:#aaa;font-size:11px'>📊 <b style="color:#FFD700">Romain</b> · Data &amp; ML</span>
            </div>
        </div>
        """)

    st.divider()

    # ── Simulateur de confrontation ───────────────────────────────────
    st.markdown("## ⚔️ Simulateur de confrontation")

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Equipe 1", equipes, key="home_accueil")
    with col2:
        away = st.selectbox("Equipe 2", equipes, key="away_accueil")

    stars_home = get_stars(home)
    stars_away = get_stars(away)
    flag_home  = get_flag(home, 60)
    flag_away  = get_flag(away, 60)

    card1, card2 = st.columns(2)
    with card1:
        render_html(f"""
        <div style="background:#112240;border-top:5px solid #006633;border-radius:12px;padding:12px;text-align:center;">
            <div style="font-size:44px;margin:4px 0">{flag_home}</div>
            <h2 style="color:white;margin:4px 0;font-size:1.2rem">{home}</h2>
            <div style="font-size:20px;color:#FFD700">{'⭐' * round(stars_home)}</div>
            <p style="color:#FFD700;font-weight:bold;margin:2px 0;font-size:13px">{stars_home}/5</p>
        </div>
        """)
    with card2:
        render_html(f"""
        <div style="background:#112240;border-top:5px solid #8a2020;border-radius:12px;padding:12px;text-align:center;">
            <div style="font-size:44px;margin:4px 0">{flag_away}</div>
            <h2 style="color:white;margin:4px 0;font-size:1.2rem">{away}</h2>
            <div style="font-size:20px;color:#FFD700">{'⭐' * round(stars_away)}</div>
            <p style="color:#FFD700;font-weight:bold;margin:2px 0;font-size:13px">{stars_away}/5</p>
        </div>
        """)

    st.divider()

    probas = predire_match(home, away)
    st.markdown("### 📊 Probabilités du match")
    c1, c2, c3 = st.columns(3)
    c1.metric(home, f"{probas['V'] * 100:.1f}%")
    c2.metric("Match nul", f"{probas['N'] * 100:.1f}%")
    c3.metric(away, f"{probas['D'] * 100:.1f}%")


# ════════════════════════════════════════════════════════════════
# FAVORIS
# ════════════════════════════════════════════════════════════════
elif page == "⭐ Favoris":
    st.markdown("<h1>⭐ Favoris — FIFA World Cup 2026</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa'>Probabilités issues de 15 000 simulations Monte Carlo — résultats figés</p>",
        unsafe_allow_html=True
    )

    # ── Résultats Monte Carlo figés (seed=404, 15 000 sims) ─────────────
    # France #1 à 10.53% — ne pas modifier
    WINS_MC_FIXE = {
        'France': 1580, 'England': 1532, 'Argentina': 1321,
        'Spain': 1295, 'Portugal': 1203, 'Brazil': 1033,
        'Germany': 988, 'Netherlands': 934, 'Belgium': 879,
        'Croatia': 469, 'Mexico': 397, 'Japan': 389,
        'Morocco': 352, 'Senegal': 334, 'USA': 303,
        'Colombia': 288, 'Uruguay': 244, 'Switzerland': 244,
        'Turkey': 232, 'Norway': 214, 'Korea Republic': 111,
        'Austria': 86, 'Canada': 83, 'Sweden': 76,
        'Bosnia': 66, 'IR Iran': 66, 'Ecuador': 61,
        'Paraguay': 58, "Cote d'Ivoire": 47, 'Egypt': 46,
        'Scotland': 43, 'South Africa': 26,
    }
    N_SIMS = 15000
    wins_mc, n = WINS_MC_FIXE, N_SIMS
    res_fav = simuler_tournoi_complet()

    champion_f  = res_fav['champion']
    finaliste_f = res_fav['finaliste']
    bronze_f    = res_fav['bronze']
    bronze_l_f  = res_fav['bronze_l']
    sf1, sf2, sf3, sf4 = res_fav['sf']
    qf_l        = res_fav['qf_loosers']
    r16_l       = res_fav['r16_loosers']

    # Etiquettes par performance dans la simulation deterministe
    labels = {}
    labels[champion_f]  = ("Champion du Monde", "#FFD700")
    labels[finaliste_f] = ("Finaliste",          "#C0C0C0")
    labels[bronze_f]    = ("3eme place",         "#CD7F32")
    labels[bronze_l_f]  = ("4eme place",         "#4FC3F7")
    for t in (sf1, sf2, sf3, sf4):
        if t not in labels:
            labels[t] = ("Demi-finaliste", "#3a7bd5")
    for t in qf_l:
        if t not in labels:
            labels[t] = ("Quart de finaliste", "#2a5090")
    for t in r16_l:
        if t not in labels:
            labels[t] = ("16eme de finale", "#1a3060")

    # Toutes les equipes ayant gagne au moins 1 simulation, triees par % decroissant
    all_teams_mc = sorted(wins_mc.items(), key=lambda x: x[1], reverse=True)

    # Barre calibree sur le max
    max_pct = all_teams_mc[0][1] / n * 100 if all_teams_mc else 1

    # En-tete du tableau
    st.markdown("""
    <div style='display:flex;align-items:center;background:#0a1220;
                border-radius:6px;padding:8px 16px;margin-bottom:4px;gap:16px'>
        <div style='width:30px'></div>
        <div style='width:32px'></div>
        <div style='flex:1;color:#aaa;font-size:11px;font-weight:bold'>EQUIPE</div>
        <div style='flex:2;color:#aaa;font-size:11px;font-weight:bold'>% VICTOIRE (15 000 sims)</div>
    </div>
    """, unsafe_allow_html=True)

    for rang, (team, wins_count) in enumerate(all_teams_mc, 1):
        pct           = wins_count / n * 100
        bw            = (pct / max_pct * 100) if max_pct > 0 else 0
        label, color  = labels.get(team, ("Qualifie", "#555555"))
        flag          = get_flag(team, 26)
        st.markdown(f"""
        <div style='display:flex;align-items:center;background:#0D1B2A;
                    border-left:4px solid {color};border-radius:6px;
                    padding:9px 16px;margin:3px 0;gap:16px'>
            <div style='display:flex;align-items:center;justify-content:center;
                        background:{color}22;border:1.5px solid {color};border-radius:50%;
                        width:28px;height:28px;flex-shrink:0;
                        font-size:11px;font-weight:bold;color:{color}'>{rang}</div>
            <div style='width:32px;flex-shrink:0'>{flag}</div>
            <div style='flex:1;min-width:120px'>
                <span style='color:white;font-size:13px;font-weight:bold'>{team}</span><br>
                <span style='color:{color};font-size:9px'>{label}</span>
            </div>
            <div style='flex:2;display:flex;align-items:center;gap:10px'>
                <div style='flex:1;background:#1a2a3a;border-radius:4px;height:10px;overflow:hidden'>
                    <div style='width:{bw:.1f}%;height:100%;
                                background:linear-gradient(90deg,{color},{color}66);
                                border-radius:4px'></div>
                </div>
                <span style='color:{color};font-size:13px;font-weight:bold;
                             min-width:50px;text-align:right'>{pct:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# EQUIPES
# ════════════════════════════════════════════════════════════════
elif page == "👕 Équipes":
    import pydeck as pdk

    @st.cache_data
    def load_joueurs_equipe():
        pf = pd.read_csv(ROOT / "data/players_final_clean.csv", encoding='utf-8')
        ov = pd.read_csv(ROOT / "data/df_joueurs_notes_a_jour.csv", encoding='utf-8')
        return pf.merge(
            ov[['nom_joueur', 'equipe_nationale', 'overall']],
            on=['nom_joueur', 'equipe_nationale'], how='left'
        )

    df_joueurs    = load_joueurs_equipe()
    equipes_dispo = sorted(df_joueurs['equipe_nationale'].unique())

    # Formations officielles : (label, n_def, lignes_mid, n_att)
    # lignes_mid = liste de nb joueurs par ligne (ex: [2,3] = 2 DM + 3 AM)
    FORMATIONS = {
        'Algeria':       ('4-3-3',   4, [3],    3),
        'Argentina':     ('4-3-3',   4, [3],    3),
        'Australia':     ('4-2-3-1', 4, [2, 3], 1),
        'Brazil':        ('4-2-3-1', 4, [2, 3], 1),
        'England':       ('4-2-3-1', 4, [2, 3], 1),
        'France':        ('4-3-3',   4, [3],    3),
        'Germany':       ('4-2-3-1', 4, [2, 3], 1),
        'Ghana':         ('4-2-3-1', 4, [2, 3], 1),
        'Japan':         ('4-2-3-1', 4, [2, 3], 1),
        'Mexico':        ('4-3-3',   4, [3],    3),
        'Morocco':       ('4-1-4-1', 4, [1, 4], 1),
        'Netherlands':   ('4-3-3',   4, [3],    3),
        'Paraguay':      ('4-4-2',   4, [4],    2),
        'Portugal':      ('4-3-3',   4, [3],    3),
        'South Africa':  ('4-3-3',   4, [3],    3),
        'South Korea':   ('4-2-3-1', 4, [2, 3], 1),
        'Spain':         ('4-3-3',   4, [3],    3),
        'United States': ('4-3-3',   4, [3],    3),
        'Uruguay':       ('4-4-2',   4, [4],    2),
    }
    DEFAULT_FORM = ('4-3-3', 4, [3], 3)

    CAMPS_BASE = {
        'Algeria':       {'ville': 'Philadelphia, Pennsylvania', 'lat': 39.9526, 'lon': -75.1652},
        'Argentina':     {'ville': 'East Rutherford, New Jersey','lat': 40.8136, 'lon': -74.0744},
        'Australia':     {'ville': 'Kansas City, Missouri',      'lat': 39.0997, 'lon': -94.5786},
        'Brazil':        {'ville': 'Los Angeles, California',    'lat': 34.0522, 'lon': -118.2437},
        'England':       {'ville': 'Boston, Massachusetts',      'lat': 42.3601, 'lon': -71.0589},
        'France':        {'ville': 'Miami, Florida',             'lat': 25.7617, 'lon': -80.1918},
        'Germany':       {'ville': 'Dallas, Texas',              'lat': 32.7767, 'lon': -96.7970},
        'Ghana':         {'ville': 'Atlanta, Georgia',           'lat': 33.7490, 'lon': -84.3880},
        'Japan':         {'ville': 'Seattle, Washington',        'lat': 47.6062, 'lon': -122.3321},
        'Mexico':        {'ville': 'Guadalajara, Mexique',       'lat': 20.6597, 'lon': -103.3496},
        'Morocco':       {'ville': 'New York, New Jersey',       'lat': 40.7128, 'lon': -74.0060},
        'Netherlands':   {'ville': 'Dallas, Texas',              'lat': 32.8500, 'lon': -96.7500},
        'Paraguay':      {'ville': 'Dallas, Texas',              'lat': 32.8000, 'lon': -96.8500},
        'Portugal':      {'ville': 'Philadelphia, Pennsylvania', 'lat': 39.9400, 'lon': -75.1500},
        'South Africa':  {'ville': 'Houston, Texas',             'lat': 29.7604, 'lon': -95.3698},
        'South Korea':   {'ville': 'Los Angeles, California',    'lat': 34.0195, 'lon': -118.4912},
        'Spain':         {'ville': 'New York, New Jersey',       'lat': 40.7128, 'lon': -74.0060},
        'United States': {'ville': 'Kansas City, Missouri',      'lat': 39.0997, 'lon': -94.5786},
        'Uruguay':       {'ville': 'Atlanta, Georgia',           'lat': 33.7700, 'lon': -84.4000},
    }

    # Selecteur + stats
    col_sel, col_flag, col_m1, col_m2, col_m3 = st.columns([3, 1, 1, 1, 1])
    with col_sel:
        equipe_choisie = st.selectbox("", equipes_dispo, label_visibility="collapsed")
    joueurs_eq = df_joueurs[df_joueurs['equipe_nationale'] == equipe_choisie].copy()

    with col_flag:
        if len(joueurs_eq) > 0:
            st.markdown(
                f"<img src='{joueurs_eq['drapeau_equipe'].iloc[0]}' style='height:34px;margin-top:6px'>",
                unsafe_allow_html=True
            )
    with col_m1:
        st.metric("Overall moy.", f"{joueurs_eq['overall'].mean():.0f}/100")
    with col_m2:
        st.metric("Buts saison", f"{joueurs_eq['buts_marques'].sum():.0f}")
    with col_m3:
        st.metric("Effectif", len(joueurs_eq))

    # Affichage par rubriques de poste
    POSTES_ORDRE = ['Gardien', 'Défenseur', 'Milieu', 'Attaquant']
    ICONS        = {'Gardien': '🧤', 'Défenseur': '🛡️', 'Milieu': '⚙️', 'Attaquant': '⚡'}
    POSTE_COLOR  = {'Gardien': '#F59E0B', 'Défenseur': '#3B82F6', 'Milieu': '#10B981', 'Attaquant': '#EF4444'}

    for poste in POSTES_ORDRE:
        joueurs_poste = joueurs_eq[joueurs_eq['poste'] == poste].sort_values('overall', ascending=False)
        if joueurs_poste.empty:
            continue

        icon  = ICONS.get(poste, '')
        color = POSTE_COLOR.get(poste, '#aaa')
        nb    = len(joueurs_poste)

        st.markdown(
            f"<div style='margin:18px 0 8px;"
            f"border-left:3px solid {color};padding-left:10px'>"
            f"<span style='color:{color};font-size:14px;font-weight:bold'>"
            f"{icon} {poste}s</span>"
            f"<span style='color:#666;font-size:11px;margin-left:8px'>({nb})</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        cols = st.columns(min(nb, 6))
        SILHOUETTE = (
            "data:image/svg+xml;charset=utf-8,"
            "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
            "%3Ccircle cx='50' cy='50' r='50' fill='%23112240'/%3E"
            "%3Ccircle cx='50' cy='36' r='20' fill='%23e8e8e8'/%3E"
            "%3Cpath d='M14 95 Q14 62 50 62 Q86 62 86 95 Z' fill='%23e8e8e8'/%3E"
            "%3C/svg%3E"
        )
        for i, (_, j) in enumerate(joueurs_poste.iterrows()):
            ov_s  = f"{j['overall']:.0f}" if pd.notna(j['overall']) else "?"
            nom   = j['nom_joueur']
            raw_photo = j['photo_joueur']
            photo = raw_photo if (isinstance(raw_photo, str) and raw_photo.startswith('http')) else SILHOUETTE
            with cols[i % min(nb, 6)]:
                st.markdown(f"""
                <div style="background:#112240;border-radius:8px;padding:10px 6px;
                            text-align:center;margin-bottom:4px">
                    <img src="{photo}"
                         style="width:52px;height:52px;border-radius:50%;
                                object-fit:cover;border:2px solid {color};
                                display:block;margin:0 auto 6px"
                         onerror="this.onerror=null;this.src='{SILHOUETTE}'">
                    <p style="color:white;font-size:10px;font-weight:bold;margin:0 0 2px;
                              line-height:1.2;overflow:hidden;text-overflow:ellipsis;
                              white-space:nowrap">{nom}</p>
                    <p style="color:#aaa;font-size:9px;margin:0 0 4px">{icon} {poste}</p>
                    <span style="color:#FFD700;font-size:13px;font-weight:bold">{ov_s}</span>
                </div>
                """, unsafe_allow_html=True)

    # Carte camp de base
    camp = CAMPS_BASE.get(equipe_choisie)
    if camp:
        st.markdown(
            f"<p style='color:#34D399;font-size:12px;margin:20px 0 4px'>"
            f"📍 Camp de base : <b style='color:white'>{camp['ville']}</b></p>",
            unsafe_allow_html=True
        )
        carte_data = pd.DataFrame([{
            'lat': camp['lat'], 'lon': camp['lon'],
            'equipe': equipe_choisie, 'ville': camp['ville'],
        }])
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer(
                'ScatterplotLayer', data=carte_data,
                get_position='[lon, lat]',
                get_color='[255, 215, 0, 220]',
                get_radius=20000, pickable=True,
            )],
            initial_view_state=pdk.ViewState(
                latitude=camp['lat'], longitude=camp['lon'], zoom=7, pitch=25
            ),
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            tooltip={'text': '{equipe}\n{ville}'},
        ))

# ════════════════════════════════════════════════════════════════
# RÉSULTATS EN DIRECT
# ════════════════════════════════════════════════════════════════
elif page == "📡 Résultats en direct":
    import requests
    from datetime import datetime, timedelta, timezone

    API_KEY  = "64efba97e0ab4f5680e42937a01d8616"
    BASE_URL = "https://api.football-data.org/v4"
    HEADERS  = {"X-Auth-Token": API_KEY}
    COMP     = "WC"

    st.markdown("<h1>📡 Résultats en direct — FIFA World Cup 2026</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa'>Données live · football-data.org · Mise à jour au chargement</p>",
                unsafe_allow_html=True)

    # ── Helpers API ───────────────────────────────────────────────
    @st.cache_data(ttl=180)
    def get_matches(date_from, date_to):
        r = requests.get(f"{BASE_URL}/competitions/{COMP}/matches", headers=HEADERS,
                         params={"dateFrom": date_from, "dateTo": date_to}, timeout=10)
        return r.json().get("matches", []) if r.status_code == 200 else []

    @st.cache_data(ttl=300)
    def get_standings():
        r = requests.get(f"{BASE_URL}/competitions/{COMP}/standings", headers=HEADERS, timeout=10)
        return r.json().get("standings", []) if r.status_code == 200 else []

    @st.cache_data(ttl=180)
    def get_knockout_matches():
        """Récupère tous les matchs de phase éliminatoire."""
        r = requests.get(f"{BASE_URL}/competitions/{COMP}/matches", headers=HEADERS,
                         params={"stage": "LAST_32"}, timeout=10)
        all_m = r.json().get("matches", []) if r.status_code == 200 else []
        for stage in ["LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "THIRD_PLACE", "FINAL"]:
            r2 = requests.get(f"{BASE_URL}/competitions/{COMP}/matches", headers=HEADERS,
                              params={"stage": stage}, timeout=10)
            if r2.status_code == 200:
                all_m += r2.json().get("matches", [])
        return all_m

    # ── Helpers affichage ─────────────────────────────────────────
    def status_badge(status):
        if status == "FINISHED":
            return "<span style='background:#1a3a1a;color:#34D399;border-radius:4px;padding:2px 7px;font-size:10px;font-weight:bold'>FT</span>"
        elif status in ("IN_PLAY", "PAUSED"):
            return "<span style='background:#3a1a00;color:#FF9800;border-radius:4px;padding:2px 7px;font-size:10px;font-weight:bold'>🔴 LIVE</span>"
        elif status in ("TIMED", "SCHEDULED"):
            return "<span style='background:#1a2a3a;color:#4FC3F7;border-radius:4px;padding:2px 7px;font-size:10px;font-weight:bold'>À venir</span>"
        return f"<span style='background:#2a2a2a;color:#aaa;border-radius:4px;padding:2px 7px;font-size:10px'>{status}</span>"

    def score_display(m):
        ft = m.get("score", {}).get("fullTime", {})
        h, a = ft.get("home"), ft.get("away")
        if h is not None and a is not None:
            return f"<span style='font-size:1.4rem;font-weight:bold;color:#FFD700'>{h} — {a}</span>"
        return "<span style='font-size:1.4rem;font-weight:bold;color:#555'>vs</span>"

    def match_card(m, highlight=False):
        home   = m["homeTeam"].get("shortName") or m["homeTeam"].get("name", "?")
        away   = m["awayTeam"].get("shortName") or m["awayTeam"].get("name", "?")
        status = m["status"]
        hflag, aflag = get_flag(home, 28), get_flag(away, 28)
        badge, score = status_badge(status), score_display(m)
        dt_str = ""
        try:
            dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            dt_str = (dt + timedelta(hours=2)).strftime("%d/%m %H:%M")
        except Exception:
            pass
        border = "#FFD700" if highlight else "#1a2a3a"
        bg     = "#16253a" if highlight else "#0D1B2A"
        group  = m.get("group", "") or ""
        stage  = m.get("stage", "").replace("_", " ").title()
        info   = stage + (f" · {group}" if group else "")
        return f"""
        <div style='background:{bg};border:1px solid {border};border-radius:10px;
                    padding:14px 18px;margin:6px 0;display:flex;align-items:center;gap:12px'>
            <div style='flex:1;text-align:right'>
                <div style='display:flex;align-items:center;justify-content:flex-end;gap:8px'>
                    <span style='color:white;font-weight:bold;font-size:13px'>{home}</span>{hflag}
                </div>
            </div>
            <div style='text-align:center;min-width:110px'>
                <div>{score}</div>
                <div style='margin-top:4px'>{badge}</div>
                <div style='color:#555;font-size:10px;margin-top:2px'>{dt_str}</div>
            </div>
            <div style='flex:1;text-align:left'>
                <div style='display:flex;align-items:center;gap:8px'>
                    {aflag}<span style='color:white;font-weight:bold;font-size:13px'>{away}</span>
                </div>
            </div>
        </div>
        <div style='color:#555;font-size:10px;text-align:center;margin:-4px 0 6px'>{info}</div>
        """

    # ── Refresh + chargement ──────────────────────────────────────
    col_t, col_r = st.columns([5, 1])
    with col_r:
        if st.button("🔄 Rafraîchir"):
            st.cache_data.clear()
            st.rerun()

    today     = datetime.now(timezone.utc)
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow  = (today + timedelta(days=2)).strftime("%Y-%m-%d")

    with st.spinner("Chargement des données..."):
        matches_window  = get_matches(yesterday, tomorrow)
        standings       = get_standings()
        knockout_matches = get_knockout_matches()

    if not matches_window and not standings:
        st.error("❌ Impossible de contacter football-data.org.")
        st.stop()

    # ── Données phase de poule ────────────────────────────────────
    live     = [m for m in matches_window if m["status"] in ("IN_PLAY", "PAUSED")]
    finished = sorted([m for m in matches_window if m["status"] == "FINISHED"],
                      key=lambda x: x.get("utcDate", ""), reverse=True)
    upcoming = [m for m in matches_window if m["status"] in ("TIMED", "SCHEDULED")]

    # ════════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════════
    tab1, tab2 = st.tabs(["🏟️ Phase de Poule", "🏆 Tableau Final"])

    # ─────────────────────────────────────────────────────────────
    # TAB 1 — PHASE DE POULE
    # ─────────────────────────────────────────────────────────────
    with tab1:

        if live:
            st.markdown("## 🔴 En cours")
            for m in live:
                st.markdown(match_card(m, highlight=True), unsafe_allow_html=True)
            st.divider()

        st.markdown("## ✅ Résultats récents")
        if finished:
            for m in finished:
                home_name = m["homeTeam"].get("shortName") or m["homeTeam"].get("name", "")
                hl = "France" in home_name or "France" in (m["awayTeam"].get("shortName") or m["awayTeam"].get("name", ""))
                st.markdown(match_card(m, highlight=hl), unsafe_allow_html=True)
        else:
            st.info("Aucun résultat récent disponible.")

        st.divider()

        st.markdown("## 📅 Prochains matchs")
        if upcoming:
            for m in upcoming[:10]:
                st.markdown(match_card(m), unsafe_allow_html=True)
        else:
            st.info("Aucun match à venir dans les prochains jours.")

        st.divider()

        st.markdown("## 📊 Classement des groupes")
        if standings:
            grp_list = [s for s in standings if s.get("type") == "TOTAL"] or standings
            for grp in grp_list:
                grp_name = grp.get("group") or grp.get("stage", "Groupe")
                table = grp.get("table", [])
                if not table:
                    continue
                st.markdown(f"<h4 style='color:#FFD700;margin:16px 0 6px'>{grp_name}</h4>",
                            unsafe_allow_html=True)
                st.markdown("""
                <div style='display:flex;background:#0a1220;border-radius:6px;
                            padding:6px 12px;margin-bottom:3px;font-size:10px;color:#aaa;font-weight:bold;gap:8px'>
                    <div style='width:22px'>#</div><div style='flex:1'>Équipe</div>
                    <div style='width:25px;text-align:center'>J</div>
                    <div style='width:25px;text-align:center'>G</div>
                    <div style='width:25px;text-align:center'>N</div>
                    <div style='width:25px;text-align:center'>D</div>
                    <div style='width:35px;text-align:center'>Diff</div>
                    <div style='width:30px;text-align:center;color:#FFD700'>Pts</div>
                </div>""", unsafe_allow_html=True)
                for row in table:
                    pos  = row.get("position", "")
                    team = row.get("team", {}).get("shortName") or row.get("team", {}).get("name", "")
                    j, g, n, d = row.get("playedGames",0), row.get("won",0), row.get("draw",0), row.get("lost",0)
                    diff = row.get("goalDifference", 0)
                    pts  = row.get("points", 0)
                    qc   = "#34D399" if pos <= 2 else ("#FFD700" if pos == 3 else "#555")
                    diff_str = f"+{diff}" if diff > 0 else str(diff)
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;background:#0D1B2A;
                                border-left:3px solid {qc};border-radius:6px;
                                padding:6px 12px;margin:2px 0;gap:8px;font-size:12px'>
                        <div style='width:22px;color:{qc};font-weight:bold'>{pos}</div>
                        <div style='flex:1;display:flex;align-items:center;gap:6px'>
                            {get_flag(team,20)}<span style='color:white;font-weight:bold'>{team}</span>
                        </div>
                        <div style='width:25px;text-align:center;color:#aaa'>{j}</div>
                        <div style='width:25px;text-align:center;color:#34D399'>{g}</div>
                        <div style='width:25px;text-align:center;color:#aaa'>{n}</div>
                        <div style='width:25px;text-align:center;color:#8a2020'>{d}</div>
                        <div style='width:35px;text-align:center;color:#aaa'>{diff_str}</div>
                        <div style='width:30px;text-align:center;color:#FFD700;font-weight:bold'>{pts}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("Classements non disponibles pour le moment.")

    # ─────────────────────────────────────────────────────────────
    # TAB 2 — TABLEAU FINAL (bracket style)
    # ─────────────────────────────────────────────────────────────
    with tab2:
        CR32, CR16, CQF, CSF = "#4FC3F7", "#FF9800", "#FFD700", "#8a2020"
        ST_W, OL_W = 14, 12   # stub width, output-line width (px)

        # ── Helpers pour les cartes du bracket (API) ──────────────
        def _tname(td):
            if not td:
                return "TBD"
            return td.get("shortName") or td.get("name") or "TBD"

        def _mc_api(m, bc="#4FC3F7"):
            """Petite carte de match depuis données API."""
            e1 = _tname(m.get("homeTeam"))
            e2 = _tname(m.get("awayTeam"))
            wraw = (m.get("score") or {}).get("winner", "")
            w = e1 if wraw == "HOME_TEAM" else (e2 if wraw == "AWAY_TEAM" else "")
            c1 = "#FFD700" if (w and w == e1) else ("white" if not w else "#444")
            c2 = "#FFD700" if (w and w == e2) else ("white" if not w else "#444")
            chk1 = "<span style='color:#34D399;font-size:9px'>&#10003;</span>" if (w and w == e1) else ""
            chk2 = "<span style='color:#34D399;font-size:9px'>&#10003;</span>" if (w and w == e2) else ""
            f1 = get_flag(e1, 14) if e1 != "TBD" else "🏳️"
            f2 = get_flag(e2, 14) if e2 != "TBD" else "🏳️"
            op = "0.4" if e1 == "TBD" else "1"
            return (
                f"<div style='background:#0D1B2A;border:1px solid {bc};border-radius:4px;"
                f"padding:4px 6px;min-width:138px;max-width:152px;margin:1px 0;opacity:{op}'>"
                f"<div style='display:flex;align-items:center;gap:3px'>"
                f"{f1}<span style='color:{c1};font-size:9px;flex:1;overflow:hidden;white-space:nowrap'>{e1[:13]}</span>{chk1}"
                f"</div><div style='height:1px;background:#1a2a3a;margin:2px 0'></div>"
                f"<div style='display:flex;align-items:center;gap:3px'>"
                f"{f2}<span style='color:{c2};font-size:9px;flex:1;overflow:hidden;white-space:nowrap'>{e2[:13]}</span>{chk2}"
                f"</div></div>"
            )

        def _mc_big_api(m, bc="#FFD700"):
            """Grande carte de match (demi/finale) depuis données API."""
            e1 = _tname(m.get("homeTeam"))
            e2 = _tname(m.get("awayTeam"))
            wraw = (m.get("score") or {}).get("winner", "")
            w = e1 if wraw == "HOME_TEAM" else (e2 if wraw == "AWAY_TEAM" else "")
            c1 = "#FFD700" if (w and w == e1) else ("white" if not w else "#777")
            c2 = "#FFD700" if (w and w == e2) else ("white" if not w else "#777")
            chk1 = "<span style='color:#34D399;font-weight:bold'>&#10003;</span>" if (w and w == e1) else ""
            chk2 = "<span style='color:#34D399;font-weight:bold'>&#10003;</span>" if (w and w == e2) else ""
            f1 = get_flag(e1, 20) if e1 != "TBD" else "🏳️"
            f2 = get_flag(e2, 20) if e2 != "TBD" else "🏳️"
            return (
                f"<div style='background:#0D1B2A;border:2px solid {bc};border-radius:6px;"
                f"padding:8px 10px;min-width:165px'>"
                f"<div style='display:flex;align-items:center;gap:5px'>"
                f"{f1}<span style='color:{c1};font-size:12px;font-weight:bold;flex:1'>{e1}</span>{chk1}"
                f"</div><div style='height:1px;background:#1a2a3a;margin:4px 0'></div>"
                f"<div style='display:flex;align-items:center;gap:5px'>"
                f"{f2}<span style='color:{c2};font-size:12px;font-weight:bold;flex:1'>{e2}</span>{chk2}"
                f"</div></div>"
            )

        def _tbd_card(bc="#4FC3F7", big=False):
            pad = "8px 10px" if big else "4px 6px"
            w   = "165px"   if big else "138px"
            fs  = "12px"    if big else "9px"
            return (
                f"<div style='background:#0D1B2A;border:{'2' if big else '1'}px solid {bc};"
                f"border-radius:{'6' if big else '4'}px;padding:{pad};min-width:{w};opacity:0.35'>"
                f"<div style='color:#555;font-size:{fs}'>TBD</div>"
                f"<div style='height:1px;background:#1a2a3a;margin:{'4' if big else '2'}px 0'></div>"
                f"<div style='color:#555;font-size:{fs}'>TBD</div>"
                f"</div>"
            )

        def _get_card(by_st, stage, idx, bc, big=False):
            ms = by_st.get(stage, [])
            if idx < len(ms):
                return _mc_big_api(ms[idx], bc) if big else _mc_api(ms[idx], bc)
            return _tbd_card(bc, big)

        # ── Connecteurs (identiques à la page Tableau final) ──────
        def _bpr(m1h, m2h, color, gap="3px"):
            top  = f"<div style='display:flex;align-items:center'>{m1h}<div style='width:{ST_W}px;height:1px;background:{color};flex-shrink:0'></div></div>"
            bot  = f"<div style='display:flex;align-items:center'>{m2h}<div style='width:{ST_W}px;height:1px;background:{color};flex-shrink:0'></div></div>"
            vbar = (f"<div style='display:flex;flex-direction:column;width:1px;flex-shrink:0'>"
                    f"<div style='flex:1'></div><div style='flex:2;background:{color}'></div><div style='flex:1'></div></div>")
            return (f"<div style='display:flex;align-items:stretch'>"
                    f"<div style='display:flex;flex-direction:column;gap:{gap}'>{top}{bot}</div>{vbar}</div>")

        def _bpl(m1h, m2h, color, gap="3px"):
            top  = f"<div style='display:flex;align-items:center'><div style='width:{ST_W}px;height:1px;background:{color};flex-shrink:0'></div>{m1h}</div>"
            bot  = f"<div style='display:flex;align-items:center'><div style='width:{ST_W}px;height:1px;background:{color};flex-shrink:0'></div>{m2h}</div>"
            vbar = (f"<div style='display:flex;flex-direction:column;width:1px;flex-shrink:0'>"
                    f"<div style='flex:1'></div><div style='flex:2;background:{color}'></div><div style='flex:1'></div></div>")
            return (f"<div style='display:flex;align-items:stretch'>{vbar}"
                    f"<div style='display:flex;flex-direction:column;gap:{gap}'>{top}{bot}</div></div>")

        def _ol(color):
            return f"<div style='width:{OL_W}px;height:1px;background:{color};flex-shrink:0;align-self:center'></div>"

        # ── Constructeurs de lignes du bracket ────────────────────
        def _rd(by_st, i1, i2, ir16):
            c1  = _get_card(by_st, "LAST_32", i1,  CR32)
            c2  = _get_card(by_st, "LAST_32", i2,  CR32)
            c16 = _get_card(by_st, "LAST_16", ir16, CR16)
            return f"<div style='display:flex;align-items:center'>{_bpr(c1,c2,CR32)}{_ol(CR32)}{c16}</div>"

        def _ra(by_st, i1, i2, ir16):
            c1  = _get_card(by_st, "LAST_32", i1,  CR32)
            c2  = _get_card(by_st, "LAST_32", i2,  CR32)
            c16 = _get_card(by_st, "LAST_16", ir16, CR16)
            return f"<div style='display:flex;align-items:center'>{c16}{_ol(CR16)}{_bpl(c1,c2,CR32)}</div>"

        def _qd(s1, s2, by_st, iqf):
            cqf = _get_card(by_st, "QUARTER_FINALS", iqf, CQF)
            return f"<div style='display:flex;align-items:center'>{_bpr(s1,s2,CR16,'6px')}{_ol(CR16)}{cqf}</div>"

        def _qa(s1, s2, by_st, iqf):
            cqf = _get_card(by_st, "QUARTER_FINALS", iqf, CQF)
            return f"<div style='display:flex;align-items:center'>{cqf}{_ol(CQF)}{_bpl(s1,s2,CR16,'6px')}</div>"

        def _sd(s1, s2, by_st, isf):
            csf = _get_card(by_st, "SEMI_FINALS", isf, CSF, big=True)
            return f"<div style='display:flex;align-items:center'>{_bpr(s1,s2,CQF,'10px')}{_ol(CQF)}{csf}</div>"

        def _sa(s1, s2, by_st, isf):
            csf = _get_card(by_st, "SEMI_FINALS", isf, CSF, big=True)
            return f"<div style='display:flex;align-items:center'>{csf}{_ol(CSF)}{_bpl(s1,s2,CQF,'10px')}</div>"

        # ── Traitement des données API ────────────────────────────
        by_st = {}
        for _m in knockout_matches:
            _s = _m.get("stage", "")
            by_st.setdefault(_s, []).append(_m)
        for _s in by_st:
            by_st[_s].sort(key=lambda x: x.get("utcDate", ""))

        # ── Construction du bracket ───────────────────────────────
        dallas = _sd(
            _qd(_rd(by_st,1,4,0), _rd(by_st,0,2,1), by_st, 0),
            _qd(_rd(by_st,10,11,2), _rd(by_st,8,9,3), by_st, 1),
            by_st, 0
        )
        atlanta = _sa(
            _qa(_ra(by_st,3,5,4), _ra(by_st,6,7,5), by_st, 2),
            _qa(_ra(by_st,13,15,6), _ra(by_st,12,14,7), by_st, 3),
            by_st, 1
        )

        fin_h = _get_card(by_st, "FINAL",       0, "#FFD700", big=True)
        bro_h = _get_card(by_st, "THIRD_PLACE", 0, "#CD7F32", big=True)

        header_d = (
            f"<div style='display:flex;justify-content:space-around;margin-bottom:8px;"
            f"padding-bottom:4px;border-bottom:1px solid #1a2a3a'>"
            f"<b style='color:{CR32};font-size:9px'>16ème</b>"
            f"<b style='color:{CR16};font-size:9px'>8ème</b>"
            f"<b style='color:{CQF};font-size:9px'>Quarts</b>"
            f"<b style='color:{CSF};font-size:9px'>Demi</b></div>"
        )
        header_a = (
            f"<div style='display:flex;justify-content:space-around;margin-bottom:8px;"
            f"padding-bottom:4px;border-bottom:1px solid #1a2a3a'>"
            f"<b style='color:{CSF};font-size:9px'>Demi</b>"
            f"<b style='color:{CQF};font-size:9px'>Quarts</b>"
            f"<b style='color:{CR16};font-size:9px'>8ème</b>"
            f"<b style='color:{CR32};font-size:9px'>16ème</b></div>"
        )
        header_c = (
            f"<div style='text-align:center;margin-bottom:8px;padding-bottom:4px;"
            f"border-bottom:1px solid #1a2a3a'>"
            f"<b style='color:#FFD700;font-size:9px'>Finale · Bronze</b></div>"
        )
        centre = (
            f"<div style='padding:0 14px;text-align:center;flex-shrink:0'>"
            f"<p style='color:#FFD700;font-size:11px;font-weight:bold;margin:0 0 6px'>FINALE</p>"
            f"{fin_h}"
            f"<p style='color:#CD7F32;font-size:10px;font-weight:bold;margin:16px 0 6px'>BRONZE</p>"
            f"{bro_h}</div>"
        )
        bracket_html = (
            "<div style='overflow-x:auto;background:#071020;border-radius:10px;padding:20px'>"
            "<div style='display:flex;align-items:flex-start;flex-wrap:nowrap'>"
            f"<div style='display:flex;flex-direction:column'>{header_d}{dallas}</div>"
            f"<div style='display:flex;flex-direction:column;align-items:center'>{header_c}{centre}</div>"
            f"<div style='display:flex;flex-direction:column'>{header_a}{atlanta}</div>"
            "</div></div>"
        )

        if not knockout_matches:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;background:#0D1B2A;
                        border-radius:12px;border:1px solid #1a2a3a;margin-top:20px'>
                <div style='font-size:3rem'>⏳</div>
                <h3 style='color:#FFD700;margin:12px 0'>Phase éliminatoire pas encore commencée</h3>
                <p style='color:#aaa'>Le tableau final se remplira automatiquement dès le début
                des 16èmes de finale (à partir du 4 juillet 2026).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(bracket_html, unsafe_allow_html=True)
