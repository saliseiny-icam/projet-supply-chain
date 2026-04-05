import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Optimisation Logistique", page_icon="📦", layout="wide")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def charger_donnees():
    return pd.read_csv("clients_france.csv")

df = charger_donnees()

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2764/2764494.png", width=100)
st.sidebar.title("⚙️ Paramètres")

volume_min = int(df['Volume_Commandes'].min())
volume_max = int(df['Volume_Commandes'].max())
filtre_volume = st.sidebar.slider(
    "Filtrer par volume minimum :",
    min_value=volume_min,
    max_value=volume_max,
    value=volume_min,
    step=10
)

# Application du filtre
df_filtre = df[df['Volume_Commandes'] >= filtre_volume].copy()

# On prépare les données géographiques pour l'Intelligence Artificielle
X = df_filtre[['Latitude', 'Longitude']]

# --- CORPS PRINCIPAL DE L'APPLICATION ---
st.title("📦 Optimisation de l'implantation d'entrepôts")
st.markdown("Outil d'aide à la décision utilisant l'algorithme **K-Means** pour déterminer le nombre et l'emplacement optimaux de nos futurs entrepôts.")
st.divider()

# --- KPI ---
col1, col2, col3 = st.columns(3)
col1.metric(label="👥 Clients analysés", value=f"{len(df_filtre)}")
col2.metric(label="📦 Volume total (Articles)", value=f"{df_filtre['Volume_Commandes'].sum():,}".replace(",", " "))
col3.metric(label="📍 Zone", value="France Métropolitaine")

# ==========================================
# PARTIE IA : MÉTHODE DU COUDE & CLUSTERING
# ==========================================

st.header("1. Méthode du coude (Trouver le nombre optimal d'entrepôts)")
st.markdown("Le graphique ci-dessous teste l'algorithme pour 1 à 10 entrepôts. **L'objectif est de trouver le 'coude'** du graphique, là où la courbe commence à s'aplatir.")

col_graph, col_explication = st.columns([2, 1])

with col_graph:
    # Calcul de l'inertie pour k allant de 1 à 10
    inertias = []
    k_valeurs = range(1, 11)
    
    for k in k_valeurs:
        # n_init='auto' pour éviter les avertissements de la nouvelle version de sklearn
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans_test.fit(X)
        inertias.append(kmeans_test.inertia_)
    
    # Création du graphique avec Matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_valeurs, inertias, marker='o', linestyle='--', color='#b30000')
    ax.set_xlabel("Nombre d'entrepôts (k)")
    ax.set_ylabel("Inertie (Distance totale)")
    ax.set_title("Graphique de la méthode du coude")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig)

with col_explication:
    st.info("💡 **Comment lire ce graphique ?**\n\nRegardez le point où la diminution de l'inertie devient moins marquée (le pli du coude). En général, pour nos données groupées autour des grandes villes, le coude se situe souvent autour de **3 ou 4**.")
    
    # Choix interactif du nombre de clusters par l'utilisateur
    k_optimal = st.slider("👉 Choisissez le nombre final d'entrepôts à implanter :", min_value=1, max_value=10, value=3)

st.divider()

# ==========================================
# RÉSULTATS : PLACEMENT DES ENTREPÔTS
# ==========================================

st.header(f"2. Emplacements optimaux pour {k_optimal} entrepôt(s)")

# Entraînement du modèle final avec le k choisi par l'utilisateur
kmeans_final = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
# On assigne chaque client à un cluster (un entrepôt)
df_filtre['Cluster'] = kmeans_final.fit_predict(X)

# On récupère les coordonnées des entrepôts (centroïdes)
centroides = kmeans_final.cluster_centers_

# Création de la nouvelle carte
carte_resultat = folium.Map(location=[46.2276, 2.2137], zoom_start=6, tiles="cartodbpositron")

# Couleurs pour différencier les clusters
couleurs = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4']

# 1. Ajouter les clients sur la carte
for index, ligne in df_filtre.iterrows():
    couleur_cluster = couleurs[int(ligne['Cluster']) % len(couleurs)]
    folium.CircleMarker(
        location=[ligne['Latitude'], ligne['Longitude']],
        radius=4,
        color=couleur_cluster,
        fill=True,
        fill_color=couleur_cluster,
        fill_opacity=0.6,
        popup=f"Client: {ligne['ID_Client']}<br>Entrepôt attribué: n°{ligne['Cluster']+1}"
    ).add_to(carte_resultat)

# 2. Ajouter les entrepôts (centroïdes) sur la carte
for i, centre in enumerate(centroides):
    couleur_entrepot = couleurs[i % len(couleurs)]
    
    # Icône spéciale pour l'entrepôt
    folium.Marker(
        location=[centre[0], centre[1]],
        popup=f"<b>🏢 ENTREPÔT {i+1}</b>",
        icon=folium.Icon(color="black", icon_color=couleur_entrepot, icon="home", prefix="fa")
    ).add_to(carte_resultat)

st_folium(carte_resultat, width=800, height=500, returned_objects=[])

with st.expander("📍 Voir les coordonnées exactes des entrepôts"):
    df_entrepots = pd.DataFrame(centroides, columns=['Latitude', 'Longitude'])
    df_entrepots.index = [f"Entrepôt {i+1}" for i in range(len(centroides))]
    st.dataframe(df_entrepots, use_container_width=True)