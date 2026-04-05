import pandas as pd
import numpy as np

# Fixer la "graine" pour avoir les mêmes résultats
np.random.seed(42)

nombre_clients = 200

# Coordonnées réelles des grandes villes françaises (Latitude, Longitude)
villes_principales = [
    (48.8566, 2.3522),   # Paris
    (45.7640, 4.8357),   # Lyon
    (43.2965, 5.3698),   # Marseille
    (43.6047, 1.4442),   # Toulouse
    (44.8378, -0.5792),  # Bordeaux
    (50.6292, 3.0573),   # Lille
    (47.2184, -1.5536),  # Nantes
    (48.5734, 7.7521),   # Strasbourg
    (48.1173, -1.6778)   # Rennes
]

latitudes = []
longitudes = []

# Pour chaque client, on choisit une ville au hasard, et on le place autour de cette ville
for _ in range(nombre_clients):
    # Choisir une ville aléatoire comme "centre"
    ville = villes_principales[np.random.randint(0, len(villes_principales))]
    
    # On réduit considérablement la dispersion (écart-type de 0.05 au lieu de 0.5)
    # 0.05 degré correspond à environ 5 km, ce qui garde les points bien sur le territoire !
    lat_client = ville[0] + np.random.normal(0, 0.05)
    lon_client = ville[1] + np.random.normal(0, 0.05)
    
    latitudes.append(lat_client)
    longitudes.append(lon_client)

# Volumes de commandes (plus réalistes, entre 10 et 1000)
volumes = np.random.randint(low=10, high=1000, size=nombre_clients)

# Création des identifiants
ids_clients = [f"Client_{i+1}" for i in range(nombre_clients)]

# Création du tableau de données
donnees = {
    'ID_Client': ids_clients,
    'Latitude': latitudes,
    'Longitude': longitudes,
    'Volume_Commandes': volumes
}

df_clients = pd.DataFrame(donnees)

# Sauvegarde des données
nom_fichier = "clients_france.csv"
df_clients.to_csv(nom_fichier, index=False)

print(f"✅ Nouveau fichier '{nom_fichier}' créé ! Les clients sont maintenant strictement autour des grandes villes françaises.")