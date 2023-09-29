import requests

url = "http://127.0.0.1:5000/predict"
data = {'size': 0.023284648356090343, 'p53_concentration': 0.0016264595117659471}
response = requests.post(url, json=data)

# Vérification du statut de la réponse
if response.status_code == 200:
    # Affichage de la prédiction reçue de l'API
    print("Prédiction reçue de l'API:", response.json())
else:
    # Affichage d'un message d'erreur si la requête a échoué
    print("Erreur:", response.status_code, response.text)