import requests

url = "http://127.0.0.1:5000/predict"
data = {'text': 'Traduit en francais ceci: I love you.'}
response = requests.post(url, json=data)

# Vérification du statut de la réponse
if response.status_code == 200:
    # Affichage de la prédiction reçue de l'API
    print("Prédiction reçue de l'API:", response.json())
else:
    # Affichage d'un message d'erreur si la requête a échoué
    print("Erreur:", response.status_code, response.text)