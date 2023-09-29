import streamlit as st
import joblib

# Chargement du modèle de régression
model = joblib.load('regression.joblib')

# Création de l'interface utilisateur avec Streamlit
st.title('Prédiction du Prix des Maisons')

# Création des champs de formulaire pour recueillir les informations de l'utilisateur
taille = st.number_input('Entrez la taille de la maison (en m²):')
nombre_de_chambre = st.number_input('Entrez le nombre de chambres:', format='%d')
jardin = st.selectbox('La maison a-t-elle un jardin?', ['Oui', 'Non'])

# Conversion de la réponse du jardin en valeur numérique
jardin = 1 if jardin == 'Oui' else 0

# Si l'utilisateur a rempli tous les champs, on effectue la prédiction
if st.button('Prédire'):
    # Création d'un tableau avec les valeurs d'entrée
    input_data = [[taille, nombre_de_chambre, jardin]]

    # Utilisation du modèle pour prédire le prix de la maison
    prediction = model.predict(input_data)

    # Affichage du résultat de la prédiction
    st.write(f'Le prix estimé de la maison est de {prediction[0]} euros.')
