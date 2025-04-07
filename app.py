import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from fpdf import FPDF
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score

# Configuration de la page
st.set_page_config(page_title="Analyse de Clustering", layout="wide", 
                   page_icon="📊", initial_sidebar_state="expanded")

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre principal avec animation
st.markdown("""
    <h1 style='text-align: center; animation: fadeIn 2s;'>
        🔍 Analyse de Clustering Interactive
    </h1>
    """, unsafe_allow_html=True)

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    st.markdown("---")

    # Upload du fichier Excel avec une zone de drag & drop personnalisée
    uploaded_file = st.file_uploader(
        "📈 Choisissez votre fichier Excel",
        type=['xlsx', 'xls'],
        help="Glissez-déposez votre fichier Excel ici"
    )

if uploaded_file is not None:
    try:
        # Lecture des données avec barre de progression
        with st.spinner('Chargement des données...'):
            df = pd.read_excel(uploaded_file)
        
        # Affichage des informations sur les données
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de lignes", df.shape[0])
        with col2:
            st.metric("Nombre de colonnes", df.shape[1])
        with col3:
            st.metric("Variables numériques", len(df.select_dtypes(include=[np.number]).columns))

        # Aperçu des données avec style
        st.subheader("📊 Aperçu des données")
        st.dataframe(df.head(), use_container_width=True)

        # Préparation des données
        X = df.select_dtypes(include=[np.number])
        if X.empty:
            st.error("⚠️ Le fichier ne contient pas de données numériques!")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Sidebar pour les paramètres d'analyse
            with st.sidebar:
                st.subheader("🎯 Paramètres d'analyse")
                max_clusters = min(10, len(df))
                k = st.slider('Nombre de clusters:', min_value=2, max_value=max_clusters, value=3)
                linkage_method = st.selectbox(
                    "Méthode de liaison hiérarchique:",
                    ["ward", "complete", "average", "single"],
                    help="Choisissez la méthode de liaison pour la classification hiérarchique"
                )

            # Analyse des clusters avec tabs
            tab1, tab2, tab3 = st.tabs(["🎯 K-means", "🌳 Classification Hiérarchique", "📈 Métriques"])

            with tab1:
                st.subheader("Analyse K-means")
                
                # K-means clustering
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                
                # PCA pour la visualisation
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                # Visualisation interactive avec Plotly
                fig_kmeans = px.scatter(
                    x=X_pca[:, 0], y=X_pca[:, 1],
                    color=[f"Cluster {c}" for c in clusters],
                    title="Visualisation des clusters K-means (PCA)",
                    labels={"x": "Premier composant principal", "y": "Second composant principal"}
                )
                st.plotly_chart(fig_kmeans, use_container_width=True)

                # Centres des clusters
                st.subheader("🎯 Centres des clusters")
                centers_df = pd.DataFrame(
                    scaler.inverse_transform(kmeans.cluster_centers_),
                    columns=X.columns
                )
                st.dataframe(centers_df.style.highlight_max(axis=0))

            with tab2:
                st.subheader("Classification Hiérarchique")
                
                # Calcul de la liaison
                linkage_matrix = linkage(X_scaled, method=linkage_method)
                
                # Dendrogramme interactif
                fig_dendro = go.Figure()
                fig_dendro.add_trace(go.Scatter(
                    x=[], y=[],
                    mode='markers',
                    showlegend=False
                ))
                
                dendro = dendrogram(linkage_matrix, no_plot=True)
                
                fig_dendro.add_trace(go.Scatter(
                    x=dendro['icoord'], y=dendro['dcoord'],
                    mode='lines',
                    line=dict(color='rgb(31, 119, 180)'),
                    name='Dendrogramme'
                ))
                
                fig_dendro.update_layout(
                    title="Dendrogramme de la classification hiérarchique",
                    xaxis_title="Index de l'échantillon",
                    yaxis_title="Distance",
                    showlegend=False
                )
                st.plotly_chart(fig_dendro, use_container_width=True)

            with tab3:
                st.subheader("📊 Métriques d'évaluation")
                
                # Calcul du score silhouette
                silhouette_avg = silhouette_score(X_scaled, clusters)
                
                # Affichage des métriques
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Score Silhouette",
                        f"{silhouette_avg:.3f}",
                        help="Score entre -1 et 1, plus il est proche de 1, meilleurs sont les clusters"
                    )
                with col2:
                    st.metric(
                        "Inertie",
                        f"{kmeans.inertia_:.2f}",
                        help="Somme des distances au carré des échantillons à leur centre de cluster le plus proche"
                    )

            # Statistiques des clusters avec expander
            st.subheader("📈 Statistiques détaillées")
            df['Cluster'] = clusters
            
            for i in range(k):
                with st.expander(f"Statistiques du Cluster {i}"):
                    cluster_data = df[df['Cluster'] == i]
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"Nombre d'éléments: {len(cluster_data)}")
                        st.dataframe(cluster_data.describe())
                    
                    with col2:
                        # Distribution des variables numériques dans le cluster
                        fig = px.box(cluster_data, 
                                   title=f"Distribution des variables dans le Cluster {i}")
                        st.plotly_chart(fig, use_container_width=True)

            # Génération du rapport PDF
            if st.sidebar.button('📑 Générer rapport PDF'):
                with st.spinner('Génération du rapport en cours...'):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 16)
                    pdf.cell(0, 10, 'Rapport d\'analyse de clustering', ln=True, align='C')
                    
                    # Sauvegarder les figures
                    for fig, title in [
                        (fig_kmeans, 'Clusters K-means'),
                        (fig_dendro, 'Dendrogramme')
                    ]:
                        img_path = f'temp_{title}.png'
                        fig.write_image(img_path)
                        pdf.add_page()
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, title, ln=True)
                        pdf.image(img_path, x=10, w=190)
                    
                    # Statistiques
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, 'Métriques d\'évaluation', ln=True)
                    pdf.set_font('Arial', '', 12)
                    pdf.cell(0, 10, f'Score Silhouette: {silhouette_avg:.3f}', ln=True)
                    pdf.cell(0, 10, f'Inertie: {kmeans.inertia_:.2f}', ln=True)
                    
                    # Générer le PDF
                    pdf_output = io.BytesIO()
                    pdf.output(pdf_output)
                    pdf_output.seek(0)
                    
                    st.sidebar.download_button(
                        label="⬇️ Télécharger le rapport PDF",
                        data=pdf_output,
                        file_name="rapport_clustering.pdf",
                        mime="application/pdf"
                    )

    except Exception as e:
        st.error(f"Une erreur s'est produite: {str(e)}")
        if st.button("Réessayer"):
            st.experimental_rerun()

else:
    # Message d'accueil
    st.info("""
        👋 Bienvenue dans l'analyseur de clustering !
        
        Pour commencer :
        1. Utilisez le panneau de gauche pour charger votre fichier Excel
        2. Ajustez les paramètres d'analyse selon vos besoins
        3. Explorez les résultats interactifs
        
        Les données doivent contenir des variables numériques pour l'analyse.
    """)