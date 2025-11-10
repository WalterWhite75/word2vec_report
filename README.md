# Projet Word2Vec - Analyse sémantique de synopsis de films

Ce projet vise à entraîner un modèle Word2Vec sur un large corpus de synopsis de films afin d'explorer les relations sémantiques entre les mots et les films. 
Il comprend les étapes de prétraitement, d'entraînement, d'évaluation et de visualisation des embeddings. 
Les résultats sont présentés sous forme de visualisations interactives et d’un rapport scientifique détaillé.

##  Structure du dépôt

Le projet est organisé de la manière suivante :

- `src/` : Contient les scripts Python pour le prétraitement des données, l'entraînement du modèle Word2Vec, les évaluations et les visualisations.
- `data/` : Regroupe les données brutes et prétraitées, notamment les synopsis et métadonnées des films.
- `w2v_out/` : Stocke les résultats générés, tels que les modèles entraînés, les fichiers de similarités, et les visualisations interactives.
- `rapport_word2vec.tex` et `rapport_word2vec.pdf` : Le rapport scientifique final en LaTeX et sa version PDF compilée.
- `README.md` : La documentation principale du projet, décrivant les objectifs, la structure et les instructions d'utilisation.
# 🎬 Projet Word2Vec — Analyse Sémantique de Synopsis de Films

## 🎯 Objectif du projet
Ce projet a pour ambition d’explorer les relations sémantiques entre les mots et les films à partir d’un vaste corpus de synopsis cinématographiques.  
Grâce à l’algorithme **Word2Vec**, il est possible de représenter les mots et les œuvres sous forme de vecteurs dans un espace sémantique afin d’identifier des similarités de sens, de ton ou de genre.

Les objectifs principaux :
- Entraîner un modèle Word2Vec adapté au domaine cinématographique.  
- Visualiser les relations entre mots et films grâce à des projections t-SNE.  
- Analyser la cohérence des genres et la proximité sémantique entre œuvres.  
- Fournir un rapport scientifique complet, reproductible et bien documenté.

---

## 🧠 Approche méthodologique

1. **Prétraitement des données**  
   - Nettoyage des synopsis (tokenisation, lemmatisation, suppression du bruit).  
   - Filtrage des tokens rares ou trop fréquents via les paramètres `min_df` et `max_df`.  

2. **Entraînement du modèle Word2Vec**  
   - Utilisation du modèle **Skip-Gram** (SG=1) avec une fenêtre contextuelle dynamique.  
   - Optimisation des hyperparamètres : taille des vecteurs, fenêtre, nombre d’époques, échantillonnage négatif.  

3. **Évaluation et visualisation**  
   - Calcul de similarités cosinus entre films.  
   - Mesure de cohérence entre genres.  
   - Visualisation t-SNE des mots et des films.  
   - Génération d’un **dashboard analytique** pour l’exploration interactive.

---

## 🗂️ Structure du dépôt

```
w2v_project/
│
├── src/                        # Scripts Python (prétraitement, entraînement, visualisations)
│   ├── w2v_movies.py           # Script principal d’entraînement et d’analyse
│   ├── utils/                  # Fonctions utilitaires
│   └── plots/                  # Fonctions de visualisation (t-SNE, heatmaps, etc.)
│
├── data/                       # Données brutes et prétraitées
│   ├── raw/                    # Données sources (synopsis, genres, métadonnées)
│   └── processed/              # Corpus nettoyé et prêt pour Word2Vec
│
├── w2v_out/                    # Résultats générés
│   ├── w2v_movies.kv           # Modèle Word2Vec entraîné
│   ├── movie_embeddings.csv    # Représentations vectorielles des films
│   ├── similar_movies_toy_story.csv
│   ├── tsne_words.png          # Visualisation t-SNE du vocabulaire
│   ├── tsne_movies_by_genre.png# Visualisation t-SNE des films par genre
│   └── dashboard.png           # Tableau de bord analytique
│
├── rapport_word2vec.tex        # Rapport LaTeX complet du projet
├── rapport_word2vec.pdf        # Rapport scientifique final
└── README.md                   # Documentation principale du dépôt
```

---

## 🚀 Exécution rapide

```bash
# 1️⃣ Lancer l’entraînement complet
python src/w2v_movies.py --no-light --window 10 --vector-size 200 --min-count 10 --negative 15 --sg 1 --epochs 10

# 2️⃣ Recalculer les visualisations t-SNE
python src/w2v_movies.py --recompute-tsne

# 3️⃣ Générer le rapport final
tectonic rapport_word2vec.tex
```

---

## 📊 Résultats clés

- **Entropie du corpus :** ~12.25 bits  
- **Sanity Check (genre agreement) :** ≈ 73 %  
- **Films similaires à “Toy Story” :**
  - *Listen, Darling*  
  - *Geek Charming*  
  - *The Toy*  
  - *Holiday Reunion*  
  - *Barbie Diaries*

Les visualisations t-SNE et le dashboard permettent d’explorer les relations sémantiques et les regroupements de films selon leur thématique et leur ton.

---

## ✍️ Auteur
**Mevlut Cakin (WalterWhite75)**  
Master 2 BIDABI — Université Sorbonne Paris Nord  
📧 [GitHub](https://github.com/WalterWhite75)