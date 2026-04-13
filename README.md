# Détection de SMS Smishing — Deep Learning & Apprentissage Fédéré

Étude expérimentale de la détection de SMS de type *smishing* (SMS phishing) par Deep Learning et apprentissage fédéré (FedAvg), développée dans le cadre d'un stage de M1 sur l'apprentissage fédéré appliqué à la détection de *network slices* en 5G.

---

## Contexte

Le **smishing** exploite les SMS pour tromper les utilisateurs (usurpation bancaire, fausses livraisons, arnaques administratives). Contrairement à un système centralisé, l'apprentissage fédéré permet de détecter ces attaques de façon collaborative sans jamais partager les données SMS brutes — une propriété essentielle dans un cadre conforme au RGPD.

---

## Hypothèses testées

- **H1** : L'apprentissage fédéré permet d'atteindre des performances comparables à un apprentissage centralisé, sans partage des données brutes.
- **H2** : La correction locale d'un nouveau type de smishing chez un client peut se propager à l'ensemble du système via l'agrégation FedAvg.

---

## Résultats principaux

| Méthode | Accuracy | F1-score | Données partagées |
|---|---|---|---|
| Centralisé (100 % data) | 0.9892 | 0.9592 | Oui |
| Fédéré — 5 clients | 0.9874 | 0.9524 | Non |
| Fédéré — 10 clients | 0.9857 | 0.9463 | Non |
| Fédéré — 20 clients | 0.9821 | 0.9329 | Non |
| Fédéré — 50 clients | 0.9695 | 0.8759 | Non |
| Fédéré — 100 clients | 0.9363 | 0.7029 | Non |

**Propagation (H2) :** la correction de 11 faux négatifs chez le client 0 améliore le F1 global de **+0.0165** après agrégation FedAvg (0.9388 → 0.9553).

---

## Structure du projet

```
.
├── smishing_federated.ipynb   # Notebook principal (exécutable sur Google Colab)
├── rapport_smishing_FL.pdf     
├── README.md
└── dataset/
    └── SMSSmishCollection.txt
└── figures/                      # Images générées par le notebook
    ├── data_explo.png
    ├── results_CL.png
    ├── results_data_quantity.png
    ├── results_client_FL.png
    ├── results_propagation_FL.png
    └── results_recap.png
```

---

## Environnement & dépendances

| Composant | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.x |
| Flower (flwr) | 1.7+ |
| scikit-learn | 1.x |
| pandas / numpy | dernières stables |
| matplotlib / seaborn | dernières stables |
| Environnement recommandé | Google Colab (GPU T4) |

Installation en une commande :

```bash
pip install flwr[simulation] torch scikit-learn pandas numpy matplotlib seaborn --upgrade --quiet
```

---

## Dataset

**SMSSmishCollection.txt** — 5 572 SMS labellisés :

- `ham` : SMS légitime (4 825 — 86.6 %)
- `smish` : SMS malveillant (747 — 13.4 %)

Téléchargement : [Kaggle — SMS Smishing Collection](https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set)

Placer le fichier `SMSSmishCollection.txt` à la racine du projet (ou l'uploader dans Colab) avant exécution.

---

## Utilisation

### Google Colab (recommandé)

1. Ouvrir `smishing_federated_v4.ipynb` dans Google Colab
2. Uploader `SMSSmishCollection.txt` dans la session Colab
3. Exécuter toutes les cellules dans l'ordre (`Runtime > Run all`)

### Environnement local

```bash
git clone https://github.com/<user>/smishing-federated-learning
cd smishing-federated-learning
pip install -r requirements.txt
jupyter notebook smishing_federated_v4.ipynb
```

---

## Architecture du modèle

Réseau MLP entraîné sur des vecteurs TF-IDF (5 000 features, bigrammes) :

```
Input(5000)
  → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64)  → ReLU
  → Linear(1)   [logit — BCEWithLogitsLoss]
```

**Paramètres entraînables :** 1 322 241

---

## Implémentation FedAvg

L'agrégation fédérée est implémentée manuellement (sans dépendance au serveur Flower) pour une compatibilité maximale avec Colab :

```python
def fedavg(weights_list, sizes):
    total = sum(sizes)
    result = []
    for layer_idx in range(len(weights_list[0])):
        agg = np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
        for w, sz in zip(weights_list, sizes):
            agg += (sz / total) * w[layer_idx].astype(np.float64)
        result.append(agg.astype(np.float32))
    return result
```

---

## Structure du notebook

| Section | Contenu |
|---|---|
| 1. Introduction | Contexte smishing, formalisation FL, hypothèses |
| 2. Installation & imports | Dépendances, reproductibilité (seed) |
| 3. Chargement & exploration | EDA — distribution, longueur SMS, mots fréquents |
| 4. Prétraitement & TF-IDF | Nettoyage, vectorisation, train/test split |
| 5. Architecture MLP | Définition du modèle, fonctions d'entraînement |
| 6. Apprentissage centralisé | Baseline 30 époques — toutes métriques + confusion matrix |
| 7. Impact quantité données | Proportions 1 % → 100 % — seuil critique à 5 % |
| 8. Apprentissage fédéré | FedAvg — 2 à 100 clients, comparaison équitable |
| 9. Propagation connaissance | Correction locale client 0 → diffusion globale |
| 10. Comparaison centralisé/fédéré | Tableau de bord synthétique |
| 11. Analyse & discussion | Validation H1/H2, limites, perspectives |

---

## Limites connues

- **Partitionnement IID uniquement** : la distribution réelle des SMS varie selon les opérateurs et pays. Des algorithmes robustes au non-IID (FedProx, SCAFFOLD) seraient plus adaptés.
- **Dégradation au-delà de 50 clients** : avec ce dataset (5 572 SMS), chaque client reçoit moins de 90 SMS à 100 clients, insuffisant pour la classe minoritaire.
- **Pas de confidentialité différentielle** : les gradients pourraient théoriquement permettre une reconstruction partielle des données (attaques par inversion). DP-SGD constituerait une amélioration naturelle.
- **Clients homogènes** : tous les clients utilisent la même architecture et les mêmes hyperparamètres.

---

## Perspectives

- Algorithmes FL robustes au non-IID : **FedProx**, **SCAFFOLD**
- Confidentialité différentielle : **DP-SGD**
- Agrégation sécurisée avec chiffrement homomorphe
- Modèles de langage légers : **TinyBERT**, **MobileBERT**
- Détection d'anomalies pour filtrer les clients byzantins

---

## Références

- McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Li, T. et al. (2020). *Federated Learning: Challenges, Methods, and Future Directions.* IEEE Signal Processing Magazine.
- Beutel, D.J. et al. (2022). *Flower: A Friendly Federated Learning Research Framework.* arXiv:2007.14390.
- Alotaibi, A. & Roussinov, D. (2016). *Automated SMS Spam Detection.* IEEE.

---

## Licence

Ce projet est développé dans un cadre académique. Le code est librement réutilisable à des fins pédagogiques et de recherche.
