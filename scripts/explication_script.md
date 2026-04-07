# Explication du script enrich_metadata.py

## 1. Les imports et les chemins
```python
import requests, pandas, re, time, argparse, os
```
On charge les librairies : `requests` pour appeler les APIs, `pandas` pour lire/ecrire les CSV, `time` pour les pauses entre requetes. Les chemins pointent vers les fichiers dans `data/`.

## 2. GENRE_KEYWORDS (dictionnaire)
Un tableau de correspondance. Les APIs renvoient des categories en anglais ("fantasy", "thriller"...). Ce dictionnaire les convertit vers les 10 genres de notre dataset original (ex: "detective" -> "mystery, thriller, crime").

## 3. clean_title(title)
Nettoie les titres avant la recherche API. Enleve le numero de tome et le nom de serie colle au titre. Exemple : "The Name of the Wind The Kingkiller Chronicle 1" devient "The Name of the Wind".

## 4. map_genres(categories)
Prend les categories renvoyees par l'API, les compare aux mots-cles, et retourne les genres au format du dataset : `"['fiction', 'romance']"`.

## 5. _search_open_library(query_params)
Appel bas-niveau a Open Library. Cherche le livre puis recupere la description via l'endpoint /works/.

## 6. fetch_open_library(title)
Essaie 3 strategies pour trouver le livre :
1. Titre complet
2. Titre nettoye (sans serie)
3. Recherche generale (plus large)

## 7. fetch_google_books(title)
Fallback : si Open Library ne trouve rien, on essaie Google Books. Limite a ~1000 requetes/jour sans cle API.

## 8. enrich(limit=None)
La boucle principale. Pour chaque livre manquant : appelle Open Library, puis Google Books si besoin, sauvegarde le resultat ligne par ligne. Si on coupe le script, il reprend ou il s'etait arrete.

## 9. merge()
Fusionne les 96 livres originaux + les livres enrichis en un seul CSV final.

## 10. validate()
Verifie la qualite : nombre de lignes, descriptions non vides, genres bien formates.

---

## Rapport d'enrichissement

| Metrique | Valeur |
|----------|--------|
| Livres avant | 96 / 898 (10.7%) |
| Livres apres | 884 / 898 (98.4%) |
| Descriptions obtenues | 680 / 884 (77%) |
| Genres obtenus | 884 / 884 (100%) |
| Nombre de pages | 751 / 884 (85%) |
| Livres echoues | 15 (titres avec series complexes) |
| Genres mal formates | 0 |
| Source principale | Open Library |
| Source fallback | Google Books |

**Avant** : le content-based (TF-IDF / Jina) ne fonctionnait que sur 96 livres (10.7%).
**Apres** : il couvre 884 livres (98.4%), dont 680 avec des descriptions reelles d'editeurs.
