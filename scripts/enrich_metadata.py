"""
Enrichissement des métadonnées des livres manquants.
Utilise Open Library (prioritaire) et Google Books (fallback).
Sauvegarde incrémentale pour reprise en cas d'interruption.

Usage:
    python scripts/enrich_metadata.py                # Lancer l'enrichissement
    python scripts/enrich_metadata.py --limit 10     # Tester sur 10 livres
    python scripts/enrich_metadata.py --validate     # Valider le résultat
"""

import argparse
import os
import re
import time
import requests
import pandas as pd

# --- Header requis par Open Library (identification de l'application) ---
HEADERS = {"User-Agent": "BookRecommenderProject/1.0 (contact@projetintegrateur.com)"}

# --- Chemins des fichiers ---
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
META_PATH = os.path.join(DATA_DIR, "collaborative_book_metadata.csv")       # metadata existante (96 livres)
RATINGS_PATH = os.path.join(DATA_DIR, "collaborative_books_df.csv")         # tous les livres (898)
PROGRESS_PATH = os.path.join(DATA_DIR, "enrichment_progress.csv")           # sauvegarde incrementale
OUTPUT_PATH = os.path.join(DATA_DIR, "enriched_collaborative_book_metadata.csv")     # fichier final enrichi
BACKUP_PATH = os.path.join(DATA_DIR, "collaborative_book_metadata_original.csv")  # backup avant ecrasement

# Les 10 genres du dataset original
CANONICAL_GENRES = [
    "children", "comics, graphic", "fantasy, paranormal", "fiction",
    "history, historical fiction, biography", "mystery, thriller, crime",
    "non-fiction", "poetry", "romance", "young-adult",
]

# Mapping : mot-cle API -> genre canonique
GENRE_KEYWORDS = {
    "fantasy": "fantasy, paranormal",
    "paranormal": "fantasy, paranormal",
    "supernatural": "fantasy, paranormal",
    "magic": "fantasy, paranormal",
    "vampire": "fantasy, paranormal",
    "werewol": "fantasy, paranormal",
    "witch": "fantasy, paranormal",
    "romance": "romance",
    "love stor": "romance",
    "mystery": "mystery, thriller, crime",
    "thriller": "mystery, thriller, crime",
    "crime": "mystery, thriller, crime",
    "detective": "mystery, thriller, crime",
    "suspense": "mystery, thriller, crime",
    "young adult": "young-adult",
    "juvenile": "young-adult",
    "teen": "young-adult",
    "children": "children",
    "comic": "comics, graphic",
    "graphic novel": "comics, graphic",
    "manga": "comics, graphic",
    "history": "history, historical fiction, biography",
    "historical": "history, historical fiction, biography",
    "biography": "history, historical fiction, biography",
    "memoir": "history, historical fiction, biography",
    "non-fiction": "non-fiction",
    "nonfiction": "non-fiction",
    "self-help": "non-fiction",
    "science": "non-fiction",
    "poetry": "poetry",
    "poem": "poetry",
    "fiction": "fiction",
}


def clean_title(title):
    """Nettoie le titre pour ameliorer la recherche API.

    Enleve le numero de tome et le nom de serie quand il est colle au titre.
    Ex: "The Name of the Wind The Kingkiller Chronicle 1" -> "The Name of the Wind"
    """
    # Enlever le numero final (ex: "Betrayed House of Night 2" -> "Betrayed House of Night")
    cleaned = re.sub(r"\s+\d+\s*$", "", title)
    # Si le titre est duplique (ex: "Anna and the French Kiss Anna and the French Kiss 1")
    # on garde seulement la premiere moitie
    half = len(cleaned) // 2
    if half > 5:
        first_half = cleaned[:half].strip()
        if cleaned[half:].strip().startswith(first_half):
            cleaned = first_half
    return cleaned


def map_genres(categories):
    """Mappe une liste de catégories API vers les genres canoniques."""
    if not categories:
        return "['fiction']"
    matched = set()
    for cat in categories:
        cat_lower = cat.lower()
        for keyword, genre in GENRE_KEYWORDS.items():
            if keyword in cat_lower:
                matched.add(genre)
    if not matched:
        matched.add("fiction")
    return str(sorted(matched))


def _search_open_library(query_params):
    """Appel bas-niveau a Open Library. Retourne un dict ou None."""
    try:
        # Etape 1 : recherche du livre
        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={**query_params, "limit": 1,
                    "fields": "key,title,author_name,number_of_pages_median,subject"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("docs"):
            return None

        doc = data["docs"][0]
        work_key = doc.get("key", "")

        # Etape 2 : recuperer la description via l'endpoint /works/
        description = ""
        if work_key:
            time.sleep(0.5)
            work_resp = requests.get(
                f"https://openlibrary.org{work_key}.json", timeout=10
            )
            if work_resp.ok:
                work_data = work_resp.json()
                desc = work_data.get("description", "")
                # La description peut etre un string ou un dict {"value": "..."}
                if isinstance(desc, dict):
                    description = desc.get("value", "")
                elif isinstance(desc, str):
                    description = desc

        # Recuperer les sujets et les mapper vers nos genres
        subjects = doc.get("subject", [])[:20]
        return {
            "description": description.replace("\n", " ").replace("\r", " "),
            "num_pages": doc.get("number_of_pages_median"),
            "genre": map_genres(subjects),
            "source": "open_library",
        }
    except requests.exceptions.RequestException:
        return None


def fetch_open_library(title):
    """Cherche un livre sur Open Library avec 3 strategies de recherche."""
    # Essai 1 : titre complet tel quel
    result = _search_open_library({"title": title})
    if result and result.get("description"):
        return result

    # Essai 2 : titre nettoye (sans nom de serie / numero)
    cleaned = clean_title(title)
    if cleaned != title:
        time.sleep(0.5)
        result2 = _search_open_library({"title": cleaned})
        if result2 and result2.get("description"):
            return result2
        if result2 and not result:
            result = result2

    # Essai 3 : recherche generale (plus large, moins precise)
    if not result:
        time.sleep(0.5)
        result = _search_open_library({"q": cleaned or title})

    return result


def fetch_google_books(title):
    """Cherche un livre sur Google Books. Utilise comme fallback si Open Library echoue."""
    try:
        resp = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params={"q": title, "maxResults": 1},
            timeout=15,
        )
        # Google limite a ~1000 requetes/jour sans cle API
        if resp.status_code == 429:
            print("    [Google Books] Rate limit atteint, skip.")
            return None
        resp.raise_for_status()
        data = resp.json()
        if not data.get("items"):
            return None

        info = data["items"][0].get("volumeInfo", {})
        description = info.get("description", "")
        categories = info.get("categories", [])
        return {
            "description": description.replace("\n", " ").replace("\r", " "),
            "num_pages": info.get("pageCount"),
            "genre": map_genres(categories),
            "source": "google_books",
        }
    except requests.exceptions.RequestException:
        return None


def get_missing_books():
    """Compare les 898 livres du dataset avec les 96 qui ont deja des metadata.
    Retourne les livres manquants."""
    meta = pd.read_csv(META_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    all_books = ratings[["title", "book_id", "book_id_mapping"]].drop_duplicates("book_id_mapping")
    missing = all_books[~all_books["book_id_mapping"].isin(meta["book_id_mapping"])]
    return missing.reset_index(drop=True)


def load_progress():
    """Charge les livres deja traites pour pouvoir reprendre apres une interruption."""
    if os.path.exists(PROGRESS_PATH):
        return set(pd.read_csv(PROGRESS_PATH)["book_id_mapping"].tolist())
    return set()


def save_row(row, first_write):
    """Sauvegarde un livre enrichi dans le fichier de progres (ajout ligne par ligne)."""
    df = pd.DataFrame([row])
    df.to_csv(PROGRESS_PATH, mode="a", header=first_write, index=False)


def enrich(limit=None):
    """Boucle principale : pour chaque livre manquant, appelle les APIs et sauvegarde."""
    missing = get_missing_books()
    done = load_progress()
    first_write = not os.path.exists(PROGRESS_PATH)

    # Exclure les livres deja traites (reprise automatique)
    todo = missing[~missing["book_id_mapping"].isin(done)]

    if limit:
        todo = todo.head(limit)

    total = len(todo)
    success, fail = 0, 0
    failed_titles = []

    print(f"Livres a enrichir : {total} (deja faits : {len(done)})\n")

    for i, (_, book) in enumerate(todo.iterrows(), 1):
        title = book["title"]
        print(f"[{i}/{total}] {title} ... ", end="", flush=True)

        # 1) Essayer Open Library en premier (pas de quota)
        result = fetch_open_library(title)

        # 2) Si pas de description, essayer Google Books en fallback
        if not result or not result.get("description"):
            gb = fetch_google_books(clean_title(title))
            if gb:
                if not result:
                    result = gb
                elif gb.get("description"):
                    # Garder les genres Open Library mais prendre la description Google
                    result["description"] = gb["description"]
                    result["source"] = "google_books"

        # 3) Sauvegarder le resultat (meme partiel, le genre seul est utile)
        if result:
            row = {
                "book_id": book["book_id"],
                "title": title,
                "num_pages": result.get("num_pages"),
                "ratings_count": None,
                "description": result.get("description", ""),
                "genre": result.get("genre", "['fiction']"),
                "book_id_mapping": book["book_id_mapping"],
            }
            save_row(row, first_write)
            first_write = False
            success += 1
            src = result.get("source", "?")
            has_desc = "+" if result.get("description") else "-desc"
            print(f"OK [{src}] {has_desc}")
        else:
            fail += 1
            failed_titles.append(title)
            print("ECHEC")

        # Pause entre chaque requete pour ne pas surcharger les APIs
        time.sleep(1)

    # Afficher le bilan
    print(f"\n--- Resume ---")
    print(f"Enrichis : {success}/{total}")
    print(f"Echoues  : {fail}/{total}")
    if failed_titles:
        print(f"\nTitres echoues :")
        for t in failed_titles:
            print(f"  - {t}")


def merge():
    """Fusionne les 96 livres originaux + les livres enrichis en un seul CSV."""
    if not os.path.exists(PROGRESS_PATH):
        print("Aucun fichier de progres trouve.")
        return

    # Sauvegarder l'original avant de l'ecraser
    if not os.path.exists(BACKUP_PATH):
        meta = pd.read_csv(META_PATH)
        meta.to_csv(BACKUP_PATH, index=False)
        print(f"Backup cree : {BACKUP_PATH}")

    # Charger l'original et supprimer les colonnes inutiles (image_url, url, name)
    meta = pd.read_csv(META_PATH)
    cols_to_drop = [c for c in ["Unnamed: 0", "image_url", "url", "name"] if c in meta.columns]
    meta = meta.drop(columns=cols_to_drop)

    # Charger les livres enrichis
    progress = pd.read_csv(PROGRESS_PATH)

    # Concatener et supprimer les doublons
    enriched = pd.concat([meta, progress], ignore_index=True)
    enriched = enriched.drop_duplicates(subset="book_id_mapping", keep="first")
    enriched = enriched.sort_values("book_id_mapping").reset_index(drop=True)

    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Fichier enrichi sauvegarde : {OUTPUT_PATH}")
    print(f"Total livres : {len(enriched)}")


def validate():
    """Verifie la qualite du fichier enrichi (nombre de lignes, descriptions, genres)."""
    if not os.path.exists(OUTPUT_PATH):
        print("Fichier enrichi non trouve.")
        return

    df = pd.read_csv(OUTPUT_PATH)
    print(f"Lignes          : {len(df)}")
    print(f"Colonnes        : {list(df.columns)}")
    print(f"Descriptions    : {df['description'].notna().sum()} / {len(df)} "
          f"({df['description'].notna().mean():.0%})")
    print(f"Genres          : {df['genre'].notna().sum()} / {len(df)}")
    print(f"Num pages       : {df['num_pages'].notna().sum()} / {len(df)}")

    # Verifier que chaque genre est bien une liste Python valide
    import ast
    errors = 0
    for g in df["genre"].dropna():
        try:
            parsed = ast.literal_eval(g)
            if not isinstance(parsed, list):
                errors += 1
        except (ValueError, SyntaxError):
            errors += 1
    print(f"Genres mal formates : {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrichissement métadonnées livres")
    parser.add_argument("--limit", type=int, help="Limiter le nombre de livres à traiter")
    parser.add_argument("--validate", action="store_true", help="Valider le fichier enrichi")
    parser.add_argument("--merge", action="store_true", help="Fusionner progrès avec l'original")
    args = parser.parse_args()

    if args.validate:
        validate()
    elif args.merge:
        merge()
    else:
        enrich(limit=args.limit)
