import argparse
import os
import re
import pandas as pd
import json
import requests
import shutil
import tempfile
from glob import glob

import torch
import clip
from PIL import Image
from icrawler.builtin import BingImageCrawler
from utils import remove_and_rename_images
from build_csv import make_record, save_character_table


IMDB_GRAPHQL_URL = "https://caching.graphql.imdb.com/"

GRAPHQL_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "x-imdb-client-name": "imdb-web-next-localized",
    "x-imdb-user-country": "US",
    "x-imdb-user-language": "en-US",
}

CAST_QUERY = """
query TitleCast($id: ID!, $numCast: Int!) {
  title(id: $id) {
    titleText { text }
    cast: credits(first: $numCast, filter: { categories: ["actor", "actress"] }) {
      edges {
        node {
          name {
            nameText { text }
          }
          ... on Cast {
            characters {
              name
            }
          }
        }
      }
    }
  }
}
"""

_FETCH_ALL_LIMIT = 9999  # IMDB requires `first`; use a large value to fetch all
_IMDBID_RE = re.compile(r"(tt\d+)")


def extract_imdbid(imdb_link):
    """Extract the `tt########` id from an IMDB url such as
    'https://www.imdb.com/title/tt0090633/'."""
    if not isinstance(imdb_link, str):
        return None
    m = _IMDBID_RE.search(imdb_link)
    return m.group(1) if m else None


def get_character_names(movie_title, imdbid, session=None, num_cast=None):
    """Fetch character names for a movie from IMDB's GraphQL cast endpoint.

    Returns a de-duplicated list of character names (preserving IMDB billing
    order). An actor can be credited for multiple roles, so we flatten all
    character entries rather than only taking the first.
    """
    if session is None:
        session = requests.Session()

    n = num_cast if num_cast is not None else _FETCH_ALL_LIMIT
    payload = {"query": CAST_QUERY, "variables": {"id": imdbid, "numCast": n}}

    try:
        resp = session.post(
            IMDB_GRAPHQL_URL, headers=GRAPHQL_HEADERS, json=payload, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  [ERROR] {movie_title} ({imdbid}): {e}")
        return []

    title_data = data.get("data", {}).get("title")
    if not title_data:
        print(f"  [WARN] {movie_title} ({imdbid}): no title data returned")
        return []

    edges = title_data.get("cast", {}).get("edges", [])

    character_names = []
    seen = set()
    for edge in edges:
        node = edge.get("node") or {}
        char_list = node.get("characters") or []
        for char in char_list:
            name = (char or {}).get("name")
            if not name:
                continue
            # Animated titles tag voice roles as "Bob Smith (voice)" — strip
            # the suffix so "Bob Smith (voice)" and "Bob Smith" dedupe cleanly.
            name = name.replace(" (voice)", "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            character_names.append(name)

    return character_names


def is_valid_profile_image(path, min_size=128, max_aspect=3.0):
    """Reject tiny, corrupt, or extreme-aspect-ratio images.
    Profile/portrait shots are roughly square-ish, so we filter panoramic
    banners and razor-thin strips that are almost never accurate profiles.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
        if w < min_size or h < min_size:
            return False
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > max_aspect:
            return False
        return True
    except Exception:
        return False


def crawl_with_engine(CrawlerClass, engine_name, query, num_images, tmp_dir):
    """Run a single search-engine crawler safely and return the list of
    downloaded file paths. Errors are caught so one failing engine does not
    abort the whole pipeline."""
    try:
        crawler = CrawlerClass(storage={'root_dir': tmp_dir})
        crawler.crawl(keyword=query, max_num=num_images)
    except Exception as e:
        print(f"[{engine_name}] Failed to crawl '{query}': {e}")
        return []
    return [
        os.path.join(tmp_dir, fname)
        for fname in sorted(os.listdir(tmp_dir))
    ]


# ---- Character description search ----

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible)"}


def search_character_description(character_name, movie_title, session=None):
    """Search for a textual description of the character from web sources.

    Tries DuckDuckGo Instant Answer API, then Wikipedia REST / search APIs.
    Returns the description string or ``None`` if nothing useful is found.
    """
    if session is None:
        session = requests.Session()

    # 1. DuckDuckGo Instant Answer API (returns wiki summaries)
    try:
        resp = session.get(
            "https://api.duckduckgo.com/",
            params={
                "q": f"{character_name} {movie_title} character",
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            headers=_HEADERS,
            timeout=10,
        )
        text = resp.json().get("AbstractText", "")
        if len(text) > 30:
            return text
    except Exception:
        pass

    # 2. Wikipedia REST API — direct page lookup
    try:
        title = character_name.replace(" ", "_")
        resp = session.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers=_HEADERS,
            timeout=10,
        )
        if resp.status_code == 200:
            text = resp.json().get("extract", "")
            if len(text) > 30:
                return text
    except Exception:
        pass

    # 3. Wikipedia search API — find mentions in any article
    try:
        resp = session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": f"{character_name} {movie_title}",
                "format": "json",
                "srlimit": 1,
            },
            headers=_HEADERS,
            timeout=10,
        )
        results = resp.json().get("query", {}).get("search", [])
        if results:
            snippet = re.sub(r"<[^>]+>", "", results[0].get("snippet", ""))
            if len(snippet) > 30:
                return snippet
    except Exception:
        pass

    return None


# ---- CLIP-based quality filter ----

_clip_model = None
_clip_preprocess = None


def _load_clip(device="cpu"):
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def compute_clip_scores(image_paths, character_name, movie_title,
                        description=None, device="cpu"):
    """Score images by CLIP similarity to the target character, penalising
    group shots so that single-character portraits rank highest.

    If a textual *description* of the character is provided it is used as the
    positive text prompt, giving CLIP a much stronger visual anchor than the
    character name alone.
    """
    model, preprocess = _load_clip(device)

    if description:
        pos_text = f"{character_name} from {movie_title}. {description}"
    else:
        pos_text = (
            f"a portrait of {character_name} from the animated film {movie_title}"
        )
    neg_text = "a group photo with multiple characters"
    text_tokens = clip.tokenize([pos_text, neg_text], truncate=True).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores = []
    for path in image_paths:
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                img_features = model.encode_image(img)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                sims = (img_features @ text_features.T).squeeze()
            # relevance to character minus group-image penalty
            scores.append(sims[0].item() - sims[1].item())
        except Exception:
            scores.append(float("-inf"))
    return scores


def crawl_profile_images_for_character(
    character_name,
    movie_title,
    save_dir,
    num_per_query=10,
    max_per_character=8,
    device="cpu",
    session=None,
):
    """Crawl candidate images for a character, score them with CLIP, and keep
    only the top `max_per_character` most relevant single-character images.

    A textual description of the character is searched online and used as the
    CLIP text prompt when available, giving much better visual matching than
    the character name alone.

    Returns `(n_saved, anchor)`, where `anchor` is the save_dir-relative path of
    the `_0` image (the best-scoring one) or None when nothing was saved.
    """
    # Search for a textual description to use as CLIP prompt
    description = search_character_description(
        character_name, movie_title, session=session
    )
    if description:
        print(f"    [CLIP] description: {description[:80]}...")

    queries = [
        f'"{character_name}" {movie_title} animated character',
        f'{character_name} {movie_title} character portrait',
    ]

    final_dir = os.path.join(save_dir, movie_title)
    os.makedirs(final_dir, exist_ok=True)

    # Stage 1: crawl candidates into a staging directory
    all_candidates = []
    with tempfile.TemporaryDirectory() as staging_dir:
        for query in queries:
            with tempfile.TemporaryDirectory() as tmp_dir:
                candidates = crawl_with_engine(
                    BingImageCrawler, "bing", query, num_per_query, tmp_dir
                )
                for src in candidates:
                    ext = os.path.splitext(src)[-1].lower()
                    if ext not in ('.jpg', '.jpeg', '.png'):
                        continue
                    if not is_valid_profile_image(src):
                        continue
                    staged = os.path.join(
                        staging_dir, f"cand_{len(all_candidates)}{ext}"
                    )
                    shutil.copy2(src, staged)
                    all_candidates.append(staged)

        if not all_candidates:
            return 0, None

        # Stage 2: rank by CLIP score and keep the best
        scores = compute_clip_scores(
            all_candidates, character_name, movie_title,
            description=description, device=device,
        )
        ranked = sorted(zip(scores, all_candidates), reverse=True)
        selected = ranked[:max_per_character]

        # Stage 3: save to final directory
        n_saved = 0
        anchor = None
        for _score, src in selected:
            ext = os.path.splitext(src)[-1].lower()
            dst = os.path.join(final_dir, f"{character_name}_{n_saved}{ext}")
            shutil.copy2(src, dst)
            if n_saved == 0:
                anchor = os.path.join(movie_title, os.path.basename(dst))
            n_saved += 1

    return n_saved, anchor


def build_online_profile_images(
    movie_imdbids, save_dir, num_per_query=10, max_per_character=8,
    num_cast=None, device="cpu", save_file=None,
):
    """For each movie in `movie_imdbids`, look up its characters via the
    IMDB cast graph and crawl profile images, filtering with CLIP to keep
    only the most relevant single-character images.

    Args:
        movie_imdbids: dict mapping movie title -> IMDB id (tt########).
        save_file: character table to flush after every movie, so an
            interrupted crawl still leaves the movies it finished on disk.

    Returns `(movie_title_to_characters, records)`, the same pair build_csv.py
    returns, so both builders describe their output the same way.
    """
    movie_title_to_characters = {}
    records = []
    session = requests.Session()

    for movie_title, imdbid in movie_imdbids.items():
        if not imdbid:
            print(f"[WARN] Skipping {movie_title}: no imdbid")
            continue

        character_names = get_character_names(
            movie_title, imdbid, session=session, num_cast=num_cast
        )
        movie_title_to_characters[movie_title] = character_names

        print(f"=== {movie_title} ({imdbid}): {len(character_names)} characters ===")
        for character_name in character_names:
            print(f"  -> crawling '{character_name}'")
            n_saved, anchor = crawl_profile_images_for_character(
                character_name=character_name,
                movie_title=movie_title,
                save_dir=save_dir,
                num_per_query=num_per_query,
                max_per_character=max_per_character,
                device=device,
                session=session,
            )
            print(f"     saved {n_saved} images for {character_name}")

            # The anchor here is the best-scoring image search result, not a
            # portrait any source declares as this character's own, so it can
            # never reach build_csv.py's 'confident'; a character the crawl
            # returned nothing for is 'missing', exactly as there.
            record = make_record(None, anchor, "bing" if anchor else None, "imdb")
            record["movie"] = movie_title
            records.append((character_name, record))

        if save_file is not None:
            save_character_table(records, save_file)

    return movie_title_to_characters, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_csv', default="src.csv", type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--save_file', default=None, type=str,
                        help="Character table to write; CSV when the path ends "
                             "in .csv, otherwise JSON rows.")
    parser.add_argument('--num_per_query', default=10, type=int,
                        help="Images to request per query.")
    parser.add_argument('--max_per_character', default=8, type=int,
                        help="Max images to keep per character after CLIP ranking.")
    parser.add_argument('--num_cast', default=None, type=int,
                        help="Top-billed cast members to fetch from IMDB per title.")
    parser.add_argument('--device', default=None, type=str,
                        help="Device for CLIP model (default: auto-detect).")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    src_csv = args.src_csv
    save_dir = args.save_dir
    save_file = args.save_file

    # Read the source CSV file
    df = pd.read_csv(src_csv)

    # Deduplicate: movie title -> imdbid (extracted from the IMDB link column)
    movie_imdbids = {}
    for _, row in df.iterrows():
        title = row["Movie title"]
        if title in movie_imdbids:
            continue
        imdbid = extract_imdbid(row.get("IMDB link"))
        movie_imdbids[title] = imdbid

    # Crawl profile images online
    movie_title_to_characters, records = build_online_profile_images(
        movie_imdbids,
        save_dir,
        num_per_query=args.num_per_query,
        max_per_character=args.max_per_character,
        num_cast=args.num_cast,
        device=device,
        save_file=save_file,
    )

    # Remove duplicates and re-index each character folder. De-duplication only
    # ever drops a repeat of an image that is already there, so the _0 anchor
    # each record names survives and keeps its name.
    for char_bank_dir in glob(os.path.join(save_dir, "*")):
        remove_and_rename_images(char_bank_dir)

    # Save the character information, in the table build_csv.py writes so the
    # two builders' output can be read the same way.
    if save_file is not None:
        save_character_table(records, save_file)
