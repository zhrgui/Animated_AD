"""Run only the retrieval stage of the character bank build.

The anchor images and the character table are already on disk, so this rebuilds
the inputs build_retrieval_images needs from that table instead of re-querying
IMDb and re-downloading every anchor. Characters that already have their full
quota of example images are skipped, so it can be re-run to pick up where an
interrupted crawl left off.
"""
import argparse
import os
import pandas as pd

from build_csv import build_retrieval_images, extract_imdbid, parse_fandom_url


def load_state(character_table, src_csv):
    """Rebuild (movie -> characters) and (movie -> (api url, character -> page))
    from the character table produced by build_csv.py."""
    table = pd.read_csv(character_table)
    links = {}
    for _, row in pd.read_csv(src_csv).iterrows():
        links.setdefault(row["Movie title"], row["Fandom link"])

    movie_title_to_characters, character_pages = {}, {}
    for movie_title, group in table.groupby("Movie title", sort=False):
        movie_title_to_characters[movie_title] = list(group["Character"])
        api_url = parse_fandom_url(links[movie_title])[0] if movie_title in links else None
        pages = {row["Character"]: row["Wiki page"]
                 for _, row in group.iterrows()
                 if isinstance(row["Wiki page"], str) and row["Wiki page"]}
        character_pages[movie_title] = (api_url, pages)
    return movie_title_to_characters, character_pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_csv', default="build_character_bank/build_appearance_bank/src.csv", type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--character_table', required=True, type=str,
                        help="The CSV written by build_csv.py")
    parser.add_argument('--k_retrieval', default=15, type=int)
    args = parser.parse_args()

    movie_title_to_characters, character_pages = load_state(
        args.character_table, args.src_csv)

    done = sum(1 for movie, names in movie_title_to_characters.items()
               for name in names
               if len([f for f in os.listdir(os.path.join(args.save_dir, movie))
                       if f.startswith(f"{name}_") and not f.endswith("_0.png")]) >= args.k_retrieval)
    total = sum(len(v) for v in movie_title_to_characters.values())
    print(f"{total} characters, {done} already complete, {total - done} to crawl")

    build_retrieval_images(movie_title_to_characters, character_pages,
                           args.save_dir, args.k_retrieval)
