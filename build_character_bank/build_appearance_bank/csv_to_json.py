"""Convert the character table into the character list JSON the voice bank reads.

build_csv.py and build_online.py write one row per character, carrying the
provenance of its anchor image. build_voice_bank/interview_crawling.py wants
{movie title: [character names]} instead, so this collapses the table into that
shape rather than the character lists being rebuilt by hand.
"""
import argparse
import json
import os
import pandas as pd


def table_to_character_lists(character_table):
    """{movie title: [character names]} from the table written by the builders.

    Row order is preserved, so the characters stay in the order the builder
    ranked them (IMDB billing order, then wiki article length).
    """
    table = pd.read_csv(character_table)
    return {movie_title: list(group["Character"])
            for movie_title, group in table.groupby("Movie title", sort=False)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--character_table', required=True, type=str,
                        help="The CSV written by build_csv.py or build_online.py")
    parser.add_argument('--save_file', required=True, type=str,
                        help="Where to write the {movie: [characters]} JSON")
    args = parser.parse_args()

    movie_title_to_characters = table_to_character_lists(args.character_table)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_file)) or ".", exist_ok=True)
    with open(args.save_file, 'w') as outfile:
        json.dump(movie_title_to_characters, outfile, indent=4)

    n_characters = sum(len(names) for names in movie_title_to_characters.values())
    print(f"{len(movie_title_to_characters)} movies, {n_characters} characters "
          f"-> {args.save_file}")
