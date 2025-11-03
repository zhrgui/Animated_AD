import argparse
import os
import pandas as pd
import json
import requests
import urllib.request
import shutil
import tempfile
from bs4 import BeautifulSoup
from glob import glob
from icrawler.builtin import BingImageCrawler
from utils import remove_and_rename_images

def get_character_names(movie_name, url):
    """Scrapes the character names from the "Trending Pages" section of a Fandom wiki page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    trending_section = soup.find("ul", class_="category-page__trending-pages")

    if trending_section is None:
        print("Trending section not found on the page.")
    else:
        trending_items = trending_section.find_all("a", href=True)
        character_names = [item['href'].split('/wiki/')[-1].replace("_", " ") for item in trending_items]
    return character_names

def download_image(image_url, save_path):
    """Download image from a specific URL to a specific path."""
    try:
        urllib.request.urlretrieve(image_url, save_path)
        print(f"Image downloaded and saved to {save_path}")
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")

def get_profile_image_url(source_url, character_name):
    """Extracts the profile image URL from a web page that contains structured data in JSON-LD format (type="application/ld+json")."""
    source_url = source_url.replace(source_url.split("/")[-1], character_name)
    print(f"Fetching from: {source_url}")

    response = requests.get(source_url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve page, status code: {response.status_code}")
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    script_tag = soup.find('script', type='application/ld+json')
    if script_tag is None:
        return None
    try:
        json_data = json.loads(script_tag.string)
        profile_image_url = json_data["image"]
    except (KeyError, json.JSONDecodeError) as e:
        raise Exception(f"Failed to parse JSON-LD data: {e}")
    return profile_image_url

def build_profile_images(src_links, save_dir):
    """Downloads profile images from the provided fandom links and saves them to a local directory."""
    movie_title_to_characters = {}

    for movie_title, fandom_link in src_links.items():
        # Retrieve the popular characters for each movie
        character_names = get_character_names(movie_title, fandom_link)
        
        movie_title_to_characters[movie_title] = character_names

        # For each character, crawl the link to its profile image
        image_url_src = {}
        for character_name in character_names:
            profile_image_url = get_profile_image_url(fandom_link, character_name)
            image_url_src[character_name] = profile_image_url

        # Download the profile images and save to the destination folder
        for character_name, profile_image_url in image_url_src.items():
            if profile_image_url is None:
                continue
            save_path = os.path.join(save_dir, movie_title, f"{character_name}_0.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            download_image(profile_image_url, save_path)

    return movie_title_to_characters

def build_retrieval_images(movie_title_to_characters, save_dir, k=15):
    """Crawl k images from the internet for each character and save to a specified directory."""
    # Download function
    def download_and_rename_images(query, num_images, save_dir, movie_title, character_name):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download images to the temp folder
            crawler = BingImageCrawler(storage={'root_dir': tmp_dir})
            crawler.crawl(keyword=query, max_num=num_images)

            # Prepare final save directory
            final_dir = os.path.join(save_dir, movie_title)
            os.makedirs(final_dir, exist_ok=True)

            # Rename and move images
            for idx, fname in enumerate(os.listdir(tmp_dir)):
                src = os.path.join(tmp_dir, fname)
                ext = os.path.splitext(fname)[-1].lower()
                new_name = f"{character_name}_{idx+1}{ext}"
                dst = os.path.join(final_dir, new_name)
                shutil.move(src, dst)

    for movie_title, character_names in movie_title_to_characters.items():
        for character_name in character_names:
            query = f"{character_name} in {movie_title} Animated Movie"
            download_and_rename_images(query, k, save_dir, movie_title, character_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_csv', default="src.csv", type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--save_file', default=None, type=str)
    args = parser.parse_args()

    src_csv = args.src_csv # Source CSV files containing links
    save_dir = args.save_dir # Target directory
    save_file = args.save_file

    # Read the source CSV file
    df = pd.read_csv(src_csv)

    # Create a dictionary to store unique movie titles and their corresponding fandom links
    src_links = {}
    for index, row in df.iterrows():
        if row["Movie title"] not in src_links:
            src_links[row["Movie title"]] = row["Fandom link"]

    # Download profile images from Fandom
    movie_title_to_characters = build_profile_images(src_links, save_dir)

    # Populate the character bank with images crawled online
    build_retrieval_images(movie_title_to_characters, save_dir)

    # Remove any repeated images
    for char_bank_dir in glob(os.path.join(save_dir, "*")):
        remove_and_rename_images(char_bank_dir)

    # Save the character information
    with open(save_file, 'w') as outfile:
        json.dump(movie_title_to_characters, outfile, indent=4)
