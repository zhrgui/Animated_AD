import argparse
import json
import os
import difflib
import yt_dlp
import requests
from bs4 import BeautifulSoup

from googleapiclient.discovery import build
from pytube import YouTube, request
from wikidata.client import Client

client = Client()
API_KEY = None

# Get the aliases for the entity using WikiData
def get_wikidata_aliases(name, lang="en"):
    try:
        entity = client.search(name, language=lang)[0]
        wd_entity = client.get(entity.id, load=True)
        labels = [wd_entity.label.lower()]
        aliases = [a.lower() for a in wd_entity.aliases.get(lang, [])]
        return set(labels + aliases)
    except Exception as e:
        print(f"[WARN] Failed to get aliases for {name}: {e}")
        return set()

# Check if there is overlap for the aliases
def is_wikidata_alias_match(name1, name2):
    aliases1 = get_wikidata_aliases(name1)
    aliases2 = get_wikidata_aliases(name2)
    return not aliases1.isdisjoint(aliases2)

# Match name using the knowledge-based sources
def find_closest_match_wikidata(character_name, character_list):
    for key_name in character_list:
        if is_wikidata_alias_match(key_name, character_name):
            return key_name
    return None

# Match name lexicographically using difflib
def find_closest_match_difflib(character_name, character_list, thres=0.6):
    closest_matches = difflib.get_close_matches(character_name, character_list, n=1, cutoff=thres)
    if closest_matches:
        return closest_matches[0]
    else:
        return None

def imdb_cast_crawling(imdbid):
    """
    Crawl the list of characters and their corresponding voice actors from IMDb.
    Return a dictionary mapping from character names to actor names.
    """
    url = f"https://www.imdb.com/title/{imdbid}/fullcredits"
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    cast_dict = {}
    cast_rows = soup.find_all('tr', class_=['odd', 'even'])
    for row in cast_rows:
        actor_tag = row.find('td', class_=None)
        character_tag = row.find('td', class_='character')
        
        if actor_tag and character_tag:
            actor_name = actor_tag.get_text(strip=True)
            character_name = character_tag.get_text(strip=True)
            character_name = character_name.replace("(voice)", "").split('/')[0].strip()
            
            if 'Additional voices(uncredited)' not in character_name:
                cast_dict[character_name] = actor_name
    return cast_dict

def search_and_download_audio(actor_name, save_dir, max_results=5):
    """
    Search the interviews of a specific actor on Youtube and download the interview audios.
    """
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    search_request = youtube.search().list(
        part='snippet',
        q=f"{actor_name} interview",
        type='video',
        maxResults=max_results
    )
    response = search_request.execute()

    for item in response["items"]:
        try:
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            print(f"Title: {title}")
            print(f"URL:   {url}")

            filename = os.path.join(save_dir, f"{video_id}")

            print(f"Downloading audio for '{title}' as '{filename}'...")
            download_audio_with_ytdlp(url, filename)
        except:
            continue

def download_audio_with_ytdlp(url, filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', default=None, type=str, help='Directory to save audio files')
    parser.add_argument('--save_file', default=None, type=str, help='File to save cast information')
    parser.add_argument('--movie_title_to_imdbid_file', required=True, type=str, help='Path to movie_title_to_imdb JSON')
    parser.add_argument('--character_bank_file', required=True, type=str, help='Path to character bank JSON')
    args = parser.parse_args()

    save_folder = args.save_folder
    save_file = args.save_file

    # Load the dictionary mapping from the movie title to its IMDb ID
    with open(args.movie_title_to_imdbid_file, 'r') as infile:
        movie_title_to_imdbid = json.load(infile)

    # Load the dictionary mapping from the movie title to its character list
    with open(args.character_bank_file, 'r') as infile:
        movie_to_character_bank = json.load(infile)

    cast_information = {}
    for movie_title in movie_title_to_imdbid:
        imdb_id = movie_title_to_imdbid[movie_title]
        character_bank = movie_to_character_bank[movie_title]
        cast_dict = imdb_cast_crawling(imdb_id)
        cast_information[movie_title] = cast_dict

        for character_name, actor_name in cast_dict.items():
            # Match the name with difflib
            matched_character_name = find_closest_match_difflib(character_name, character_bank)
            # or matched_character_name = find_closest_match_wikidata(character_name, character_bank)

            if matched_character_name is None:
                continue
            else:
                save_dir = os.path.join(save_folder, movie_title, actor_name)
                os.makedirs(save_dir, exist_ok=True)
                search_and_download_audio(actor_name, save_dir, max_results=5)

    # Save the cast information to a JSON file
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, 'w') as outfile:
        json.dump(cast_information, outfile, indent=4)
