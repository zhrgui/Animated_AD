import argparse
import json
import os
import difflib
import yt_dlp
import requests
from bs4 import BeautifulSoup

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pytube import YouTube, request
from wikidata.client import Client

client = Client()
# YouTube Data API v3 key. Set YOUTUBE_API_KEY in the environment or pass
# --youtube_api_key; without one the cast information is still collected and
# only the interview downloads are skipped.
API_KEY = os.environ.get("YOUTUBE_API_KEY")

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

IMDB_GRAPHQL_URL = "https://caching.graphql.imdb.com/"
IMDB_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "x-imdb-client-name": "imdb-web-next-localized",
    "x-imdb-user-country": "US",
    "x-imdb-user-language": "en-US",
}
CAST_QUERY = """
query TitleCast($id: ID!, $numCast: Int!) {
  title(id: $id) {
    cast: credits(first: $numCast, filter: { categories: ["actor", "actress"] }) {
      edges {
        node {
          name { nameText { text } }
          ... on Cast { characters { name } }
        }
      }
    }
  }
}
"""


def imdb_cast_crawling(imdbid, num_cast=200):
    """
    Fetch the list of characters and their corresponding voice actors from IMDb.
    Return a dictionary mapping from character names to actor names.

    This uses the GraphQL endpoint the IMDb site itself calls. Scraping
    /fullcredits no longer works: the request is answered with 403 without a
    browser user agent, and the markup it looked for (tr.odd/tr.even with a
    td.character) belongs to a retired layout, so it silently returned an empty
    cast for every title.
    """
    payload = {"query": CAST_QUERY, "variables": {"id": imdbid, "numCast": num_cast}}
    try:
        response = requests.post(IMDB_GRAPHQL_URL, headers=IMDB_HEADERS,
                                 json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"[WARN] IMDb cast lookup failed for {imdbid}: {e}")
        return {}

    title_data = (data.get("data") or {}).get("title")
    if not title_data:
        print(f"[WARN] No cast data returned for {imdbid}")
        return {}

    cast_dict = {}
    for edge in title_data.get("cast", {}).get("edges", []):
        node = edge.get("node") or {}
        actor_name = (((node.get("name") or {}).get("nameText") or {})
                      .get("text") or "").strip()
        if not actor_name:
            continue
        for character in (node.get("characters") or []):
            character_name = (character or {}).get("name", "")
            character_name = character_name.replace("(voice)", "").split('/')[0].strip()
            if not character_name or "Additional voices" in character_name:
                continue
            cast_dict[character_name] = actor_name
    return cast_dict

class QuotaExceeded(Exception):
    """Raised when the YouTube Data API daily quota is used up."""


def search_youtube_ytdlp(query, max_results=5):
    """Search YouTube through yt-dlp instead of the Data API.

    The Data API charges 100 quota units per search against a default allowance
    of 10,000 per day, which caps a run at about 100 actors. yt-dlp reads the
    same result page the browser does, so it needs no key and has no daily
    allowance. It also reports each video's duration before anything is
    downloaded, which lets over-long uploads be skipped for free.
    """
    opts = {'quiet': True, 'no_warnings': True,
            'extract_flat': True, 'skip_download': True, **_cookie_opts()}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    except Exception as e:
        print(f"[WARN] yt-dlp search failed for '{query}': {e}")
        return []
    return [e for e in (info or {}).get('entries', []) if e and e.get('id')]


def search_and_download_audio_ytdlp(actor_name, save_dir, max_results=5):
    """Search an actor's interviews with yt-dlp and download the audio."""
    entries = search_youtube_ytdlp(f"{actor_name} interview", max_results)
    for entry in entries:
        duration = entry.get('duration')
        if duration and duration > MAX_DURATION_S:
            print(f"Skipping '{entry.get('title', '')[:60]}' ({duration}s, too long)")
            continue
        url = f"https://www.youtube.com/watch?v={entry['id']}"
        print(f"Downloading audio for '{entry.get('title', '')[:60]}'...")
        try:
            download_audio_with_ytdlp(url, os.path.join(save_dir, entry['id']))
        except Exception as e:
            print(f"[WARN] Download failed for {entry['id']}: {str(e)[:120]}")
            continue


def search_and_download_audio(actor_name, save_dir, max_results=5):
    """
    Search the interviews of a specific actor on Youtube and download the interview audios.

    A search costs 100 quota units against a default allowance of 10,000 per
    day, so the quota - not the network - is the limit on how many actors can be
    covered in one run. Exhausting it raises QuotaExceeded so the caller can
    stop cleanly and keep what has already been written.
    """
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    search_request = youtube.search().list(
        part='snippet',
        q=f"{actor_name} interview",
        type='video',
        maxResults=max_results
    )
    try:
        response = search_request.execute()
    except HttpError as e:
        reason = ""
        try:
            reason = json.loads(e.content).get("error", {}).get("errors", [{}])[0].get("reason", "")
        except Exception:
            pass
        # The API reports an exhausted daily allowance as 429 as well as 403,
        # so both have to stop the run. Treating 429 as a transient warning
        # burns through every remaining actor with searches that cannot succeed.
        if e.resp.status == 429 or (e.resp.status == 403 and "quota" in reason.lower()):
            raise QuotaExceeded(reason or f"HTTP {e.resp.status}") from e
        print(f"[WARN] Search failed for {actor_name}: {e}")
        return

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

MAX_DURATION_S = 45 * 60

# Path to a Netscape-format cookies.txt, or a browser spec such as "firefox"
# that yt-dlp can read a profile from. YouTube starts asking downloaders to
# "sign in to confirm you're not a bot" after sustained use; supplying cookies
# from a logged-in session answers that. Set via --cookies / --cookies_from_browser.
COOKIE_FILE = os.environ.get("YTDLP_COOKIES")
COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")


def _cookie_opts():
    """yt-dlp options selecting whichever cookie source is configured."""
    if COOKIE_FILE:
        return {'cookiefile': COOKIE_FILE}
    if COOKIES_FROM_BROWSER:
        # yt-dlp wants a tuple: (browser, profile, keyring, container)
        parts = COOKIES_FROM_BROWSER.split(':')
        return {'cookiesfrombrowser': (parts[0], *(parts[1:] or [None]))}
    return {}


def download_audio_with_ytdlp(url, filename, sample_rate=16000, channels=1):
    """Download a video's audio as WAV, downmixed to 16 kHz mono.

    The format stays .wav because clustering.py builds its transcript paths by
    swapping that extension. Writing 44.1 kHz stereo, as this did, costs 192 KB
    per second: the full actor list would have taken about 717 GB against 555 GB
    free, with single files reaching 3.7 GB. WhisperX and the speaker embedding
    both resample to 16 kHz mono anyway, so storing that directly is ~6x smaller
    and loses nothing they use.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename,
        'match_filter': yt_dlp.utils.match_filter_func(f"duration < {MAX_DURATION_S}"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        # Applied by the extract-audio post-processor when re-encoding.
        'postprocessor_args': {
            'extractaudio': ['-ac', str(channels), '-ar', str(sample_rate)],
        },
        'quiet': True,
        'no_warnings': True,
        **_cookie_opts(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', default=None, type=str, help='Directory to save audio files')
    parser.add_argument('--save_file', default=None, type=str, help='File to save cast information')
    parser.add_argument('--movie_title_to_imdbid_file', required=True, type=str, help='Path to movie_title_to_imdb JSON')
    parser.add_argument('--character_bank_file', required=True, type=str, help='Path to character bank JSON')
    parser.add_argument('--youtube_api_key', default=None, type=str,
                        help='YouTube Data API v3 key (overrides YOUTUBE_API_KEY)')
    parser.add_argument('--cookies', default=None, type=str,
                        help='Path to a Netscape-format cookies.txt exported from a '
                             'logged-in YouTube session (or set YTDLP_COOKIES).')
    parser.add_argument('--cookies_from_browser', default=None, type=str,
                        help='Read cookies from a local browser profile, e.g. "firefox". '
                             'Only works where that browser is installed.')
    parser.add_argument('--use_api_search', action='store_true',
                        help='Search via the YouTube Data API instead of yt-dlp. '
                             'Needs a key and is capped at ~100 searches per day.')
    args = parser.parse_args()

    use_api_search = args.use_api_search
    if args.cookies:
        COOKIE_FILE = args.cookies
    if args.cookies_from_browser:
        COOKIES_FROM_BROWSER = args.cookies_from_browser

    if args.youtube_api_key:
        API_KEY = args.youtube_api_key

    save_folder = args.save_folder
    save_file = args.save_file

    # Load the dictionary mapping from the movie title to its IMDb ID
    with open(args.movie_title_to_imdbid_file, 'r') as infile:
        movie_title_to_imdbid = json.load(infile)

    # Load the dictionary mapping from the movie title to its character list
    with open(args.character_bank_file, 'r') as infile:
        movie_to_character_bank = json.load(infile)

    if use_api_search and API_KEY is None:
        print("[WARN] --use_api_search given but no API key (set YOUTUBE_API_KEY "
              "or pass --youtube_api_key). Cast information will still be "
              "written; interview downloads are skipped.")

    cast_information = {}
    searched = 0
    quota_hit = False
    for movie_title in movie_title_to_imdbid:
        imdb_id = movie_title_to_imdbid[movie_title]
        character_bank = movie_to_character_bank.get(movie_title)
        if character_bank is None:
            print(f"[WARN] {movie_title} is not in the character bank, skipping")
            continue
        cast_dict = imdb_cast_crawling(imdb_id)
        cast_information[movie_title] = cast_dict
        print(f"=== {movie_title} ({imdb_id}): {len(cast_dict)} credited roles ===")

        # Several characters can share a voice actor, and the interviews are
        # per actor, so download each actor only once per film.
        downloaded_actors = set()
        for character_name, actor_name in cast_dict.items():
            # Match the name with difflib
            matched_character_name = find_closest_match_difflib(character_name, character_bank)
            # or matched_character_name = find_closest_match_wikidata(character_name, character_bank)

            if matched_character_name is None:
                continue
            print(f"  {character_name} -> bank '{matched_character_name}' (voice: {actor_name})")
            if (use_api_search and API_KEY is None) or actor_name in downloaded_actors:
                continue
            downloaded_actors.add(actor_name)
            save_dir = os.path.join(save_folder, movie_title, actor_name)

            # Resume: an actor already downloaded in a previous run is skipped
            # so a quota-limited crawl can be continued day by day.
            if os.path.isdir(save_dir) and os.listdir(save_dir):
                continue
            os.makedirs(save_dir, exist_ok=True)
            try:
                if use_api_search:
                    search_and_download_audio(actor_name, save_dir, max_results=5)
                else:
                    search_and_download_audio_ytdlp(actor_name, save_dir, max_results=5)
                searched += 1
            except QuotaExceeded as e:
                print(f"\n[STOP] YouTube daily quota exhausted after {searched} "
                      f"searches this run ({e}). Cast information is still "
                      f"written; re-run tomorrow to continue where this left off.")
                quota_hit = True
                break
        if quota_hit:
            break

    # Save the cast information to a JSON file
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, 'w') as outfile:
        json.dump(cast_information, outfile, indent=4)
