import argparse
import io
import os
import re
import time
import urllib.parse
import numpy as np
import pandas as pd
import json
import requests
import shutil
import tempfile
from bs4 import BeautifulSoup
from collections import Counter
from glob import glob
from PIL import Image
from icrawler.builtin import BingImageCrawler
from utils import remove_and_rename_images

# Fandom sits behind Cloudflare and rejects the default python-requests agent
# with a 403 "Just a moment..." challenge page, so every request needs a
# browser-like UA. The rendered HTML is challenged on most wikis even with one,
# which is why character lookup goes through the MediaWiki API instead.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Pages that live in a character category but are not characters themselves.
_JUNK_TITLE_RE = re.compile(
    r"(^(list of|category:|file:|template:)|"
    r"(gallery|galleries|images|soundtrack|transcript|credits|commentary|"
    r"filmography|merchandise|episodes?|\(species\))\b)",
    re.IGNORECASE,
)

# Bit-part credits that are roles rather than named characters. Matched against
# the whole credit only: 'Boy' is a bit part, 'Beast Boy' is a character, and
# the same goes for 'Man'/'Spider-Man' and 'Woman'/'Wonder Woman'.
_GENERIC_ROLES = {
    "additional voice", "additional voices", "various", "ensemble", "chorus",
    "narrator", "narration", "announcer", "voice", "voices", "singer", "dancer",
    "man", "woman", "boy", "girl", "kid", "child", "baby", "guest", "clerk",
    "extra", "crowd", "villager", "soldier", "guard", "reporter", "waiter",
    "bystander", "passenger", "customer", "patron", "self", "himself", "herself",
}
# Numbered or parenthesised bit parts: 'Grape #3', 'Wedgie Bergen #2'.
_BIT_PART_RE = re.compile(r"#\s*\d|\b(additional|uncredited)\b", re.IGNORECASE)

IMDB_GRAPHQL_URL = "https://caching.graphql.imdb.com/"
IMDB_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": HEADERS["User-Agent"],
    "x-imdb-client-name": "imdb-web-next-localized",
    "x-imdb-user-country": "US",
    "x-imdb-user-language": "en-US",
}
CAST_QUERY = """
query TitleCast($id: ID!, $numCast: Int!) {
  title(id: $id) {
    cast: credits(first: $numCast, filter: { categories: ["actor", "actress"] }) {
      edges { node { ... on Cast { characters { name } } } }
    }
  }
}
"""
_IMDBID_RE = re.compile(r"(tt\d+)")

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def api_request(api_url, params, retries=3):
    """GET the MediaWiki API, returning the parsed JSON or None on failure."""
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("action", "query")
    for attempt in range(retries):
        try:
            response = SESSION.get(api_url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            print(f"  [WARN] API {response.status_code} for {api_url} {params.get('list') or params.get('prop')}")
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  [WARN] API request failed ({e})")
        time.sleep(1.5 * (attempt + 1))
    return None


def parse_fandom_url(url):
    """Split a Fandom wiki url into its API endpoint and page title.

    'https://shrek.fandom.com/wiki/Category:Shrek_2_Characters'
        -> ('https://shrek.fandom.com/api.php', 'Category:Shrek 2 Characters')
    """
    parsed = urllib.parse.urlparse(url)
    title = urllib.parse.unquote(parsed.path.split("/wiki/", 1)[-1]).replace("_", " ")
    return f"https://{parsed.netloc}/api.php", title


def get_trending_names(url):
    """Best-effort scrape of the 'Trending Pages' section of a category page.

    This is the most faithful popularity signal Fandom exposes, but the section
    is absent on many category pages and the HTML is Cloudflare-challenged on
    most wikis. Both cases return an empty list rather than failing.
    """
    try:
        response = SESSION.get(url, timeout=20)
    except requests.RequestException as e:
        print(f"  [INFO] trending pages unavailable ({e})")
        return []

    if response.status_code != 200:
        print(f"  [INFO] trending pages unavailable (HTTP {response.status_code})")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    trending_section = soup.find("ul", class_="category-page__trending-pages")
    if trending_section is None:
        print("  [INFO] no trending section on this page")
        return []

    names = []
    for item in trending_section.find_all("a", href=True):
        title = urllib.parse.unquote(item["href"].split("/wiki/")[-1]).replace("_", " ")
        names.append(title)
    return names


def get_category_members(api_url, category, descend_if_fewer_than=8):
    """List the article pages of a category, following subcategories one level
    deep when the category itself holds too few articles to be useful."""
    def _members(title, namespace):
        out, cont = [], {}
        while True:
            data = api_request(api_url, {
                "list": "categorymembers", "cmtitle": title,
                "cmnamespace": namespace, "cmlimit": "500", **cont,
            })
            if data is None:
                return out
            out += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
            if "continue" in data:
                cont = data["continue"]
            else:
                return out

    pages = _members(category, "0")
    if len(pages) < descend_if_fewer_than:
        for subcat in _members(category, "14")[:10]:
            pages += _members(subcat, "0")
    return list(dict.fromkeys(pages))


def get_linked_characters(api_url, page_title):
    """Fallback for source links that point at an article instead of a category:
    take the article's outgoing links and keep the ones whose own categories
    mark them as characters."""
    links, cont = [], {}
    while True:
        data = api_request(api_url, {
            "titles": page_title, "prop": "links",
            "plnamespace": "0", "pllimit": "500", **cont,
        })
        if data is None:
            break
        for page in data.get("query", {}).get("pages", {}).values():
            links += [l["title"] for l in page.get("links", [])]
        if "continue" in data:
            cont = data["continue"]
        else:
            break

    links = list(dict.fromkeys(links))
    if len(links) <= 20:
        # Short, hand-curated character lists need no filtering, and asking for
        # categories would drop characters whose pages are uncategorised.
        return links

    characters = []
    for batch in _batched(links, 50):
        data = api_request(api_url, {
            "titles": "|".join(batch), "prop": "categories", "cllimit": "500",
        })
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            cats = [c["title"].lower() for c in page.get("categories", [])]
            if any("character" in c or "protagonist" in c or "villain" in c for c in cats):
                characters.append(page["title"])
    return characters


def _batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def get_page_lengths(api_url, titles):
    """Map page title -> article byte length. Article length is a good proxy for
    how central a character is: leads have long pages, walk-ons have stubs."""
    lengths = {}
    for batch in _batched(titles, 50):
        data = api_request(api_url, {"titles": "|".join(batch), "prop": "info"})
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" not in page:
                lengths[page["title"]] = page.get("length", 0)
    return lengths


# The image parameter of a character infobox. Its value may be a bare filename,
# a [[File:...]] link, or a <gallery>/<tabber> block whose first entry is the
# canonical portrait, so the filename is taken from the value that follows.
_INFOBOX_IMAGE_RE = re.compile(
    r"\|\s*(?:image|img|portrait|photo|picture)\s*\d*\s*=", re.IGNORECASE)
_FILENAME_RE = re.compile(
    r"([^\|\[\]\n<>=]+?\.(?:png|jpe?g|gif|webp|svg))", re.IGNORECASE)


def get_infobox_image_titles(api_url, page_titles):
    """Map page -> infobox image file, read from the article wikitext.

    prop=pageimages returns the page's *first* image, which is the portrait only
    when the infobox actually carries one. Where it does not, that first image
    is whatever screenshot leads the article - on The Iron Giant wiki it makes
    a picture of Kent Mansley the anchor for Hogarth Hughes. Reading the infobox
    parameter directly gets the character the page is actually about.
    """
    found = {}
    for batch in _batched(page_titles, 20):
        data = api_request(api_url, {
            "titles": "|".join(batch), "prop": "revisions",
            "rvprop": "content", "rvslots": "main",
        })
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                continue
            try:
                text = page["revisions"][0]["slots"]["main"]["*"]
            except (KeyError, IndexError):
                continue
            match = _INFOBOX_IMAGE_RE.search(text)
            if not match:
                continue
            # Look only just past the parameter so a later field cannot supply
            # an unrelated file.
            filename = _FILENAME_RE.search(text[match.end():match.end() + 400])
            if filename:
                # Strip any File:/Image: prefix. Not lstrip(), which removes
                # characters rather than the prefix and turns Fiona.png into
                # ona.png.
                name = re.sub(r"^\s*(?:File|Image)\s*:", "", filename.group(1), flags=re.I)
                found[page["title"]] = "File:" + name.strip()
    return found


def resolve_file_urls(api_url, file_titles, width=800):
    """Map File: titles to fetchable urls."""
    urls = {}
    for batch in _batched(list(file_titles), 50):
        data = api_request(api_url, {
            "titles": "|".join(batch), "prop": "imageinfo",
            "iiprop": "url|size", "iiurlwidth": str(width),
        })
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("width", 0) < 120 or info.get("height", 0) < 120:
                continue
            source = info.get("thumburl") or info.get("url")
            if source:
                urls[page["title"]] = source
    return urls


def get_anchor_image_urls(api_url, page_titles):
    """Best available portrait per page as {page: (url, source)}.

    'infobox' means the image the article itself declares as the character's
    portrait. 'pageimage' is the fallback and is weaker evidence: it is only the
    first image on the page, which is the portrait when the infobox has one and
    an arbitrary screenshot otherwise. The distinction is carried through to the
    output so low-confidence anchors can be reviewed.
    """
    page_titles = list(dict.fromkeys(page_titles))
    infobox = get_infobox_image_titles(api_url, page_titles)
    file_urls = resolve_file_urls(api_url, set(infobox.values())) if infobox else {}

    anchors = {}
    for page, file_title in infobox.items():
        if file_title in file_urls:
            anchors[page] = (file_urls[file_title], "infobox")

    missing = [p for p in page_titles if p not in anchors]
    if missing:
        for page, url in get_profile_image_urls(api_url, missing).items():
            anchors[page] = (url, "pageimage")
    return anchors


def get_profile_image_urls(api_url, titles):
    """Map page title -> lead/infobox image url via the MediaWiki pageimages API.

    This replaces scraping the JSON-LD block out of each character page, which
    needs one Cloudflare-challenged request per character; here 50 characters
    resolve in a single API call.
    """
    urls = {}
    for batch in _batched(titles, 50):
        data = api_request(api_url, {
            "titles": "|".join(batch), "prop": "pageimages",
            "piprop": "original", "pilicense": "any",
        })
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            source = (page.get("original") or {}).get("source")
            if source:
                urls[page["title"]] = source
    return urls


def extract_imdbid(imdb_link):
    """Pull the 'tt########' id out of an IMDB url."""
    if not isinstance(imdb_link, str):
        return None
    match = _IMDBID_RE.search(imdb_link)
    return match.group(1) if match else None


def get_imdb_characters(imdbid, num_cast=60):
    """Character names for a title in IMDB billing order.

    Billing order is the principal-cast signal: leads are credited first. Unlike
    a Fandom category it is also specific to this film rather than to the whole
    franchise the wiki covers.
    """
    payload = {"query": CAST_QUERY, "variables": {"id": imdbid, "numCast": num_cast}}
    try:
        response = SESSION.post(IMDB_GRAPHQL_URL, headers=IMDB_HEADERS,
                                json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  [WARN] IMDB lookup failed for {imdbid}: {e}")
        return []

    title_data = (data.get("data") or {}).get("title")
    if not title_data:
        return []

    names = []
    for edge in title_data.get("cast", {}).get("edges", []):
        for character in (edge.get("node", {}).get("characters") or []):
            name = (character or {}).get("name", "")
            # Animated titles credit roles as "Fievel (voice)".
            name = name.replace("(voice)", "").strip()
            if not name or _BIT_PART_RE.search(name):
                continue
            if _normalise(name) in _GENERIC_ROLES:
                continue
            if name not in names:
                names.append(name)
    return names


def _normalise(name):
    """Casefold a name for matching: drop parentheticals and punctuation."""
    name = re.sub(r"\s*\([^)]*\)", "", name).lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"^(the|a) ", "", name)
    return re.sub(r"\s+", " ", name).strip()


def resolve_fandom_pages(api_url, names):
    """Map character names onto wiki pages, following redirects, and return
    {name: (page_title, image_url or None)} for the ones that exist.

    Redirect resolution is what makes IMDB names line up with wiki titles:
    'Beast Boy' redirects to 'Garfield Logan', 'Poppy' to 'Queen Poppy'.
    """
    resolved = {}
    for batch in _batched(names, 40):
        data = api_request(api_url, {
            "titles": "|".join(batch), "redirects": "1", "prop": "pageimages",
            "piprop": "original", "pilicense": "any",
        })
        if data is None:
            continue
        query = data.get("query", {})
        normalised = {n["from"]: n["to"] for n in query.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        pages = {p["title"]: p for p in query.get("pages", {}).values()}
        for name in batch:
            title = normalised.get(name, name)
            title = redirects.get(title, title)
            page = pages.get(title)
            if page is None or "missing" in page:
                continue
            resolved[name] = (title, (page.get("original") or {}).get("source"))
    return resolved


def match_by_tokens(names, candidates):
    """Loose fallback match of names onto category members.

    Covers the cases redirects miss: credits that spell a name out in full
    ('Lillian Deville' for the page 'Lil DeVille', 'Princess Fiona' for 'Fiona').
    """
    lookup = {}
    for title in candidates:
        lookup.setdefault(_normalise(title), title)

    matches = {}
    for name in names:
        key = _normalise(name)
        if not key:
            continue
        if key in lookup:
            matches[name] = lookup[key]
            continue
        name_tokens = key.split()
        best = None
        for cand_key, title in lookup.items():
            cand_tokens = cand_key.split()
            # One name's words are a subset of the other's, allowing shortened
            # forms so 'Lil' matches 'Lillian' but 'Lou' never matches 'Louise'.
            short, long_ = sorted((name_tokens, cand_tokens), key=len)
            if not short:
                continue
            paired = all(
                any(l.startswith(s) or s.startswith(l) for l in long_) for s in short
            )
            if paired and (best is None or len(title) < len(best)):
                best = title
        if best is not None:
            matches[name] = best
    return matches


# Wiki furniture that is attached to every page and is never character art.
_CHROME_FILE_RE = re.compile(
    r"(site-?logo|wiki-?wordmark|wordmark|favicon|placeholder|nav-?bar|"
    r"badge|award|icon|button|spoiler|stub|edit|quote|watermark)",
    re.IGNORECASE,
)


def get_gallery_image_urls(api_url, page_title, cap=60, width=800):
    """Every usable image on a character's page and its /Gallery subpage.

    Curated wiki art beats image search on the two things that matter here:
    it is guaranteed to be the right film, and it carries no stock-photo
    watermarks. Measured diversity matches web search, so it does not
    narrow the pool.
    """
    file_titles = []
    for source in (page_title, f"{page_title}/Gallery"):
        data = api_request(api_url, {"titles": source, "prop": "images", "imlimit": "300"})
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                continue
            file_titles += [i["title"] for i in page.get("images", [])]

    file_titles = [t for t in dict.fromkeys(file_titles) if not _CHROME_FILE_RE.search(t)]
    file_titles = file_titles[:cap]

    urls = []
    for batch in _batched(file_titles, 50):
        data = api_request(api_url, {
            "titles": "|".join(batch), "prop": "imageinfo",
            "iiprop": "url|size", "iiurlwidth": str(width),
        })
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            # Skip sprites and banners; the bank wants character shots.
            if info.get("width", 0) < 200 or info.get("height", 0) < 200:
                continue
            urls.append(info.get("thumburl") or info.get("url"))
    return urls


def perceptual_hash(image, size=8):
    """Difference hash, used to drop near-duplicate candidates.

    utils.remove_and_rename_images de-duplicates on SHA-256, which only catches
    byte-identical files, so the same still re-encoded by two sites survives.
    """
    grey = image.convert("L").resize((size + 1, size), Image.BICUBIC)
    pixels = np.asarray(grey, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def is_near_duplicate(candidate, seen_hashes, threshold=6):
    return any(int((candidate != h).sum()) <= threshold for h in seen_hashes)


def clean_character_name(title):
    """Normalise a wiki page title into a character name usable as a filename.

    Disambiguation suffixes ('Shrek (character)', 'Toothless (Franchise)') are
    dropped, and underscores are avoided because the de-duplication step keys
    on the text before the first underscore.
    """
    name = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip() or title.strip()
    name = name.replace("/", "-").replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


def make_record(page, image_url, source, match):
    """Describe one character's anchor and how much to trust it.

    'confident'  - the page's own declared infobox portrait, on a page the wiki
                   itself ties to the credited name (exact title or redirect).
    'low'        - usable but unverified: either the image is only the first one
                   on the page (the failure mode that made a picture of Kent
                   Mansley the anchor for Hogarth Hughes), or the page was
                   matched to the credit by token similarity rather than by the
                   wiki. Still written to disk, but flagged.
    'missing'    - no anchor image at all. Nothing is written for this character,
                   because the downstream bank is keyed on the _0 image.
    """
    if image_url is None:
        confidence = "missing"
    elif source == "infobox" and match in ("exact", "redirect"):
        confidence = "confident"
    else:
        confidence = "low"
    return {"page": page, "image": image_url, "source": source or "none",
            "match": match, "confidence": confidence}


def get_character_names(movie_name, url, imdbid=None, max_characters=10):
    """Return {character name: record} for a movie's principal cast, where each
    record carries the wiki page, the anchor image url, and how confident that
    anchor is.

    The cast list comes from IMDB billing order, which is per-film and ranks
    leads first; the wiki only supplies the artwork. Fandom categories are the
    fallback, because many of these wikis cover a whole franchise and their
    'Characters' category is neither film-specific nor ordered by prominence.

    Confidence combines the two ways an anchor goes wrong: the image may not be
    the page's declared portrait, and the page itself may be the wrong one.
    Characters whose anchor is unusable are still returned, marked 'missing', so
    they can be dealt with later rather than silently disappearing.
    """
    api_url, title = parse_fandom_url(url)

    if title.lower().startswith("category:"):
        candidates = get_category_members(api_url, title)
    else:
        candidates = get_linked_characters(api_url, title)
    candidates = [c for c in candidates if "/" not in c and not _JUNK_TITLE_RE.search(c)]

    characters = {}
    used_pages = set()

    # 1. Principal cast from IMDB, matched onto wiki pages for their artwork.
    cast = get_imdb_characters(imdbid) if imdbid else []
    if cast:
        # How the credit was tied to a page. An exact title or a wiki redirect
        # is the wiki's own assertion that the two names denote one character;
        # a token match is our guess and is treated as weaker.
        resolved, match_kind = {}, {}
        for name, (page, _) in resolve_fandom_pages(api_url, cast).items():
            resolved[name] = page
            match_kind[name] = "exact" if _normalise(name) == _normalise(page) else "redirect"
        for name, page in match_by_tokens(
                [n for n in cast if n not in resolved], candidates).items():
            resolved[name] = page
            match_kind[name] = "fuzzy"

        # Re-resolve every anchor from the infobox rather than trusting the
        # page's lead image.
        anchors = get_anchor_image_urls(api_url, list(resolved.values()))

        # A credit whose page yields no anchor is often pointing at a
        # disambiguation page rather than the character: on the Shrek wiki
        # 'Shrek' lists the ogre, the book and the film, and the character
        # lives at 'Shrek (character)'. Retry those against the disambiguated
        # title, then against the film's own category.
        stranded = [n for n, p in resolved.items() if p not in anchors]
        if stranded:
            alternatives = {n: f"{n} (character)" for n in stranded}
            repaired = resolve_fandom_pages(api_url, list(alternatives.values()))
            by_token = match_by_tokens(stranded, candidates)
            retry = {}
            for name in stranded:
                page = None
                if alternatives[name] in repaired:
                    page = repaired[alternatives[name]][0]
                elif by_token.get(name) and by_token[name] != resolved[name]:
                    page = by_token[name]
                if page:
                    retry[name] = page
            if retry:
                found = get_anchor_image_urls(api_url, list(retry.values()))
                for name, page in retry.items():
                    if page in found:
                        resolved[name] = page
                        anchors[page] = found[page]

        # Billing order is kept exactly as IMDB gives it. Re-ranking by whether
        # a page sits in the film's wiki category was tried and is worse: on
        # franchise wikis the leads live on shared pages outside the film's
        # category, so it demotes Red and Robin below one-scene bit parts.
        for name in cast:
            if len(characters) >= max_characters:
                break
            if name not in resolved:
                continue
            page = resolved[name]
            if page in used_pages:  # 'Grinch' and 'The Grinch' are one page
                continue
            clean = clean_character_name(name)
            if not clean or clean in characters:
                continue
            used_pages.add(page)
            image_url, source = anchors.get(page, (None, None))
            characters[clean] = make_record(page, image_url, source, match_kind[name])

    # 2. Top up from the wiki itself, ranked by article length. This carries
    #    films whose IMDB credits are unusable (bit parts billed first, or the
    #    cast credited under actor names rather than character names).
    if len(characters) < max_characters and candidates:
        lengths = get_page_lengths(api_url, candidates)
        ranked = [t for t in sorted(candidates, key=lambda t: -lengths.get(t, 0))
                  if t not in used_pages]
        pool = ranked[:max_characters * 3]
        images = get_anchor_image_urls(api_url, pool)
        for page in pool:
            if len(characters) >= max_characters:
                break
            if page not in images:
                continue
            clean = clean_character_name(page)
            if not clean or clean in characters:
                continue
            image_url, source = images[page]
            characters[clean] = make_record(page, image_url, source, "category")

    if not characters:
        print(f"  [WARN] {movie_name}: no characters found ({url})")
    return characters


def rasterise_url(image_url, width=800):
    """Point an SVG at the wiki CDN's raster renderer.

    PIL cannot open SVG, and some infobox art is vector only (Angry Birds
    credits Red, Chuck and Bomb this way). Adding the scale-to-width-down
    segment makes the CDN return a bitmap instead.
    """
    if ".svg" not in image_url.lower():
        return image_url
    base, _, query = image_url.partition("?")
    base = base.rstrip("/")
    if "scale-to-width-down" not in base:
        base += f"/scale-to-width-down/{width}"
    return base + (f"?{query}" if query else "")


def download_image(image_url, save_path):
    """Download an image and normalise it to PNG.

    The CDN rejects urllib's default agent, and Fandom serves webp/gif behind
    .png urls, so the bytes are fetched with a browser UA and re-encoded.
    Transparency is kept rather than flattened, since most Fandom character
    art is a cut-out on an alpha background.
    """
    try:
        response = SESSION.get(rasterise_url(image_url), timeout=30)
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as img:
            mode = "RGBA" if (img.mode in ("RGBA", "LA", "P") and "transparency" in img.info
                              or img.mode in ("RGBA", "LA")) else "RGB"
            img.convert(mode).save(save_path, "PNG")
        print(f"Image downloaded and saved to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return False


def build_profile_images(src_links, save_dir, max_characters=10):
    """Downloads profile images from the provided fandom links and saves them to a local directory."""
    movie_title_to_characters = {}

    character_pages = {}
    records = []

    for movie_title, (fandom_link, imdbid) in src_links.items():
        print(f"=== {movie_title} ===")
        api_url, _ = parse_fandom_url(fandom_link)

        # Retrieve the principal characters for each movie, with their anchor
        # image urls resolved in the same pass.
        image_url_src = get_character_names(movie_title, fandom_link, imdbid, max_characters)

        # Download the anchor images. A character is only kept once its anchor
        # is on disk, because every retrieval image is later scored against it.
        names, pages = [], {}
        for character_name, record in image_url_src.items():
            if record["image"] is not None:
                save_path = os.path.join(save_dir, movie_title, f"{character_name}_0.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if not download_image(record["image"], save_path):
                    # The url did not yield a usable image, so there is no
                    # anchor on disk no matter what the wiki claimed.
                    record["confidence"] = "missing"
            # Every character is carried into the retrieval stage. One without
            # an anchor still gets example images; it simply has no _0 and is
            # marked in the table so it can be anchored later.
            names.append(character_name)
            if record["page"]:
                pages[character_name] = record["page"]
            record["movie"] = movie_title
            records.append((character_name, record))

        movie_title_to_characters[movie_title] = names
        character_pages[movie_title] = (api_url, pages)
        by_conf = Counter(r["confidence"] for _, r in records if r["movie"] == movie_title)
        print(f"  -> {len(names)} characters; anchors written "
              f"confident={by_conf['confident']}, low={by_conf['low']}, "
              f"none={by_conf['missing']}")

    return movie_title_to_characters, character_pages, records


def build_retrieval_images(movie_title_to_characters, character_pages, save_dir, k=15):
    """Collect k example images per character, from the character's own wiki
    gallery first and image search second.

    Wiki galleries are the better source: right film guaranteed, no stock-photo
    watermarks, and typically far more images than a search returns. Measured
    pairwise diversity matches image search, so preferring them does not make
    the bank repetitive. Search is still used to top up, and near-duplicates are
    dropped across both sources.
    """
    def crawl_bing(query, num_images, tmp_dir):
        try:
            crawler = BingImageCrawler(storage={'root_dir': tmp_dir})
            crawler.crawl(keyword=query, max_num=num_images)
        except Exception as e:
            print(f"Failed to crawl '{query}': {e}")
            return []
        return [os.path.join(tmp_dir, f) for f in sorted(os.listdir(tmp_dir))]

    for movie_title, character_names in movie_title_to_characters.items():
        api_url, pages = character_pages.get(movie_title, (None, {}))
        final_dir = os.path.join(save_dir, movie_title)
        os.makedirs(final_dir, exist_ok=True)

        for character_name in character_names:
            # Resume support: a character that already has its full quota is
            # left alone, and a partially crawled one is cleared and redone so
            # the numbering stays contiguous.
            existing = sorted(glob(os.path.join(
                final_dir, f"{character_name}_[1-9]*.png")))
            if len(existing) >= k:
                continue
            for path in existing:
                os.remove(path)

            # Seed the duplicate filter with the anchor so it is never repeated
            # back as one of the character's own examples.
            hashes = []
            anchor_path = os.path.join(final_dir, f"{character_name}_0.png")
            if os.path.exists(anchor_path):
                try:
                    with Image.open(anchor_path) as img:
                        hashes.append(perceptual_hash(img))
                except Exception:
                    pass

            saved = 0
            page = pages.get(character_name)
            gallery = get_gallery_image_urls(api_url, page) if (api_url and page) else []
            for url in gallery:
                if saved >= k:
                    break
                try:
                    response = SESSION.get(rasterise_url(url), timeout=30)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                    digest = perceptual_hash(image)
                    if is_near_duplicate(digest, hashes):
                        continue
                    mode = "RGBA" if image.mode in ("RGBA", "LA") else "RGB"
                    image.convert(mode).save(
                        os.path.join(final_dir, f"{character_name}_{saved+1}.png"), "PNG")
                    hashes.append(digest)
                    saved += 1
                except Exception:
                    continue

            # Top up from image search when the wiki is thin.
            if saved < k:
                query = f"{character_name} in {movie_title} Animated Movie"
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for src in crawl_bing(query, (k - saved) * 2, tmp_dir):
                        if saved >= k:
                            break
                        try:
                            with Image.open(src) as image:
                                digest = perceptual_hash(image)
                                if is_near_duplicate(digest, hashes):
                                    continue
                                mode = "RGBA" if image.mode in ("RGBA", "LA") else "RGB"
                                image.convert(mode).save(
                                    os.path.join(final_dir, f"{character_name}_{saved+1}.png"), "PNG")
                            hashes.append(digest)
                            saved += 1
                        except Exception:
                            continue

            print(f"  {movie_title} / {character_name}: {saved} examples "
                  f"({len(gallery)} gallery candidates)")


def save_character_table(records, save_file):
    """Write every character found, with the provenance of its anchor image.

    Characters whose anchor is 'missing' or 'low' are listed here even when
    nothing was written to the image directory, so they can be revisited.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_file)) or ".", exist_ok=True)
    columns = ["Movie title", "Character", "Anchor confidence", "Anchor source",
               "Page match", "Wiki page", "Anchor URL"]
    rows = [{
        "Movie title": record["movie"],
        "Character": name,
        "Anchor confidence": record["confidence"],
        "Anchor source": record["source"],
        "Page match": record["match"],
        "Wiki page": record["page"],
        "Anchor URL": record["image"] or "",
    } for name, record in records]

    if save_file.lower().endswith(".csv"):
        pd.DataFrame(rows, columns=columns).to_csv(save_file, index=False)
    else:
        with open(save_file, 'w') as outfile:
            json.dump(rows, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_csv', default="build_character_bank/build_appearance_bank/src.csv", type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--save_file', default=None, type=str)
    parser.add_argument('--max_characters', default=10, type=int,
                        help="Max characters to keep per movie, ranked by prominence.")
    parser.add_argument('--k_retrieval', default=15, type=int,
                        help="Images to crawl per character in the retrieval stage.")
    parser.add_argument('--skip_retrieval', action='store_true',
                        help="Only build the Fandom profile image bank.")
    args = parser.parse_args()

    src_csv = args.src_csv # Source CSV files containing links
    save_dir = args.save_dir # Target directory
    save_file = args.save_file

    # Read the source CSV file
    df = pd.read_csv(src_csv)

    # Create a dictionary to store unique movie titles and their corresponding
    # fandom link (artwork) and IMDB id (principal cast list)
    src_links = {}
    for index, row in df.iterrows():
        if row["Movie title"] not in src_links:
            src_links[row["Movie title"]] = (row["Fandom link"],
                                             extract_imdbid(row.get("IMDB link")))

    # Download profile images from Fandom
    movie_title_to_characters, character_pages, records = build_profile_images(
        src_links, save_dir, args.max_characters)

    # Save the character information before the (slow) crawling stage
    if save_file is not None:
        save_character_table(records, save_file)

    if not args.skip_retrieval:
        # Populate the character bank with images crawled online
        build_retrieval_images(movie_title_to_characters, character_pages,
                               save_dir, args.k_retrieval)

        # De-duplication already happened during retrieval, on a perceptual hash
        # rather than SHA-256, so it also catches the same still re-encoded by
        # two sites. remove_and_rename_images is deliberately not called here:
        # it renumbers each character's files from 0, which would promote the
        # first retrieval image of an anchor-less character to _0 and make an
        # arbitrary crawled image look like a verified anchor.

    # Save the character information
    if save_file is not None:
        save_character_table(records, save_file)
