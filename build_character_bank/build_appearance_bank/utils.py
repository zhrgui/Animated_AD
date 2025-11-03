import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from uuid import uuid4

FNAME_RE = re.compile(r'^(?P<cls>[^_]+)_(?P<idx>\d+)\.(?P<ext>jpg|jpeg|png)$', re.IGNORECASE)

def hash_file(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def remove_and_rename_images(folder_path):
    """
    Work per 'class' (prefix before the first underscore).
    Deduplicate by content (hash) within each class, keeping the lowest index.
    Then rename remaining images in each class to sequential <class>_0.ext, <class>_1.ext, ...
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(folder_path)

    # Gather files per class
    class_files = defaultdict(list)
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        m = FNAME_RE.match(entry.name)
        if not m:
            continue
        cls = m.group('cls')
        idx = int(m.group('idx'))
        ext = m.group('ext').lower()
        class_files[cls].append((idx, entry, ext))

    # Process each class separately
    for cls, files in class_files.items():
        # Deduplicate by hash: keep smallest index file if hash is repeated
        hash_map = {}  # hash -> (idx, path)
        to_remove = []

        for idx, path, ext in files:
            file_hash = hash_file(path)
            if file_hash in hash_map:
                prev_idx, prev_path = hash_map[file_hash]
                if idx > prev_idx:
                    to_remove.append(path)
                else:
                    to_remove.append(prev_path)
                    hash_map[file_hash] = (idx, path)
            else:
                hash_map[file_hash] = (idx, path)

        # Remove duplicates
        for p in to_remove:
            p.unlink()
            print(f"[{cls}] Removed duplicate: {p.name}")

        # Remaining files (re-scan to ensure we have the updated list)
        remaining = []
        for entry in folder.iterdir():
            if not entry.is_file():
                continue
            m = FNAME_RE.match(entry.name)
            if not m or m.group('cls') != cls:
                continue
            idx = int(m.group('idx'))
            ext = m.group('ext').lower()
            remaining.append((idx, entry, ext))

        # Sort by original idx
        remaining.sort(key=lambda t: t[0])

        # Two-pass rename to avoid collisions (rename to temp names first)
        temp_map = {}
        for new_idx, (_, old_path, ext) in enumerate(remaining):
            tmp_name = f"__tmp__{uuid4().hex}.{ext}"
            tmp_path = old_path.with_name(tmp_name)
            old_path.rename(tmp_path)
            temp_map[tmp_path] = (new_idx, ext)

        # Final pass to desired names
        for tmp_path, (new_idx, ext) in temp_map.items():
            new_name = f"{cls}_{new_idx}.{ext}"
            final_path = tmp_path.with_name(new_name)
            tmp_path.rename(final_path)
            print(f"[{cls}] Renamed to: {final_path.name}")

if __name__ == "__main__":
    remove_and_rename_images("/path/to/your/folder")
