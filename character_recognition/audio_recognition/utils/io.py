import json


def load_json(file_path):
    with open(file_path, 'r') as infile:
        return json.load(infile)


def load_jsonl(file_path):
    """Load a JSONL file into a list of json objects."""
    data = []
    with open(file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_json(obj, file_path, indent=4):
    with open(file_path, 'w') as outfile:
        json.dump(obj, outfile, indent=indent)


def split_list(lst, n):
    """
    Split a list into n (roughly) equal-sized chunks.
    """
    if n <= 0:
        raise ValueError("Number of chunks must be positive")
    if not lst:
        return [[] for _ in range(n)]

    q, r = divmod(len(lst), n)
    chunks = []
    start = 0

    for i in range(n):
        end = start + q + (1 if i < r else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def get_chunk(lst, n, k):
    """
    Get the k-th chunk from n chunks.
    """
    chunks = split_list(lst, n)
    if k >= len(chunks):
        raise IndexError(f"Chunk index {k} is out of range for {len(chunks)} chunks.")
    return chunks[k]
