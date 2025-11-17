# Narrative Building CLI

This repository contains a narrative building utility that ingests a large JSON (or JSONL/NDJSON) news dataset, filters for source quality, and produces a structured storyline for any topic supplied on the command line.

## Features
- Filters articles by `source_rating` (strictly greater than 8 by default)
- Topic-aware relevance ranking via TF-IDF cosine similarity
- Chronological timeline with heuristic "why it matters" notes
- Narrative clusters with automatically generated theme labels
- Narrative graph connecting articles with `builds_on`, `adds_context`, `contradicts`, and `escalates` relations

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python narrative_builder.py --json path/to/news.json --topic "AI regulation"
python narrative_builder.py --json path/to/news.json --topic "Israel-Iran conflict" --topk 200
python narrative_builder.py --json path/to/news.json --topic "Jubilee Hills elections" --out outputs/jubilee_hills.json
python narrative_builder.py --json news.json.gz --topic "AI regulation" --progress
```

## Frontend dashboard
An interactive Next.js dashboard lives in `frontend/`. It shells out to `narrative_builder.py`, making it easy to explore results from the browser.

### Getting started
```bash
cd frontend
npm install
npm run dev
```

The UI is available at `http://localhost:3000` and accepts either a path to a local JSON/NDJSON dataset (resolved relative to `Narrative_Building`) or pasted JSON content.

Set `NARRATIVE_PYTHON` to point at your virtualenv interpreter if needed, e.g. `export NARRATIVE_PYTHON="$VIRTUAL_ENV/bin/python"`.

### Arguments
- `--topic` (required): text string describing the storyline you want to build
- `--json` (required): path to the JSON, JSONL, NDJSON, or gzipped JSON dataset
- `--min_rating`: minimum `source_rating` (default `8.0`, retained articles must exceed this)
- `--topk`: maximum number of relevant articles to keep in the narrative (default `300`)
- `--out`: optional path to persist the JSON output in addition to standard output
- `--progress`: show a progress bar during normalization when `tqdm` is installed

## Output Schema
The script prints (and optionally saves) a JSON object with the following structure:
```json
{
  "narrative_summary": "...",
  "timeline": [
    {"date": "2025-04-28", "headline": "...", "url": "...", "score": 0.54, "why_it_matters": "..."}
  ],
  "clusters": [
    {"cluster_id": 0, "theme": "policy, reform", "top_terms": ["policy", "reform"], "size": 8, ...}
  ],
  "graph": {
    "nodes": [{"id": 0, "headline": "...", "score": 0.62, ...}],
    "edges": [{"source": 0, "target": 2, "relation": "builds_on", "score": 0.48}]
  },
  "metadata": {"topic": "AI regulation", "source_count": 1423, "selected_count": 180, ...}
}
```

## Tips
- For very large datasets, consider preprocessing to ensure consistent field naming (`title`, `content`, `source_rating`, `published_at`, etc.).
- Increase `--topk` if the topic is broad and you want more context; decrease it for tighter narratives.
- The narrative graph can be visualized with tools that accept node/edge lists (for example, Gephi or NetworkX scripts).
