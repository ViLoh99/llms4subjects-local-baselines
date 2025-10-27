#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jsonld2jsonl_zst.py  –  wandelt einen Ordner mit JSON-LD-Dateien
in eine einzige .jsonl.zst-Datei um.

Aufruf:
    python jsonld2jsonl_zst.py  RAW_DIR  OUT_FILE.zst  [-j 8]

RAW_DIR       = Verzeichnis, das die .json-Dateien (rekursiv) enthält
OUT_FILE.zst  = Zielpfad (wird überschrieben, wenn vorhanden)
-j/--jobs N   = optionale Prozessanzahl für Parallel-Parsing
"""

import argparse, re, sys, os
from pathlib import Path
from functools import partial
import orjson, zstandard as zstd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ---------------- Hilfsfunktionen -----------------------------------------
def load_excluded_json(json_path: str) -> set[str]:
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(Path(p).name for p in data.get("Records", []))


def language_code(uri: str | None) -> str:
    """http://id.loc.gov/vocabulary/iso639-1/de → 'de'  |  None → 'und'"""
    if not uri:
        return "und"
    return Path(uri).name or "und"

'''
def extract_one(path: Path) -> bytes | None:
    """
    Lädt eine JSON-Datei, sucht den 'Hauptknoten' (hat @type UND title),
    baut das Ausgabeschema und liefert eine BYTES-Zeile (inkl. '\n').
    """
    try:
        with path.open("rb") as f:
            data = orjson.loads(f.read())
    except Exception as e:
        sys.stderr.write(f"⚠️  Fehler beim Laden {path}: {e}\n")
        return None

    graph = data.get("@graph", [])
    main = next((n for n in graph if n.get("@type") and n.get("title")), None)
    if main is None:
        # keine passenden Infos vorhanden
        return None

    out = {
        "id"      : Path(path).stem,                       # Dateiname als Fallback-ID
        "type"    : main["@type"].split(":")[-1],          # z.B. 'Book'
        "language": language_code(main.get("language")),
        "title"   : main["title"],
        "abstract": main.get("abstract", ""),
    }

    # ----- Subjects -------------------------------------------------------
    subj_list = list(main.get("subject", []))
    subj_field = main.get("dcterms:subject", {})
    subj_ids = []

    if isinstance(subj_field, dict):
        subj_ids = [subj_field.get("@id")]
    elif isinstance(subj_field, list):
        subj_ids = [d.get("@id") for d in subj_field if isinstance(d, dict)]

    # Jetzt iterieren wir über mögliche IDs und holen Labels
    for subj_id in subj_ids:
        if not subj_id:
            continue
        label = next((n.get("sameAs") for n in graph
                    if n.get("@id") == subj_id and "sameAs" in n), None)
        if label:
            subj_list.append(label)



    # Klassifikations-Strings wegfiltern & Duplikate entfernen
    out["subjects"] = list({
        s for s in subj_list
        if not s.startswith("(") and s.strip()
    })

    return orjson.dumps(out, option=orjson.OPT_NON_STR_KEYS) + b"\n"
'''

def extract_one(path: Path) -> bytes | None:
    """
    Loads a JSON-LD file, finds the main node (with @type and title),
    extracts all required fields, and outputs a JSON line (as bytes).
    """
    try:
        with path.open("rb") as f:
            data = orjson.loads(f.read())
    except Exception as e:
        sys.stderr.write(f"⚠️  Fehler beim Laden {path}: {e}\n")
        return None

    graph = data.get("@graph", [])
    main = next((n for n in graph if n.get("@type") and n.get("title")), None)
    if main is None:
        return None

    out = {
        "id": Path(path).stem,  # fallback ID
        "type": main["@type"].split(":")[-1],  # e.g., 'Book'
        "language": language_code(main.get("language")),
        "title": main["title"],
        "abstract": main.get("abstract", ""),
    }

    # ----- Task 2: Extract all GND IDs from dcterms:subject -----
    subj_field = main.get("dcterms:subject", [])
    gnd_ids = []
    if isinstance(subj_field, dict):
        gnd_id = subj_field.get("@id")
        if gnd_id:
            gnd_ids.append(gnd_id.replace("gnd:", ""))
    elif isinstance(subj_field, list):
        for d in subj_field:
            if isinstance(d, dict) and "@id" in d:
                gnd_ids.append(d["@id"].replace("gnd:", ""))

    out["gnd_subject_ids"] = gnd_ids

    # ----- Task 1: Extract LINSEARCH mapping codes from subject -----
    subject_list = main.get("subject", [])
    lim_codes = []
    if isinstance(subject_list, str):
        subject_list = [subject_list]
    for subj in subject_list:
        if "classificationName=linsearch:mapping" in subj:
            code = subj.split(")")[-1].strip()
            if code != "rest":
                lim_codes.append(code)
    out["linsearch_codes"] = lim_codes

    return orjson.dumps(out, option=orjson.OPT_NON_STR_KEYS) + b"\n"


# ---------------- Hauptprogramm ------------------------------------------
def main(raw_dir: str, out_file: str, n_jobs: int, exclude_path: str | None):
    raw_dir = Path(raw_dir)
    files   = list(raw_dir.rglob("*.jsonld"))

    excluded_files = load_excluded_json(exclude_path) if exclude_path else set()
    if excluded_files:
        files = [f for f in files if f.name not in excluded_files]
        print(f"📛 {len(excluded_files)} files excluded from input")

    if not files:
        print("Keine .json-Dateien gefunden – Pfad prüfen!", file=sys.stderr)
        sys.exit(1)

    cctx   = zstd.ZstdCompressor(level=19, threads=max(1, n_jobs))
    writer = cctx.stream_writer(open(out_file, "wb"))

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for line in tqdm(pool.map(extract_one, files, chunksize=256),
                         total=len(files), unit="file"):
            if line:
                writer.write(line)

    writer.flush()
    writer.close()
    print(f"✅  Fertig! → {out_file} ({len(files)} Dateien verarbeitet)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON-LD → JSONL.zst-Konverter")
    parser.add_argument("raw_dir",  help="Wurzel­verzeichnis mit JSON-Dateien")
    parser.add_argument("out_file", help="Zieldatei, z. B. corpus.jsonl.zst")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                        help="Anzahl Prozesse (Default = alle Kerne)")
    parser.add_argument("--exclude", type=str,
                        help="Optional JSON file with list of files to exclude (format: { 'Records': [...] })")

    args = parser.parse_args()
    main(args.raw_dir, args.out_file, args.jobs, args.exclude)