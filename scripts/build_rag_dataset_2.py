"""
Build a RAG retrieval dataset from test_2.csv for the 5G RCA multi-agent system.

test_2.csv format:
  - Columns: question, answer
  - Each question embeds: problem description, 8 root-cause options (C1-C8),
    domain knowledge (beam/downtilt rules), user-plane drive-test data,
    and engineering parameters.
  - Answer is one of C1..C8.

Optimised for RAG retrieval quality:
  - Global singletons for domain knowledge (1 chunk) and diagnostic option
    templates (8 chunks) — avoids massive duplication.
  - Cell configurations deduplicated — each unique cell emitted once with a
    list of scenario IDs it appears in.
  - Per-scenario chunks focus on data-specific content: combined analysis
    (KPI + neighbor + cell summary), raw time-series, and full scenario doc.
  - Rich metadata on every chunk for hybrid search / metadata filtering.

Output: data/rag_dataset_2.jsonl  (one JSON object per line)
"""

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "test_2.csv"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "rag_dataset_2.jsonl"

ROOT_CAUSE_LABELS = {
    "C1": "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    "C2": "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    "C3": "A neighboring cell provides higher throughput.",
    "C4": "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    "C5": "Frequent handovers degrade performance.",
    "C6": "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    "C7": "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    "C8": "Average scheduled RBs are below 160, affecting throughput.",
}

ROOT_CAUSE_CATEGORIES = {
    "C1": "coverage_tilt",
    "C2": "overshooting",
    "C3": "neighbor_throughput",
    "C4": "overlapping_coverage",
    "C5": "handover_degradation",
    "C6": "pci_collision",
    "C7": "mobility_speed",
    "C8": "rb_scheduling",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk_id(prefix: str, chunk_type: str, sub: int = 0) -> str:
    raw = f"{prefix}:{chunk_type}:{sub}"
    return hashlib.md5(raw.encode()).hexdigest()


def parse_pipe_table(raw_text: str) -> tuple[list[str], list[dict]]:
    """Parse pipe-delimited text into (headers, list-of-row-dicts)."""
    if not raw_text:
        return [], []
    lines = [ln.strip() for ln in raw_text.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return [], []
    headers = [h.strip() for h in lines[0].split("|")]
    rows = []
    for line in lines[1:]:
        vals = [v.strip() for v in line.split("|")]
        if len(vals) == len(headers):
            rows.append(dict(zip(headers, vals)))
    return headers, rows


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def stats_str(vals: list[float], unit: str = "") -> str:
    if not vals:
        return "N/A"
    return f"min={min(vals):.1f}{unit} avg={statistics.mean(vals):.1f}{unit} max={max(vals):.1f}{unit}"


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two GPS points."""
    R = 6_371_000
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Question parser
# ---------------------------------------------------------------------------

def parse_question(question: str) -> dict:
    result = {
        "problem_desc": "",
        "domain_knowledge": "",
        "user_plane_raw": "",
        "eng_params_raw": "",
    }

    up_marker = re.search(
        r"User plane drive[- ]?test data as follows\s*[\uff1a:]?\s*\n",
        question, re.IGNORECASE,
    )
    eng_marker = re.search(
        r"Eng[ei]neering parameters data as follows\s*[\uff1a:]?\s*\n",
        question, re.IGNORECASE,
    )

    if up_marker and eng_marker:
        preamble = question[:up_marker.start()].strip()
        result["user_plane_raw"] = question[up_marker.end():eng_marker.start()].strip()
        result["eng_params_raw"] = question[eng_marker.end():].strip()
    elif up_marker:
        preamble = question[:up_marker.start()].strip()
        result["user_plane_raw"] = question[up_marker.end():].strip()
    else:
        preamble = question.strip()

    given_match = re.search(r"\nGiven:\s*\n", preamble)
    if given_match:
        result["problem_desc"] = preamble[:given_match.start()].strip()
        result["domain_knowledge"] = preamble[given_match.start():].strip()
    else:
        result["problem_desc"] = preamble

    return result


# ---------------------------------------------------------------------------
# Global (singleton) chunk builders
# ---------------------------------------------------------------------------

def build_global_domain_knowledge(sample_dk_text: str) -> dict:
    """Single domain-knowledge chunk for the whole dataset."""
    content = (
        "5G Domain Knowledge — Beam Scenario & Downtilt Interpretation Rules\n\n"
        f"{sample_dk_text}\n\n"
        "Application guidance:\n"
        "- Use these rules to interpret Digital Tilt values in cell configs "
        "(255 = 6 degrees default).\n"
        "- Beam Scenario determines vertical beamwidth which affects coverage "
        "footprint and tilt sensitivity.\n"
        "- Large downtilt + narrow beamwidth = aggressive coverage limiting "
        "(potential weak far-end).\n"
        "- Small downtilt + wide beamwidth = broad coverage "
        "(potential overshooting / interference)."
    )
    return {
        "chunk_id": make_chunk_id("global", "domain_knowledge"),
        "scenario_id": "global",
        "chunk_type": "domain_knowledge",
        "content": content,
        "metadata": {
            "scope": "global",
            "topics": [
                "electronic_downtilt", "beam_scenario",
                "vertical_beamwidth", "coverage_analysis",
            ],
        },
    }


def build_global_diagnostic_chunks() -> list[dict]:
    """One chunk per root cause category with detailed retrieval signal."""
    chunks = []
    for cid, label in ROOT_CAUSE_LABELS.items():
        cat = ROOT_CAUSE_CATEGORIES[cid]

        if cid == "C1":
            indicators = (
                "Look for: weak RSRP at cell edge, large mechanical/digital downtilt, "
                "narrow vertical beamwidth (6 deg), high site height amplifying tilt effect."
            )
        elif cid == "C2":
            indicators = (
                "Look for: serving cell distance > 1km from UE (use GPS vs cell location), "
                "low site height, small downtilt, wide beamwidth, strong signal at distance."
            )
        elif cid == "C3":
            indicators = (
                "Look for: neighbor cell with stronger RSRP than serving, "
                "missing or delayed handover, neighbor on same or different frequency."
            )
        elif cid == "C4":
            indicators = (
                "Look for: multiple non-colocated neighbors with RSRP close to serving, "
                "high interference (low SINR despite adequate RSRP), co-frequency cells."
            )
        elif cid == "C5":
            indicators = (
                "Look for: multiple serving PCI changes in the time-series, "
                "throughput drops coinciding with handover events, ping-pong patterns."
            )
        elif cid == "C6":
            indicators = (
                "Look for: serving PCI mod 30 equals a neighbor PCI mod 30, "
                "DMRS sequence collision causing interference despite adequate RSRP/SINR."
            )
        elif cid == "C7":
            indicators = (
                "Look for: GPS speed consistently > 40 km/h across samples, "
                "Doppler-induced degradation, throughput drops correlating with speed."
            )
        else:
            indicators = (
                "Look for: average DL RB count < 160 across samples, "
                "scheduler not allocating enough resources despite good RF conditions."
            )

        content = (
            f"Root Cause: {cid} — {label}\n"
            f"Category: {cat}\n"
            f"Diagnostic indicators:\n{indicators}"
        )
        chunks.append({
            "chunk_id": make_chunk_id("global", f"root_cause_{cid}"),
            "scenario_id": "global",
            "chunk_type": "root_cause_reference",
            "content": content,
            "metadata": {
                "scope": "global",
                "root_cause_id": cid,
                "root_cause_label": label,
                "category": cat,
            },
        })
    return chunks


# ---------------------------------------------------------------------------
# Per-scenario chunk builders
# ---------------------------------------------------------------------------

def build_combined_analysis(idx: int, parsed: dict, answer: str,
                            serving_pcis: list[str],
                            cell_ids: list[str]) -> dict:
    """
    Single rich chunk combining KPI analysis, neighbor analysis, and cell
    summary — the primary retrieval target for a scenario query.
    """
    _, up_rows = parse_pipe_table(parsed["user_plane_raw"])
    _, eng_rows = parse_pipe_table(parsed["eng_params_raw"])

    # --- KPI stats ---
    rsrp_vals, sinr_vals, tp_vals, rb_vals, speed_vals = [], [], [], [], []
    handover_count, prev_pci = 0, None
    for r in up_rows:
        rsrp_vals.append(safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]", ""), -999))
        sinr_vals.append(safe_float(r.get("5G KPI PCell RF Serving SS-SINR [dB]", ""), -999))
        tp_vals.append(safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]", ""), 0))
        rb_vals.append(safe_float(r.get("5G KPI PCell Layer1 DL RB Num (Including 0)", ""), 0))
        speed_vals.append(safe_float(r.get("GPS Speed (km/h)", ""), 0))
        pci = r.get("5G KPI PCell RF Serving PCI", "")
        if pci and prev_pci and pci != prev_pci:
            handover_count += 1
        if pci:
            prev_pci = pci

    rsrp_clean = [v for v in rsrp_vals if v > -900]
    sinr_clean = [v for v in sinr_vals if v > -900]
    low_tp = sum(1 for t in tp_vals if t < 600)
    low_rb = sum(1 for r in rb_vals if r < 160)
    high_speed = sum(1 for s in speed_vals if s > 40)
    weak_rsrp = sum(1 for r in rsrp_clean if r < -100)

    # --- Neighbor analysis ---
    pci_to_cell = {r.get("PCI", ""): f"{r.get('gNodeB ID', '')}_{r.get('Cell ID', '')}" for r in eng_rows}
    pci_to_gnb = {r.get("PCI", ""): r.get("gNodeB ID", "") for r in eng_rows}
    neighbor_stats: dict[str, list[float]] = {}
    pci_mod30: dict[str, int] = {}
    neighbor_stronger = 0

    for r in up_rows:
        serv_pci = r.get("5G KPI PCell RF Serving PCI", "")
        serv_rsrp = safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]", ""), -999)
        if serv_pci:
            try:
                pci_mod30[serv_pci] = int(serv_pci) % 30
            except ValueError:
                pass
        for n in range(1, 6):
            npci = r.get(f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {n} PCI", "")
            nrsrp_str = r.get(
                f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {n} Filtered Tx BRSRP [dBm]", ""
            )
            if npci and npci != "-" and nrsrp_str and nrsrp_str != "-":
                nrsrp = safe_float(nrsrp_str, -999)
                if nrsrp > -900:
                    neighbor_stats.setdefault(npci, []).append(nrsrp)
                    if nrsrp > serv_rsrp and serv_rsrp > -900:
                        neighbor_stronger += 1
                try:
                    pci_mod30[npci] = int(npci) % 30
                except ValueError:
                    pass

    mod30_groups: dict[int, list[str]] = {}
    for pci_val, mod in pci_mod30.items():
        mod30_groups.setdefault(mod, []).append(pci_val)
    pci_collisions = [(mod, pcis) for mod, pcis in mod30_groups.items() if len(pcis) > 1]

    neighbor_lines = []
    for npci, rsrps in sorted(neighbor_stats.items(), key=lambda x: -statistics.mean(x[1])):
        cell_name = pci_to_cell.get(npci, "unknown")
        gnb = pci_to_gnb.get(npci, "")
        coloc = "colocated" if gnb and sum(1 for p, g in pci_to_gnb.items() if g == gnb) > 1 else "non-colocated"
        neighbor_lines.append(
            f"  PCI {npci} ({cell_name}, {coloc}): avg={statistics.mean(rsrps):.1f}dBm"
        )

    # --- Coverage distance estimation ---
    coverage_distances = []
    for r in up_rows:
        serv_pci = r.get("5G KPI PCell RF Serving PCI", "")
        ue_lat = safe_float(r.get("Latitude", ""), 0)
        ue_lon = safe_float(r.get("Longitude", ""), 0)
        if serv_pci and ue_lat and ue_lon:
            for er in eng_rows:
                if er.get("PCI", "") == serv_pci:
                    cell_lat = safe_float(er.get("Latitude", ""), 0)
                    cell_lon = safe_float(er.get("Longitude", ""), 0)
                    if cell_lat and cell_lon:
                        d = haversine_m(ue_lat, ue_lon, cell_lat, cell_lon)
                        coverage_distances.append(d)
                    break

    over_1km = sum(1 for d in coverage_distances if d > 1000)

    # --- Cell config summary ---
    config_lines = []
    for r in eng_rows:
        gnb = r.get("gNodeB ID", "")
        cell = r.get("Cell ID", "")
        dt = r.get("Digital Tilt", "")
        if dt == "255":
            dt = "6(def)"
        beam = r.get("Beam Scenario", "")
        config_lines.append(
            f"  {gnb}_{cell}: PCI={r.get('PCI', '')} Az={r.get('Mechanical Azimuth', '')} "
            f"MTilt={r.get('Mechanical Downtilt', '')} DTilt={dt} "
            f"Beam={beam} H={r.get('Height', '')}m {r.get('TxRx Mode', '')}"
        )

    # --- Compose chunk ---
    content = (
        f"5G Throughput RCA — Scenario test2_{idx:04d}\n"
        f"Root cause: {answer} — {ROOT_CAUSE_LABELS.get(answer, '')}\n"
        f"Category: {ROOT_CAUSE_CATEGORIES.get(answer, '')}\n\n"
        f"KPI Summary ({len(up_rows)} drive-test samples):\n"
        f"  RSRP: {stats_str(rsrp_clean, 'dBm')}\n"
        f"  SINR: {stats_str(sinr_clean, 'dB')}\n"
        f"  Throughput: {stats_str(tp_vals, 'Mbps')}\n"
        f"  RB Count: {stats_str(rb_vals)}\n"
        f"  Speed: {stats_str(speed_vals, 'km/h')}\n\n"
        f"Degradation Flags:\n"
        f"  Throughput <600Mbps: {low_tp}/{len(tp_vals)}\n"
        f"  RBs <160: {low_rb}/{len(rb_vals)}\n"
        f"  Speed >40km/h: {high_speed}/{len(speed_vals)}\n"
        f"  RSRP <-100dBm: {weak_rsrp}/{len(rsrp_clean)}\n"
        f"  Neighbor stronger than serving: {neighbor_stronger}\n"
        f"  Handovers: {handover_count}\n"
        f"  Coverage distance >1km: {over_1km}/{len(coverage_distances)}\n"
        f"  PCI mod30 collisions: {len(pci_collisions)}\n\n"
        f"Serving PCIs: {', '.join(serving_pcis)}\n"
        f"Neighbor cells:\n" + "\n".join(neighbor_lines[:10]) + "\n\n"
        f"Cell Configurations ({len(eng_rows)} cells):\n" + "\n".join(config_lines)
    )

    return {
        "chunk_id": make_chunk_id(f"test2:{idx}", "combined_analysis"),
        "scenario_id": f"test2_{idx:04d}",
        "chunk_type": "combined_analysis",
        "content": content,
        "metadata": {
            "source": "test_2.csv",
            "row_index": idx,
            "answer": answer,
            "root_cause_category": ROOT_CAUSE_CATEGORIES.get(answer, ""),
            "root_cause_label": ROOT_CAUSE_LABELS.get(answer, ""),
            "serving_pcis": serving_pcis,
            "cell_ids": cell_ids,
            "num_samples": len(up_rows),
            "num_cells": len(eng_rows),
            "avg_rsrp": round(statistics.mean(rsrp_clean), 2) if rsrp_clean else None,
            "avg_sinr": round(statistics.mean(sinr_clean), 2) if sinr_clean else None,
            "avg_throughput": round(statistics.mean(tp_vals), 2) if tp_vals else None,
            "avg_rb_count": round(statistics.mean(rb_vals), 2) if rb_vals else None,
            "avg_speed": round(statistics.mean(speed_vals), 2) if speed_vals else None,
            "handover_count": handover_count,
            "low_throughput_ratio": round(low_tp / max(len(tp_vals), 1), 3),
            "low_rb_ratio": round(low_rb / max(len(rb_vals), 1), 3),
            "high_speed_ratio": round(high_speed / max(len(speed_vals), 1), 3),
            "weak_rsrp_ratio": round(weak_rsrp / max(len(rsrp_clean), 1), 3),
            "neighbor_stronger_count": neighbor_stronger,
            "pci_mod30_collisions": len(pci_collisions),
            "over_1km_ratio": round(over_1km / max(len(coverage_distances), 1), 3),
        },
    }


def build_raw_timeseries(idx: int, parsed: dict, answer: str) -> list[dict]:
    """
    Raw time-series drive-test data as a chunk for fine-grained retrieval.
    Preserves per-sample detail that aggregated stats lose.
    """
    _, up_rows = parse_pipe_table(parsed["user_plane_raw"])
    if not up_rows:
        return []

    lines = []
    for r in up_rows:
        ts = r.get("Timestamp", "")
        pci = r.get("5G KPI PCell RF Serving PCI", "")
        rsrp = r.get("5G KPI PCell RF Serving SS-RSRP [dBm]", "")
        sinr = r.get("5G KPI PCell RF Serving SS-SINR [dB]", "")
        tp = r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]", "")
        speed = r.get("GPS Speed (km/h)", "")
        rb = r.get("5G KPI PCell Layer1 DL RB Num (Including 0)", "")
        lon = r.get("Longitude", "")
        lat = r.get("Latitude", "")

        n_pcis, n_rsrps = [], []
        for n in range(1, 6):
            np_val = r.get(f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {n} PCI", "")
            nr_val = r.get(
                f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {n} Filtered Tx BRSRP [dBm]", ""
            )
            if np_val and np_val != "-":
                n_pcis.append(np_val)
                n_rsrps.append(nr_val if nr_val and nr_val != "-" else "?")

        nb_str = " ".join(f"N{i+1}:PCI{p}={r_}" for i, (p, r_) in enumerate(zip(n_pcis, n_rsrps)))
        lines.append(
            f"{ts} | PCI={pci} RSRP={rsrp} SINR={sinr} TP={tp}Mbps "
            f"Speed={speed}km/h RBs={rb} ({lon},{lat}) [{nb_str}]"
        )

    content = (
        f"Drive Test Time-Series — Scenario test2_{idx:04d} "
        f"(answer: {answer})\n" + "\n".join(lines)
    )

    return [{
        "chunk_id": make_chunk_id(f"test2:{idx}", "raw_timeseries"),
        "scenario_id": f"test2_{idx:04d}",
        "chunk_type": "raw_timeseries",
        "content": content,
        "metadata": {
            "answer": answer,
            "num_samples": len(up_rows),
        },
    }]


# ---------------------------------------------------------------------------
# Cell configuration deduplication
# ---------------------------------------------------------------------------

def cell_config_key(row: dict) -> str:
    """Unique key for a cell config row."""
    fields = [
        row.get("gNodeB ID", ""), row.get("Cell ID", ""),
        row.get("Longitude", ""), row.get("Latitude", ""),
        row.get("Mechanical Azimuth", ""), row.get("Mechanical Downtilt", ""),
        row.get("Digital Tilt", ""), row.get("Digital Azimuth", ""),
        row.get("Beam Scenario", ""), row.get("Height", ""),
        row.get("PCI", ""), row.get("TxRx Mode", ""),
        row.get("Max Transmit Power", ""), row.get("Antenna Model", ""),
    ]
    return "|".join(fields)


def build_deduped_cell_config(key: str, row: dict, scenario_ids: list[str]) -> dict:
    """Build a single cell config chunk linked to all scenarios that use it."""
    gnb = row.get("gNodeB ID", "")
    cell = row.get("Cell ID", "")
    cell_label = f"{gnb}_{cell}"
    pci = row.get("PCI", "")
    dig_tilt_raw = row.get("Digital Tilt", "")
    dig_tilt = "6 (default=255)" if dig_tilt_raw == "255" else dig_tilt_raw
    beam = row.get("Beam Scenario", "")

    beam_upper = beam.upper()
    if "DEFAULT" in beam_upper or beam_upper in [f"SCENARIO_{i}" for i in range(1, 6)]:
        vbw = "6 degrees"
    elif beam_upper in [f"SCENARIO_{i}" for i in range(6, 12)]:
        vbw = "12 degrees"
    else:
        vbw = "25 degrees"

    content = (
        f"Cell Configuration: {cell_label} (PCI {pci})\n"
        f"Location: ({row.get('Latitude', '')}, {row.get('Longitude', '')}) | "
        f"Height: {row.get('Height', '')}m\n"
        f"Antenna: {row.get('TxRx Mode', '')} | Model: {row.get('Antenna Model', '')}\n"
        f"Azimuth: {row.get('Mechanical Azimuth', '')}deg "
        f"(digital: {row.get('Digital Azimuth', '')}deg)\n"
        f"Mechanical Downtilt: {row.get('Mechanical Downtilt', '')}deg | "
        f"Digital Tilt: {dig_tilt}deg\n"
        f"Beam Scenario: {beam} | Vertical Beamwidth: {vbw}\n"
        f"Max Transmit Power: {row.get('Max Transmit Power', '')}dBm\n"
        f"Used in {len(scenario_ids)} scenarios"
    )

    return {
        "chunk_id": make_chunk_id(f"cell:{key}", "cell_configuration"),
        "scenario_id": "shared",
        "chunk_type": "cell_configuration",
        "content": content,
        "metadata": {
            "scope": "shared",
            "gnb_id": gnb,
            "cell_id": cell,
            "cell_label": cell_label,
            "pci": pci,
            "height_m": row.get("Height", ""),
            "azimuth": row.get("Mechanical Azimuth", ""),
            "mechanical_downtilt": row.get("Mechanical Downtilt", ""),
            "digital_tilt": dig_tilt_raw,
            "beam_scenario": beam,
            "vertical_beamwidth": vbw,
            "txrx_mode": row.get("TxRx Mode", ""),
            "antenna_model": row.get("Antenna Model", ""),
            "scenario_ids": scenario_ids,
            "scenario_count": len(scenario_ids),
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print(f"Reading {INPUT_CSV} ...")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # --- First pass: collect cell configs for deduplication ---
    cell_registry: dict[str, dict] = {}  # key -> first row dict
    cell_scenarios: dict[str, list[str]] = {}  # key -> list of scenario_ids

    domain_knowledge_text = ""
    all_scenario_chunks: list[dict] = []
    total_rows = 0
    answer_counts: dict[str, int] = {}

    with open(INPUT_CSV, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for idx, row in enumerate(reader):
            question = row.get("question", "")
            answer = row.get("answer", "").strip()
            total_rows += 1
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

            parsed = parse_question(question)
            sid = f"test2_{idx:04d}"

            if not domain_knowledge_text and parsed["domain_knowledge"]:
                domain_knowledge_text = parsed["domain_knowledge"]

            _, up_rows = parse_pipe_table(parsed["user_plane_raw"])
            serving_pcis = sorted({
                r.get("5G KPI PCell RF Serving PCI", "")
                for r in up_rows if r.get("5G KPI PCell RF Serving PCI", "")
            })

            _, eng_rows = parse_pipe_table(parsed["eng_params_raw"])
            cell_ids = [f"{r.get('gNodeB ID', '')}_{r.get('Cell ID', '')}" for r in eng_rows]

            for er in eng_rows:
                key = cell_config_key(er)
                if key not in cell_registry:
                    cell_registry[key] = er
                cell_scenarios.setdefault(key, []).append(sid)

            try:
                all_scenario_chunks.append(
                    build_combined_analysis(idx, parsed, answer, serving_pcis, cell_ids)
                )
                all_scenario_chunks.extend(
                    build_raw_timeseries(idx, parsed, answer)
                )
            except Exception as e:
                print(f"  ERROR row {idx}: {e}", file=sys.stderr)

    # --- Write output ---
    chunk_type_counts: dict[str, int] = {}
    total_chunks = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        def write_chunk(chunk: dict):
            nonlocal total_chunks
            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            ctype = chunk.get("chunk_type", "unknown")
            chunk_type_counts[ctype] = chunk_type_counts.get(ctype, 0) + 1
            total_chunks += 1

        # Global chunks
        if domain_knowledge_text:
            write_chunk(build_global_domain_knowledge(domain_knowledge_text))

        for rc_chunk in build_global_diagnostic_chunks():
            write_chunk(rc_chunk)

        # Deduplicated cell configs
        for key, er in cell_registry.items():
            write_chunk(build_deduped_cell_config(key, er, cell_scenarios[key]))

        # Per-scenario chunks
        for chunk in all_scenario_chunks:
            write_chunk(chunk)

    file_mb = OUTPUT_JSONL.stat().st_size / (1024 * 1024)
    print(f"\nDone! {total_rows} scenarios -> {total_chunks} chunks")
    print(f"Output: {OUTPUT_JSONL} ({file_mb:.1f} MB)")
    print("\nChunk type distribution:")
    for ctype, count in sorted(chunk_type_counts.items()):
        print(f"  {ctype}: {count}")
    print(f"\nDeduplication savings:")
    print(f"  Cell configs: {sum(len(v) for v in cell_scenarios.values())} -> {len(cell_registry)} (deduped)")
    print(f"  Domain knowledge: 864 -> 1")
    print(f"  Diagnostic options: 864 -> 8")
    print(f"\nAnswer distribution:")
    for ans, count in sorted(answer_counts.items()):
        print(f"  {ans}: {count}")


if __name__ == "__main__":
    main()
