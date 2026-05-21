"""
Build a RAG retrieval dataset from test.csv for the 5G RCA multi-agent system.

Parses each 5G network troubleshooting scenario and produces semantically
meaningful document chunks with rich metadata, ready for embedding and
vector-store ingestion.

Output: data/rag_dataset.jsonl  (one JSON object per line)
"""

import ast
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "test.csv"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "rag_dataset.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_parse(raw: str) -> Any:
    """Parse a Python-literal or JSON string."""
    if not raw or raw.strip() in ("", "To be determined"):
        return raw.strip() if raw else ""
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return raw.strip()


def chunk_id(scenario_id: str, chunk_type: str, index: int = 0) -> str:
    raw = f"{scenario_id}:{chunk_type}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def parse_pipe_table(raw_text: str) -> list[dict]:
    """Parse pipe-delimited data into list of dicts."""
    if not raw_text:
        return []
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split("|")]
    rows = []
    for line in lines[1:]:
        vals = [v.strip() for v in line.split("|")]
        if len(vals) == len(headers):
            rows.append(dict(zip(headers, vals)))
    return rows


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Chunk builders – each returns a list of document dicts
# ---------------------------------------------------------------------------

def build_scenario_overview(sid: str, task: dict, context: dict,
                            data: dict, cell_ids: list[str]) -> dict:
    """High-level scenario description for broad retrieval."""
    ctx = context if isinstance(context, dict) else {}
    net_info = ctx.get("wireless_network_information", {})
    num_bs = net_info.get("num_base_stations", "unknown")
    net_type = net_info.get("network_type", "5G")
    mobility = net_info.get("mobility_scenario", "drive test")
    description = ctx.get("description", "")

    task_desc = task.get("description", "") if isinstance(task, dict) else str(task)
    options = task.get("options", []) if isinstance(task, dict) else []
    option_summary = "; ".join(
        f"{o['id']}: {o['label']}" for o in options[:5]
    )
    if len(options) > 5:
        option_summary += f" ... and {len(options)-5} more options"

    content = (
        f"5G Network Troubleshooting Scenario\n"
        f"Network type: {net_type} | Base stations: {num_bs} | "
        f"Mobility: {mobility}\n"
        f"Scenario: {description}\n"
        f"Problem: {task_desc}\n"
        f"Candidate solutions: {option_summary}\n"
        f"Cells involved: {', '.join(cell_ids)}"
    )
    return {
        "chunk_id": chunk_id(sid, "scenario_overview"),
        "scenario_id": sid,
        "chunk_type": "scenario_overview",
        "content": content,
        "metadata": {
            "network_type": net_type,
            "num_base_stations": num_bs,
            "mobility_scenario": mobility,
            "num_options": len(options),
            "cell_ids": cell_ids,
        },
    }


def build_cell_config_chunks(sid: str, data: dict) -> list[dict]:
    """One chunk per cell with full configuration details."""
    raw = data.get("network_configuration_data", "")
    rows = parse_pipe_table(raw)
    if not rows:
        return []

    chunks = []
    for i, row in enumerate(rows):
        gnb = row.get("gNodeB ID", "")
        cell = row.get("Cell ID", "")
        cell_label = f"{gnb}_{cell}"
        pci = row.get("PCI", "")
        band = row.get("Band", "")
        arfcn = row.get("DL ARFCN", "")
        bw = row.get("BW [MHz]", "")
        duplex = row.get("Duplex Mode", "")
        txrx = row.get("TxRx Mode", "")
        power = row.get("Transmission Power", "")
        max_power = row.get("Max Transmit Power", "")
        height = row.get("Height", "")
        azimuth = row.get("Mechanical Azimuth", "")
        dig_azimuth = row.get("Digital Azimuth", "")
        mech_tilt = row.get("Mechanical Downtilt", "")
        dig_tilt = row.get("Digital Tilt", "")
        pdcch = row.get("PdcchOccupiedSymbolNum", "")
        neighbors = row.get("PCell Neighbor Cell (gNodeBID_ARFCN_PCI)", "")

        ho_type = row.get("InterFreqHoEventType", "")
        a2_thresh = row.get("CovInterFreqA2RsrpThld [dBm]", "")
        a5_thresh1 = row.get("CovInterFreqA5RsrpThld1 [dBm]", "")
        a5_thresh2 = row.get("CovInterFreqA5RsrpThld2 [dBm]", "")
        a3_offset = row.get("IntraFreqHoA3Offset [0.5dB]", "")
        a3_hyst = row.get("IntraFreqHoA3Hyst [0.5dB]", "")
        a3_ttt = row.get("IntraFreqHoA3TimeToTrig", "")

        lon = row.get("Longitude", "")
        lat = row.get("Latitude", "")

        content = (
            f"Cell Configuration: {cell_label} (PCI {pci})\n"
            f"Location: ({lat}, {lon}) | Height: {height}m\n"
            f"Band: {band} | ARFCN: {arfcn} | BW: {bw} | Duplex: {duplex}\n"
            f"Antenna: {txrx} | Azimuth: {azimuth}deg (digital: {dig_azimuth}) | "
            f"Mechanical Downtilt: {mech_tilt}deg | Digital Tilt: {dig_tilt}deg\n"
            f"Power: {power}dBm (max {max_power}dBm)\n"
            f"PDCCH Symbol: {pdcch}\n"
            f"Handover Config: {ho_type} | A2 Threshold: {a2_thresh}dBm | "
            f"A5 Thresholds: {a5_thresh1}/{a5_thresh2}dBm | "
            f"A3 Offset: {a3_offset} (x0.5dB) | A3 Hysteresis: {a3_hyst} (x0.5dB) | "
            f"A3 TTT: {a3_ttt}\n"
            f"Neighbor Cells: {neighbors}"
        )

        chunks.append({
            "chunk_id": chunk_id(sid, "cell_config", i),
            "scenario_id": sid,
            "chunk_type": "cell_configuration",
            "content": content,
            "metadata": {
                "gnb_id": gnb,
                "cell_id": cell,
                "cell_label": cell_label,
                "pci": pci,
                "band": band,
                "arfcn": arfcn,
                "duplex_mode": duplex,
                "transmission_power": power,
                "height_m": height,
                "azimuth": azimuth,
                "mechanical_downtilt": mech_tilt,
                "digital_tilt": dig_tilt,
            },
        })

    return chunks


def build_drive_test_analysis(sid: str, data: dict) -> list[dict]:
    """Summarize user-plane drive test KPIs into an analysis chunk."""
    raw = data.get("user_plane_data", "")
    rows = parse_pipe_table(raw)
    if not rows:
        return []

    rsrp_vals, sinr_vals, tp_vals, bler_vals, mcs_vals = [], [], [], [], []
    serving_pcis = set()

    for r in rows:
        rsrp_vals.append(safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]", ""), -999))
        sinr_vals.append(safe_float(r.get("5G KPI PCell RF Serving SS-SINR [dB]", ""), -999))
        tp_vals.append(safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]", ""), 0))
        bler_vals.append(safe_float(r.get("Initial BLER(%)", ""), 0))
        mcs_vals.append(safe_float(r.get("Avg MCS", ""), 0))
        pci = r.get("5G KPI PCell RF Serving PCI", "")
        if pci:
            serving_pcis.add(pci)

    rsrp_vals = [v for v in rsrp_vals if v > -900]
    sinr_vals = [v for v in sinr_vals if v > -900]

    def stats_str(vals, unit=""):
        if not vals:
            return "N/A"
        mn, mx, avg = min(vals), max(vals), statistics.mean(vals)
        return f"min={mn:.1f}{unit} avg={avg:.1f}{unit} max={mx:.1f}{unit}"

    low_tp_count = sum(1 for t in tp_vals if t < 100)
    high_bler_count = sum(1 for b in bler_vals if b > 10)
    low_sinr_count = sum(1 for s in sinr_vals if s < 0)

    content = (
        f"Drive Test KPI Analysis ({len(rows)} samples)\n"
        f"Serving PCIs observed: {', '.join(sorted(serving_pcis))}\n"
        f"SS-RSRP: {stats_str(rsrp_vals, 'dBm')}\n"
        f"SS-SINR: {stats_str(sinr_vals, 'dB')}\n"
        f"DL Throughput: {stats_str(tp_vals, 'Mbps')}\n"
        f"Initial BLER: {stats_str(bler_vals, '%')}\n"
        f"Avg MCS: {stats_str(mcs_vals)}\n"
        f"Degradation indicators: {low_tp_count}/{len(tp_vals)} samples <100Mbps | "
        f"{high_bler_count}/{len(rows)} samples >10% BLER | "
        f"{low_sinr_count}/{len(rows)} samples negative SINR"
    )

    return [{
        "chunk_id": chunk_id(sid, "drive_test_analysis"),
        "scenario_id": sid,
        "chunk_type": "drive_test_analysis",
        "content": content,
        "metadata": {
            "num_samples": len(rows),
            "serving_pcis": sorted(serving_pcis),
            "avg_rsrp": round(statistics.mean(rsrp_vals), 2) if rsrp_vals else None,
            "avg_sinr": round(statistics.mean(sinr_vals), 2) if sinr_vals else None,
            "avg_throughput": round(statistics.mean(tp_vals), 2) if tp_vals else None,
            "avg_bler": round(statistics.mean(bler_vals), 2) if bler_vals else None,
            "low_throughput_ratio": round(low_tp_count / max(len(tp_vals), 1), 3),
            "high_bler_ratio": round(high_bler_count / max(len(rows), 1), 3),
            "negative_sinr_ratio": round(low_sinr_count / max(len(rows), 1), 3),
        },
    }]


def build_signaling_analysis(sid: str, data: dict) -> list[dict]:
    """Structured summary of signaling events."""
    raw = data.get("signaling_plane_data", "")
    rows = parse_pipe_table(raw)
    if not rows:
        return []

    event_counts: dict[str, int] = {}
    ho_attempts, ho_successes = 0, 0
    rach_attempts, rach_successes = 0, 0
    reestab_count = 0
    events_narrative = []

    for r in rows:
        name = r.get("Event Name", "")
        content_field = r.get("Event Content", "")
        event_counts[name] = event_counts.get(name, 0) + 1

        if "HandoverAttempt" in name:
            ho_attempts += 1
            events_narrative.append(f"Handover attempt: {content_field}")
        elif "HandoverSuc" in name:
            ho_successes += 1
        elif "RandomAccessAttempt" in name:
            rach_attempts += 1
        elif "RandomAccessSuc" in name:
            rach_successes += 1
        elif "Reestablish" in name:
            reestab_count += 1
            events_narrative.append(f"RRC reestablishment: {content_field}")
        elif "EventA2" in name and "Config" not in name:
            events_narrative.append(f"A2 event triggered: {content_field}")
        elif "EventA3" in name and "Config" not in name:
            events_narrative.append(f"A3 event triggered: {content_field}")
        elif "EventA5" in name and "Config" not in name:
            events_narrative.append(f"A5 event triggered: {content_field}")

    event_summary = "; ".join(f"{k}: {v}" for k, v in sorted(event_counts.items()))
    narrative = "\n".join(events_narrative[:15])
    if len(events_narrative) > 15:
        narrative += f"\n... and {len(events_narrative)-15} more events"

    content = (
        f"Signaling Event Analysis ({len(rows)} events)\n"
        f"Event counts: {event_summary}\n"
        f"Handover: {ho_attempts} attempts, {ho_successes} successful\n"
        f"RACH: {rach_attempts} attempts, {rach_successes} successful\n"
        f"RRC Reestablishments: {reestab_count}\n"
        f"Key events:\n{narrative}"
    )

    return [{
        "chunk_id": chunk_id(sid, "signaling_analysis"),
        "scenario_id": sid,
        "chunk_type": "signaling_analysis",
        "content": content,
        "metadata": {
            "num_events": len(rows),
            "handover_attempts": ho_attempts,
            "handover_successes": ho_successes,
            "rach_attempts": rach_attempts,
            "rach_successes": rach_successes,
            "reestablishment_count": reestab_count,
            "event_types": list(event_counts.keys()),
        },
    }]


def build_traffic_analysis(sid: str, data: dict) -> list[dict]:
    """Cell-level traffic and utilization summary."""
    raw = data.get("traffic_data", "")
    rows = parse_pipe_table(raw)
    if not rows:
        return []

    lines = []
    cell_labels = []
    for r in rows:
        gnb = r.get("gNodeB_ID", "")
        cell = r.get("Cell_ID", "")
        cell_label = f"{gnb}_{cell}"
        cell_labels.append(cell_label)

        ul_prb = r.get("Uplink PRB utilization(%)", "")
        dl_prb = r.get("Downlink PRB utilization(%)", "")
        ul_tp = r.get("User Uplink Throughput(Mbps)", "")
        dl_tp = r.get("User Downlink Throughput(Mbps)", "")
        weak_cov = r.get("Downlink Weak Coversge Ratio", "")
        ta_ratio = r.get("TA>1KM Ratio", "")
        ul_interf = r.get("Uplink PRB Interference(dBm)", "")
        dl_cce_rate = r.get("Downlink CCE Allocation Success Rate(%)", "")

        lines.append(
            f"  {cell_label}: UL/DL PRB={ul_prb}%/{dl_prb}% | "
            f"UL/DL TP={ul_tp}/{dl_tp}Mbps | "
            f"Weak Coverage={weak_cov} | TA>1km={ta_ratio} | "
            f"UL Interference={ul_interf}dBm | DL CCE Success={dl_cce_rate}%"
        )

    content = (
        f"Traffic & Utilization Analysis ({len(rows)} cells)\n"
        + "\n".join(lines)
    )

    return [{
        "chunk_id": chunk_id(sid, "traffic_analysis"),
        "scenario_id": sid,
        "chunk_type": "traffic_analysis",
        "content": content,
        "metadata": {
            "num_cells": len(rows),
            "cell_labels": cell_labels,
        },
    }]


def build_mr_analysis(sid: str, data: dict) -> list[dict]:
    """Measurement report summary with inter-cell signal relationships."""
    raw = data.get("mr_data", "")
    rows = parse_pipe_table(raw)
    if not rows:
        return []

    serving_stats: dict[str, list[float]] = {}
    tp_vals = []
    for r in rows:
        pci = r.get("Serving PCI", "")
        rsrp = safe_float(r.get("Serving RSRP(dBm)", ""), -999)
        tp = safe_float(r.get("Throughput(Mbps)", ""), 0)
        if rsrp > -900:
            serving_stats.setdefault(pci, []).append(rsrp)
        tp_vals.append(tp)

    lines = []
    for pci, rsrps in serving_stats.items():
        avg_r = statistics.mean(rsrps)
        lines.append(f"  PCI {pci}: avg RSRP={avg_r:.1f}dBm ({len(rsrps)} reports)")

    neighbor_lines = []
    for r in rows[:5]:
        parts = []
        for n in range(1, 4):
            npci = r.get(f"Neighbor {n} PCI", "")
            nrsrp = r.get(f"Neighbor {n} RSRP(dBm)", "")
            if npci:
                parts.append(f"N{n}:PCI{npci}={nrsrp}dBm")
        if parts:
            neighbor_lines.append(f"  ServPCI {r.get('Serving PCI','')}: " + " | ".join(parts))

    content = (
        f"Measurement Report Analysis ({len(rows)} reports)\n"
        f"Serving cell statistics:\n" + "\n".join(lines) + "\n"
        f"MR Throughput: min={min(tp_vals):.1f} avg={statistics.mean(tp_vals):.1f} "
        f"max={max(tp_vals):.1f} Mbps\n"
        f"Sample neighbor relationships:\n" + "\n".join(neighbor_lines)
    )

    return [{
        "chunk_id": chunk_id(sid, "mr_analysis"),
        "scenario_id": sid,
        "chunk_type": "mr_analysis",
        "content": content,
        "metadata": {
            "num_reports": len(rows),
            "serving_pcis": list(serving_stats.keys()),
            "avg_throughput": round(statistics.mean(tp_vals), 2) if tp_vals else None,
        },
    }]


def build_diagnostic_options(sid: str, task: dict, context: dict) -> dict:
    """Chunk listing all candidate optimization actions for retrieval."""
    if not isinstance(task, dict):
        return None

    options = task.get("options", [])
    if not options:
        return None

    categories: dict[str, list[str]] = {}
    for o in options:
        label = o.get("label", "")
        oid = o.get("id", "")
        if "neighbor" in label.lower():
            categories.setdefault("Neighbor Optimization", []).append(f"{oid}: {label}")
        elif "azimuth" in label.lower():
            categories.setdefault("Antenna Azimuth Adjustment", []).append(f"{oid}: {label}")
        elif "tilt" in label.lower():
            categories.setdefault("Antenna Tilt Adjustment", []).append(f"{oid}: {label}")
        elif "power" in label.lower():
            categories.setdefault("Power Adjustment", []).append(f"{oid}: {label}")
        elif "a3" in label.lower() or "a2" in label.lower() or "a5" in label.lower():
            categories.setdefault("Handover Threshold Tuning", []).append(f"{oid}: {label}")
        elif "pdcch" in label.lower():
            categories.setdefault("PDCCH Configuration", []).append(f"{oid}: {label}")
        elif "server" in label.lower() or "transmission" in label.lower():
            categories.setdefault("Infrastructure Check", []).append(f"{oid}: {label}")
        elif "insufficient" in label.lower():
            categories.setdefault("Data Sufficiency", []).append(f"{oid}: {label}")
        else:
            categories.setdefault("Other", []).append(f"{oid}: {label}")

    lines = []
    for cat, items in categories.items():
        lines.append(f"[{cat}]")
        for item in items:
            lines.append(f"  {item}")

    content = (
        f"Diagnostic Options ({len(options)} candidates)\n"
        + "\n".join(lines)
    )

    return {
        "chunk_id": chunk_id(sid, "diagnostic_options"),
        "scenario_id": sid,
        "chunk_type": "diagnostic_options",
        "content": content,
        "metadata": {
            "num_options": len(options),
            "categories": list(categories.keys()),
            "option_ids": [o["id"] for o in options],
        },
    }


def build_full_scenario_document(sid: str, task: dict, context: dict,
                                 data: dict, cell_ids: list[str]) -> dict:
    """
    A single comprehensive document combining all data for the scenario.
    Useful as a 'parent document' in parent-document retrieval strategies.
    """
    ctx = context if isinstance(context, dict) else {}
    net_info = ctx.get("wireless_network_information", {})
    task_desc = task.get("description", "") if isinstance(task, dict) else str(task)

    sections = [
        f"=== 5G RCA Scenario {sid[:8]} ===",
        f"Network: {net_info.get('network_type','5G')} | "
        f"BSs: {net_info.get('num_base_stations','?')} | "
        f"Cells: {', '.join(cell_ids)}",
        f"Problem: {task_desc}",
    ]

    raw_config = data.get("network_configuration_data", "")
    if raw_config:
        config_rows = parse_pipe_table(raw_config)
        config_lines = []
        for r in config_rows:
            gnb = r.get("gNodeB ID", "")
            cell = r.get("Cell ID", "")
            config_lines.append(
                f"  {gnb}_{cell}: PCI={r.get('PCI','')} Band={r.get('Band','')} "
                f"Power={r.get('Transmission Power','')}dBm "
                f"Tilt={r.get('Mechanical Downtilt','')}/{r.get('Digital Tilt','')} "
                f"Az={r.get('Mechanical Azimuth','')}"
            )
        sections.append("Network Config:\n" + "\n".join(config_lines))

    raw_up = data.get("user_plane_data", "")
    if raw_up:
        up_rows = parse_pipe_table(raw_up)
        tp_vals = [safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]",""),0) for r in up_rows]
        rsrp_vals = [safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]",""),-999) for r in up_rows]
        rsrp_vals = [v for v in rsrp_vals if v > -900]
        sinr_vals = [safe_float(r.get("5G KPI PCell RF Serving SS-SINR [dB]",""),-999) for r in up_rows]
        sinr_vals = [v for v in sinr_vals if v > -900]
        sections.append(
            f"Drive Test Summary ({len(up_rows)} samples): "
            f"TP avg={statistics.mean(tp_vals):.1f}Mbps | "
            f"RSRP avg={statistics.mean(rsrp_vals):.1f}dBm | "
            f"SINR avg={statistics.mean(sinr_vals):.1f}dB"
        )

    raw_sig = data.get("signaling_plane_data", "")
    if raw_sig:
        sig_rows = parse_pipe_table(raw_sig)
        ho_att = sum(1 for r in sig_rows if "HandoverAttempt" in r.get("Event Name",""))
        reest = sum(1 for r in sig_rows if "Reestablish" in r.get("Event Name",""))
        sections.append(f"Signaling: {len(sig_rows)} events | HO attempts: {ho_att} | Reestablishments: {reest}")

    raw_traffic = data.get("traffic_data", "")
    if raw_traffic:
        tr_rows = parse_pipe_table(raw_traffic)
        dl_tps = [safe_float(r.get("User Downlink Throughput(Mbps)",""),0) for r in tr_rows]
        sections.append(
            f"Cell Traffic: {len(tr_rows)} cells | "
            f"Avg cell DL TP={statistics.mean(dl_tps):.1f}Mbps"
        )

    options = task.get("options", []) if isinstance(task, dict) else []
    if options:
        opt_text = "; ".join(f"{o['id']}:{o['label']}" for o in options)
        sections.append(f"Options: {opt_text}")

    content = "\n".join(sections)

    return {
        "chunk_id": chunk_id(sid, "full_scenario"),
        "scenario_id": sid,
        "chunk_type": "full_scenario",
        "content": content,
        "metadata": {
            "cell_ids": cell_ids,
            "num_base_stations": net_info.get("num_base_stations", ""),
            "network_type": net_info.get("network_type", "5G"),
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_cell_ids(data: dict) -> list[str]:
    """Extract cell IDs from network config data."""
    raw = data.get("network_configuration_data", "")
    rows = parse_pipe_table(raw)
    return [f"{r.get('gNodeB ID','')}_{r.get('Cell ID','')}" for r in rows]


def process_scenario(row: dict) -> list[dict]:
    """Process a single CSV row into multiple RAG document chunks."""
    sid = row.get("scenario_id", "")
    task = safe_parse(row.get("task", ""))
    context = safe_parse(row.get("context", ""))
    data = safe_parse(row.get("data", ""))
    if not isinstance(data, dict):
        data = {}

    cell_ids = extract_cell_ids(data)

    chunks = []

    overview = build_scenario_overview(sid, task, context, data, cell_ids)
    chunks.append(overview)

    chunks.extend(build_cell_config_chunks(sid, data))
    chunks.extend(build_drive_test_analysis(sid, data))
    chunks.extend(build_signaling_analysis(sid, data))
    chunks.extend(build_traffic_analysis(sid, data))
    chunks.extend(build_mr_analysis(sid, data))

    diag = build_diagnostic_options(sid, task, context)
    if diag:
        chunks.append(diag)

    full_doc = build_full_scenario_document(sid, task, context, data, cell_ids)
    chunks.append(full_doc)

    return chunks


def main():
    print(f"Reading {INPUT_CSV} ...")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_scenarios = 0
    chunk_type_counts: dict[str, int] = {}

    with open(INPUT_CSV, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        for row in reader:
            try:
                chunks = process_scenario(row)
                for chunk in chunks:
                    fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    ctype = chunk.get("chunk_type", "unknown")
                    chunk_type_counts[ctype] = chunk_type_counts.get(ctype, 0) + 1
                total_chunks += len(chunks)
                total_scenarios += 1
            except Exception as e:
                print(f"  ERROR processing scenario {row.get('scenario_id','?')}: {e}",
                      file=sys.stderr)

    print(f"\nDone! Processed {total_scenarios} scenarios -> {total_chunks} chunks")
    print(f"Output: {OUTPUT_JSONL}")
    print(f"File size: {OUTPUT_JSONL.stat().st_size / (1024*1024):.1f} MB")
    print("\nChunk type distribution:")
    for ctype, count in sorted(chunk_type_counts.items()):
        print(f"  {ctype}: {count}")


if __name__ == "__main__":
    main()
