"""
Memory-efficient multi-day IEX DEEP collector + incremental calibration.

daily_iex_collector.py downloads a full day's compressed pcap.gz to disk
(~11GB) and, via iex_cppparser's parse_file(), converts it to an
intermediate classic-pcap file (~40GB) before parsing -- both get deleted
right after, but each day peaks at ~50GB of transient disk.

This script never writes either of those to disk at all. Each day streams
straight through:

    curl <url> | editcap -F pcap - - | <C++ parser> /dev/stdin <prefix> <symbols>

editcap and the parser both accept '-'/'/dev/stdin' and read sequentially,
so the whole chain is one pipe from network socket to filtered CSV -- only
the small per-symbol _prl.csv/_trd.csv files (a few hundred MB/day) ever
touch disk. After each day finishes, it re-fits the queue-reactive model on
every day archived so far (via iex_calibrate.calibrate_multi) and reports
the updated parameters, so the estimate visibly firms up as more days are
pulled in, and a valid (if noisier) fit already sits on disk if you stop
partway through.

Usage:
    python iex_stream_collect.py --symbols symbols.txt --archive ./archive \\
        --dates 2026-08-13,2026-08-14,2026-08-17,2026-08-18,2026-08-19,2026-08-20,2026-08-21 \\
        --calibrate-symbol INTC
"""
import argparse
import glob
import json
import logging
import os
import subprocess
import sys

import iex_cppparser as iex_pkg
from iex_cppparser.download import get_hist_data

import iex_calibrate

IEX_PARSER = os.path.join(os.path.dirname(iex_pkg.__file__), 'bin', 'iex_parser_threaded.out')
MIN_OUTPUT_BYTES = 50_000  # a real trading day is MBs+; anything smaller means the stream failed silently


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def already_have(archive_dir, date_str):
    day_dir = os.path.join(archive_dir, date_str)
    if not os.path.isdir(day_dir):
        return False
    files = os.listdir(day_dir)
    return any(f.endswith('_prl.csv') for f in files) and any(f.endswith('_trd.csv') for f in files)


def stream_one_day(date_str, symbols_file, archive_dir, retries=3):
    """Stream-download + convert + parse one day with nothing but the final
    filtered CSVs ever touching disk."""
    if already_have(archive_dir, date_str):
        logging.info(f"{date_str}: already archived, skipping")
        return True

    date_compact = date_str.replace('-', '')
    try:
        file_info = get_hist_data(date_compact)
    except Exception:
        logging.exception(f"{date_str}: no DEEP file available (holiday/weekend/not yet published?)")
        return False
    url = file_info['link']
    expected_gb = int(file_info.get('size', 0)) / 1e9
    logging.info(f"{date_str}: streaming ~{expected_gb:.1f} GB compressed -- "
                 f"nothing but the filtered CSVs will touch disk")

    day_dir = os.path.join(archive_dir, date_str)
    os.makedirs(day_dir, exist_ok=True)
    prefix = os.path.join(day_dir, date_str)

    cmd = (
        'set -o pipefail; '
        f'curl -fsSL --retry 3 --retry-delay 5 --connect-timeout 15 --max-time 1800 "{url}" '
        f'| editcap -F pcap - - '
        f'| "{IEX_PARSER}" /dev/stdin "{prefix}" "{symbols_file}"'
    )

    for attempt in range(1, retries + 1):
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        produced = [f for f in os.listdir(day_dir) if f.endswith('_prl.csv') or f.endswith('_trd.csv')]
        sizes_ok = all(os.path.getsize(os.path.join(day_dir, f)) >= MIN_OUTPUT_BYTES for f in produced)
        if result.returncode == 0 and produced and sizes_ok:
            logging.info(f"{date_str}: archived {len(produced)} file(s), streamed end to end, nothing kept on disk")
            return True
        logging.warning(f"{date_str}: attempt {attempt}/{retries} failed "
                         f"(exit {result.returncode}, produced={produced}), retrying")
        for f in produced:
            os.remove(os.path.join(day_dir, f))
    return False


def pooled_day_files(archive_dir):
    day_files = []
    for prl_path in sorted(glob.glob(os.path.join(archive_dir, '*', '*_prl.csv'))):
        day_dir = os.path.dirname(prl_path)
        trd_candidates = glob.glob(os.path.join(day_dir, '*_trd.csv'))
        if trd_candidates:
            day_files.append((prl_path, trd_candidates[0]))
    return day_files


def report_progress(archive_dir, symbol, out_path):
    day_files = pooled_day_files(archive_dir)
    if not day_files:
        return None
    res = iex_calibrate.calibrate_multi(day_files, symbol)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', required=True, help='path to symbols.txt (one ticker per line)')
    p.add_argument('--archive', default='./archive')
    p.add_argument('--dates', required=True, help='comma-separated YYYY-MM-DD list')
    p.add_argument('--calibrate-symbol', required=True, help='which symbol to fit the queue-reactive model for')
    p.add_argument('--out', default='calibrated_params_multiday.json')
    p.add_argument('--log', default='stream_collect.log')
    args = p.parse_args()

    os.makedirs(args.archive, exist_ok=True)
    setup_logging(args.log)

    dates = [d.strip() for d in args.dates.split(',') if d.strip()]
    logging.info(f"=== streaming collection start: {len(dates)} day(s), symbol {args.calibrate_symbol} ===")

    progression = []
    for date_str in dates:
        ok = stream_one_day(date_str, args.symbols, args.archive)
        if not ok:
            logging.error(f"{date_str}: giving up after retries, continuing to next date")
            continue
        res = report_progress(args.archive, args.calibrate_symbol, args.out)
        if res:
            progression.append(res)
            logging.info(
                f"[{res['n_days']} day(s) pooled] Q0={res.get('Q0')} "
                f"a_LO={res.get('a_LO'):.4f} b_LO={res.get('b_LO'):.5f} "
                f"a_C={res.get('a_C'):.4f} a_MO={res.get('a_MO'):.5f} "
                f"(n_LO={res['n_LO_events']} n_C={res['n_C_events']} n_MO={res['n_MO_events']})"
            )

    logging.info(f"=== streaming collection complete: {len(progression)}/{len(dates)} day(s) contributed ===")
    with open('calibration_progression.json', 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != '_diagnostics'} for r in progression], f, indent=2)
    logging.info("Saved calibration_progression.json (parameter estimates after each additional day)")
