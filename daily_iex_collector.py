"""
Daily IEX DEEP collector.

IMPORTANT DESIGN NOTE: this is deliberately a once-a-day job, not a
continuously-running scraper. IEX's free HIST archive publishes each
trading day's full DEEP feed (every symbol) as a single pcap file on a
T+1 basis -- there is nothing to "poll" intraday, and IEX's *live* feed
requires a formal Data Subscriber Agreement and direct/colocated
connectivity that a generic cloud VM won't have anyway. So: one cron run
per day, each run grabs whatever new trading day has become available,
filters it down to the tickers you care about, and immediately deletes
the (large, multi-symbol) raw file so disk usage stays flat over weeks.

Requires: pip install iex-cppparser --break-system-packages

Usage (manual test run):
    python daily_iex_collector.py --symbols symbols.txt --archive ./archive

Usage (intended, via cron -- see DEPLOY.md):
    0 9 * * * cd /path/to/project && python3 daily_iex_collector.py \
        --symbols symbols.txt --archive ./archive >> collector.log 2>&1
"""
import argparse
import datetime as dt
import json
import logging
import os
import shutil
import sys
import time

import iex_cppparser as iex


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def already_have(archive_dir, date_str):
    """Idempotency check: skip re-downloading a day we've already archived."""
    day_dir = os.path.join(archive_dir, date_str)
    if not os.path.isdir(day_dir):
        return False
    files = os.listdir(day_dir)
    return any(f.endswith('_prl.csv') for f in files) and any(f.endswith('_trd.csv') for f in files)


def disk_free_gb(path):
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def process_one_day(date_str, symbols_file, work_dir, archive_dir, min_free_gb=5.0):
    """Downloads + parses one trading day, archives the small filtered CSVs,
    then deletes the large raw pcap regardless of success/failure."""
    if already_have(archive_dir, date_str):
        logging.info(f"{date_str}: already archived, skipping")
        return True

    free_gb = disk_free_gb(work_dir)
    if free_gb < min_free_gb:
        logging.error(f"{date_str}: only {free_gb:.1f} GB free, refusing to download "
                       f"(need at least {min_free_gb} GB headroom)")
        return False

    download_dir = os.path.join(work_dir, 'raw')
    parsed_dir = os.path.join(work_dir, 'parsed')
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)

    ok = False
    try:
        logging.info(f"{date_str}: downloading + parsing (this can take a while for a "
                      f"full-market DEEP file)")
        iex.parse_date(
            date_str=date_str,
            download_dir=download_dir,
            parsed_folder=parsed_dir,
            symbol=symbols_file,
            download=True,
            split=False,
        )
        produced = [f for f in os.listdir(parsed_dir)
                    if f.endswith('_prl.csv') or f.endswith('_trd.csv')]
        if not produced:
            logging.warning(f"{date_str}: parser produced no output files "
                             f"(no trading day published yet, or filter matched nothing)")
        else:
            day_archive = os.path.join(archive_dir, date_str)
            os.makedirs(day_archive, exist_ok=True)
            for f in produced:
                shutil.move(os.path.join(parsed_dir, f), os.path.join(day_archive, f))
            logging.info(f"{date_str}: archived {len(produced)} file(s) to {day_archive}")
            ok = True
    except Exception:
        logging.exception(f"{date_str}: failed")
        ok = False
    finally:
        # ALWAYS reclaim the large raw download regardless of outcome --
        # this is what keeps disk usage flat across a multi-week run.
        if os.path.isdir(download_dir):
            freed = sum(os.path.getsize(os.path.join(download_dir, f))
                        for f in os.listdir(download_dir)
                        if os.path.isfile(os.path.join(download_dir, f)))
            shutil.rmtree(download_dir, ignore_errors=True)
            os.makedirs(download_dir, exist_ok=True)
            logging.info(f"{date_str}: freed {freed / (1024**2):.0f} MB of raw pcap data")
        if os.path.isdir(parsed_dir):
            shutil.rmtree(parsed_dir, ignore_errors=True)

    return ok


def backfill_recent_days(symbols_file, work_dir, archive_dir, lookback_days=10, retries=3):
    """Each run, walk backward over the last `lookback_days` calendar days and
    attempt any that aren't archived yet. This makes the job self-healing:
    if a VM reboot or a network blip causes a missed day, the next run picks
    it up automatically, and it naturally backfills the first time you run it."""
    today = dt.date.today()
    results = {}
    for i in range(1, lookback_days + 1):
        d = today - dt.timedelta(days=i)
        if d.weekday() >= 5:  # skip weekends, IEX has no equity trading then
            continue
        date_str = d.isoformat()
        ok = False
        for attempt in range(1, retries + 1):
            ok = process_one_day(date_str, symbols_file, work_dir, archive_dir)
            if ok:
                break
            logging.warning(f"{date_str}: attempt {attempt}/{retries} failed, retrying")
            time.sleep(5 * attempt)
        results[date_str] = ok
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', required=True, help='path to symbols.txt (one ticker per line)')
    p.add_argument('--archive', default='./archive', help='where finished per-day CSVs accumulate')
    p.add_argument('--work-dir', default='./work', help='scratch space for raw downloads (cleaned each run)')
    p.add_argument('--lookback-days', type=int, default=10)
    p.add_argument('--log', default='collector.log')
    args = p.parse_args()

    os.makedirs(args.archive, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    setup_logging(args.log)

    logging.info("=== daily collector run start ===")
    results = backfill_recent_days(args.symbols, args.work_dir, args.archive,
                                    lookback_days=args.lookback_days)
    n_ok = sum(results.values())
    logging.info(f"=== run complete: {n_ok}/{len(results)} day(s) newly ok or already archived ===")
    with open(os.path.join(args.archive, '_status.json'), 'w') as f:
        json.dump({'last_run': dt.datetime.now().isoformat(), 'results': results}, f, indent=2)
