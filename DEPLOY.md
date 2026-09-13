# Deploying the IEX collector to a free-tier VM

## Why Oracle Cloud "Always Free," specifically

For a job that needs to sit unattended for weeks, the free tiers matter more
than usual:

- **AWS/GCP free tiers** are time-limited trials (12 months) or very small
  instances (~30GB disk), and it's easy to accidentally spin up something
  that isn't covered and get billed.
- **Oracle Cloud's "Always Free"** tier is free indefinitely (not a trial),
  and its Ampere A1 shape gives up to 4 CPUs / 24GB RAM / 200GB block
  storage at no cost — the 200GB matters here since each day's raw IEX file
  gets deleted right after parsing, but you want headroom for the transient
  download plus weeks of small archived CSVs.

You can use AWS/GCP instead if you already have an account you trust — the
steps below (Python setup, cron) are the same regardless of cloud, only the
"create a VM" step differs.

**Note:** Oracle's signup usually asks for a card for identity verification
even for Always Free resources. Make sure whatever instance you launch is
explicitly marked "Always Free eligible" before you create it.

## 1. Create the VM

1. Sign up at oracle.com/cloud/free (this part is you, not me).
2. Console → Compute → Instances → Create Instance.
3. Choose an **Always Free eligible** shape (Ampere A1, 1-4 OCPUs is enough
   for this — it's I/O and disk bound, not CPU bound).
4. Ubuntu as the image (simplest for the steps below).
5. Add your SSH key during creation (or let Oracle generate one and download it).
6. Launch, note the public IP.

## 2. Connect and set up Python

```bash
ssh -i /path/to/your_key.pem ubuntu@<VM_PUBLIC_IP>

sudo apt update
sudo apt install -y python3-pip
pip3 install iex-cppparser pandas numpy --break-system-packages
```

## 3. Upload the project files

From your own machine (not the VM):

```bash
scp -i /path/to/your_key.pem daily_iex_collector.py symbols.txt ubuntu@<VM_PUBLIC_IP>:~/lob_collector/
```

(Create the `~/lob_collector/` directory on the VM first with `mkdir -p ~/lob_collector`.)

Edit `symbols.txt` there if you want different tickers than INTC/CSCO/MU —
one ticker per line.

## 4. Test it manually before automating

```bash
cd ~/lob_collector
python3 daily_iex_collector.py --symbols symbols.txt --archive ./archive --lookback-days 3
```

Watch `collector.log`. First run will attempt to backfill the last few
trading days. This is the point to sanity-check the third-party parser's
output before trusting it — open one of the `_trd.csv` files and eyeball
whether the prices are plausible for that ticker on that date.

## 5. Automate with cron

```bash
crontab -e
```

Add (runs once daily at 9am server time — well after T+1 data would be
published, and outside market hours so it doesn't compete with anything):

```
0 9 * * * cd /home/ubuntu/lob_collector && /usr/bin/python3 daily_iex_collector.py --symbols symbols.txt --archive ./archive >> cron.log 2>&1
```

The script is idempotent and self-healing (`--lookback-days 10` by default,
configurable) — if the VM reboots or a day fails, the next run picks up
anything missing over the trailing 10 calendar days automatically. You don't
need to babysit it.

## 6. Retrieving your data

Periodically (or after a couple of weeks), pull the archive back down:

```bash
scp -i /path/to/your_key.pem -r ubuntu@<VM_PUBLIC_IP>:~/lob_collector/archive ./iex_archive
```

Each `archive/<date>/` folder will have `<SYMBOL>_prl.csv` and
`<SYMBOL>_trd.csv` — feed those straight into `iex_calibrate.py`.

## 7. Shutting it down

When you have enough data, terminate the instance from the Oracle console
(Compute → Instances → your instance → Terminate) so it's not just sitting
there — Always Free doesn't charge you, but there's no reason to leave it
running once you have what you need.
