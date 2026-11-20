"""
Runs a list of shell jobs on several machines, each with a number of slots, one job per slot.

Each line of --jobs is one command, run from the machine's project directory. The result
directory is copied back after each job. A job already in --log as DONE, FAIL or dropped is
skipped, so the dispatcher can be restarted. Touching <log>.stop stops new jobs; running ones
finish. A file <log>.slots.<host> with a number limits that host's active slots, and is read
again while the dispatcher runs. A host named "local" runs on this machine, from this directory.

A slot starts a job only when its machine answers. A connection lost during a job (ssh exit 255)
puts the job back up to --infra-retries times; after that it is logged as INFRA dropped. Once the
machine answers again, what is left of the job there is killed, so that it does not run twice.
Any other non-zero exit is logged as FAIL, and the job is not run again.

Usage: dispatch.py --jobs JOBS --log LOG --results DIR --project PATH --env "A=1 B=2"
                   [--infra-retries N] --host name:slots:key:user@ip ... [--host local:slots]
"""

import argparse
import os
import re
import shlex
import subprocess
import threading
import time
from collections import Counter, deque

SSH_OPTS = ["-F", "/dev/null", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]


def ssh_cmd(key):
    return ["ssh", "-i", key] + SSH_OPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--results", required=True, help="Result directory, relative to the project.")
    ap.add_argument("--project", required=True, help="Project directory on the remote machines.")
    ap.add_argument("--env", default="")
    ap.add_argument("--infra-retries", type=int, default=1)
    ap.add_argument("--host", action="append", required=True)
    args = ap.parse_args()

    with open(args.jobs) as f:
        jobs = [(i, line.strip()) for i, line in enumerate(f, 1) if line.strip()]
    done = set()
    if os.path.exists(args.log):
        with open(args.log) as f:
            for line in f:
                parts = line.split()
                if len(parts) > 3 and (parts[2] in ("DONE", "FAIL") or (parts[2] == "INFRA" and parts[-1] == "dropped")):
                    done.add(int(parts[3]))
    queue = deque(j for j in jobs if j[0] not in done)
    retried, lock = Counter(), threading.Lock()
    stop = args.log + ".stop"

    def log(event, *fields):
        line = " ".join([time.strftime("%F %T"), event] + [str(x) for x in fields])
        with lock, open(args.log, "a") as f:
            f.write(line + "\n")

    def allowed(name, index):
        try:
            with open(f"{args.log}.slots.{name}") as f:
                return index < int(f.read().strip())
        except (OSError, ValueError):
            return True

    log("DISPATCH_START", len(queue), "of", len(jobs))

    def local_slot(name, index):
        while not os.path.exists(stop):
            if not allowed(name, index):
                time.sleep(30)
                continue
            with lock:
                if not queue:
                    return
                n, cmd = queue.popleft()
            log("START", n, name)
            t0 = time.time()
            os.makedirs(f"{args.results}/logs", exist_ok=True)
            shell = f"{{ {args.env} {cmd} ; }} > {args.results}/logs/job_{n}.log 2>&1"
            rc = subprocess.call(["bash", "-c", shell], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log("DONE" if rc == 0 else "FAIL", n, name, f"rc={rc}", f"{time.time() - t0:.0f}s")

    def slot(name, index, key, target):
        ssh = ssh_cmd(key) + [target]
        while not os.path.exists(stop):
            if not allowed(name, index):
                time.sleep(30)
                continue
            if subprocess.call(ssh + ["true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
                time.sleep(60)
                continue
            with lock:
                if not queue:
                    return
                n, cmd = queue.popleft()
            log("START", n, name)
            t0 = time.time()
            remote = f"cd {args.project} && mkdir -p {args.results}/logs && " \
                     f"{{ {args.env} {cmd} ; }} > {args.results}/logs/job_{n}.log 2>&1"
            rc = subprocess.call(ssh + [remote], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc == 255:
                pattern = shlex.quote("[" + cmd[0] + "]" + re.escape(cmd[1:]))
                while subprocess.call(ssh + [f"pkill -f -- {pattern}; true"], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL) != 0:
                    time.sleep(60)
                with lock:
                    again = retried[n] < args.infra_retries
                    if again:
                        retried[n] += 1
                        queue.append((n, cmd))
                log("INFRA", n, name, "requeued" if again else "dropped")
                continue
            sync = ["rsync", "-az", "-e", shlex.join(ssh_cmd(key)), f"{target}:{args.project}/{args.results}/",
                    f"{args.results}/"]
            if subprocess.call(sync, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
                log("SYNC_FAILED", n, name)
            log("DONE" if rc == 0 else "FAIL", n, name, f"rc={rc}", f"{time.time() - t0:.0f}s")

    threads = []
    for spec in args.host:
        fields = spec.split(":")
        for i in range(int(fields[1])):
            if fields[0] == "local":
                threads.append(threading.Thread(target=local_slot, args=(fields[0], i)))
            else:
                threads.append(threading.Thread(target=slot, args=(fields[0], i, fields[2], fields[3])))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("DISPATCH_END")


if __name__ == "__main__":
    main()
