"""Run both demo servers; stop them after the last dashboard disconnects."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def port_in_use(port):
    with socket.socket() as connection:
        return connection.connect_ex(('127.0.0.1', port)) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idle-seconds', type=float, default=60)
    parser.add_argument('--startup-seconds', type=float, default=120)
    args = parser.parse_args()
    if args.idle_seconds <= 0 or args.startup_seconds <= 0:
        parser.error('Timeouts must be positive')
    if any(port_in_use(port) for port in (8000, 5173)):
        parser.error('Port 8000 or 5173 is already occupied. Stop the old demo before starting this launcher.')
    node = shutil.which('node')
    vite = ROOT / 'frontend/node_modules/vite/bin/vite.js'
    if not node or not vite.exists():
        parser.error('Install Node.js and run npm install --prefix frontend first.')
    python = ROOT / 'backend/.venv/bin/python'
    python = str(python) if python.exists() else sys.executable
    processes = []
    # Signals request cleanup; cleanup also runs if either server exits unexpectedly.
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    with tempfile.TemporaryDirectory(prefix='safedrive-session-') as directory:
        heartbeat = Path(directory) / 'heartbeat'
        env = dict(os.environ, DEMO_HEARTBEAT_FILE=str(heartbeat))
        try:
            commands = [([python, '-m', 'uvicorn', 'main:app', '--app-dir', 'backend',
                          '--host', '127.0.0.1', '--port', '8000'], ROOT),
                        ([node, str(vite), '--host', '127.0.0.1'], ROOT / 'frontend')]
            for command, cwd in commands:
                processes.append(subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True))
            started = time.time()
            print(f'Open http://127.0.0.1:5173 within {args.startup_seconds:g} seconds. '
                  f'Both servers stop {args.idle_seconds:g} seconds after the last dashboard closes. '
                  'Control+C also stops both.', flush=True)
            while all(process.poll() is None for process in processes):
                if heartbeat.exists():
                    expired = time.time() - heartbeat.stat().st_mtime >= args.idle_seconds
                else:
                    expired = time.time() - started >= args.startup_seconds
                if expired:
                    print('No dashboard connection. Stopping demo servers.', flush=True)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            print('\nStopping demo servers.', flush=True)
        finally:
            # Stop only the process groups this launcher created, including Vite helpers.
            for process in processes:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == '__main__':
    main()
