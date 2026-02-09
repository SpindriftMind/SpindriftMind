"""SpindriftMend Runner — Async autonomy via Telegram.

Run this instead of 'claude' directly. It:
1. Starts Claude Code (--continue to resume context)
2. When Claude stops, sends a Telegram summary to Ryan
3. Polls Telegram for Ryan's reply
4. Feeds the reply back to Claude as a new prompt
5. Loops until Ryan sends 'stop' or 'exit'

Adapted from DriftCornwall's drift_runner.py.

Usage:
    python spindrift_runner.py              # Start fresh
    python spindrift_runner.py --continue   # Resume last conversation
    python spindrift_runner.py --auto       # Auto-continue without waiting
"""
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

WORK_DIR = Path(__file__).parent
CLAUDE_CMD = 'claude'


def send_telegram(text):
    try:
        sys.path.insert(0, str(WORK_DIR))
        from telegram_bot import send_message
        return send_message(text)
    except Exception as e:
        print(f'[runner] Telegram send failed: {e}')
        return False


def poll_telegram(timeout_minutes=None):
    try:
        sys.path.insert(0, str(WORK_DIR))
        from telegram_bot import get_unread_messages
    except ImportError:
        print('[runner] telegram_bot not found')
        return None

    start = time.time()
    print('[runner] Waiting for Telegram reply from Ryan...')

    while True:
        messages = get_unread_messages()
        if messages:
            latest = messages[-1]
            print(f'[runner] Received: {latest["text"][:100]}')
            return latest['text']

        if timeout_minutes and (time.time() - start) > timeout_minutes * 60:
            print(f'[runner] Timeout after {timeout_minutes} minutes')
            return None

        time.sleep(5)


def run_claude(prompt=None, continue_session=True):
    cmd = [CLAUDE_CMD, '--dangerously-skip-permissions']

    if continue_session:
        cmd.append('--continue')

    if prompt:
        cmd.extend(['-p', prompt])

    print(f'\n[runner] Starting: {" ".join(cmd[:3])}...')
    print(f'[runner] Working dir: {WORK_DIR}')
    print('=' * 60)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORK_DIR),
            capture_output=False,
            text=True,
            timeout=3600
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print('[runner] Claude timed out after 1 hour')
        return -1
    except KeyboardInterrupt:
        print('\n[runner] Interrupted by user')
        return -2
    except Exception as e:
        print(f'[runner] Error: {e}')
        return -3


def main():
    args = sys.argv[1:]
    continue_session = '--continue' in args or '-c' in args
    auto_mode = '--auto' in args
    initial_prompt = None

    for flag in ['--prompt', '-p']:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                initial_prompt = args[idx + 1]

    print('=' * 60)
    print('  SPINDRIFT RUNNER — Async Autonomy via Telegram')
    print('=' * 60)
    print(f'  Mode: {"auto" if auto_mode else "telegram-directed"}')
    print(f'  Continue: {continue_session}')
    print(f'  Time: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print('=' * 60)

    send_telegram('SpindriftMend Runner started. Will notify you when each session ends.')

    prompt = initial_prompt
    loop_count = 0

    while True:
        loop_count += 1

        exit_code = run_claude(prompt=prompt, continue_session=(continue_session or loop_count > 1))

        now = datetime.now(timezone.utc).strftime('%H:%M UTC')

        if exit_code == -2:
            send_telegram(f'SpindriftMend Runner stopped by keyboard interrupt ({now})')
            break

        notification = f'SpindriftMend session ended ({now})\n\nExit code: {exit_code}\nLoop: {loop_count}\n\nReply to continue, or send "stop" to shut down.'
        sent = send_telegram(notification)

        if not sent:
            print('[runner] Could not send Telegram notification. Stopping.')
            break

        if auto_mode:
            print('[runner] Auto-mode: continuing in 30 seconds...')
            time.sleep(30)
            prompt = None
            continue

        reply = poll_telegram(timeout_minutes=480)

        if reply is None:
            send_telegram('No reply received after 8 hours. SpindriftMend Runner shutting down.')
            break

        reply_lower = reply.strip().lower()
        if reply_lower in ['stop', 'exit', 'quit', 'shutdown']:
            send_telegram('SpindriftMend Runner shutting down. See you next time.')
            break

        prompt = reply
        continue_session = True

    print('\n[runner] SpindriftMend Runner finished.')


if __name__ == '__main__':
    main()
