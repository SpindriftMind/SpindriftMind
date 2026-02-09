"""Telegram bot for async communication with Ryan.

Send session summaries, receive directions between sessions.
Adapted from DriftCornwall's implementation.

Usage:
    python telegram_bot.py test              # Send test message
    python telegram_bot.py poll              # Check for messages
    python telegram_bot.py send <message>    # Send a message
    python telegram_bot.py setup <token> <chat_id>  # Save credentials
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CREDS_FILE = Path(os.path.expanduser('~/.config/telegram/spindrift-credentials.json'))
STATE_FILE = Path(__file__).parent / 'memory' / '.telegram_state.json'
BASE_URL = 'https://api.telegram.org/bot{token}'


def load_creds():
    if not CREDS_FILE.exists():
        return None
    with open(CREDS_FILE, 'r') as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'last_update_id': 0}


def send_message(text, parse_mode='Markdown'):
    """Send a message to Ryan via Telegram."""
    import urllib.request
    import urllib.error
    creds = load_creds()
    if not creds:
        print('[telegram] No credentials found')
        return False

    url = BASE_URL.format(token=creds['bot_token']) + '/sendMessage'
    if len(text) > 4000:
        text = text[:3990] + '\n...(truncated)'

    # Try with parse_mode first
    payload = json.dumps({
        'chat_id': creds['chat_id'],
        'text': text,
        'parse_mode': parse_mode
    }).encode('utf-8')

    try:
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read())
        if result.get('ok'):
            return True
    except urllib.error.HTTPError:
        pass  # Fall through to plain text retry
    except Exception as e:
        print(f'[telegram] Send with {parse_mode} failed: {e}')

    # Retry without parse_mode (plain text)
    try:
        payload2 = json.dumps({
            'chat_id': creds['chat_id'],
            'text': text
        }).encode('utf-8')
        req2 = urllib.request.Request(url, data=payload2, headers={'Content-Type': 'application/json'})
        resp2 = urllib.request.urlopen(req2, timeout=10)
        return json.loads(resp2.read()).get('ok', False)
    except Exception as e:
        print(f'[telegram] Send failed: {e}')
        return False


def get_unread_messages():
    """Poll for messages received since last check."""
    import urllib.request
    creds = load_creds()
    if not creds:
        return []

    state = load_state()
    url = BASE_URL.format(token=creds['bot_token']) + '/getUpdates'

    try:
        params = f'?timeout=1'
        if state['last_update_id'] > 0:
            params += f'&offset={state["last_update_id"] + 1}'

        req = urllib.request.Request(url + params)
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())

        if not data.get('ok'):
            return []

        updates = data.get('result', [])
        messages = []
        max_id = state['last_update_id']

        for update in updates:
            uid = update.get('update_id', 0)
            if uid > max_id:
                max_id = uid
            msg = update.get('message', {})
            if str(msg.get('chat', {}).get('id')) == str(creds['chat_id']):
                text = msg.get('text', '')
                ts = msg.get('date', 0)
                if text:
                    messages.append({
                        'text': text,
                        'timestamp': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        'update_id': uid
                    })

        if max_id > state['last_update_id']:
            state['last_update_id'] = max_id
            save_state(state)

        return messages

    except Exception as e:
        print(f'[telegram] Poll failed: {e}')
        return []


if __name__ == '__main__':
    args = sys.argv[1:]

    if not args or args[0] == 'test':
        print('[telegram] Sending test message...')
        ok = send_message('SpindriftMend online. Telegram integration working.')
        print(f'[telegram] Send: {"OK" if ok else "FAILED"}')

    elif args[0] == 'poll':
        print('[telegram] Checking for messages...')
        msgs = get_unread_messages()
        if msgs:
            for m in msgs:
                print(f'  [{m["timestamp"]}] {m["text"]}')
        else:
            print('  No new messages.')

    elif args[0] == 'send':
        text = ' '.join(args[1:]) if len(args) > 1 else 'No message specified'
        ok = send_message(text)
        print(f'[telegram] Send: {"OK" if ok else "FAILED"}')

    elif args[0] == 'setup':
        print('[telegram] Creating credentials file...')
        CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if len(args) > 2:
            creds = {'bot_token': args[1], 'chat_id': args[2]}
        else:
            creds = {
                'bot_token': input('Bot token: ').strip(),
                'chat_id': input('Chat ID: ').strip()
            }
        with open(CREDS_FILE, 'w') as f:
            json.dump(creds, f, indent=2)
        print(f'[telegram] Saved to {CREDS_FILE}')

    else:
        print('Usage: telegram_bot.py [test|poll|send <msg>|setup <token> <chat_id>]')
