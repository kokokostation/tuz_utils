import json
import os
import re
from tqdm import tqdm
import numpy as np
import datetime as dt
import pytz
from collections import Counter

from dataset import reddit


LINK_RE = re.compile('@(\d+)')
SPACE_RE = re.compile(' +')
RUSSIAN_RE = re.compile('[А-я]')


def parse_message(message):
    raw_text = message['text']
    parent_ids = []
    for match in LINK_RE.finditer(raw_text):
        parent_ids.append(int(match.group(1)))

    text = SPACE_RE.sub(' ', LINK_RE.sub('', raw_text)).strip()
    id_ = int(message['id'])
    normalized_time = dt.datetime.strptime(
        message['time'].replace(':', ''), '%Y-%m-%dT%H%M%S%z'
    ).astimezone(pytz.UTC).replace(tzinfo=None).isoformat()
    user_id = message['user_id']

    return {
        'id': id_,
        'text': text.replace('\n', ' '),
        'user_id': user_id,
        'time': normalized_time,
        'parent_ids': parent_ids,
    }


def get_minute(message):
    return message['time'][:16]


def get_length(message):
    return len(message['text'])


def prepare_dataset(logs_path, output_path, context_len=3, batch_size=100000):
    messages = []

    with open(logs_path) as infile:
        for line in tqdm(infile):
            messages.extend(map(parse_message, json.loads(line)['data']))

    minute_counter = Counter(map(get_minute, messages))

    messages = [message for message in messages if minute_counter[get_minute(message)] <= 20]
    messages = [message for message in messages if len(message['text']) < 200]
    messages = [message for message in messages if '://' not in message['text']]
    messages = [message for message in messages if RUSSIAN_RE.search(message['text'])]

    message_ids = {message['id'] for message in messages}
    for message in messages:
        message['parent_ids'] = list(set(message['parent_ids']) & message_ids)

    samples = []
    for index in range(context_len + 1, len(messages)):
        samples.append([
            {'body': m['text'], 'processed_body': reddit.convert(m['text']), 'author': m['user_id']}
            for m in messages[index - context_len - 1:index]
        ])

    np.random.shuffle(samples)

    for batch_num, index in enumerate(range(0, len(samples), batch_size)):
        with open(os.path.join(output_path, f'pack_{batch_num}'), 'w') as outfile:
            json.dump(samples[index:index + batch_size], outfile)
