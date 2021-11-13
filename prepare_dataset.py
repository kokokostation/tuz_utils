import datetime as dt
import json
import os
import re
from collections import Counter

import numpy as np
import pytz
from dataset import reddit
from tqdm import tqdm

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


def get_filtered_messages(logs_path):
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

    return messages


def get_context_reply_seq(messages, context_len=3):
    result = []
    for index in range(context_len + 1, len(messages)):
        result.append(messages[index - context_len - 1:index])

    return result


def dfs(messages, id_to_index, result, sample, context_len):
    if len(sample) == context_len + 1:
        result.append([messages[index] for index in reversed(sample)])

        return

    parent_ids = messages[sample[-1]]['parent_ids']
    if parent_ids:
        parent_inds = [id_to_index[id_] for id_ in parent_ids]
    else:
        parent_inds = [sample[-1] - 1]

    for parent_index in parent_inds:
        if parent_index < 0:
            continue

        sample.append(parent_index)
        dfs(messages, id_to_index, result, sample, context_len)
        sample.pop()


def get_context_reply_with_links(messages, context_len=3):
    id_to_index = {m['id']: index for index, m in enumerate(messages)}
    result = []

    for index, _ in tqdm(enumerate(messages)):
        dfs(messages, id_to_index, result, [index], context_len)

    return result


def prepare_dataset(samples):
    prepared_samples = []
    for sample in samples:
        prepared_sample = [
            {'body': m['text'], 'processed_body': reddit.convert(m['text']), 'author': m['user_id']}
            for m in sample
        ]
        prepared_samples.append(prepared_sample)

    np.random.shuffle(prepared_samples)

    return prepared_samples


def write_dataset(samples, output_path, batch_size=100000):
    for batch_num, index in tqdm(enumerate(range(0, len(samples), batch_size))):
        with open(os.path.join(output_path, f'pack_{batch_num}'), 'w') as outfile:
            json.dump(samples[index:index + batch_size], outfile)
