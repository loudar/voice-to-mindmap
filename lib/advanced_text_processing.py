import math
import multiprocessing

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from collections import defaultdict
import itertools

from lib.word_categorization_wordnet import get_word_category_wordnet, save_word_category_cache
from queue import Queue
from threading import Thread

stopwords_map = {
    'en': stopwords.words('english'),
    'de': stopwords.words('german')
}

nlp = None
nltk.download('stopwords')


def link_worker(q, results, selected_lang, live_mode):
    while not q.empty():
        work = q.get()
        i = 0
        for pair, count in work:
            i += 1
            pair_list = list(pair)
            first_word = pair_list[0]
            second_word = pair_list[1]

            if not live_mode:
                source_category = get_word_category_wordnet(first_word, selected_lang)
                target_category = get_word_category_wordnet(second_word, selected_lang)
            else:
                source_category = None
                target_category = None
            results.append({
                'source': first_word,
                'target': second_word,
                'source_category': source_category,
                'target_category': target_category,
                'weight': count
            })
        q.task_done()


def extract_logical_links_advanced(text, selected_lang, live_mode=False):
    # could use keyPhrase extraction here: https://language.cognitive.azure.com/tryout/keyPhrases
    cooccurrence = get_cooccurrence(text, selected_lang)

    if len(cooccurrence) == 0:
        return []

    chunk_size = 1000
    chunks = [list(cooccurrence.items())[i:i + chunk_size] for i in range(0, len(cooccurrence), chunk_size)]
    print(f"Total link chunks: {len(chunks)}")

    q = Queue(maxsize=0)
    for chunk in chunks:
        q.put(chunk)
    results = []
    worker_threads = []
    for i in range(len(chunks)):
        worker_thread = Thread(target=link_worker, args=(q, results, selected_lang, live_mode))
        worker_thread.setDaemon(True)
        worker_thread.start()
        worker_threads.append(worker_thread)

    print("Waiting for link tasks to complete...")
    for worker_thread in worker_threads:
        worker_thread.join()

    q.join()
    save_word_category_cache(selected_lang)
    print(f"Extracted {len(results)} logical links")
    return results


def get_core_count(text):
    available_cores = multiprocessing.cpu_count() - 4
    if available_cores < 1:
        available_cores = 1
    desired_cores = max(int(len(text) / 10000), 1)
    return min(available_cores, desired_cores)


def cooccurrence_worker(q, results):
    while not q.empty():
        work = q.get()
        result = calculate_cooccurrence(work)
        for k, v in result.items():
            results[k] += v
        q.task_done()


def get_cooccurrence(text, selected_lang):
    texts = preprocess_text(text, selected_lang)
    parallelism = max(math.ceil(len(text) / 2000), 1)
    split_texts = [texts[i::parallelism] for i in range(parallelism)]
    print(f"Working with {parallelism} threads...")

    q = Queue(maxsize=0)
    for text_chunk in split_texts:
        q.put(text_chunk)

    results = defaultdict(int)
    worker_threads = []
    for i in range(parallelism):
        worker_thread = Thread(target=cooccurrence_worker, args=(q, results))
        worker_thread.setDaemon(True)
        worker_thread.start()
        worker_threads.append(worker_thread)

    print("Waiting for cooccurrence tasks to complete...")
    for worker_thread in worker_threads:
        worker_thread.join()

    q.join()
    print("All cooccurrence tasks completed.")
    return results


def calculate_cooccurrence(tokens, window_size=10):
    cooccurrence = defaultdict(int)
    for i in range(len(tokens)):
        increment = 1
        window_start = max(0, i - window_size)
        window = tokens[window_start: i + window_size + 1]
        for pair in itertools.combinations(window, 2):
            if pair[0] != pair[1]:
                if len(pair[0]) < 4 or len(pair[1]) < 4:
                    increment = 0.2
                cooccurrence[frozenset(pair)] += increment
    return cooccurrence


def preprocess_text(text, selected_lang):
    print(f"Preprocessing text...")
    # Convert to lowercase
    text = text.lower()

    # Split into sentences
    sentences = sent_tokenize(text)

    # Initialize list to hold preprocessed tokens
    tokens = []

    for sentence in sentences:
        # Remove non-alphabetic characters
        sentence = re.sub(r'[^\w]', ' ', sentence)
        # Remove words with 1 character
        sentence = re.sub(r'\b\w\b', '', sentence)

        # Tokenize
        sentence_tokens = word_tokenize(sentence)

        # Remove stopwords
        stop_words = stopwords_map[selected_lang]
        sentence_tokens = [token for token in sentence_tokens if token not in stop_words]

        # Merge tokens into the final list
        tokens.extend(sentence_tokens)

    return tokens


def get_preprocessed_sentences(text):
    # Convert to lowercase
    text = text.lower()

    # Split into sentences
    sentences = sent_tokenize(text)

    for i in range(len(sentences)):
        # Remove non-alphabetic characters
        sentences[i] = re.sub(r'[^\w]', ' ', sentences[i])
        # Remove words with 1 character
        sentences[i] = re.sub(r'\b\w\b', '', sentences[i])

    return sentences