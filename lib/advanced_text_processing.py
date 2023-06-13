import multiprocessing

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from collections import defaultdict
import itertools

from lib.word_categorization_wordnet import get_word_category_wordnet

stopwords_map = {
    'en': stopwords.words('english'),
    'de': stopwords.words('german')
}

nlp = None
nltk.download('stopwords')


def extract_logical_links_advanced(text, selected_lang, live_mode=False):
    logical_links = []
    cooccurrence = get_cooccurrence(text, selected_lang)

    if len(cooccurrence) == 0:
        return logical_links
    for pair, count in sorted(cooccurrence.items(), key=lambda x: -x[1])[:len(cooccurrence)]:
        pair_list = list(pair)
        first_word = pair_list[0]
        second_word = pair_list[1]

        if not live_mode:
            source_category = get_word_category_wordnet(first_word, selected_lang)
            target_category = get_word_category_wordnet(second_word, selected_lang)
        else:
            source_category = None
            target_category = None
        logical_links.append({
            'source': first_word,
            'target': second_word,
            'source_category': source_category,
            'target_category': target_category,
            'weight': count
        })

    print(f"Extracted {len(logical_links)} logical links")
    return logical_links


def get_cooccurrence(text, selected_lang):
    texts = preprocess_text(text, selected_lang)
    available_cores = multiprocessing.cpu_count() - 4
    if available_cores < 1:
        available_cores = 1
    desired_cores = max(int(len(texts) / 100000), 1)
    used_cores = min(available_cores, desired_cores)
    split_texts = [texts[i::used_cores] for i in range(used_cores)]
    print(f"Using {used_cores} cores to calculate cooccurrence within {len(texts)} ({int(len(texts) / used_cores)} each core) tokens...")
    cooccurrence = []
    if __name__ == '__main__':
        multiprocessing.freeze_support()
        with multiprocessing.Pool(used_cores) as pool:
            cooccurrence = pool.map(calculate_cooccurrence, [split_texts], chunksize=1)
    return cooccurrence


def calculate_cooccurrence(tokens, window_size=10):
    print(f"Received {len(tokens)} tokens...")
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