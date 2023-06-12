import nltk

from deepmultilingualpunctuation import PunctuationModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from collections import defaultdict
import itertools

from lib.word_categorization import get_word_category_wordnet

stopwords_map = {
    'en': stopwords.words('english'),
    'de': stopwords.words('german')
}

nlp = None
punctuation_model = PunctuationModel()
nltk.download('stopwords')


def extract_logical_links_advanced(text, selected_lang, live_mode=False):
    logical_links = []
    result = punctuation_model.restore_punctuation(text)
    texts = preprocess_text(result, selected_lang)
    cooccurrence = calculate_cooccurrence(texts, window_size=10)

    if len(cooccurrence) == 0:
        return logical_links
    max_count = max(cooccurrence.values())
    for pair, count in sorted(cooccurrence.items(), key=lambda x: -x[1])[:len(cooccurrence)]:
        pair_list = list(pair)
        first_word = pair_list[0]
        second_word = pair_list[1]

        relative_weight = count / max_count
        weight = 4.5 * relative_weight + 0.5
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
            'weight': weight
        })

    print(f"Extracted {len(logical_links)} logical links")
    return logical_links


def calculate_cooccurrence(tokens, window_size):
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
