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


def extract_logical_links_advanced(text, selected_lang, pair_count=50):
    logical_links = []
    result = punctuation_model.restore_punctuation(text)
    texts = preprocess_text(result, selected_lang)
    cooccurrence = calculate_cooccurrence(texts, window_size=5)

    for pair, count in sorted(cooccurrence.items(), key=lambda x: -x[1])[:min(pair_count, len(cooccurrence))]:
        relative_weight = count / len(texts)
        weight = relative_weight * 100
        logical_links.append({
            'source': list(pair)[0],
            'target': list(pair)[1],
            'source_category': get_word_category_wordnet(list(pair)[0], selected_lang),
            'target_category': get_word_category_wordnet(list(pair)[1], selected_lang),
            'weight': weight
        })

    return logical_links


def calculate_cooccurrence(tokens, window_size):
    cooccurrence = defaultdict(int)
    for i in range(len(tokens)):
        window_start = max(0, i - window_size)
        window = tokens[window_start: i + window_size + 1]
        for pair in itertools.combinations(window, 2):
            cooccurrence[frozenset(pair)] += 1
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
