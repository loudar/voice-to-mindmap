import json
import os
import sqlite3
from datetime import datetime

import Levenshtein as Levenshtein
import wn

CAT_CACHE_NOT_CACHED = '__not_cached__'
CAT_CACHE_ERROR = '__error__'
CAT_CACHE_UNKNOWN = 'unknown'

net = None
word_category_cache = {}


def get_word_category_wordnet(word, language='en', debug=False, save_cache=False):
    if not word:
        return CAT_CACHE_UNKNOWN

    word_lower = word.lower()
    if save_cache:
        save_word_category_cache(language)

    initialize_word_category_cache(language)
    cached_category = check_for_cached_word_categories(word_lower, language, debug)
    if cached_category is not None:
        return cached_category

    try:
        return get_word_category_wordnet_internal(word, language, debug)
    except sqlite3.ProgrammingError as e:
        if 'SQLite objects created in a thread can only be used in that same thread' in str(e):
            # Strange error I don't know how to fix
            return CAT_CACHE_UNKNOWN
        else:
            raise e


def get_word_category_wordnet_internal(word, language='en', debug=False):
    global net
    language_map = {
        'en': 'oewn:2022',
        'de': 'odenet:1.4'
    }
    if language not in language_map:
        raise ValueError(f"Language '{language}' is not supported.")
    else:
        wordnet_language = language_map[language]
        if debug:
            print(f"Language is '{language}'.")

    word_lower = word.lower()
    # Query Wordnet
    if net is None:
        net = wn.Wordnet(wordnet_language)
    try:
        synsets = net.synsets(word)
    except sqlite3.ProgrammingError as e:
        # Strange error I don't know how to fix
        raise e


    print(f"Querying wordnet({wordnet_language}) for '{word}'.")
    if len(synsets) == 0:
        # maybe add translation here to account for bilingual text
        if debug:
            print(f"No synsets found for '{word}'.")
        cache_word_categories(word_lower, CAT_CACHE_UNKNOWN)
        return CAT_CACHE_UNKNOWN
    else:
        synset = synsets[0]
        lemmas = synset.lemmas()
        closest_lemma = closest_match(word_lower, lemmas)
        if word_lower == closest_lemma.lower():
            hypernyms = synset.hypernyms()
            if len(hypernyms) == 0:
                if debug:
                    print(f"No hypernyms found for {word}.")
                cache_word_categories(word_lower, CAT_CACHE_UNKNOWN)
                return CAT_CACHE_UNKNOWN

            lemmas = hypernyms[0].lemmas()
            print(f"Exact match for {word} (hypernym: '{lemmas[0]}').")
            cache_word_categories(word_lower, lemmas[0])
            return lemmas[0]

        hypernyms = synset.hypernyms()
        if len(hypernyms) == 0:
            if debug:
                print(f"No hypernyms found for {word}.")
            cache_word_categories(word_lower, CAT_CACHE_UNKNOWN)
            return CAT_CACHE_UNKNOWN

        hypernym = hypernyms[0]
        lemmas = hypernym.lemmas()
        print(f"Closest match for {word} (hypernym: '{lemmas[0]}').")
        cache_word_categories(word_lower, lemmas[0])
        return lemmas[0]


def check_for_cached_word_categories(word, language='en', debug=False):
    cached_category = get_cached_word_category(word)
    if cached_category != CAT_CACHE_NOT_CACHED and cached_category != CAT_CACHE_ERROR:
        if debug:
            print(f"Cached category for '{word}' is '{cached_category}'.")
        return cached_category
    if cached_category == CAT_CACHE_ERROR:
        if debug:
            print(f"Querying category for '{word}' returned an error last time.")
        return CAT_CACHE_UNKNOWN

    return None


def closest_match(word, str_list):
    min_distance = 100
    min_word = None
    for w in str_list:
        distance = Levenshtein.distance(word, w)
        if distance < min_distance:
            min_distance = distance
            min_word = w

    return min_word


def cache_word_categories(word, category):
    if word not in word_category_cache:
        word_category_cache[word] = {
            'cachetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'category': category
        }

    for key in word_category_cache:
        if word_category_cache[key] is None:
            word_category_cache[key] = {}
            word_category_cache[key]['cachetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            word_category_cache[key]['category'] = CAT_CACHE_UNKNOWN

        if isinstance(word_category_cache[key], str):
            print(f"Converting cache for {key}...")
            temp_category = word_category_cache[key]
            word_category_cache[key] = {}
            word_category_cache[key]['cachetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            word_category_cache[key]['category'] = temp_category


def get_cached_word_category(word):
    if word in word_category_cache:
        if 'cachetime' in word_category_cache[word]:
            cachetime = datetime.strptime(word_category_cache[word]['cachetime'], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - cachetime).total_seconds() > 60 * 60 * 24 * 30:
                print(f"Cache for '{word}' is maybe outdated (last updated {cachetime}).")
                return CAT_CACHE_NOT_CACHED

        if 'category' in word_category_cache[word]:
            return word_category_cache[word]['category']

        return word_category_cache[word]
    else:
        return CAT_CACHE_NOT_CACHED


def create_word_cache(language):
    folder = "cache"
    filename = f"{folder}/{language}.json"
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("{}")

    return filename


def initialize_word_category_cache(language):
    global word_category_cache
    if word_category_cache is not None:
        return
    filename = create_word_cache(language)
    with open(filename, 'r') as f:
        word_category_cache = json.load(f)


def save_word_category_cache(language):
    global word_category_cache
    filename = create_word_cache(language)
    with open(filename, 'w') as f:
        json.dump(word_category_cache, f)
