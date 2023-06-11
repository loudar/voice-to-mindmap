import json
import os
import sqlite3
from datetime import datetime

import Levenshtein as Levenshtein

try:
    import babelnet as bn
except RuntimeError as e_babelnet:
    print("Babelnet API key is not valid at the moment.")
import wn

CAT_CACHE_NOT_CACHED = '__not_cached__'
CAT_CACHE_ERROR = '__error__'
CAT_CACHE_UNKNOWN = 'unknown'

net = None


def get_word_category_wordnet(word, language='en', debug=False):
    if not word:
        return CAT_CACHE_UNKNOWN

    word_lower = word.lower()
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
        cache_word_categories(word_lower, CAT_CACHE_UNKNOWN, language)
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
                cache_word_categories(word_lower, CAT_CACHE_UNKNOWN, language)
                return CAT_CACHE_UNKNOWN

            lemmas = hypernyms[0].lemmas()
            print(f"Exact match for {word} (hypernym: '{lemmas[0]}').")
            cache_word_categories(word_lower, lemmas[0], language)
            return lemmas[0]

        hypernyms = synset.hypernyms()
        if len(hypernyms) == 0:
            if debug:
                print(f"No hypernyms found for {word}.")
            return CAT_CACHE_UNKNOWN

        hypernym = hypernyms[0]
        lemmas = hypernym.lemmas()
        print(f"Closest match for {word} (hypernym: '{lemmas[0]}').")
        cache_word_categories(word_lower, lemmas[0], language)
        return lemmas[0]


def get_word_category_babelnet(word, language='en', debug=False):
    if not word:
        return CAT_CACHE_UNKNOWN

    word_lower = word.lower()
    cached_category = check_for_cached_word_categories(word_lower, language, debug)
    if cached_category != None:
        return cached_category

    # Query BabelNet
    print(f"Querying BabelNet for '{word}'.")
    babel_lang = None
    if language == 'en':
        babel_lang = bn.Language.EN
    elif language == 'de':
        babel_lang = bn.Language.DE
    else:
        Exception(f"Language {language} not supported.")

    try:
        synsets = bn.get_synsets(word, from_langs=[babel_lang], to_langs=[babel_lang])

        if not synsets:
            print(f"No synsets found for '{word}'.")
            cache_word_categories(word_lower, CAT_CACHE_UNKNOWN, language)
            return CAT_CACHE_UNKNOWN

        word_category_map = {}
        for synset in synsets:
            raw_categories = synset.categories(babel_lang)
            string_categories = [category.category.lower() for category in raw_categories]
            lemma_lower = synset.main_sense_preferably_in(babel_lang).full_lemma.lower()
            if word_lower in string_categories:
                string_categories.remove(word_lower)

            if len(string_categories) == 0:
                continue

            word_category_map[lemma_lower] = string_categories

        if word_lower in word_category_map:
            closest = closest_match(word_lower, word_category_map[word_lower])
            cache_word_categories(word_lower, closest, language)
            print(f"Word category for '{word}' is '{closest}' (out of {word_category_map[word_lower]}).")
            return closest
        else:
            word_keys = word_category_map.keys()
            min_word = closest_match(word_lower, word_keys)

            if min_word:
                print(f"Word '{word}' not found in BabelNet. Closest match is '{min_word}' (out of {word_keys}).")
                closest = closest_match(word_lower, word_category_map[min_word])
                cache_word_categories(word_lower, closest, language)
                cache_word_categories(min_word, closest, language)
                print(
                    f"Word category for closest match '{min_word}' is '{closest}' (out of {word_category_map[min_word]}).")
                return closest
            else:
                print(f"No closest match found for '{word}' (in {word_keys}).")
                cache_word_categories(word_lower, CAT_CACHE_UNKNOWN, language)
                return CAT_CACHE_UNKNOWN
    except RuntimeError as e_babel:
        print(f"Error querying BabelNet for '{word}': {e_babel}")
        cache_word_categories(word_lower, CAT_CACHE_ERROR, language)
        return CAT_CACHE_UNKNOWN


def check_for_cached_word_categories(word, language='en', debug=False):
    cached_category = get_cached_word_category(word, language)
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


def cache_word_categories(word, category, language):
    filename = create_word_cache(language)
    with open(filename, 'r') as f:
        data = json.load(f)

    if word not in data:
        data[word] = {
            'cachetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'category': category
        }

    for key in data:
        if data[key] is None:
            data[key] = {}
            data[key]['cachetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data[key]['category'] = CAT_CACHE_UNKNOWN

        if isinstance(data[key], str):
            print(f"Converting cache for {key}...")
            temp_category = data[key]
            data[key] = {}
            data[key]['cachetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data[key]['category'] = temp_category

    with open(filename, 'w') as f:
        json.dump(data, f)


def get_cached_word_category(word, language):
    filename = create_word_cache(language)
    with open(filename, 'r') as f:
        data = json.load(f)

    if word in data:
        if 'cachetime' in data[word]:
            cachetime = datetime.strptime(data[word]['cachetime'], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - cachetime).total_seconds() > 60 * 60 * 24 * 30:
                print(f"Cache for '{word}' is maybe outdated (last updated {cachetime}).")
                return CAT_CACHE_NOT_CACHED

        if 'category' in data[word]:
            return data[word]['category']

        return data[word]
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
