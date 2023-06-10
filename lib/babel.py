import json
import os
import Levenshtein as Levenshtein
try:
    import babelnet as bn
except RuntimeError as e_babelnet:
    print("Babelnet API key is not valid at the moment.")

def get_word_category(word, language='en'):
    if not word:
        return 'unknown'

    word_lower = word.lower()
    cached_category = get_cached_word_categories(word_lower, language)
    if cached_category != '__not_cached__':
        print(f"Cached category for '{word}' is '{cached_category}'.")
        return cached_category
    if cached_category != '__error__':
        print(f"Querying category for '{word}' returned an error last time.")
        return 'unknown'

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
            cache_word_categories(word_lower, 'unknown', language)
            return 'unknown'

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
                cache_word_categories(word_lower, 'unknown', language)
                return 'unknown'
    except RuntimeError as e_babel:
        print(f"Error querying BabelNet for '{word}': {e_babel}")
        cache_word_categories(word_lower, '__error__', language)
        return 'unknown'


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
        data[word] = category

    with open(filename, 'w') as f:
        json.dump(data, f)


def get_cached_word_categories(word, language):
    filename = create_word_cache(language)
    with open(filename, 'r') as f:
        data = json.load(f)

    if word in data:
        return data[word]
    else:
        return '__not_cached__'


def create_word_cache(language):
    folder = "cache"
    filename = f"{folder}/{language}.json"
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("{}")

    return filename