import sys
import spacy

from lib.word_categorization import get_word_category_wordnet

models = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm'
}

nlp = None


def load_spacy_model_if_needed(selected_lang):
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load(models[selected_lang])
        except OSError as e:
            print(f"Error loading model: {e} \nPlease run 'python -m spacy download {models[selected_lang]}'")
            sys.exit(1)

    return nlp


def extract_logical_links(text, selected_lang):
    doc = load_spacy_model_if_needed(selected_lang)(text)
    logical_links = []

    for sentence in doc.sents:
        subjects = []
        for token in sentence:
            if token.pos_ == 'NOUN' or token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
                subjects.append((token.text, token.pos_))  # include the token's POS as the category
            elif token.pos_ == 'ADJ':
                subjects.append((token.text, token.pos_))  # include the token's POS as the category

        for i in range(len(subjects) - 1):
            logical_links.append({
                'source': subjects[i][0],
                'target': subjects[i + 1][0],
                'source_category': get_word_category_wordnet(subjects[i][0], selected_lang),
                'target_category': get_word_category_wordnet(subjects[i + 1][0], selected_lang),
            })

    # calculate a weight for each link based on the count of the same link
    for link in logical_links:
        link['weight'] = logical_links.count(link)

    return logical_links
