import sys
import spacy

from deepmultilingualpunctuation import PunctuationModel
from lib.word_categorization_wordnet import get_word_category_wordnet

spacy_models = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm'
}

nlp = None
model = PunctuationModel()


def load_spacy_model_if_needed(selected_lang):
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load(spacy_models[selected_lang])
        except OSError as e:
            print(f"Error loading model: {e} \nPlease run 'python -m spacy download {spacy_models[selected_lang]}'")
            sys.exit(1)

    return nlp


def extract_logical_links(text, selected_lang):
    result = model.restore_punctuation(text)
    doc = load_spacy_model_if_needed(selected_lang)(result)
    logical_links = []
    enabled_pos = ['NOUN']

    for sentence in doc.sents:
        print(f"Processing sentence: {sentence}")
        subjects = []
        for token in sentence:
            if 'NOUN' in enabled_pos and (token.pos_ == 'NOUN' or token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass'):
                subjects.append((token.text, token.pos_))  # include the token's POS as the category
            elif 'ADJ' in enabled_pos and token.pos_ == 'ADJ':
                subjects.append((token.text, token.pos_))  # include the token's POS as the category
            elif 'VERB' in enabled_pos and (token.pos_ == 'VERB' or token.dep_ == 'ROOT'):
                subjects.append((token.text, token.pos_))

        for i in range(len(subjects) - 1):
            logical_links.append({
                'source': subjects[i][0],
                'target': subjects[i + 1][0],
                'source_category': get_word_category_wordnet(subjects[i][0], selected_lang),
                'target_category': get_word_category_wordnet(subjects[i + 1][0], selected_lang),
                'source_type': subjects[i][1],
                'target_type': subjects[i + 1][1],
            })

    # calculate a weight for each link based on the count of the same link
    for link in logical_links:
        link['weight'] = logical_links.count(link)

    return logical_links
