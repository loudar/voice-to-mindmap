import sys
import nltk as nltk
from deepmultilingualpunctuation import PunctuationModel

from lib.advanced_text_processing import extract_logical_links_advanced
from lib.classes.timer import Timer
from lib.mapper import create_mind_map_force
from lib.plotly_wrapper import create_plot

languages = ['en', 'de']
selected_lang = 'de'

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

if len(sys.argv) > 1:
    source_file = sys.argv[1]
else:
    source_file = "transcripts/transcript_latest.txt"


def generate_plot(text):
    punctuation_model = PunctuationModel()
    timer = Timer()
    timer.start(f"{len(text)}")
    text = punctuation_model.restore_punctuation(text)
    logical_links = extract_logical_links_advanced(text, selected_lang)
    G, positions = create_mind_map_force(logical_links)
    timer.stop()
    create_plot(G, positions)


def main():
    with open(source_file, 'r', encoding='utf-8') as f:
        text = f.read()

    generate_plot(text)


main()
