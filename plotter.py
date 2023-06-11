import sys
import nltk as nltk

from lib.advanced_text_processing import extract_logical_links_advanced
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


def update_plot(text):
    # Extract logical links from the new text
    logical_links = extract_logical_links_advanced(text, selected_lang)

    # Create the mind map using the accumulated logical links
    G, subgraph_positions = create_mind_map_force(logical_links)
    create_plot(G, subgraph_positions)


def main():
    with open(source_file, 'r', encoding='windows-1252') as f:
        text = f.read()

    update_plot(text)


main()
