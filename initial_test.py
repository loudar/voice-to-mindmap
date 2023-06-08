import speech_recognition as sr
import networkx as nx
import matplotlib.pyplot as plt
import keyboard
import os
import spacy
import uuid
from matplotlib.patches import Ellipse

nlp = spacy.load('en_core_web_sm')

# Initialize recognizer
r = sr.Recognizer()

def voice_to_text():
    audio_file = "input.mp3"

    if not os.path.isfile(audio_file):
        # Record audio
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Recording audio...")
            audio = r.listen(source)
        print("Audio recording complete.")

        # Save audio to file
        with open(audio_file, "wb") as f:
            f.write(audio.get_wav_data())

    # Convert audio to text
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    text = r.recognize_google(audio)

    return text

def extract_logical_links(text):
    doc = nlp(text)
    logical_links = []

    for sentence in doc.sents:
        subjects = []
        for token in sentence:
            if token.pos_ == 'NOUN' or token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
                subjects.append(token.text)

        for i in range(len(subjects)-1):
            logical_links.append((subjects[i], subjects[i+1]))

    return logical_links

def create_mind_map(proximity_links):
    G = nx.Graph()

    for word1, word2 in proximity_links:
        if word1 not in G.nodes():
            G.add_node(word1, size=1)
        else:
            G.nodes[word1]['size'] += 1

        if word2 not in G.nodes():
            G.add_node(word2, size=1)
        else:
            G.nodes[word2]['size'] += 1

        if not G.has_edge(word1, word2):
            G.add_edge(word1, word2, weight=1)
        else:
            G[word1][word2]['weight'] += 1

    return G

def draw_mind_map(G, filename):
    # Set background color
    plt.gca().set_facecolor('white')

    # Calculate scaling factor based on the total number of words
    total_words = sum(G.nodes[node]['size'] for node in G.nodes())
    scaling_factor = 5000 / total_words  # Adjust this value as needed for desired scaling

    # Scale and adjust node positions based on the scaling factor
    pos = nx.spring_layout(G, seed=42, scale=scaling_factor)

    # Draw nodes with labels and circles
    for n in G.nodes():
        x, y = pos[n]
        node_size = G.nodes[n]['size'] * scaling_factor * 0.01

        # Create and add the circle patch using the scaled node size
        circle = plt.Circle((x, y), node_size, color='green', alpha=0.3)
        plt.gca().add_patch(circle)

        # Draw the node label
        plt.text(x, y, n, color='black', ha='center', va='center')

    # Draw edges
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='blue', arrows=True, arrowstyle='-|>', arrowsize=12)

    # Adjust plot limits
    x_values = [pos[node][0] for node in G.nodes()]
    y_values = [pos[node][1] for node in G.nodes()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_margin = (x_max - x_min) * 0.3
    y_margin = (y_max - y_min) * 0.3
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.gcf().set_size_inches(12, 12)

    # Save the figure
    plt.savefig(f"./maps/{filename}.png", dpi=300)
    plt.show()

text = voice_to_text()
print(f"Converted Text: {text}")
logical_links = extract_logical_links(text)
print(f"Logical Links: {logical_links}")
G = create_mind_map(logical_links)

# check if the directory exists. If not, create it
if not os.path.exists('live/maps'):
    os.makedirs('live/maps')

# Generate a random ID
random_id = str(uuid.uuid4().hex)

# Create the filename with the random ID
filename = f"mindmap_{random_id}"

# Save and display the mind map
draw_mind_map(G, filename)