import os
import math
import string

SPORT_FOLDER = 'bbc-news/sport'
STOP_WORDS = {"and", "the", "is", "in", "to"}

# List and numerically sort .txt files in the folder
def list_sorted_txt_files(folder):
    return sorted(
        [file for file in os.listdir(folder) if file.endswith('.txt')],
        key=lambda x: int(os.path.splitext(x)[0])
    )

# Read all text files and return a list of (filename, content) tuple
def read_texts_from_folder(folder, filenames):
    texts = []
    for filename in filenames:
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='utf-8') as file:
            texts.append((filename, file.read()))
    return texts

# Strip punctuation from a single word
def remove_punctuation(word):
    return word.strip(string.punctuation)

# Split text into words, lowercase them, remove punctuation and stopwords
def tokenize(text):
    words = text.lower().split()
    words = [remove_punctuation(word) for word in words]
    # return words
    # comment below for BoW without filtering out stop words
    return [word for word in words if word and word not in STOP_WORDS]

# Create a sorted vocabulary from a list of tokenized texts
def build_vocabulary(texts):
    vocab = set()
    for _, text in texts:
        vocab.update(tokenize(text))
    return sorted(vocab)

# Create a Bag-of-Words vector for a given text
def create_bow_vector(text, vocabulary, word_to_index):
    bow = [0] * len(vocabulary)
    for word in tokenize(text):
        if word in word_to_index:
            bow[word_to_index[word]] += 1
    return bow

# Compute Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def main():
    sorted_files = list_sorted_txt_files(SPORT_FOLDER)
    sport_texts = read_texts_from_folder(SPORT_FOLDER, sorted_files)

    print(f"\n--- First 2000 characters of the first text ({sorted_files[0]}) ---\n")
    print(sport_texts[0][1][:2000])
    print("\n---\n")

    vocabulary = build_vocabulary(sport_texts)
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

    print("\n--- First 50 words in sorted vocabulary ---\n")
    print(vocabulary[:50])
    print("\n---\n")

    bow_vectors = [create_bow_vector(text, vocabulary, word_to_index) for _, text in sport_texts]

    target_idx = 497 - 1
    target_bow = bow_vectors[target_idx]

    distances = [
        (idx, euclidean_distance(target_bow, vec))
        for idx, vec in enumerate(bow_vectors)
        if idx != target_idx
    ]

    closest_texts = sorted(distances, key=lambda x: x[1])[:3]

    print("--- Top 3 closest texts to 497.txt are ---\n")
    for idx, dist in closest_texts:
        print(f"Text Index: {idx + 1}, Distance: {dist}")

    print("\n--- Closest texts ---\n")
    for idx, _ in closest_texts:
        print(f"\nText Index: {idx + 1} \n")
        print(sport_texts[idx][1][:2000])
        print("\n---\n")

if __name__ == "__main__":
    main()
