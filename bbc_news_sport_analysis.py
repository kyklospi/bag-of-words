import os
import math

sport_folder = 'bbc-news/sport'
# list to hold the text content of each file
sport_texts = []

# read each file from sport folder
for filename in os.listdir(sport_folder):
    file_path = os.path.join(sport_folder, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        sport_texts.append(file.read())

# print first 1000 characters of the first text to check
print(f"--- First 1000 characters of the first text ---\n\n {sport_texts[0][:1000]} \n\n---")

# function to preprocess and split text into words
def preprocess(text):
    words = text.lower().split()
    # remove common punctuations
    words = [word.strip('.,!?()[]{}":;') for word in words]
    return words

# create a vocabulary
vocabulary = set()
for text in sport_texts:
    words = preprocess(text)
    vocabulary.update(words)

# convert vocabulary to a sorted list
vocabulary = sorted(vocabulary)

# print the first 50 words in vocabulary
print(f"\n--- First 50 words in sorted vocabulary ---\n\n {vocabulary[:50]} \n\n---")

# create mapping of word to index in vocabulary
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

# function to create Bag of Words (BoW) vector with word counts
def create_bow(text):
    # initialize a zero vector for BoW model
    bow_vector = [0] * len(vocabulary)

    words = preprocess(text)

    # update BoW vector with word counts
    for word in words:
        if word in word_to_index:
            idx = word_to_index[word]
            bow_vector[idx] += 1

    return bow_vector

bow_vectors = [create_bow(text) for text in sport_texts]

# function to calculate Euclidean distance between 2 vectors
def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((x-y) ** 2 for x, y in zip(vec1, vec2)))

# get the BoW vector for the text "497.txt"
target_text_idx = 497-1     # index starting with 0
target_bow = bow_vectors[target_text_idx]

# calculate the Euclidean distance of each text to "497.txt"
distances = []
for idx, bow_vector in enumerate(bow_vectors):
    if idx != target_text_idx:      # exclude the target text itself
        dist = euclidean_distance(target_bow, bow_vector)
        distances.append((idx, dist))

# sort the distances and get the top 3 closest texts
distances.sort(key=lambda x: x[1])
closest_texts = distances[:3]

print("\n--- Top 3 closest texts to 497.txt are ---\n")
for idx, dist in closest_texts:
    print(f"Text Index: {idx + 1}, Distance: {dist}")

print("\n--- Closest texts ---\n")
for idx, _ in closest_texts:
    print(f"\nText Index: {idx + 1} \n")
    # print the first 1000 characters of the closest texts
    print(sport_texts[idx][:1000])
    print("\n---\n")