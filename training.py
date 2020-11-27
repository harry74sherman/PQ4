import time
import numpy as np
import csv
import datetime
import json
import os
from nltk.stem.lancaster import LancasterStemmer
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')


def get_raw_training_data(filename):
    '''creates a list of dictionaries from an input csv file'''

    training_data = []

    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)

        for person, sentence in reader:
            author = {}
            author["person"] = person
            author["sentence"] = sentence
            training_data.append(author)

    return training_data


def preprocess_words(words, stemmer):
    '''Stem all words'''

    # initialize list of stemmed words
    stemmed_words = []

    # remove punctuation
    no_punctuation = [word for word in words if word.isalpha()]

    # stem tokens
    stemmed_words = [stemmer.stem(word).lower()
                     for word in no_punctuation]

    # remove duplicates
    stemmed_words = set(stemmed_words)
    stemmed_words = list(stemmed_words)

    return stemmed_words


def organize_raw_training_data(raw_training_data, stemmer):
    '''Generates words, classes, and classes'''

    # initialize list of all tokens
    words = []

    # initialize list for actors we haven't heard before (no duplicates)
    classes = []

    # initialize list that will contain tuples of actors and their tokens
    documents = []

    for data in raw_training_data:

        # tokenize sentence
        tokenized_sentence = nltk.word_tokenize(data["sentence"])

        # add tokens to words
        words.extend(tokenized_sentence)

        actor = data["person"]

        # add tuple  (tokens, actor) to documents
        documents.append((tokenized_sentence, actor))

        # if actor not in classes, add to classes
        if actor not in classes:
            classes.append(actor)

    # preprocess words to only contain stems
    stemmed_words = preprocess_words(words, stemmer)

    return stemmed_words, classes, documents


def create_training_data(stemmed_words, classes, documents, stemmer):
    '''Creates training data'''

    # initialize training data (list I think?)
    training_data = []

    # initialize output
    output = []

    # fill training_data_variable
    print(type(documents))
    for doc in documents:
        sentence = doc[0]  # index tuple for sentence

        # initialize and fill the bag
        bag_of_words = [1 if word in sentence else 0 for word in stemmed_words]

        # append new bag to old bag
        training_data.append(bag_of_words)

        # initialize list of 0s of len classes
        actor_arr = [0] * len(classes)

        actor = doc[1]  # index tuple for actor

        # mark the actor
        try:
            index = classes.index(actor)
            actor_arr[index] = 1
        except:
            pass  # actor not in list

        # append actor to output
        output.append(actor_arr)

    return training_data, output


def sigmoid(z):
    '''calculates sigmoid of z'''

    return 1 / (1 + np.exp(-z))


def sigmoid_output_to_derivative(output):
    '''Convert the sigmoid function's output to its derivative.'''

    return output * (1-output)


"""* * * TRAINING * * *"""


def init_synapses(X, hidden_neurons, classes):
    """Initializes our synapses (using random values)."""
    # Ensures we have a "consistent" randomness for convenience.
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    return synapse_0, synapse_1


def feedforward(X, synapse_0, synapse_1):
    """Feed forward through layers 0, 1, and 2."""
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    return layer_0, layer_1, layer_2


def get_synapses(epochs, X, y, alpha, synapse_0, synapse_1):
    """Update our weights for each epoch."""
    # Initializations.
    last_mean_error = 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    synapse_0_direction_count = np.zeros_like(synapse_0)

    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    # Make an iterator out of the number of epochs we requested.
    for j in iter(range(epochs+1)):
        layer_0, layer_1, layer_2 = feedforward(X, synapse_0, synapse_1)

        # How much did we miss the target value?
        layer_2_error = y - layer_2

        if (j % 10000) == 0 and j > 5000:
            # If this 10k iteration's error is greater than the last iteration,
            # break out.
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print("delta after "+str(j)+" iterations:" +
                      str(np.mean(np.abs(layer_2_error))))
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print("break:", np.mean(np.abs(layer_2_error)),
                      ">", last_mean_error)
                break

        # In what direction is the target value?  How much is the change for layer_2?
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # How much did each l1 value contribute to the l2 error (according to the weights)?
        # (Note: .T means transpose and can be accessed via numpy!)
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # In what direction is the target l1?  How much is the change for layer_1?
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        # Manage updates.
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if j > 0:
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    return synapse_0, synapse_1


def save_synapses(filename, words, classes, synapse_0, synapse_1):
    """Save our weights as a JSON file for later use."""
    now = datetime.datetime.now()

    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print("Saved synapses to:", synapse_file)


def train(X, y, words, classes, hidden_neurons=10, alpha=1, epochs=50000):
    """Train using specified parameters."""
    print("Training with {0} neurons and alpha = {1}".format(
        hidden_neurons, alpha))

    synapse_0, synapse_1 = init_synapses(X, hidden_neurons, classes)

    # For each epoch, update our weights
    synapse_0, synapse_1 = get_synapses(
        epochs, X, y, alpha, synapse_0, synapse_1)

    # Save our work
    save_synapses("synapses.json", words, classes, synapse_0, synapse_1)


def start_training(words, classes, training_data, output):
    """Initialize training process and keep track of processing time."""
    start_time = time.time()
    X = np.array(training_data)
    y = np.array(output)

    train(X, y, words, classes, hidden_neurons=20, alpha=0.1, epochs=100000)

    elapsed_time = time.time() - start_time
    print("Processing time:", elapsed_time, "seconds")


"""* * * CLASSIFICATION * * *"""


def bow(sentence, words):
    """Return bag of words for a sentence."""
    stemmer = LancasterStemmer()

    # Break each sentence into tokens and stem each token.
    sentence_words = [stemmer.stem(word.lower())
                      for word in nltk.word_tokenize(sentence)]

    # Create the bag of words.
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return (np.array(bag))


def get_output_layer(words, sentence):
    """Open our saved weights from training and use them to predict based on
    our bag of words for the new sentence to classify."""

    # Load calculated weights.
    synapse_file = 'synapses.json'
    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])

    # Retrieve our bag of words for the sentence.
    x = bow(sentence.lower(), words)
    # This is our input layer (which is simply our bag of words for the sentence).
    l0 = x
    # Perform matrix multiplication of input and hidden layer.
    l1 = sigmoid(np.dot(l0, synapse_0))
    # Create the output layer.
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


def classify(words, classes, sentence):
    """Classifies a sentence by examining known words and classes and loading our calculated weights (synapse values)."""
    error_threshold = 0.2
    results = get_output_layer(words, sentence)
    results = [[i, r] for i, r in enumerate(results) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    print("\nSentence to classify: {0}\nClassification: {1}".format(
        sentence, return_results))
    return return_results


def main():
    """TODO: more instructions here..."""

    # initialize stemmer
    stemmer = LancasterStemmer()

    # get raw training data
    raw_training_data = get_raw_training_data('dialogue_data.csv')

    # organize data and get words, classes, and documents
    words, classes, documents = organize_raw_training_data(
        raw_training_data, stemmer)

    # create training data and output
    training_data, output = create_training_data(
        words, classes, documents, stemmer)

    # Comment this out if you have already trained once and don't want to re-train.
    start_training(words, classes, training_data, output)

    # Classify new sentences.
    classify(words, classes, "will you look into the mirror?")
    classify(words, classes,
             "mithril, as light as a feather, and as hard as dragon scales.")
    classify(words, classes, "the thieves!")


if __name__ == "__main__":
    main()
