import time
import numpy as np
import csv
import datetime
import json
import os
from nltk.stem.lancaster import LancasterStemmer
import nltk
nltk.download('punkt')


def get_raw_training_data(filename):
    # TODO: Read assignment for clarification...I'm how she wants this dictionary written
    '''Reads in raw training data file, and extracts data into multi-key dictionary
    Key: Person --> Value: Actor Speaking
    Key: Actor Speaking --> Sentence said
    I think this is what she means?
    '''

    training_data = {}

    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)

        for person, sentence in reader:
            training_data["person"] = person
            training_data["sentence"] = sentence

    return training_data


def preprocess_words(words, stemmer):
    '''Stem all words'''

    # initialize list of stemmed words
    stemmed_words = []

    # remove punctuation
    # TODO: Check this
    no_punctuation = [word for word in words if word.isalpha()]

    # stem tokens
    stemmed_words = [stemmer.stem(word).lower()
                     for word in no_punctuation]  # TODO: Check this

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

    # for sentence in sentences (loop thru dictionary?)
    # TODO: Need to finish the dictionary to work on this step
    # TODO: add tokenize each sentence
    # TODO: add each token to 'words'
    # TODO: add tuple of (tokenized_sentence, actor) to documents or classes as needed

    # preprocess words to only contain stems
    stemmed_words = preprocess_words(words, stemmer)

    return stemmed_words, classes, stemmer


def create_training_data(stemmed_words, classes, documents, stemmer):
    '''Creates training data'''

    # initialize training data (list I think?)
    training_data = []

    # initialize output
    output = []

    # fill training_data_variable
    for sentence in documents:
        sentence = sentence[0]  # index tuple for sentence

        # initialize and fill the bag
        bag_of_words = [1 if word in stemmed_words else 0 for word in sentence]

        # append new bag to old bag
        training_data.append(bag_of_words)

        # initialize list of 0s of len classes
        actor_arr = []
        for i in len(classes):
            output.append(0)

        actor = sentence[1]  # index tuple for actor

        # mark the actor
        try:
            index = classes.index(actor)  # TODO: Check this
            output[index] = 1
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


def main():

    # initialize stemmer
    stemmer = LancasterStemmer()

    # get raw training data
    raw_training_data = get_raw_training_data('dialogue_data.csv')

    # organize data and get words, classes, and documents
    stemmed_words, classes, documents = organize_raw_training_data(
        organize_raw_training_data, stemmer)

    # create training data and output
    training_data, output = create_training_data(
        stemmed_words, classes, documents, stemmer)


if __name__ == "__main__":
    main()
