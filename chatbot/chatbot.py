import numpy as np
import tensorflow as tf
import re
import time

lines = open('dataset/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('dataset/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all the conversations
conversations_ids = []
for  conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) -1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text) # replace i'm with i am
    text = re.sub(r"he's", "he is", text) # replace he's with he is
    text = re.sub(r"she's", "she is", text) # replace she's with she is
    text = re.sub(r"that's", "that is", text) # replace that's with that is
    text = re.sub(r"what's", "what is", text) # replace what's with what is
    text = re.sub(r"where's", "where is", text) # replace where's with where is
    text = re.sub(r"\'ll", "will", text) # replace 'll with will
    text = re.sub(r"\'ve", "have", text) # replace 've with have
    text = re.sub(r"\'re", "are", text) # replace 're with are
    text = re.sub(r"\'d", "would", text) # replace 'd with would
    text = re.sub(r"won't", "will not", text) # replace won't with will not
    text = re.sub(r"can't", "cannot", text) # replace can't with cannot
    text = re.sub(r"[-()\"/@;:<>{}+=~|.?!,]", "", text) # remove all special characters
    return text


# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word to its number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Creating two dictionaries that map the questions words and the answers words to a unique integer
threshold = 20
questionsword2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionsword2int[word] = word_number
        word_number += 1

answersword2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answersword2int[word] = word_number
        word_number += 1