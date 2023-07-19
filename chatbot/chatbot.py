import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_addons as tfa
import re
import time


########## PART 1 - DATA PREPROCESSING ##########


# Importing the dataset
lines = open('dataset/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('dataset/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)  # replace i'm with i am
    text = re.sub(r"he's", "he is", text)  # replace he's with he is
    text = re.sub(r"she's", "she is", text)  # replace she's with she is
    text = re.sub(r"that's", "that is", text)  # replace that's with that is
    text = re.sub(r"what's", "what is", text)  # replace what's with what is
    text = re.sub(r"where's", "where is", text)  # replace where's with where is
    text = re.sub(r"\'ll", "will", text)  # replace 'll with will
    text = re.sub(r"\'ve", "have", text)  # replace 've with have
    text = re.sub(r"\'re", "are", text)  # replace 're with are
    text = re.sub(r"\'d", "would", text)  # replace 'd with would
    text = re.sub(r"won't", "will not", text)  # replace won't with will not
    text = re.sub(r"can't", "cannot", text)  # replace can't with cannot
    text = re.sub(r"[-()\"/@;:<>{}+=~|.?!,]", "", text)  # remove all special characters
    return text


# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word to its number of occurrences
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
threshold_questions = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        word_number += 1

threshold_answers = 20
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number
        word_number += 1

# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']  # PAD: padding, EOS: end of string, OUT: out of vocabulary, SOS: start of string
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}

# Adding the End Of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):  # 25 is the maximum length of questions
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:  # i[1] is the question
            sorted_clean_questions.append(questions_into_int[i[0]])  # i[0] is the index of the question
            sorted_clean_answers.append(answers_into_int[i[0]])  # i[0] is the index of the question


########## PART 2 - BUILDING THE SEQ2SEQ MODEL ##########


# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')  # None: batch size, None: sequence length
    targets = tf.placeholder(tf.int32, [None, None], name='target')  # None: batch size, None: sequence length
    lr = tf.placeholder(tf.float32, name='learning_rate')  # learning rate
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout rate
    return inputs, targets, lr, keep_prob



# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])  # fill a tensor with a specific value
    # strided_slice: extract a subset of a tensor
    # first argument: tensor, second argument: begin, third argument: end, fourth argument: stride
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    # concatenate two tensors
    # horizontal concatenation: axis = 1, vertical concatenation: axis = 0
    # horizontal example: [[1, 2, 3], [4, 5, 6]] + [[7], [8]] = [[1, 2, 3, 7], [4, 5, 6, 8]]
    # vertical example: [[1, 2, 3], [4, 5, 6]] + [[7, 8, 9]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    preprocessed_targets = tf.concat([left_side, right_side], 1)  # horizontal concatenation
    return preprocessed_targets


# Creating the Encoder RNN Layer
# rnn_inputs: inputs of the RNN layer
# rnn_size: number of input tensors
# num_layers: number of layers
# keep_prob: dropout rate
# sequence_length: list of the length of each question in the batch
# def encoder_rnn(
#         rnn_inputs,
#         rnn_size,
#         num_layers,
#         keep_prob,
#         sequence_length):
#     lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)  # create a LSTM cell
#     lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)  # add dropout to the LSTM cell
#     encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)  # create multiple LSTM cells
#     # cell_fw: forward cell, cell_bw: backward cell
#     # sequence_length: list of the length of each question in the batch
#     # inputs: inputs of the RNN layer
#     # dtype: type of the rnn inputs
#     # _encoder_output: output of the encoder, don't need it
#     # encoder_state: final state of the encoder
#     _encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
#         cell_fw=encoder_cell,
#         cell_bw=encoder_cell,
#         sequence_length=sequence_length,
#         inputs=rnn_inputs,
#         dtype=tf.float32)  # create a bidirectional RNN
#
#     return encoder_state

def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    # Create a LSTM cell
    lstm = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True, dropout=1-keep_prob)

    # Stack multiple LSTM layers
    stacked_lstm = tf.keras.Sequential([lstm]*num_layers)

    # Use the Bidirectional layer of Keras to make it bidirectional
    encoder_cell = tf.keras.layers.Bidirectional(stacked_lstm, merge_mode='concat')

    # In the Bidirectional layer in Keras, you should put the sequence_length in the mask
    mask = tf.sequence_mask(sequence_length)

    # Now rnn_inputs are the inputs of the RNN layer
    _encoder_output = encoder_cell(rnn_inputs, mask=mask)

    # Since the Bidirectional layer in Keras returns the final states of the forward and the backward
    # LSTM separately, we should concatenate them to get the final encoder state
    encoder_state = tf.concat(_encoder_output[1:], axis=-1)

    return encoder_state

# Decoding the training set
# encoder_state: final state of the encoder
# decoder_cell: decoder RNN cell
# decoder_embedded_input: embedded input of the decoder
# sequence_length: list of the length of each question in the batch
# decoding_scope: scope of the decoding layer
# output_function: function to return the output of the decoding layer
# keep_prob: dropout rate
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])  # initialize the attention states
    # attention_keys: keys to be compared with the target state
    # attention_values: values used to construct the context vector
    # attention_score_function: used to compute the similarity between the keys and the target state
    # attention_construct_function: used to build the attention state
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.keras.layers.AdditiveAttention.prepare_attention(
        attention_states=attention_states,
        attention_option='bahdanau',
        num_units=decoder_cell.output_size)  # create the attention mechanism

    training_decoder_function = tf.keras.layers.AdditiveAttention.attention_decoder_fn_train(
        encoder_state[0],  # encoder_state[0]: forward state of the encoder
        attention_keys,  # keys to be compared with the target state
        attention_values,  # values used to construct the context vector
        attention_score_function,  # used to compute the similarity between the keys and the target state
        attention_construct_function,  # used to build the attention state
        name='attn_dec_train')  # name of the attention decoder function

    # decoder_output: output of the decoder
    # _decoder_final_state: final state of the decoder, don't need it
    # _decoder_final_context_state: final state of the decoder context, don't need it
    decoder_output, _decoder_final_state, _decoder_final_context_state = tf.keras.layers.AdditiveAttention.dynamic_rnn_decoder(
        decoder_cell,  # decoder RNN cell
        training_decoder_function,  # attention decoder function
        decoder_embedded_input,  # embedded input of the decoder
        sequence_length, # list of the length of each question in the batch
        scope=decoding_scope)  # scope of the decoding layer

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)  # add dropout to the decoder output
    return output_function(decoder_output_dropout)  # return the output of the decoding layer


def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])  # initialize the attention states
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.keras.layers.AdditiveAttention.prepare_attention(
        attention_states=attention_states,
        attention_option='bahdanau',
        num_units=decoder_cell.output_size)  # create the attention mechanism

    test_decoder_function = tf.keras.layers.AdditiveAttention.attention_decoder_fn_inference(
        output_function,  # output function to use
        encoder_state[0],  # encoder_state[0]: forward state of the encoder
        attention_keys,  # keys to be compared with the target state
        attention_values,  # values used to construct the context vector
        attention_score_function,  # used to compute the similarity between the keys and the target state
        attention_construct_function,  # used to build the attention state
        decoder_embeddings_matrix,  # embeddings matrix of the decoder
        sos_id,  # start of sentence id
        eos_id,  # end of sentence id
        maximum_length,  # maximum length of the sentence
        num_words,  # number of words in the vocabulary
        name='attn_dec_inf')  # name of the attention decoder function

    test_predictions, _decoder_final_state, _decoder_final_context_state = tf.keras.layers.AdditiveAttention.dynamic_rnn_decoder(
        decoder_cell,  # decoder RNN cell
        test_decoder_function,  # attention decoder function
        scope=decoding_scope)  # scope of the decoding layer

    return test_predictions


# Creating the Decoder RNN
# def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
#     with tf.variable_scope('decoding') as decoding_scope:
#         lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)  # create a LSTM cell
#         lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)  # add dropout to the LSTM cell
#         decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)  # create multiple LSTM cells
#         weights = tf.truncated_normal_initializer(stddev=0.1)  # initialize the weights, stddev: standard deviation
#         biases = tf.zeros_initializer()
#         output_function = lambda x: tf.keras.layers.Dense(
#             x,  # input tensor
#             num_words,  # number of outputs
#             None,  # activation function
#             scope=decoding_scope,  # scope of the decoding layer
#             weights_initializer=weights,  # weights initializer
#             biases_initializer=biases)  # biases initializer
#
#         training_predictions = decode_training_set(
#             encoder_state,  # final state of the encoder
#             decoder_cell,  # decoder RNN cell
#             decoder_embedded_input,  # embedded input of the decoder
#             sequence_length,  # list of the length of each question in the batch
#             decoding_scope,  # scope of the decoding layer
#             output_function,  # function to return the output of the decoding layer
#             keep_prob,  # dropout rate
#             batch_size)  # batch size
#
#         decoding_scope.reuse_variables()  # reuse the variables
#         test_predictions = decode_test_set(
#             encoder_state,  # final state of the encoder
#             decoder_cell,  # decoder RNN cell
#             decoder_embeddings_matrix,  # embeddings matrix of the decoder
#             word2int['<SOS>'],  # start of sentence id
#             word2int['<EOS>'],  # end of sentence id
#             sequence_length - 1,  # maximum length of the sentence
#             num_words,  # number of words in the vocabulary
#             decoding_scope,  # scope of the decoding layer
#             output_function,  # function to return the output of the decoding layer
#             keep_prob,  # dropout rate
#             batch_size)  # batch size
#
#     return training_predictions, test_predictions

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size,
                num_layers, word2int, keep_prob, batch_size):
    # Create a LSTM cell
    lstm_cells = [tf.keras.layers.LSTMCell(rnn_size, dropout=1 - keep_prob) for _ in range(num_layers)]
    decoder_cell = tf.keras.layers.StackedRNNCells(lstm_cells)

    # Initialize the weights and biases
    weights_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.1)
    biases_initializer = tf.keras.initializers.Zeros()

    # Define output layer
    output_layer = tf.keras.layers.Dense(
        num_words,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer)

    # This function now returns the final output layer
    output_function = lambda x: output_layer(x)

    # Here, you might need to modify your decode_training_set and decode_test_set functions to work with TF 2.x
    # I'm assuming these functions are compatible with TF 2.x after modifications
    training_predictions = decode_training_set(
        encoder_state,
        decoder_cell,
        decoder_embedded_input,
        sequence_length,
        output_function,
        keep_prob,
        batch_size)

    # decoding_scope.reuse_variables() is not needed in TF 2.x because of eager execution

    test_predictions = decode_test_set(
        encoder_state,
        decoder_cell,
        decoder_embeddings_matrix,
        word2int['<SOS>'],
        word2int['<EOS>'],
        sequence_length - 1,
        num_words,
        output_function,
        keep_prob,
        batch_size)

    return training_predictions, test_predictions


# Building the seq2seq model
def seq2seq_model(
        inputs,  # questions of the Colnell movie data set
        targets,  # the real answer
        keep_prob,  # dropout rate
        batch_size,  # batch size
        sequence_length,  # list of the length of each question in the batch
        answers_num_words,  # number of words in the answers vocabulary
        questions_num_words,  # number of words in the questions vocabulary
        encoder_embedding_size,  # size of the encoder embedding layer
        decoder_embedding_size,  # size of the decoder embedding layer
        rnn_size,  # number of units in the RNN
        num_layers,  # number of layers in the RNN
        questionswords2int):  # word2int dictionary of the questions vocabulary

    # Instantiate the layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=answers_num_words + 1,
                                                output_dim=encoder_embedding_size,
                                                embeddings_initializer=tf.random_uniform_initializer(0, 1))

    # If you have sequence data (a batch of sequences, where each sequence is a list of integers)
    # Apply the layer
    encoder_embedded_input = embedding_layer(inputs)

    # encoder_embedded_input = tf.contrib.layers.embed_sequence(
    #     inputs,  # questions of the Colnell movie data set
    #     answers_num_words + 1,  # number of words in the answers vocabulary
    #     encoder_embedding_size,  # number of dimensions for each word in the encoder embedding layer
    #     initializer=tf.random_uniform_initializer(0, 1))  # initialize the encoder embedding layer

    encoder_state = encoder_rnn(
        encoder_embedded_input,  # inputs that formatted for the encoder
        rnn_size,  # number of units in the encoder RNN
        num_layers,  # number of layers in the encoder RNN
        keep_prob,  # dropout rate
        sequence_length)  # list of the length of each question in the batch

    preprocessed_targets = preprocess_targets(
        targets,  # the real answer
        questionswords2int,  # word2int dictionary of the questions vocabulary
        batch_size)  # batch size

    decoder_embeddings_matrix = tf.Variable(
        tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))  # initialize the decoder embedding layer
    decoder_embedded_input = tf.nn.embedding_lookup(
        decoder_embeddings_matrix,  # embeddings matrix of the decoder
        preprocessed_targets)  # targets that formatted for the decoder

    training_predictions, test_predictions = decoder_rnn(
        decoder_embedded_input,  # embedded input of the decoder
        decoder_embeddings_matrix,  # embeddings matrix of the decoder
        encoder_state,  # final state of the encoder
        questions_num_words,  # number of words in the questions vocabulary
        sequence_length,  # list of the length of each question in the batch
        rnn_size,  # number of units in the encoder RNN
        num_layers,  # number of layers in the encoder RNN
        questionswords2int,  # word2int dictionary of the questions vocabulary
        keep_prob,  # dropout rate
        batch_size)  # batch size

    return training_predictions, test_predictions


########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########

# Setting the Hyperparameters
epochs = 100  # number of epochs
batch_size = 64  # batch size
rnn_size = 512  # number of units in the RNN
num_layers = 3  # number of encoder and decoder layers in the RNN
encoding_embedding_size = 512  # size of the encoder embedding layer
decoding_embedding_size = 512  # size of the decoder embedding layer
learning_rate = 0.01  # learning rate
learning_rate_decay = 0.9  # learning rate decay 90%
min_learning_rate = 0.0001  # minimum learning rate
keep_probability = 0.5  # dropout rate

# Defining a session
# tf.reset_default_graph()  # reset the graph
# session = tf.InteractiveSession()  # create a session

context = tf.compat.v1.get_default_graph()

session = tf.compat.v1.Session()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()  # get the inputs

print('done')





























