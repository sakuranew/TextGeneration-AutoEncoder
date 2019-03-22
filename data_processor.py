import numpy as np
import tensorflow as tf
from tensorflow import keras
import config
import json
import copy


def generate_sentence_from_indices(sequence, inverse_word_index):
    words = (inverse_word_index[x] for x in sequence if x != 0 and
             x != config.predefined_word_index[config.eos_token] and
             x != config.predefined_word_index[config.sos_token])
    sentence = " ".join(words)
    return sentence


def load_glove_embedding(glove_embedding_path, word_index):
    embedding_model = dict()
    with open(glove_embedding_path) as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            embedding_model[word] = embedding

    # 不在词列表的正好向量可以为随机初始化的值
    encoder_embedding_matrix = np.random.uniform(
        size=(config.vocab_size + 1, config.embedding_size),
        low=-0.05, high=0.05).astype(dtype=np.float32)
    # eos之类的词向量的能够保证是相等
    decoder_embedding_matrix = copy.deepcopy(encoder_embedding_matrix)
    tf.logging.info("encoder_embedding_matrix: {}".format(encoder_embedding_matrix.shape))
    # i = 0
    i = 1
    for word in word_index:
        try:
            assert (i == word_index[word])
            word_embedding = embedding_model[word]
            encoder_embedding_matrix[i] = word_embedding
            decoder_embedding_matrix[i] = word_embedding
        except KeyError:
            pass

        i += 1
        if i > config.vocab_size:
            break
    del embedding_model
    return encoder_embedding_matrix, decoder_embedding_matrix
    #
    # decoder_embedding_matrix = np.random.uniform(
    #     size=(global_config.vocab_size, global_config.embedding_size),
    #     low=-0.05, high=0.05).astype(dtype=np.float32)
    # logging.debug("decoder_embedding_matrix: {}".format(decoder_embedding_matrix.shape))


def get_labels(label_file_path, store_labels, store_path):
    all_labels = list(open(label_file_path, "r").readlines())
    all_labels = [label.strip() for label in all_labels]

    labels = sorted(list(set(all_labels)))
    num_labels = len(labels)

    counter = 0
    label_to_index_map = dict()
    index_to_label_map = dict()
    for label in labels:
        label_to_index_map[label] = counter
        index_to_label_map[counter] = label
        counter += 1

    # if store_labels:
    #     with open(os.path.join(store_path, global_config.index_to_label_dict_file), 'w') as file:
    #         json.dump(index_to_label_map, file)
    #     with open(os.path.join(store_path, global_config.label_to_index_dict_file), 'w') as file:
    #         json.dump(label_to_index_map, file)
    # logging.info("labels: {}".format(label_to_index_map))

    one_hot_labels = list()
    for label in all_labels:
        one_hot_label = np.zeros(shape=num_labels, dtype=np.int32)
        one_hot_label[label_to_index_map[label]] = 1
        one_hot_labels.append(one_hot_label)

    return [np.asarray(one_hot_labels), num_labels]


def get_text_sequences(text_file_path, ):
    word_index = config.predefined_word_index
    num_predefined_tokens = len(word_index)

    text_tokenizer = keras.preprocessing.text.Tokenizer(config.vocab_size - num_predefined_tokens)
    with open(text_file_path, "r") as text:
        text = text.readlines()[:config.data_size]
        # text=text[:config.data_size0]
        text_tokenizer.fit_on_texts(text)
    available_vocab = len(text_tokenizer.word_index)
    tf.logging.info("available_vocab:%d" % available_vocab)

    for index, word in enumerate(text_tokenizer.word_index, start=1):
        new_index = index + num_predefined_tokens
        if new_index == config.vocab_size + 1:
            break
        word_index[word] = new_index
    text_tokenizer.word_index = word_index

    with open(text_file_path) as text_file:
        text_file = text_file.readlines()[:config.data_size]

        actual_sequences = text_tokenizer.texts_to_sequences(text_file)
    text_seq_len = np.asarray(a=[len(x) if len(x) < config.max_seq_len else config.max_seq_len
                                 for x in actual_sequences], dtype=np.int32)
    # text_seq_len = np.asarray(a=[len(x) + 1 if len(x) < config.max_seq_len else config.max_seq_len
    #                              for x in actual_sequences], dtype=np.int32)   # x + 1 to accomodate a single EOS token
    trimmed_text_seq = [[x if x <= config.vocab_size else word_index[config.unk_token] for x in seq]
                        for seq in actual_sequences]
    inverse_word_index = {v: k for k, v in word_index.items()}
    padded_seq = keras.preprocessing.sequence.pad_sequences(trimmed_text_seq, maxlen=config.max_seq_len,
                                                            padding="post",
                                                            truncating="post",
                                                            value=0)
    # 应该以结束符来pad 不对。以0然后embed层进行mask，后面加eos就好了

    # value=word_index[config.eos_token])
    with open(config.vocab_save_path, 'w') as json_file:
        json.dump(word_index, json_file)
    return padded_seq, text_seq_len, word_index, inverse_word_index, text_tokenizer


def get_test_sequences(text_file_path, text_tokenizer, word_index, inverse_word_index):
    with open(text_file_path) as text_file:
        actual_sequences = text_tokenizer.texts_to_sequences(text_file)
    actual_sentences = 0
    # actual_sentences=[generate_sentence_from_indices(x, inverse_word_index)
    #          for x in actual_sequences]
    text_seq_len = np.asarray(a=[len(x) if len(x) < config.max_seq_len else config.max_seq_len
                                 for x in actual_sequences], dtype=np.int32)
    # text_seq_len = np.asarray(a=[len(x) + 1 if len(x) < config.max_seq_len else config.max_seq_len
    #                              for x in actual_sequences], dtype=np.int32)   # x + 1 to accomodate a single EOS token

    trimmed_text_seq = [[x if x <= config.vocab_size else word_index[config.unk_token] for x in seq]
                        for seq in actual_sequences]

    padded_seq = keras.preprocessing.sequence.pad_sequences(trimmed_text_seq, maxlen=config.max_seq_len,
                                                            padding="post",
                                                            truncating="post",
                                                            # value=0) 应该以结束符来pad
                                                            value=0)
    with open(config.vocab_save_path, 'w') as json_file:
        json.dump(word_index, json_file)
    return padded_seq, text_seq_len, actual_sentences
