from tensorflow import keras
import numpy as np
import tensorflow as tf
import config
import data_processor
import os
import copy


def build(text_seq, label, text_seq_len, word_index, inverse_word_index,
          encoder_embedding_matrix, decoder_embedding_matrix,
          val_text_seq, val_label, val_text_seq_len, test_text_seq, test_text_seq_len):
    # load data
    encoder_embedding_init = keras.initializers.constant(encoder_embedding_matrix, )
    decoder_embedding_init = keras.initializers.constant(decoder_embedding_matrix, )

    # encoder
    encoder_input = keras.layers.Input(shape=(None,), name="encoder_input")
    encoder_embedding = keras.layers.Embedding(input_dim=config.vocab_size + 1, output_dim=config.embedding_size,
                                               embeddings_initializer=encoder_embedding_init,
                                               mask_zero=True,
                                               # trainable=False,
                                               name="encoder_embedding")
    encoder_embedding = encoder_embedding(encoder_input)
    encoder = keras.layers.LSTM(units=config.encoder_num_units,
                                activation="tanh",
                                return_state=True,
                                name="encoder")
    encoder_output, encoder_state_h, encoder_state_c = encoder(encoder_embedding)
    encoder_state = [encoder_state_h, encoder_state_c]

    # decoder
    decoder_inputs = keras.layers.Input(shape=(None,), name="decoder_input")
    decoder_embedding = keras.layers.Embedding(input_dim=config.vocab_size + 1, output_dim=config.embedding_size,
                                               embeddings_initializer=decoder_embedding_init,
                                               mask_zero=True,
                                               # trainable=False,
                                               name="decoder_embedding")
    decoder_embedding = decoder_embedding(decoder_inputs)
    decoder = keras.layers.LSTM(units=config.decoder_num_units,
                                activation="tanh",
                                return_state=True,
                                return_sequences=True,
                                name="decoder")
    decoder_output, decoder_state_h, decoder_state_c = decoder(decoder_embedding,
                                                               initial_state=encoder_state)
    decoder_dense = keras.layers.Dense(config.vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_output)

    s2s_model = keras.Model([encoder_input, decoder_inputs], decoder_outputs)
    print(s2s_model.summary(line_length=200))
    # module 'tensorflow._api.v1.keras.utils' has no attribute 'print_summary'
    # keras.utils.print_summary(s2s_model, line_length=200)

    # train
    _encoder_input = text_seq
    _decoder_inputs = np.concatenate((np.full((len(_encoder_input), 1),
                                              fill_value=word_index[config.sos_token],
                                              dtype=np.int32),
                                      _encoder_input
                                      ), axis=1)
    temp_decoder_outputs = copy.deepcopy(_encoder_input)
    temp_decoder_outputs = [[x if x != 0 else word_index[config.eos_token] for x in seq]
                            for seq in temp_decoder_outputs]
    _decoder_outputs = np.concatenate((temp_decoder_outputs,
                                       np.full((len(temp_decoder_outputs), 1),
                                               fill_value=word_index[config.eos_token],
                                               dtype=np.int32),
                                       ), axis=1)
    one_hot_decoder_outputs = keras.utils.to_categorical(_decoder_outputs, num_classes=config.vocab_size)
    s2s_model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy")
    s2s_model.fit(x=[_encoder_input, _decoder_inputs],
                  y=one_hot_decoder_outputs,
                  batch_size=config.batch_size,
                  epochs=config.epoch)

    # inference model
    encoder_model = keras.Model(encoder_input, encoder_state)
    decoder_state_input_h = keras.layers.Input(shape=(config.encoder_num_units,))
    decoder_state_input_c = keras.layers.Input(shape=(config.encoder_num_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_output, decoder_state_h, decoder_state_c = decoder(decoder_embedding,
                                                               initial_state=decoder_states_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_output)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    print(decoder_model.summary())

    # inference
    generated_sentences = []
    inference_seqs = text_seq
    # inference_seqs=test_text_seq
    for seq_index in range(len(inference_seqs)):
        input_seq = inference_seqs[seq_index:seq_index + 1]
        predict_output_seq = []
        temp_state = encoder_model.predict(input_seq)

        # Error when checking input: expected decoder_input to have 2 dimensions, but got array with shape (1, 1, 10000)
        # decoder_single_input=np.zeros(shape=[1,1,config.vocab_size],
        #                               dtype=np.int32)
        # decoder_single_input[0,0,word_index[config.sos_token]]=1
        decoder_single_input = np.zeros(shape=[1, 1],
                                        dtype=np.int32)
        decoder_single_input[0, 0] = word_index[config.sos_token]
        end_flag = False
        while (not end_flag):
            decoder_output, decoder_state_h, decoder_state_c = decoder_model.predict(
                [decoder_single_input] + temp_state)
            decoder_single_output_word_index = np.argmax(decoder_output, axis=-1, )
            decoder_single_output_word_index = decoder_single_output_word_index[0][0]
            # decoder_single_output_word_index=decoder_single_output_word_index.reshape((1))
            predict_output_seq.append(decoder_single_output_word_index)
            if decoder_single_output_word_index == word_index[config.eos_token] \
                    or len(predict_output_seq) > config.max_seq_len:
                end_flag = True
            temp_state = [decoder_state_h, decoder_state_c]
            decoder_single_input = np.zeros(shape=[1, 1],
                                            dtype=np.int32)
            decoder_single_input[0, 0,] = decoder_single_output_word_index

        sentence = data_processor.generate_sentence_from_indices(predict_output_seq, inverse_word_index)
        # sentence = [word for word in sentence if word != config.eos_token]
        generated_sentences.append(sentence)
    raw_sentences = \
        [data_processor.generate_sentence_from_indices(x, inverse_word_index)
         for x in inference_seqs]
    # raw_sentences = \
    #     [[word for word in sentence
    #       if word != config.eos_token]
    #      for sentence in raw_sentences]

    dir_name = os.path.dirname(config.generated_sentences_save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(config.generated_sentences_save_path, "w") as o:
        o.write("raw_sentence           generated_sentences\n")
        count = 0
        for a, b in zip(raw_sentences, generated_sentences):
            o.write(a + " ---> " + b + "\n")
            if count < 5:
                print(a + " ---> " + b)
            count += 1
