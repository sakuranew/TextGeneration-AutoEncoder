# import tensorflow as tf
from tensorflow import keras
import os
import data_processor
import model

if __name__ == "__main__":
    text_train_path = "data/reviews-train.txt"
    # text_train_path = "data/reviews-test.txt"
    text_val_path = "data/reviews-val.txt"
    text_test_path = "data/reviews-test.txt"
    glove_embedding_path = "data/glove.6B.100d.txt"
    # tf.logging.set_verbosity(tf.logging.DEBUG)
    # os.environ["CUDA_VISIABLE"]
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # gpu_options = tf.GPUOptions(allow_growth=True)
    # config_proto = tf.ConfigProto(
    #     log_device_placement=False, allow_soft_placement=True,
    #     gpu_options=gpu_options)
    # sess = tf.Session(config=config_proto)

    text_seq, text_seq_len, word_index, inverse_word_index, text_tokenizer = data_processor.get_text_sequences(
        text_train_path,
    )
    val_text_seq, val_text_seq_len, _ = data_processor.get_test_sequences(
        text_val_path, text_tokenizer, word_index, inverse_word_index)
    test_text_seq, test_text_seq_len, _ = data_processor.get_test_sequences(
        text_test_path, text_tokenizer, word_index, inverse_word_index)

    encoder_embedding_matrix, decoder_embedding_matrix = data_processor.load_glove_embedding(glove_embedding_path,
                                                                                             word_index)

    my_model = model.build(text_seq=text_seq,
                           label=None,
                           text_seq_len=text_seq_len,
                           word_index=word_index,
                           inverse_word_index=inverse_word_index,
                           encoder_embedding_matrix=encoder_embedding_matrix,
                           decoder_embedding_matrix=decoder_embedding_matrix,
                           val_text_seq=val_text_seq,
                           val_label=None,
                           val_text_seq_len=val_text_seq_len,
                           test_text_seq=test_text_seq,
                           test_text_seq_len=test_text_seq_len)
