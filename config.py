# model config
batch_size = 128
epoch = 100
learning_rate = 0.01
data_size = 20000
encoder_num_units = 256
decoder_num_units = 256
recurrent_state_keep_prob = 0.2
fully_connected_keep_prob = 0.2
sequence_word_keep_prob = 0.2

vocab_size = 5000
embedding_size = 100
max_seq_len = 30
validation_interval = 2

unk_token = "<unk>"
sos_token = "<sos>"
eos_token = "<eos>"
predefined_word_index = {
    unk_token: 1,
    sos_token: 2,
    eos_token: 3,
}

# path
log_directory = "log"
model_save_path = "checkpoint"
vocab_save_path = "data/vocab.json"
# generated_sentences_save_path = "output/generated_sentences-"+".txt"

# generated_sentences_save_path = "output/generated_sentences-"+str(data_size)+"-"+str(epoch)+".txt"
generated_sentences_save_path = "output/generated_sentences0.8-" + str(data_size) + "-" + str(vocab_size) + "-" + str(
    epoch) + ".txt"
