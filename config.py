# model config
batch_size = 128
epoch = 20
learning_rate=0.001

encoder_num_units=256
decoder_num_units=256
recurrent_state_keep_prob=0.5
fully_connected_keep_prob=0.5
sequence_word_keep_prob=0.5

vocab_size=3000
embedding_size=100
max_seq_len=30
validation_interval=1

unk_token = "<unk>"
sos_token = "<sos>"
eos_token = "<eos>"
predefined_word_index = {
    unk_token: 0,
    sos_token: 1,
    eos_token: 2,
}

# path
log_directory="log"
model_save_path="checkpoint"
vocab_save_path="data/vocab.json"
generated_sentences_save_path="output/generated_sentences.txt"