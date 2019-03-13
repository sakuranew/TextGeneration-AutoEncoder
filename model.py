import math

import numpy as np
import tensorflow as tf

import config
import data_processor
import os
class Model:
    def __init__(self, text_seq, label, text_seq_len, word_index,inverse_word_index,
                 encoder_embedding_matrix, decoder_embedding_matrix,
                 val_text_seq,val_label,val_text_seq_len,test_text_seq,):
        self.text_seq = text_seq
        self.label = label
        self.text_seq_len = text_seq_len
        self.word_index = word_index
        self.inverse_word_index = inverse_word_index
        self.encoder_embedding_matrix = encoder_embedding_matrix
        self.decoder_embedding_matrix = decoder_embedding_matrix

        self.val_text_seq = val_text_seq
        self.val_text_seq_len = val_text_seq_len
        self.val_label = val_label

        self.test_text_seq = test_text_seq
        self.inference_ids=None
        self.final_seq_len=None

        self._loss = 0

        self.inference_mode = tf.placeholder(dtype=tf.bool, name="inference_mode")
        self.generation_mode = tf.placeholder(dtype=tf.bool, name="generation_mode")
        self.recurrent_state_keep_prob = tf.cond(
            pred=tf.math.logical_or(self.inference_mode, self.generation_mode),
            true_fn=lambda: 1.0,
            false_fn=lambda: config.recurrent_state_keep_prob)
        self.fully_connected_keep_prob = tf.cond(
            pred=tf.math.logical_or(self.inference_mode, self.generation_mode),
            true_fn=lambda: 1.0,
            false_fn=lambda: config.fully_connected_keep_prob)
        self.sequence_word_keep_prob = tf.cond(
            pred=tf.math.logical_or(self.inference_mode, self.generation_mode),
            true_fn=lambda: 1.0,
            false_fn=lambda: config.sequence_word_keep_prob)
        # input
        self.encoder_input_seq = tf.placeholder(tf.int32, [None, config.max_seq_len],
                                                name="encoder_input_seq")
        tf.logging.info("encoder_input_seq:{}".format(self.encoder_input_seq))
        self.encoder_input_seq_len = tf.placeholder(tf.int32, [None, 1], name="encoder_input_seq_len")
        tf.logging.info("encoder_input_seq_len:{}".format(self.encoder_input_seq_len))
        self.decoder_input_seq = tf.concat([tf.fill([self.encoder_input_seq.shape[0], 1], self.word_index[config.sos_token]),
                                            self.encoder_input_seq], axis=-1, name="decoder_input_seq")
        tf.logging.info("decoder_input_seq:{}".format(self.decoder_input_seq))
        self.decoder_input_seq_len = tf.placeholder(tf.int32, [None, 1], name="decoder_input_seq_len")
        tf.logging.info("decoder_input_seq_len:{}".format(self.decoder_input_seq_len))
        # target outputs

        # 不用在后面加eos，处理encoder输入的时候把不足最大长度的补eos，超出的直接截断而不是截断后添加eos，
        # 这样不会改变说这个句子从这里结束了，而是被截断，这样就看decoder的最大解码次数了。
        # self.decoder_output_seq = tf.concat([self.encoder_input_seq,
        #                                      tf.fill([config.batch_size, 1], self.word_index[config.eos_token]),
        #                                     ], axis=-1, name="decoder_output_seq")

        self.decoder_output_seq = self.encoder_input_seq
        tf.logging.info("decoder_output_seq:{}".format(self.decoder_output_seq))
        # target weight,decoder_input_len=encoder_input_len+1,shape should be same as decoder_input_len
        self.decoder_output_seq_mask = tf.sequence_mask(lengths=tf.add(self.encoder_input_seq_len, 1),
                                                        maxlen=config.max_seq_len,
                                                        dtype=tf.int32)
        tf.logging.info("decoder_output_seq_mask:{}".format(self.decoder_output_seq_mask))

        # embedding
        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            self.encoder_embedding = tf.get_variable(
                initializer=self.encoder_embedding_matrix,
                dtype=tf.float32, trainable=True, name="encoder_embeddings"
            )
            tf.logging.info("encoder_embedding:{}".format(self.encoder_embedding))
            self.decoder_embedding = tf.get_variable(
                initializer=self.decoder_embedding_matrix,
                dtype=tf.float32, trainable=True, name="decoder_embeddings"
            )
            tf.logging.info("decoder_embedding:{}".format(self.decoder_embedding))

            # encoder_embedded_sequence = tf.nn.dropout(
            #     x=tf.nn.embedding_lookup(params=encoder_embeddings, ids=self.input_sequence),
            #     keep_prob=self.sequence_word_keep_prob,
            #     name="encoder_embedded_sequence")
            self.encoder_input = tf.nn.embedding_lookup(params=self.encoder_embedding,
                                                        ids=self.encoder_input_seq,
                                                        name="encoder_input")
            tf.logging.info("encoder_input:{}".format(self.encoder_input))
            self.decoder_input = tf.nn.embedding_lookup(params=self.decoder_embedding,
                                                        ids=self.decoder_input_seq,
                                                        name="decoder_input")
            tf.logging.info("decoder_input:{}".format(self.decoder_input))

    def loss(self):
        return self._loss

    def validate(self,sess,cur_epoch,):
        tf.logger.info("Running Validation {}:".format(cur_epoch // config.validation_interval))
        val_batches = math.ceil(len(self.val_text_seq) / config.batch_size)
        tf.logging.info("Training - texts shape: {}; labels shape {}"
                        .format(self.val_text_seq.shape, self.val_text_seq.shape))
        val_loss=0.0
        for val_batch_number in range(val_batches):
            start_index = val_batch_number * config.batch_size
            end_index = min((val_batch_number + 1) * config.batch_size, len(self.val_text_seq))

            fetchs = [
                reconstruction_loss,
            ]
            [ reconstruction_loss] = sess.run(fetchs=fetchs,
                                                           feed_dict={
                                                               self.encoder_input_seq: self.val_text_seq[
                                                                   start_index, end_index],
                                                               self.encoder_input_seq_len:
                                                                   self.val_text_seq_len[
                                                                       start_index, end_index],
                                                               self.decoder_input_seq_len:
                                                                   self.val_text_seq_len[
                                                                       start_index, end_index] + 1,
                                                               self.inference_mode: False,
                                                               self.generation_mode: False,
                                                           })
            val_loss+=reconstruction_loss
        val_loss/=val_batches
        log_msg = "Validation : " \
                  "Reconstruction Loss: {:.4f}, \n"
        tf.logger.info(log_msg.format(
            val_loss,
        ))
    def test(self,sess,):
        tf.logger.info("Running Test :")
        test_batches = math.ceil(len(self.test_text_seq) / config.batch_size)
        tf.logging.info("Test - texts shape: {}; "
                        .format(self.test_text_seq.shape,) )
        test_generated_sequences=list()
        test_generated_sequences_len=list()
        for test_batch_number in range(test_batches):
            start_index = test_batch_number * config.batch_size
            end_index = min((test_batch_number + 1) * config.batch_size, len(self.test_text_seq))

            fetchs = [
                self.inference_ids,
                self.final_seq_len
            ]
            [ inference_ids,final_seq_len] = sess.run(fetchs=fetchs,
                                           feed_dict={
                                               self.encoder_input_seq: self.test_text_seq[
                                                   start_index, end_index],
                                               self.encoder_input_seq_len:
                                                   self.test_text_seq_len[
                                                       start_index, end_index],
                                               self.decoder_input_seq_len:
                                                   self.test_text_seq_len[
                                                       start_index, end_index] + 1,
                                               self.inference_mode: True,
                                               self.generation_mode: False,
                                           })
            test_generated_sequences.extend(inference_ids)
            test_generated_sequences_len.extend(final_seq_len)
        trimmed_raw_sequences = \
            [[index for index in sequence
              if index != config.predefined_word_index[config.eos_token]]
             # for sequence in [x[:(y - 1)] for (x, y) in zip(
             for sequence in [x[:y] for (x, y) in zip(
                self.test_text_seq, self.test_text_seq_len)]]
        raw_sentences = \
            [data_processor.generate_sentence_from_indices(x, self.inverse_word_index)
             for x in trimmed_raw_sequences]
        trimmed_generated_sequences = \
            [[index for index in sequence
              if index != config.predefined_word_index[config.eos_token]]
             # for sequence in [x[:(y - 1)] for (x, y) in zip(
             for sequence in [x[:y] for (x, y) in zip(
                test_generated_sequences, test_generated_sequences_len)]]
        generated_sentences = \
            [data_processor.generate_sentence_from_indices(x, self.inverse_word_index)
             for x in trimmed_generated_sequences]
        dir_name=os.path.dirname(config.generated_sentences_save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(config.generated_sentences_save_path,"w") as o:
            o.write("raw_sentence           generated_sentences\n")
            for a,b in zip(raw_sentences,generated_sentences):
                o.write(a+" ---> "+b+"\n")


    def build(self, sess,inference_mode=False, generation_mode=False):
        # encoder
        with tf.name_scope(name="encoder"):
            encoder_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(num_units=config.encoder_num_units,
                                       kernel_initializer=tf.initializers.truncated_normal,
                                       name="encoder_cell_fw"),
                input_keep_prob=config.recurrent_state_keep_prob,
                output_keep_prob=config.recurrent_state_keep_prob,
                state_keep_prob=config.recurrent_state_keep_prob
            )
            encoder_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(num_units=config.encoder_num_units,
                                       kernel_initializer=tf.initializers.truncated_normal,
                                       name="encoder_cell_bw"),
                input_keep_prob=config.recurrent_state_keep_prob,
                output_keep_prob=config.recurrent_state_keep_prob,
                state_keep_prob=config.recurrent_state_keep_prob
            )
            _, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                                cell_bw=encoder_cell_bw,
                                                                inputs=self.encoder_input,
                                                                sequence_length=self.encoder_input_seq_len,
                                                                dtype=tf.float32)
            tf.logging.info("encoder_states:{}".format(encoder_states))
            hidden_code = tf.concat(values=encoder_states, axis=1, name="hidden_code")
            tf.logging.info("hidden_code:{}".format(hidden_code))

            decoder_init_state = tf.layers.dense(inputs=hidden_code,
                                                 units=config.decoder_num_units,
                                                 activation=tf.nn.relu,
                                                 name="decoder_init_state")
            tf.logging.info("decoder_init_state:{}".format(decoder_init_state))

            # decoder
            with tf.name_scope(name="encoder"):

                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.GRUCell(
                    num_units=config.decoder_num_units,
                    kernel_initializer=tf.initializers.truncated_normal,
                    name="decoder_cell"
                ),
                    input_keep_prob=config.recurrent_state_keep_prob,
                    output_keep_prob=config.recurrent_state_keep_prob,
                    state_keep_prob=config.recurrent_state_keep_prob
                )
                if not self.inference_mode and not self.generation_mode:
                    with tf.name_scope("decoder_train"):
                        helper = tf.contrib.seq2seq.TrainingHelper(
                            input=self.decoder_input,
                            sequence_length=self.decoder_input_seq_len
                        )
                else:
                    with tf.name_scope("decoder_inference"):
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            embedding=self.decoder_embedding,
                            start_tokens=tf.tile(input=self.word_index[config.sos_token],
                                                 multiples=[config.batch_size]),
                            end_token=self.word_index[config.eos_token]
                        )
                projection_layer = tf.layers.Dense(units=config.vocab_size,
                                                   # activation=tf.nn.softmax,
                                                   use_bias=False)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                          helper=helper,
                                                          initial_state=decoder_init_state,
                                                          output_layer=projection_layer)
                final_decoder_outputs, final_decoder_state, self.final_seq_len = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, impute_finished=True, maximum_iterations=config.max_seq_len,
                )
            training_output, self.inference_ids = final_decoder_outputs.rnn_output, final_decoder_outputs.sample_id
            tf.logging.info("training_output:{}".format(training_output))
            tf.logging.info("inference_ids:{}".format(self.inference_ids))
            tf.logging.info("final_seq_len:{}".format(self.final_seq_len))

        # reconstruction loss
        with tf.name_scope('reconstruction_loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_output_seq,
                                                                           logits=training_output)
            reconstruction_loss = tf.reduce_mean(cross_entropy * self.decoder_output_seq_mask)
            tf.logging.info("reconstruction_loss:{}".format(reconstruction_loss))

        # with tf.name_scope('reconstruction_loss'):
        #     batch_maxlen = tf.reduce_max(self.sequence_lengths)
        #     tf.logging.info("batch_maxlen: {}".format(batch_maxlen))
        #
        #     # the training decoder only emits outputs equal in time-steps to the
        #     # max time-steps in the current batch
        #     target_sequence = tf.slice(
        #         input_=self.input_sequence,
        #         begin=[0, 0],
        #         size=[config.batch_size, batch_maxlen],
        #         name="target_sequence")
        #     tf.logging.info("target_sequence: {}".format(target_sequence))
        #
        #     output_sequence_mask = tf.sequence_mask(
        #         lengths=tf.add(x=self.sequence_lengths, y=1),
        #         maxlen=batch_maxlen,
        #         dtype=tf.float32)
        #
        #     self.reconstruction_loss = tf.contrib.seq2seq.sequence_loss(
        #         logits=training_output, targets=target_sequence,
        #         weights=output_sequence_mask)
        #     tf.logging.info("reconstruction_loss: {}".format(self.reconstruction_loss))
        tf.summary.scalar(tensor=reconstruction_loss, name="reconstruction_loss_summary")
        all_summaries = tf.summary.merge_all()

        self._loss = reconstruction_loss

        # optimaization
        trainable_var = tf.trainable_variables()

        # gradients=tf.gradients(reconstruction_loss,trainable_var)
        # max_gradient_norm=1
        # clipped_gradients, _ = tf.clip_by_global_norm(
        #     gradients, max_gradient_norm)
        # optimazer=tf.train.AdamOptimizer(config.learning_rate)
        # update_step=optimazer.apply_gradients(zip(clipped_gradients,trainable_var))

        update_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss=reconstruction_loss,
                                                                            var_list=trainable_var)

        # train

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(logdir=config.log_directory, graph=sess.graph)

        num_batches = math.ceil(len(self.text_seq) / config.batch_size)
        tf.logging.info("Training - texts shape: {}; labels shape {}"
                        .format(self.text_seq.shape, self.text_seq.shape))
        for cur_epoch in range(1, config.epoch + 1):
            epoch_loss = 0.0
            shuffle_indices = np.random.permutation(np.arange(len(self.text_seq)))
            shuffled_text_seq = self.text_seq[shuffle_indices]
            # shuffled_label_seq=self.text_seq[shuffle_indices]
            shuffled_text_seq_len = self.text_seq_len[shuffle_indices]

            for cur_batch in range(num_batches):
                start_index = cur_batch * config.batch_size
                end_index = min((cur_batch + 1) * config.batch_size, len(self.text_seq))

                fetchs = [
                    update_step,
                    reconstruction_loss,
                    all_summaries
                ]
                [_, reconstruction_loss, all_summaries] = sess.run(fetchs=fetchs,
                                                                   feed_dict={
                                                                       self.encoder_input_seq: shuffled_text_seq[
                                                                           start_index, end_index],
                                                                       self.encoder_input_seq_len:
                                                                           shuffled_text_seq_len[
                                                                               start_index, end_index],
                                                                       self.decoder_input_seq_len:
                                                                           shuffled_text_seq_len[
                                                                               start_index, end_index] + 1,
                                                                       self.inference_mode: inference_mode,
                                                                       self.generation_mode: generation_mode,
                                                                   })
                log_msg = "Epoch {}-{} : " \
                          "Reconstruction Loss: {:.4f}, "
                tf.logger.debug(log_msg.format(
                    cur_epoch, cur_batch,
                    reconstruction_loss,
                ))
                epoch_loss += reconstruction_loss
            epoch_loss /= num_batches
            log_msg = "----------------------------------" \
                      "Epoch {} : " \
                      "Reconstruction Loss: {:.4f}, " \
                      "----------------------------------"
            tf.logger.info(log_msg.format(
                cur_epoch,
                epoch_loss,
            ))

            writer.add_summary(all_summaries, cur_epoch)
            writer.flush()
            # saver.save(sess=sess, save_path=config.model_save_path)
            if not cur_epoch % config.validation_interval:
                self.validate(sess,cur_epoch)
        writer.close()
