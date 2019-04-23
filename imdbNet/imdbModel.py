import tensorflow as tf

class imdbModel:

    def __init__(self, action="is_training", num_steps=100, hidden_size1=64, hidden_size2=32,
                 output_size=2, vocab_size=10000, embedding_dim=50,
                 max_grad_norm=2, keep_prob=1.0):

        self.num_steps = num_steps
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm
        self.action = action

        # Placeholders
        self._input_data = tf.placeholder(tf.int32, shape=[None, self.num_steps])
        self._target = tf.placeholder(tf.int32, shape=[None])

        # Embedding
        with tf.variable_scope("embedding"):
            embedding_matrix = tf.get_variable("embedding_matrix", shape=[self.vocab_size, self.embedding_dim])
            self._embed_out = tf.nn.embedding_lookup(embedding_matrix, self._input_data)

        # Dropout
        if self.action == "is_training" and self.keep_prob < 1:
            tf.nn.dropout(self._embed_out, self.keep_prob)

        # RNN model using GRU
        with tf.variable_scope("multi-rnn"):
            cell1 = tf.nn.rnn_cell.GRUCell(self.hidden_size1)
            cell2 = tf.nn.rnn_cell.GRUCell(self.hidden_size2)

            stacked_cells = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2], state_is_tuple=True)
            # self._initial_state = stacked_cells.zero_state(batch_size=tf.shape(self._input_data)[0], dtype=tf.float32)
            self._outputs, self._state = tf.nn.dynamic_rnn(stacked_cells, self._embed_out, dtype=tf.float32)

        # Logits
        with tf.variable_scope("Logits"):
            self._logits = tf.layers.dense(inputs=self._state[1], units=self.output_size, activation=None)

        # Loss
        with tf.variable_scope("loss"):
            self._losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._target, logits=self._logits)
            # average over batch
            self._loss = tf.reduce_mean(self._losses)

        # prediction
        with tf.variable_scope('prediction'):
            self._predictions = tf.argmax(tf.nn.softmax(self._logits), axis=1)

        if self.action != "is_training":
            return

        # Optimizer
        with tf.variable_scope("train_op"):
            # get all trainable variables
            params = tf.trainable_variables()
            params_grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, params), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer()
            self._train_op = optimizer.apply_gradients(zip(params_grads, params))

    # Helper functions
    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._target

    @property
    def logits(self):
        return self._logits

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def predictions(self):
        return self._predictions









