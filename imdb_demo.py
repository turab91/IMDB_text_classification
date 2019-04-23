from imdbNet.imdb_utils import load_imdb, imdb_word_dic, pad_data
from imdbNet.dataGenerator import DataGenerator
from imdbNet.imdbModel import imdbModel
from imdbNet.train import Train
import tensorflow as tf


if __name__ == "__main__":
    # train parameter

    hparam = {
        'train_size': 200,
        'valid_size': 100,
        'test_size': 100,
        'batch_size': 32,
        'num_epochs': 5,
        'num_steps': 100,
        'hidden_size1': 16,
        'hidden_size2': 32,
        'output_size': 2,
        'keep_prob': 0.5,
        'vocab_size': 10000,
        'embedding_dim': 50,
        'max_grad_norm': 2
    }
    # load data
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_imdb(num_training=hparam['train_size'],
                                                                   num_validation=hparam['valid_size'],
                                                                   num_test=hparam['test_size'],
                                                                   num_words=hparam['vocab_size'])
    word2index, index2word = imdb_word_dic()

    # Pad data
    train_x, valid_x, test_x = pad_data(train_x, valid_x, test_x, word2index, max_len=hparam['num_steps'])
    print(f"\ntrain_x: {train_x.shape}  valid_x: {valid_x.shape} test_x: {test_x.shape}")
    print(f"\ntrain_y: {train_y.shape} valid_y: {valid_y.shape} test_y: {test_y.shape}")

    tf.reset_default_graph()
    # Define Models
    with tf.variable_scope("model", reuse=None):
        train_model = imdbModel("is_training", hparam['num_steps'], hparam['hidden_size1'],hparam['hidden_size2'],
                                hparam['output_size'], hparam['vocab_size'], hparam['output_size'],
                                hparam['max_grad_norm'], hparam['keep_prob'])

    with tf.variable_scope("model", reuse=True):
        valid_model = imdbModel("is_valid", hparam['num_steps'], hparam['hidden_size1'],hparam['hidden_size2'],
                                hparam['output_size'], hparam['vocab_size'], hparam['output_size'],
                                hparam['max_grad_norm'], keep_prob=1.0)

        test_model = imdbModel("is_testing", hparam['num_steps'], hparam['hidden_size1'],hparam['hidden_size2'],
                                hparam['output_size'], hparam['vocab_size'], hparam['output_size'],
                                hparam['max_grad_norm'], keep_prob=1.0)

    # Data Generators
    trainGenerator = DataGenerator(train_x, train_y, batch_size=hparam['batch_size'], shuffle=True)
    validGenerator = DataGenerator(valid_x, valid_y, batch_size=hparam['batch_size'], shuffle=False)
    testGenerator = DataGenerator(test_x, test_y, batch_size=hparam['batch_size'], shuffle=False)


    # Train Model
    with tf.Session() as sess:
        train = Train(session=sess, num_epochs=hparam['num_epochs'])
        train(trainGenerator, train_model,
              validGenerator, valid_model, verbose=False)

        print("Predictions: \n")
        train.predict(testGenerator, test_model)


