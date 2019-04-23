import tensorflow as tf
import numpy as np
import time


class Train:

    def __init__(self, session, num_epochs: int = 1):
        self.num_epochs = num_epochs
        self.session = session

    def __call__(self, train_dset, train_model,
            valid_daset, valid_model, verbose=False):

        self.session.run(tf.global_variables_initializer())
        for i in range(self.num_epochs):
            start = time.time()
            if valid_model:
                train_cost = self.run_one_epoch(train_model, train_model.train_op, train_dset, verbose)
                valid_cost = self.run_one_epoch(valid_model, tf.no_op(), valid_daset, verbose)
                print("epoch: {}/{}  train loss: {:.4f} valid loss: {:.4f}  time: {:.2f} secs".format(i + 1, self.num_epochs,
                                                                                                      train_cost, valid_cost,
                                                                                                      time.time() - start))
            else:
                train_cost = self.run_one_epoch(train_model, train_model.train_op, train_dset, verbose)
                print("epoch: {}/{}  train loss: {:.4f} time: {:.2f} secs".format(i + 1, self.num_epochs,
                                                                                  train_cost, time.time() - start))
    def predict(self, dset, model):

        num_correct, num_samples = 0, 0
        for x_batch, y_batch in dset:
            feed_dict = {model.input_data: x_batch, model.targets: y_batch}
            y_pred = self.session.run(model.predictions, feed_dict=feed_dict)
            num_samples += x_batch.shape[0]
            num_correct += (y_pred == y_batch).sum()
        acc = float(num_correct) / num_samples
        print("{} / {} correct {:.2f}".format(num_correct, num_samples, 100 * acc))



    def run_one_epoch(self, model, train_op, dset, verbose=False):
        costs = 0
        verbose_count = 1
        for step, (batch_x, batch_y) in enumerate(dset):
            feed_dict = {model.input_data: batch_x, model.targets: batch_y}
            _, loss = self.session.run([train_op, model.loss], feed_dict=feed_dict)
            costs += loss
            if verbose and step in np.linspace(0, dset.num_batch+1, 5, dtype=np.int32):
                print("{}/{} cost: {:.4f}".format(verbose_count, 5, costs / (step+1)))
                verbose_count += 1
        return costs / (step+1)



