import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import models.ltc_model as ltc
from models.ctrnn_model import CTRNN, NODE, CTGRU
from typing import Tuple, Iterable 



class TS_Data:
    
    def __init__(self, 
                 x: np.ndarray, 
                 y: np.ndarray, 
                 seq_len: int=32,
                 seq_gap: int=4, 
                 verbose: int=0, 
                 val_ratio: float=0.1,
                 test_ratio: float=0.15,
                 rand_state: int = 123):
        
        self.verbose = verbose
        self.seq_gap = seq_gap
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rand_state = rand_state
        
        # Get input sequences
        x_input_seq, y_input_seq = self.cut_in_sequences(x, y, seq_len, seq_gap, self.verbose)

        self.x_input_seq = np.stack(x_input_seq, axis=0)
        self.y_input_seq = np.stack(y_input_seq, axis=0)
        
        self.total_seqs = self.x_input_seq.shape[1]
        
        print("Total number of training sequences: {}".format(self.total_seqs))
        
        # Order of the sequences is lost, and therefore 
        # for TS forecast, data lekeage problem appears if indeces are shuffled.
        # TODO: fix it
        permutation = np.random.RandomState(rand_state).permutation(self.total_seqs)
        valid_size = int(self.val_ratio * self.total_seqs)
        test_size = int(self.test_ratio * self.total_seqs)

        self.valid_x = self.x_input_seq[:, permutation[:valid_size]]
        self.valid_y = self.y_input_seq[:, permutation[:valid_size]]
        self.test_x = self.x_input_seq[:, permutation[valid_size : valid_size + test_size]]
        self.test_y = self.y_input_seq[:, permutation[valid_size : valid_size + test_size]]
        self.x_input_seq = self.x_input_seq[:, permutation[valid_size + test_size :]]
        self.y_input_seq = self.y_input_seq[:, permutation[valid_size + test_size :]]
        

    @staticmethod
    def cut_in_sequences(x: np.ndarray, 
                         y: np.ndarray, 
                         seq_len: int, 
                         seq_gap: int=1, 
                         verbose: int=0) -> Tuple[np.ndarray, np.ndarray]:
        """Generates the input sequences for the model.
        Variable "seq_gap" corresponds to the gap for selecting
        the input sequences. E.g. if seq_gap=4, then the indices to 
        use for cutting the sequences will be:
        
            start: 0 end: 6
            start: 4 end: 10
            start: 8 end: 14
            start: 12 end: 18
            ...
        Output array is ()

        Args:
            x (np.ndarray): input features
            y (np.ndarray): target variable
            seq_len (int): sequence length for training
            seq_gap (int, optional): Gap for selecting the sequences
                    .Defaults to 1

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                x_input_seq -> size: (seq_len, x.shape[0]/step_gap, x.shape[1]) 
                y_input_seq -> size: (seq_len, y.shape[0]/step_gap)
                
                total_seqs = x.shape[0]/step_gap 
                num_features = x.shape[1]
        """
        sequences_x = []
        sequences_y = []

        for s in range(0, x.shape[0] - seq_len, seq_gap):
            start = s
            end = start + seq_len
            
            if verbose > 0:
                print("start:", s, "end:", end)
                
            sequences_x.append(x[start:end])
            sequences_y.append(y[start:end])

        return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)

        
    def iterate_train(self, 
                      batch_size:int = 16
                      ):
        """Iterable to generate the training batches.
        

        Args:
            batch_size (int, optional): Defaults to 16.

        Yields:
            Iterator[Iterable[np.ndarray, np.ndarray]]: 
                batch_x: (seq_len, batch_size, num_features)
                batch_y: (seq_len, batch_size)
            
        """
        permutation = np.random.permutation(self.total_seqs)
        total_batches = self.total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.x_input_seq[:, permutation[start:end]]
            batch_y = self.y_input_seq[:, permutation[start:end]]
            yield (batch_x, batch_y)


class TSModel:
    def __init__(self, 
                 model_type: str, 
                 model_size: int, 
                 learning_rate : float=0.001,
                 batch_size : int = 16):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 7])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.model_size = model_size
        head = self.x
        if model_type == "lstm":
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type.startswith("ltc"):
            learning_rate = 0.01  # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if model_type.endswith("_rk"):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif model_type.endswith("_ex"):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head, _ = tf.nn.dynamic_rnn(
                self.wm, head, dtype=tf.float32, time_major=True
            )
            self.constrain_op = self.wm.get_param_constrain_op()
        elif model_type == "node":
            self.fused_cell = NODE(model_size, cell_clip=10)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type == "ctgru":
            self.fused_cell = CTGRU(model_size, cell_clip=-1)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        elif model_type == "ctrnn":
            self.fused_cell = CTRNN(model_size, cell_clip=-1, global_feedback=True)
            head, _ = tf.nn.dynamic_rnn(
                self.fused_cell, head, dtype=tf.float32, time_major=True
            )
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        target_y = tf.expand_dims(self.target_y, axis=-1)
        self.y = tf.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(),
        )(head)
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(target_y - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.abs(target_y - self.y))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join(
            "results", "traffic", "{}_{}.csv".format(model_type, model_size)
        )
        if not os.path.exists("results/traffic"):
            os.makedirs("results/traffic")
        if not os.path.isfile(self.result_file):
            with open(self.result_file, "w") as f:
                f.write(
                    "best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n"
                )

        self.checkpoint_path = os.path.join(
            "tf_sessions", "traffic", "{}".format(model_type)
        )
        if not os.path.exists("tf_sessions/traffic"):
            os.makedirs("tf_sessions/traffic")

        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)

    def fit(self, gesture_data, epochs, verbose=True, log_period=50):

        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        self.save()
        for e in range(epochs):
            if verbose and e % log_period == 0:
                test_acc, test_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gesture_data.test_x, self.target_y: gesture_data.test_y},
                )
                valid_acc, valid_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: gesture_data.valid_x, self.target_y: gesture_data.valid_y},
                )
                # MSE metric -> less is better
                if (valid_loss < best_valid_loss and e > 0) or e == 1:
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x, batch_y in gesture_data.iterate_train(self.batch_size):
                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_step],
                    {self.x: batch_x, self.target_y: batch_y},
                )
                if not self.constrain_op is None:
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if verbose and e % log_period == 0:
                print(
                    "Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                )
            if e > 0 and (not np.isfinite(np.mean(losses))):
                break
        self.restore()
        (
            best_epoch,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            test_loss,
            test_acc,
        ) = best_valid_stats
        print(
            "Best epoch {:03d}, train loss: {:0.3f}, train mae: {:0.3f}, valid loss: {:0.3f}, valid mae: {:0.3f}, test loss: {:0.3f}, test mae: {:0.3f}".format(
                best_epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                test_loss,
                test_acc,
            )
        )
        with open(self.result_file, "a") as f:
            f.write(
                "{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
                    best_epoch,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    test_loss,
                    test_acc,
                )
            )

    def predict(self, gesture_data):
        losses = []
        accs = []
        preds = []
        for batch_x, batch_y in gesture_data.iterate_train(self.batch_size):
            acc, loss, y_hat = self.sess.run(
                [self.accuracy, self.loss, self.y],
                {self.x: batch_x, self.target_y: batch_y},
            )
            if not self.constrain_op is None:
                self.sess.run(self.constrain_op)

            losses.append(loss)
            accs.append(acc)
            preds.append(y_hat)
            
            #print("Input data ", batch_x)
            #print("y prediction ", y_hat)
        return preds     
    
