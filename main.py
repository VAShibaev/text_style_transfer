# -*- coding: utf-8 -*-

# Copyright 2019 "Style Transfer for Texts: to Err is Human, but Error Margins Matter" Authors. All Rights Reserved.
#
# It's a modified code from
# Toward Controlled Generation of Text, ICML2017
# Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing
# https://github.com/asyml/texar/tree/master/examples/text_style_transfer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Text style transfer

This is a implementation of:

Style Transfer for Texts: to Err is Human, but Error Margins Matter, EMNLP-2019
Alexey Tikhonov, Viacheslav Shibaev, Aleksander Nagaev, Aigul Nugmanova and Ivan P. Yamshchikov

Download the data with the cmd:

$ python prepare_data.py

Train the model with the cmd:

$ python main.py --config config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, no-member

import os
import importlib
import numpy as np
import tensorflow as tf
import texar as tx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from ctrl_gen_model import CtrlGenModel

flags = tf.flags

flags.DEFINE_string('config', 'config', 'The config to use.')

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

def _main(_):
    # Data
    train_data = tx.data.MultiAlignedData(config.train_data)
    val_data = tx.data.MultiAlignedData(config.val_data)
    test_data = tx.data.MultiAlignedData(config.test_data)
    if config.manual:
        manual_data = tx.data.MultiAlignedData(config.manual_data)
    vocab = train_data.vocab(0)

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    if config.manual:
        iterator = tx.data.FeedableDataIterator(
            {'train_g': train_data, 'train_d': train_data, 'train_z': train_data,
             'val': val_data, 'test': test_data, 'manual': manual_data})
    else:
        iterator = tx.data.FeedableDataIterator(
            {'train_g': train_data, 'train_d': train_data, 'train_z': train_data,
             'val': val_data, 'test': test_data})
    batch = iterator.get_next()

    # Model
    gamma = tf.placeholder(dtype=tf.float32, shape=[], name='gamma')
    lambda_g = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_g')
    lambda_z = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_z')
    lambda_z1 = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_z1')
    lambda_z2 = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_z2')
    lambda_ae = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_ae')
    model = CtrlGenModel(batch, vocab, gamma, lambda_g, lambda_z, lambda_z1, lambda_z2, lambda_ae, config.model)

    def _train_epoch(sess, gamma_, lambda_g_, lambda_z_, lambda_z1_, lambda_z2_, lambda_ae_, epoch, verbose=True):
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        avg_meters_z = tx.utils.AverageRecorder(size=10)

        step = 0
        while True:
            try:
                step += 1
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train_d'),
                    gamma: gamma_,
                    lambda_g: lambda_g_,
                    lambda_z: lambda_z_,
                    lambda_z1: lambda_z1_,
                    lambda_z2: lambda_z2_,
                    lambda_ae: lambda_ae_
                }

                vals_d = sess.run(model.fetches_train_d, feed_dict=feed_dict)
                avg_meters_d.add(vals_d)

                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train_g'),
                    gamma: gamma_,
                    lambda_g: lambda_g_,
                    lambda_z: lambda_z_,
                    lambda_z1: lambda_z1_,
                    lambda_z2: lambda_z2_,
                    lambda_ae: lambda_ae_
                }
                vals_g = sess.run(model.fetches_train_g, feed_dict=feed_dict)
                avg_meters_g.add(vals_g)

                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train_z'),
                    gamma: gamma_,
                    lambda_g: lambda_g_,
                    lambda_z: lambda_z_,
                    lambda_z1: lambda_z1_,
                    lambda_z2: lambda_z2_,
                    lambda_ae: lambda_ae_
                }
                vals_z = sess.run(model.fetches_train_z, feed_dict=feed_dict)
                avg_meters_z.add(vals_z)


                if verbose and (step == 1 or step % config.display == 0):
                    print('epoch: {}, step: {}, {}'.format(epoch, step, avg_meters_d.to_str(4)))
                    print('epoch: {}, step: {}, {}'.format(epoch, step, avg_meters_z.to_str(4)))
                    print('epoch: {}, step: {}, {}'.format(epoch, step, avg_meters_g.to_str(4)))

                if verbose and step % config.display_eval == 0:
                    iterator.restart_dataset(sess, 'val')
                    _eval_epoch(sess, gamma_, lambda_g_, lambda_z_, lambda_z1_, lambda_z2_, lambda_ae_, epoch)

            except tf.errors.OutOfRangeError:
                print('epoch: {}, {}'.format(epoch, avg_meters_d.to_str(4)))
                print('epoch: {}, {}'.format(epoch, avg_meters_z.to_str(4)))
                print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
                break



    def _eval_epoch(sess, gamma_, lambda_g_, lambda_z_, lambda_z1_,lambda_z2_, lambda_ae_, epoch, val_or_test='val',
                    plot_z=False, plot_max_count=1000, spam=False, repetitions=False, write_text=True,
                    write_labels=False):
        avg_meters = tx.utils.AverageRecorder()

        if plot_z:
            z_vectors = []
            labels = []
            tsne = TSNE(n_components=2)
        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, val_or_test),
                    gamma: gamma_,
                    lambda_g: lambda_g_,
                    lambda_z: lambda_z_,
                    lambda_z1: lambda_z1_,
                    lambda_z2: lambda_z2_,
                    lambda_ae: lambda_ae_,
                    tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                }

                vals = sess.run(model.fetches_eval, feed_dict=feed_dict)

                batch_size = vals.pop('batch_size')

                # Computes BLEU
                samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)

                refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
                refs = np.expand_dims(refs, axis=1)

                bleu = tx.evals.corpus_bleu_moses(refs, hyps)
                vals['bleu'] = bleu

                if spam or repetitions:
                    target_labels = samples['labels_target']
                    predicted_labels = samples['labels_predicted']

                    results = [(r, h, t, p) for r, h, t, p in zip(refs, hyps, target_labels, predicted_labels)]

                # Computes repetitions
                if repetitions:
                    count_equal_strings = 0
                    remain_samples_e = []
                    for r, h, t, p in results:
                        if r == h:
                            count_equal_strings += 1
                        else:
                            remain_samples_e.append((r, h, t, p))
                    vals['equal'] = count_equal_strings / len(hyps)

                # Computes spam
                if spam:
                    count_spam = 0
                    remain_samples_s = []
                    for r, h, t, p in results:
                        words = h.split()
                        if len(words) > 2 and words[-1] == words[-2]:
                            count_spam += 1
                        elif len(words) > 4 and words[-1] == words[-3] and words[-2] == words[-4]:
                            count_spam += 1
                        else:
                            remain_samples_s.append((r, h, t, p))
                    vals['spam'] = count_spam / len(hyps)

                if repetitions and spam:
                    remain_samples = [semple for semple in remain_samples_e if semple in remain_samples_s]
                    remain_samples = list(remain_samples)
                elif not repetitions and spam:
                    remain_samples = remain_samples_s
                elif repetitions and not spam:
                    remain_samples = remain_samples_e

                if repetitions and spam:
                    refs_remain = [r for r, h, t, p in remain_samples]
                    hyps_remain = [h for r, h, t, p in remain_samples]
                    bleu_remain = tx.evals.corpus_bleu_moses(refs_remain, hyps_remain)
                    vals['bleu_remain'] = bleu_remain

                    if len(remain_samples) != 0:
                        true_labels = 0
                        for _, _, t, p in remain_samples:
                            if t == p:
                                true_labels += 1
                        vals['acc_remain'] = true_labels / len(remain_samples)
                    else:
                        vals['acc_remain'] = 0.

                avg_meters.add(vals, weight=batch_size)

                if plot_z:
                    z_vectors += samples['z_vector'].tolist()
                    labels += samples['labels_source'].tolist()

                # Writes samples
                if write_text:
                    tx.utils.write_paired_text(
                        refs.squeeze(), hyps,
                        os.path.join(config.sample_path, 'text_{}.{}'.format(val_or_test, epoch)),
                        append=True, mode='v')

                # Writes labels samples
                if write_labels:
                    tx.utils.write_paired_text(
                        [str(l) for l in samples['labels_target'].tolist()],
                        [str(l) for l in samples['labels_predicted'].tolist()],
                        os.path.join(config.sample_path, 'labels_{}.{}'.format(val_or_test, epoch)),
                        append=True, mode='v')

            except tf.errors.OutOfRangeError:
                print('epoch: {}, {}: {}'.format(
                    epoch, val_or_test, avg_meters.to_str(precision=4)))
                break

        if plot_z:
            if plot_max_count == 0:
                z_vectors = z_vectors
                labels = labels
            else:
                z_vectors = z_vectors[:plot_max_count]
                labels = labels[:plot_max_count]
            tsne_result = tsne.fit_transform(np.array(z_vectors))
            x_data = tsne_result[:, 0]
            y_data = tsne_result[:, 1]
            plt.scatter(x_data, y_data, c=np.array(labels), s=1, cmap=plt.cm.get_cmap('jet', 2))
            plt.clim(0.0, 1.0)
            if not os.path.exists('./images'):
                os.makedirs('./images')
            plt.savefig('./images/{}_{}.png'.format(val_or_test, epoch))
            plt.clf()

        return avg_meters.avg()

    tf.gfile.MakeDirs(config.sample_path)
    tf.gfile.MakeDirs(config.checkpoint_path)

    # Runs the logics
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        if config.restore:
            print('Restore from: {}'.format(config.restore))
            saver.restore(sess, config.restore)

        iterator.initialize_dataset(sess)

        gamma_ = 1.
        lambda_g_ = 0.
        lambda_z_ = 0.
        lambda_ae_ = 1.
        lambda_z1_ = config.lambda_z1
        lambda_z2_ = config.lambda_z2
        for epoch in range(1, config.max_nepochs+1):
            if epoch > config.pretrain_ae_nepochs:
                # Anneals the gumbel-softmax temperature
                gamma_ = max(0.001, gamma_ * config.gamma_decay)
                lambda_g_ = config.lambda_g
                lambda_z_ = config.lambda_z
            if epoch > config.chage_lambda_ae_epoch:
                lambda_ae_ = lambda_ae_ - config.change_lambda_ae
            print('gamma: {}, lambda_g: {}, lambda_z: {}, lambda_z1: {}, lambda_z2: {}, lambda_ae: {}'.format(
                gamma_, lambda_g_, lambda_z_, lambda_z1_, lambda_z2_, lambda_ae_))

            # Train
            iterator.restart_dataset(sess, ['train_g', 'train_d', 'train_z'])
            _train_epoch(sess, gamma_, lambda_g_, lambda_z_, lambda_z1_, lambda_z2_, lambda_ae_, epoch)

            # Val
            iterator.restart_dataset(sess, 'val')
            _eval_epoch(sess, gamma_, lambda_g_, lambda_z_, lambda_z1_, lambda_z2_, lambda_ae_, epoch, 'val', plot_z=config.plot_z,
                        plot_max_count=config.plot_max_count, spam=config.spam, repetitions=config.repetitions,
                        write_text=config.write_text, write_labels=config.write_labels)

            saver.save(
                sess, os.path.join(config.checkpoint_path, 'ckpt'), epoch)

            # Test
            iterator.restart_dataset(sess, 'test')
            _eval_epoch(sess, gamma_, lambda_g_, lambda_z_, lambda_z1_, lambda_z2_, lambda_ae_, epoch, 'test', plot_z=config.plot_z,
                        plot_max_count=config.plot_max_count, spam=config.spam, repetitions=config.repetitions,
                        write_text=config.write_text, write_labels=config.write_labels)

            if config.manual:
                iterator.restart_dataset(sess, 'manual')
                _eval_epoch(sess, gamma_, lambda_g_, lambda_z_, lambda_z1_, lambda_z2_, lambda_ae_, epoch, 'manual', plot_z=config.plot_z,
                        plot_max_count=config.plot_max_count, spam=config.spam, repetitions=config.repetitions,
                        write_text=config.write_text, write_labels=config.write_labels)

if __name__ == '__main__':
    tf.app.run(main=_main)
