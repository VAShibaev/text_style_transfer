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
Config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import copy

max_nepochs = 12 # Total number of training epochs
                 # (including pre-train and full-train)
pretrain_ae_nepochs = 10 # Number of pre-train epochs (training as autoencoder)
display = 500  # Display the training results every N training steps.
display_eval = 1e10 # Display the dev results every N training steps (set to a
                    # very large value to disable it).
sample_path = './samples'
checkpoint_path = './checkpoints'
restore = ''   # Model snapshot to restore from

lambda_g = 0.1    # Weight of the classification loss
lambda_z = 0.5
gamma_decay = 0.5   # Gumbel-softmax temperature anneal rate

change_lambda_ae = 0.
chage_lambda_ae_epoch = 12

plot_z = True
plot_max_count = 1000

spam = True
repetitions = True

write_text = True
write_labels = True

manual = True

train_data = {
    'batch_size': 64,
    # "max_dataset_size": 256,
    #'seed': 123,
    'datasets': [
        {
            'files': './data/yelp/sentiment.train.text',
            'vocab_file': './data/yelp/vocab',
            'data_name': ''
        },
        {
            'files': './data/yelp/sentiment.train.labels',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}

val_data = copy.deepcopy(train_data)
val_data['datasets'][0]['files'] = './data/yelp/sentiment.dev.text'
val_data['datasets'][1]['files'] = './data/yelp/sentiment.dev.labels'

test_data = copy.deepcopy(train_data)
test_data['datasets'][0]['files'] = './data/yelp/sentiment.test.text'
test_data['datasets'][1]['files'] = './data/yelp/sentiment.test.labels'

if manual:
    manual_data = copy.deepcopy(train_data)
    manual_data['datasets'][0]['files'] = './data/yelp/sentiment.manual.text'
    manual_data['datasets'][1]['files'] = './data/yelp/sentiment.manual.labels'

model = {
    'dim_c': 200,
    'dim_z': 500,
    'num_classes': 2,
    'embedder': {
        'dim': 100,
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': 21,
        'max_decoding_length_infer': 20,
    },
    'classifier': {
        'kernel_size': [3, 4, 5],
        'filters': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
    'z_classifier_l1': {
        'activation_fn': 'sigmoid'
    },
    'z_classifier_l2': {
        'activation_fn': 'sigmoid'
    }
}
