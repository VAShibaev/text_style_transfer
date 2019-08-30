# -*- coding: utf-8 -*-

# Copyright 2019 "Style Transfer for Texts: to Err is Human, but Error Margins Matter" Authors. All Rights Reserved.
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
Prepare manual dataset compiled by Li et al. for model evaluation.
"""

import re
import nltk
import pickle


INPUT_MANUAL_FILE_0 = './manual/reference.0'
INPUT_MANUAL_FILE_1 = './manual/reference.1'

LABELS_FILE = './data/yelp/sentiment.manual.labels'
TEXT_FILE = './data/yelp/sentiment.manual.text'
REFERENCE_FILE = './data/yelp/sentiment.manual.reference'
SOURCE_TARGET_DICT = './data/yelp/dict.pkl'

def clean_text(string):
    string = re.sub(r'\d+', " _num_ ", string)
    string = string.strip().lower()
    string = ' '.join(nltk.word_tokenize(string))
    return string

source_target_0 = []
with open(INPUT_MANUAL_FILE_0, 'r') as input_file:
    for line in input_file.readlines():
        source_sentence, target_sentence = line.strip().split('\t')
        source_target_0.append((source_sentence, target_sentence, 0))

source_target_1 = []
with open(INPUT_MANUAL_FILE_1, 'r') as input_file:
    for line in input_file.readlines():
        source_sentence, target_sentence = line.strip().split('\t')
        source_target_1.append((source_sentence, target_sentence, 1))

source_target = source_target_0 + source_target_1

source_target_dict = {}
with open(LABELS_FILE, 'w') as labels_file, \
    open(TEXT_FILE, 'w') as text_file, \
    open(REFERENCE_FILE, 'w') as reference_file:
    first = True
    for s, t, l in source_target:
        if first:
            labels_file.write(str(l))
            text_file.write(clean_text(s))
            reference_file.write(clean_text(t))
            first = False
        else:
            labels_file.write('\n' + str(l))
            text_file.write('\n' + clean_text(s))
            reference_file.write('\n' + clean_text(t))
        source_target_dict.update({clean_text(s): clean_text(t)})

with open(SOURCE_TARGET_DICT, 'wb') as pkl_file:
    pickle.dump(source_target_dict, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)





