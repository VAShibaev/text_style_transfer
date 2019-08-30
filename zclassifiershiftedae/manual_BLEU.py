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

import pickle
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

GENERATED_FILE = './samples/text_manual.1'
SOURCE_TARGET_DICT_PATH = './data/yelp/dict.pkl'

with open(SOURCE_TARGET_DICT_PATH, 'rb') as pkl_file:
    source_target_dict = pickle.load(pkl_file)

semples = []
with open(GENERATED_FILE, 'r') as input_file:
    lines = input_file.readlines()
    for i in range(0, len(lines), 2):
        source = lines[i].strip()
        generated = lines[i + 1].strip()

        source_clean = re.sub(r'<UNK> ', "", source)
        key = ''
        for target_str in source_target_dict.keys():
            word_count = 0
            for word1 in source_clean.split():
                if word1 in target_str.split():
                    word_count += 1
            if word_count == len(source_clean.split()):
                key = target_str
                break

        target = source_target_dict[key]
        semples.append((source, target, generated))

total_bleu = 0
smoothing_foonction = SmoothingFunction()
for _, t, g in semples:
    target_words = t.strip().split()
    generated_words = g.strip().split()
    score = sentence_bleu([target_words], generated_words, smoothing_function=smoothing_foonction.method4)
    total_bleu += score

print('BLEU: {:.4}'.format(total_bleu / len(semples)))