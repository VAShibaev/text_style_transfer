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

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import csv

TEXT_FILE = './samples/text_val.12'
LABELS_FILE = './samples/labels_val.12'


smoothing_foonction = SmoothingFunction()
semples = []
semples.append(['Input Sentence', 'Input Label', 'Generated Sentence', 'Predicted Label', 'BLEU'])
with open(TEXT_FILE, 'r') as input_file_text:
    lines_text = input_file_text.readlines()
    with open(LABELS_FILE, 'r') as input_file_labels:
        lines_labels = input_file_labels.readlines()
        for i in range(0, len(lines_text)-1, 2):
            input_sentence = lines_text[i].strip()
            generated_sentence = lines_text[i + 1].strip()
            input_label = 1 - int(lines_labels[i])
            predicted_label = int(lines_labels[i + 1])

            words_input = input_sentence.split()
            words_generated = generated_sentence.split()
            try:
                score = sentence_bleu([words_input], words_generated, smoothing_function=smoothing_foonction.method4)
            except Exception:
                print(input_sentence)
                print(generated_sentence)

            semples.append([input_sentence, input_label, generated_sentence, predicted_label, score])


with open('semples.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in semples:
        writer.writerow(line)
