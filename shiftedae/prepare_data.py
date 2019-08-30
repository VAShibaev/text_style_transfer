# -*- coding: utf-8 -*-

# It's a code from
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

"""Downloads data.
"""
import texar as tx

# pylint: disable=invalid-name

def prepare_data():
    """Downloads data.
    """
    tx.data.maybe_download(
        urls='https://drive.google.com/file/d/'
             '1HaUKEYDBEk6GlJGmXwqYteB-4rS9q8Lg/view?usp=sharing',
        path='./',
        filenames='yelp.zip',
        extract=True)

def main():
    """Entrypoint.
    """
    prepare_data()

if __name__ == '__main__':
    main()
