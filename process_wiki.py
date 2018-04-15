#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys

import jieba
from gensim.corpora import WikiCorpus
from hanziconv import HanziConv


def tokenizer_func(content, token_min_len=None, token_max_len=None, lower=True):
    """
    tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str
    :return:
    """
    stext = HanziConv.toSimplified(content)
    tokens = list(jieba.cut(stext))
    if len(tokens) < 5:
        return []
    if " " in tokens:
        tokens.remove(" ")
    if "=" in tokens:
        tokens.remove("=")
    if "." in tokens:
        tokens.remove(".")
    return tokens


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(sys.argv)
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    wiki = WikiCorpus(inp, lemmatize=False, dictionary={}, token_min_len=1, lower=False, tokenizer_func=tokenizer_func)
    with open(outp, 'w', encoding="UTF-8") as output:
        for text in wiki.get_texts():
            output.write(space.join(text) + "\n")
            i += 1
            if i % 10000 == 0:
                logger.info("Saved " + str(i) + " articles")
                output.flush()

    logger.info("Finished Saved " + str(i) + " articles")
