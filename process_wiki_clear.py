#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import string
import sys

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
    i = 0
    puntuation_trans = str.maketrans('　', '，',
                                     string.punctuation + string.digits + string.ascii_lowercase + string.ascii_uppercase
                                     + '“”？。…！·「」—（）《》')
    with open(inp, 'r', encoding="UTF-8") as inpf, open(outp, 'w', encoding="UTF-8") as output:
        for text in inpf:
            text = text.translate(puntuation_trans)
            txtarr = text.split()
            if len(txtarr) < 5:
                continue
            output.write(' '.join(txtarr) + "\n")
            i += 1
            if i % 10000 == 0:
                logger.info("Saved " + str(i) + " articles")
                output.flush()

    logger.info("Finished Saved " + str(i) + " articles")
