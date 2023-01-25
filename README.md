# iHS #

*repository under construction*

iHS (short for illegal hate speech) is a dataset of German Twitter messages annotated for potentially illegal hate speech and mentions of identity terms.
 the full dataset upon motivated request.

We describe the annotation guidelines in our [previous work](https://dtct.eu/wp-content/uploads/2021/10/DTCT-TR3-CL.pdf) ([Bibtex](https://johannes-schaefer.github.io/files/JS-KB_techrep2021.txt)) and provide them [online (in German)](https://dtct.eu/wp-content/uploads/2021/10/DTCT-TR3-CL.pdf).

This repository provides the anonymized dataset and supplementary material:

    data/iHS-corpus_anonymized.xml  -- the iHS dataset (anonymized by only providing the Tweet-ID for instances and masked username mentions)
    keywords  -- list of keywords used to query Twitter data
    src/ -- source code for the experiments described in the paper
    logs/ -- log files of the conducted experiments 
