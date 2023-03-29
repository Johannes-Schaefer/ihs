# iHS #

--- NOTE: repo still under construction

iHS (short for illegal hate speech) is a dataset of German Twitter messages annotated for potentially illegal hate speech and mentions of identity terms.
I am happy to provide the complete data to you, feel free to contact me with a motivated request.

We describe the annotation guidelines in our [previous work](https://dtct.eu/wp-content/uploads/2021/10/DTCT-TR3-CL.pdf) ([Bibtex](https://johannes-schaefer.github.io/files/JS-KB_techrep2021.txt)) and provide them [online (in German)](https://dtct.eu/wp-content/uploads/2021/10/DTCT-TR3-CL.pdf).

This repository provides the anonymized dataset and supplementary material:

    data/iHS-corpus_anonymized.xml  -- the iHS dataset (anonymized by only providing the Tweet-ID for instances and masked username mentions)
    keywords  -- list of keywords used to query Twitter data
    src/ -- source code for the experiments described in the paper
    logs/ -- log files of the conducted experiments 

# Citation # 

If you use the dataset, please cite the paper Schäfer, J. 2023. Bias Mitigation for Capturing Potentially Illegal Hate Speech. Datenbank Spektrum. https://doi.org/10.1007/s13222-023-00439-0

Bibtex:
```text
@article{schaefer2023bias,
  title={Bias Mitigation for Capturing Potentially Illegal Hate Speech},
  author={Sch{\"a}fer, Johannes},
  journal={Datenbank-Spektrum},
  pages={1610--1995},
  doi={10.1007/s13222-023-00439-0},
  year={2023}
}
```
