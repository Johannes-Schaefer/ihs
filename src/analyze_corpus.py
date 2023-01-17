import pathlib
from collections import defaultdict
import xml.etree.ElementTree as ET

IHS_LABELS = ('A', 'V', 'N', 'B')


def count_labels(ds_path):
    t = ET.parse(ds_path).getroot()
    labels = defaultdict(int)
    for tw in t:
        labels[tw.attrib["ihs_label"]] += 1
    print('\nLabel distribution in the dataset:')
    for label, freq in labels.items():
        print(f'{label}\t{freq}')


def count_idts_overall(ds_path):
    t = ET.parse(ds_path).getroot()
    idt_freqs = defaultdict(int)
    for tw in t:
        for el in tw:
            if el.tag == 'identity_terms':
                for idt in el:
                    idt.text = idt.text.strip()
                    idt_freqs[idt.text] += 1
    print(f'\nOverall number of annotated identity terms: {sum(idt_freqs.values())}')
    print(f'Number of annotated @-marked username identity terms:'
          f' {sum([idtf for idt, idtf in idt_freqs.items() if idt.startswith("@")])}')


def count_top_idts(ds_path, min_freq=11):
    t = ET.parse(ds_path).getroot()
    idt_freqs = defaultdict(int)
    hs_idt = defaultdict(int)
    for tw in t:
        for el in tw:
            if el.tag == 'identity_terms':
                tw_idts = []
                for idt in el:
                    idt.text = idt.text.strip()
                    # ignore duplicate entries per tweet
                    if idt.text in tw_idts:
                        continue
                    tw_idts.append(idt.text)
                    if tw.attrib["ihs_label"] in IHS_LABELS:
                        hs_idt[idt.text] += 1
                    idt_freqs[idt.text] += 1
    si = sorted(idt_freqs.items(), key=lambda x: x[1], reverse=True)
    print('\nIDT | Frequency | %HS')
    for el in si:
        freq = el[1]
        if freq >= min_freq:
            idt = el[0]
            hs_percentage = 100*hs_idt[el[0]]/el[1]
            print(f'{idt:15}\t{freq:2}\t{hs_percentage:2.2f}')


def analyze_idt_per_label(ds_path):
    corpus = ET.parse(ds_path).getroot()
    label_idts = defaultdict(list)
    for tweet in corpus:
        label = tweet.attrib["ihs_label"].strip()
        tw_names = []
        for child in tweet:
            if child.tag == 'identity_terms':
                for name in child:
                    if name.text.strip():
                        name_str = name.text.strip()
                        tw_names.append(name_str)
        label_idts[label].append(len(tw_names))
    print('\nAvg. # idts per label:')
    for label in label_idts:
        fl = sum(label_idts[label]) / len(label_idts[label])
        print(f"{label} {fl:.2f}")
    fl = sum([sum(label_idts[label]) for label in label_idts]) / sum([len(label_idts[label]) for label in label_idts])
    print(f"Overall {fl:.2f}")


def analyze_keyword_distribution(ds_path, keywords, top_n=9):
    tree = ET.parse(ds_path)
    corpus = tree.getroot()
    kws = {}
    for tweet in corpus:
        label = tweet.attrib["ihs_label"].strip()
        tweet_text = ''
        for child in tweet:
            if child.tag == 'text':
                tweet_text = child.text.strip()
                continue
        for kw in keywords:
            if kw in tweet_text or kw.lower() in tweet_text:
                if kw not in kws:
                    kws[kw] = defaultdict(int)
                kws[kw][label] += 1
    s_kws = sorted(kws.keys(), key=lambda kw: sum([freq for _, freq in kws[kw].items()]), reverse=True)
    res = []
    for i in range(top_n):
        if i >= len(s_kws):
            continue
        kw = s_kws[i]
        f_kw = sum([freq for _, freq in kws[kw].items()])
        p_ihs = sum([freq for label, freq in kws[kw].items() if label in IHS_LABELS]) / f_kw
        res.append((kw, f_kw, p_ihs))
    print('\nKeyword | Frequency | %HS')
    for kw in res:
        print(f"{kw[0]:15}\t{kw[1]}\t{100*kw[2]:5.1f}")


def analyze_idt_keywords_distribution(ds_path, keywords, top_n=20):
    tree = ET.parse(ds_path)
    corpus = tree.getroot()
    idts = {}
    for tweet in corpus:
        label = tweet.attrib["ihs_label"].strip()
        tw_names = []
        for child in tweet:
            if child.tag == 'identity_terms':
                for name in child:
                    if name.text.strip():
                        name_str = name.text.strip()
                        if name_str in tw_names:
                            # ignore name duplicates in one tweet
                            continue
                        tw_names.append(name_str)
                        if name_str not in idts:
                            idts[name_str] = defaultdict(int)
                        idts[name_str][label] += 1
    s_idts = sorted(idts.keys(), key=lambda idt: sum([freq for _, freq in idts[idt].items()]), reverse=True)
    kws = {}
    for tweet in corpus:
        label = tweet.attrib["ihs_label"].strip()
        tweet_text = ''
        for child in tweet:
            if child.tag == 'text':
                tweet_text = child.text.strip()
                continue
        for kw in keywords:
            if kw in tweet_text or kw.lower() in tweet_text:
                if kw not in kws:
                    kws[kw] = defaultdict(int)
                kws[kw][label] += 1
    print('\nIdTs also used as keywords:')
    freq_total = 0
    for i in range(top_n):
        idt = s_idts[i]
        if idt in kws or idt.lower() in kws or idt[0].upper() + idt[1:] in kws:
            f_idt = sum([freq for _, freq in idts[idt].items()])
            print(f"{idt:15}\t{f_idt}")
            freq_total += f_idt
    print(f"Sum:\t{freq_total}")


if __name__ == '__main__':
    keywords_path = pathlib.Path(__file__).absolute().parent.parent / 'data' / 'keywords'
    keywords_ihs = [kw.strip() for kw in open(keywords_path).readlines() if kw.strip()]

    # # analyze iHS corpus
    # ihs_ds_path = pathlib.Path(__file__).absolute().parent.parent / 'data' / 'iHS-corpus.xml'
    # count_labels(ihs_ds_path)
    # count_idts_overall(ihs_ds_path)
    # count_top_idts(ihs_ds_path)
    # analyze_idt_per_label(ihs_ds_path)
    # analyze_keyword_distribution(ihs_ds_path, keywords_ihs)
    # analyze_idt_keywords_distribution(ihs_ds_path, keywords_ihs)

    # analyze anonymized iHS corpus
    ihs_ds_path = pathlib.Path(__file__).absolute().parent.parent / 'data' / 'iHS-corpus_anonymized.xml'
    count_labels(ihs_ds_path)
    count_idts_overall(ihs_ds_path)
    count_top_idts(ihs_ds_path)
    analyze_idt_per_label(ihs_ds_path)

