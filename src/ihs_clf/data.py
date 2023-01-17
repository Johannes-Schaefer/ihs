import xml.etree.ElementTree as ET
import random
import torch
from collections import Counter
import sklearn.model_selection

SEED = 36
IHS_LABEL_SET_6CLASS = {'Other': 0, 'OL': 1, 'B': 2, 'N': 3, 'A': 4, 'V': 5}
IHS_LABEL_SET_3CLASS = {'Other': 0, 'OL': 1, 'HS': 2}


def read_ihs_corpus(ds_path, mask_identity_terms=False, combine_ihs_labels=False):
    tree = ET.parse(ds_path)
    corpus = tree.getroot()
    data = []
    labels = []
    for tweet in corpus:
        label = tweet.attrib["ihs_label"].strip()
        if combine_ihs_labels and label in ('B', 'N', 'A', 'V'):
            label = 'HS'
        tweet_text = ''
        for child in tweet:
            if child.tag == 'text':
                tweet_text = child.text.strip()
                break
        if mask_identity_terms:
            for child in tweet:
                if child.tag == 'identity_terms':
                    for name in child:
                        if name.text.strip():
                            name_str = name.text.strip()
                            tweet_text = tweet_text.replace(name_str, '*Name*')
        data.append(tweet_text)
        labels.append(label)
    return data, labels


def prepare_train_test_ds(ds_path, ratio=(.8, .2), shuffle_data=True, mask_identity_terms=False,
                          stratified_sampling=False, combine_ihs_labels=False):
    if sum(ratio) != 1:
        raise ValueError('Error: DS split ratio sum not 1.')
    instances, labels = read_ihs_corpus(ds_path, mask_identity_terms=mask_identity_terms,
                                        combine_ihs_labels=combine_ihs_labels)
    if stratified_sampling:
        train_instances, test_instances, train_labels, test_labels = sklearn.model_selection.train_test_split(
            instances, labels, test_size=ratio[1], shuffle=True, random_state=SEED, stratify=labels)
    else:
        if shuffle_data:
            combined_data = list(zip(instances, labels))
            random.seed(SEED)
            random.shuffle(combined_data)
            instances, labels = zip(*combined_data)
        train_instances = instances[:int(round(len(instances) * ratio[0]))]
        train_labels = labels[:int(round(len(instances) * ratio[0]))]
        test_instances = instances[int(round(len(instances) * ratio[0])):]
        test_labels = labels[int(round(len(instances) * ratio[0])):]

    print('train label distr: ' + str(Counter(train_labels)))
    print('test label distr: ' + str(Counter(test_labels)))
    return train_instances, train_labels, test_instances, test_labels


class HSDataset(torch.utils.data.Dataset):

    def __init__(self, instances, labels, model_tokenizer, hs_label_set, max_len=100):
        self.labels = [hs_label_set[label] for label in labels]
        print('Tokenizing %d instances' % len(instances))
        self.texts = [model_tokenizer.tokenizer(text, padding='max_length', max_length=max_len, truncation=True,
                                                return_tensors="pt") for text in instances]
        print('done.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_labels = self.labels[idx]
        return batch_texts, batch_labels


def prepare_clf_datasets(data_path, tokenizer, max_len, mask_identity_terms=False, combine_ihs_labels=False,
                         print_train_percentile=False):
    ihs_label_set = IHS_LABEL_SET_3CLASS if combine_ihs_labels else IHS_LABEL_SET_6CLASS
    train_instances, train_labels, test_instances, test_labels = prepare_train_test_ds(
        data_path, shuffle_data=True, mask_identity_terms=mask_identity_terms, stratified_sampling=True,
        combine_ihs_labels=combine_ihs_labels)
    train_dataset = HSDataset(train_instances, train_labels, tokenizer, ihs_label_set, max_len=max_len)
    test_dataset = HSDataset(test_instances, test_labels, tokenizer, ihs_label_set, max_len=max_len)

    if print_train_percentile:
        nth_percentile = 0.99
        instance_lengths = [len([v for v in torch.squeeze(instance['attention_mask']) if v])
                            for instance in train_dataset.texts]
        sorted_lens = sorted(instance_lengths)
        print(str(sorted_lens[round(len(instance_lengths) * nth_percentile)]))

    return train_dataset, test_dataset
