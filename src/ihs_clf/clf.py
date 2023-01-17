import numpy as np
import torch
import torch.nn as nn
import transformers
from urllib.error import HTTPError
from time import sleep
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from datetime import datetime


class GBERTTokenizer(object):

    def __init__(self, model_name):
        tokenizer_loaded = False
        while not tokenizer_loaded:
            try:
                self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)
                tokenizer_loaded = True
            except HTTPError:
                print('HTTPError when loading loading tokenizer, sleeping for 10 minutes and trying again...')
                sleep(60 * 10)


class IHSPredictorBERT(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.1, bert_model_name='deepset/gbert-base', out_size=6,
                 clf_linear_size=300):
        super(IHSPredictorBERT, self).__init__()
        self.input_size = input_size
        self.bert_model_name = bert_model_name
        bert_model_loaded = False
        while not bert_model_loaded:
            try:
                self.bert = transformers.AutoModel.from_pretrained(bert_model_name)
                bert_model_loaded = True
            except HTTPError:
                print('HTTPError when loading BERT model, sleeping for 10 minutes and trying again...')
                sleep(60*10)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.ihs_linear1 = nn.Linear(hidden_size, clf_linear_size)
        self.ihs_linear1_activation = nn.ReLU()
        self.ihs_linear2 = nn.Linear(clf_linear_size, out_size)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        _, encoding = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=False)
        encoding = self.dropout_layer(encoding)
        out_ihs = self.ihs_linear1(encoding)
        out_ihs = self.ihs_linear1_activation(out_ihs)
        out_ihs = self.ihs_linear2(out_ihs)
        return out_ihs


def apply_model(texts, model, device):
    mask = texts['attention_mask'].to(device)
    input_id = texts['input_ids'].squeeze(1).to(device)
    output = model(input_id, mask)
    return output


def get_class_weights(labels):
    y = torch.as_tensor(labels)
    unique_labels = np.unique(y)
    task_class_weights = class_weight.compute_class_weight('balanced', classes=unique_labels, y=y.numpy())
    task_class_weights = torch.tensor(task_class_weights, dtype=torch.float)
    print(f'using class weights: {str(task_class_weights)}')
    return task_class_weights


def get_loss(output, labels, device, criterion):
    labels = labels.to(device)
    loss = criterion(torch.squeeze(output), torch.squeeze(labels))
    return loss


def eval_pred(pred, gold, labels=(0, 1, 2, 3, 4, 5)):
    y_pred = torch.as_tensor([torch.argmax(instance_pred) for instance_pred in torch.cat(pred)])
    y_true = torch.cat(gold)
    acc = torch.sum(y_pred == y_true)/len(y_true)
    class_f1_scores = f1_score(y_true, y_pred, labels=labels, average=None)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro')
    return acc, class_f1_scores, macro_f1


def print_eval(mode, acc, class_f1_scores, macro_f1, loss=None, epoch_num=None, log_file=None):
    out = f'Epochs: {epoch_num + 1} | ' if epoch_num is not None else ''
    out += f'{mode} Loss: {loss: .5f} | ' if loss is not None else ''
    out += f'{mode} acc: {acc: .3f} | {mode} macro_f1: {macro_f1: .3f}'
    class_f1_out = [f'{score: .3f}' for score in class_f1_scores]
    out += f' | {mode} f1 scores of classes: {class_f1_out}'
    print(out)
    if log_file is not None:
        with open(log_file, mode='a') as logfile:
            logfile.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ': ' + out + '\n')


def train(model, train_data, learning_rate, epochs, batch_size, criterion, labels=(0, 1, 2, 3, 4, 5), log_file=None):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    model.train()
    for epoch_num in range(epochs):
        total_pred_train = []
        total_label_train = []
        total_loss_train = 0.
        for train_input, train_label in train_dataloader:
            output = apply_model(train_input, model, device)
            total_pred_train.append(output)
            total_label_train.append(train_label)
            batch_loss = get_loss(output, train_label, device, criterion)
            total_loss_train += batch_loss.item()
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        if log_file is not None:
            acc, class_f1_scores, macro_f1 = eval_pred(total_pred_train, total_label_train, labels=labels)
            print_eval('Train',
                       acc,
                       class_f1_scores,
                       macro_f1,
                       loss=batch_size * total_loss_train / len(train_data.texts),
                       epoch_num=epoch_num,
                       log_file=log_file)


def evaluate(model, test_data, batch_size, criterion=None, labels=(0, 1, 2, 3, 4, 5), log_file=None):
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    model.eval()
    total_pred_test = []
    total_label_test = []
    total_loss_test = 0.
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            output = apply_model(test_input, model, device)
            if criterion is not None:
                batch_loss = get_loss(output, test_label, device, criterion)
                total_loss_test += batch_loss.item()
            total_pred_test.append(output)
            total_label_test.append(test_label)
    acc, class_f1_scores, macro_f1 = eval_pred(total_pred_test, total_label_test, labels=labels)
    print_eval('Test', acc, class_f1_scores, macro_f1, loss=batch_size * total_loss_test / len(test_data.texts),
               log_file=log_file)


def prepare_model(model_name, dropout, input_size, hidden_size, out_size=6):
    model = IHSPredictorBERT(input_size, hidden_size, dropout=dropout, bert_model_name=model_name, out_size=out_size)
    # freeze pre-trained BERT
    for param in model.bert.parameters():
        param.requires_grad = False
    return model


def prepare_tokenizer(model_name):
    model_tokenizer = GBERTTokenizer(model_name)
    return model_tokenizer


def do_clf(model, num_epochs, learning_rate, train_dataset, test_dataset, class_weights, log_file=None,
           combine_ihs_labels=False):
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    labels = (0, 1, 2) if combine_ihs_labels else (0, 1, 2, 3, 4, 5)
    train(model, train_dataset, learning_rate, num_epochs, batch_size, criterion, labels=labels)
    evaluate(model, test_dataset, batch_size, criterion=criterion, labels=labels, log_file=log_file)
