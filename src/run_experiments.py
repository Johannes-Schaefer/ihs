import pathlib
from datetime import datetime
from ihs_clf import data
from ihs_clf import clf


def optim_ihs_clf(params, mask_identity_terms=False, combine_ihs_labels=False):
    ihs_ds_path = pathlib.Path(__file__).absolute().parent.parent / 'data' / 'iHS-corpus.xml'
    log_file = pathlib.Path(__file__).absolute().parent.parent / 'logs' / \
        f'optim_ihs_clf_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.log'

    num_classes = 3 if combine_ihs_labels else 6
    tokenizer = clf.prepare_tokenizer(params['model_name'])
    train_dataset, test_dataset = data.prepare_clf_datasets(ihs_ds_path,
                                                            tokenizer,
                                                            params['input_size'],
                                                            mask_identity_terms=mask_identity_terms,
                                                            combine_ihs_labels=combine_ihs_labels)
    class_weights = clf.get_class_weights(train_dataset.labels)
    for dropout in params['hyperparameters']['dropout']:
        for num_epochs in params['hyperparameters']['num_epochs']:
            for learning_rate in params['hyperparameters']['learning_rate']:
                model = clf.prepare_model(params['model_name'],
                                          dropout,
                                          params['input_size'],
                                          params['hidden_size'],
                                          out_size=num_classes)
                with open(log_file, mode='a') as logfile:
                    logfile.write(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}: Starting run with config: '
                                  f'dropout={dropout}, num_epochs={num_epochs}, learning_rate={learning_rate}\n')
                clf.do_clf(model,
                           num_epochs,
                           learning_rate,
                           train_dataset,
                           test_dataset,
                           class_weights,
                           log_file=log_file,
                           combine_ihs_labels=combine_ihs_labels)


if __name__ == '__main__':
    # parameters
    exp_params = {
        'input_size': 100,  # determined as 99th percentile on data
        'model_name': 'deepset/gbert-base',  # https://huggingface.co/deepset/gbert-base
        'hidden_size': 768,
        'hyperparameters': {
            'dropout': (0.01, 0.1, 0.2, 0.3),
            'num_epochs': (3, 5, 7, 10),
            'learning_rate': (0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001)
        }
    }

    # fine-grained 6 label iHS experiments
    optim_ihs_clf(exp_params)
    optim_ihs_clf(exp_params, mask_identity_terms=True)
    #
    # # coarse-grained 3 label iHS experiments
    optim_ihs_clf(exp_params, combine_ihs_labels=True)
    optim_ihs_clf(exp_params, mask_identity_terms=True, combine_ihs_labels=True)
