'''
change January 31 2022
    - type parameter -> input_type
    - added clf_type: svm, logreg, mlp
    - removed proba
'''

import pandas as pd
import numpy as np
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from utils.training import preprocess_t

def classify_words(word_fragments, train_targets, val_targets, val_predictions, t_mean, t_std,
                                                                        input_type='default',
                                                                        clf_type = 'svm_linear', # svm_linear, svm_rbf, logreg, mlp
                                                                        return_proba=False): # for confusion martix plots

    # if 'svm' in clf_type:
    #     assert return_proba==False, 'should not be asked to return probability for svm'

    if 'svm' in clf_type:
        if '_' in clf_type:
            clf_type, kernel = clf_type.split('_')
        else:
            kernel = 'linear'
    else:
        kernel = None

    labels = pd.factorize(word_fragments['text'])[0]
    val_labels = labels[word_fragments[word_fragments['subset']=='validation'].index]
    train_labels = np.delete(labels, word_fragments[word_fragments['subset']=='validation'].index)

    # normalize and reshape
    x_train = preprocess_t(Tensor(train_targets), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
    x_val = preprocess_t(Tensor(val_targets), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
    pred_val = preprocess_t(Tensor(val_predictions), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()

    if input_type == 'default':
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        pred_val = x_val.reshape(pred_val.shape[0], -1)
    elif input_type == 'avg_time':
        x_train = np.mean(x_train, -1)
        x_val = np.mean(x_val, -1)
        pred_val = np.mean(pred_val, -1)
    elif input_type == 'avg_chan':
        x_train = np.mean(x_train, 1)
        x_val = np.mean(x_val, 1)
        pred_val = np.mean(pred_val, 1)

    # train classifier
    seed = np.random.randint(0, 1000, 1)[0]
    if clf_type == 'svm':
        clf = SVC(random_state=seed, C=1, kernel=kernel, probability=return_proba)
    elif clf_type == 'logreg':
        clf = LogisticRegression(random_state=seed, C=1, solver='saga')
    elif clf_type == 'mlp':
        clf = MLPClassifier(random_state=seed, hidden_layer_sizes=50)
    else:
        raise ValueError
    clf.fit(x_train, train_labels)

    # test on targets
    acc_targets = clf.score(x_val, val_labels)

    # test on predictions
    acc_predictions = clf.score(pred_val, val_labels)

    # prepare output
    out = {'input':['target_audio', 'reconstructed'],
           'target_label': [word_fragments.loc[labels==val_labels]['text'].values[0]] * 2,
           'target_label_id': list(val_labels) * 2,
           'predicted_label': [word_fragments.loc[labels==clf.predict(x_val)]['text'].values[0],
                                word_fragments.loc[labels == clf.predict(pred_val)]['text'].values[0]],
           'predicted_label_id': list(clf.predict(x_val)) + list(clf.predict(pred_val)),
           'accuracy': [acc_targets, acc_predictions]}

    if not return_proba:
        return pd.DataFrame(out)
    else:
        return pd.DataFrame(out), clf.predict_proba(pred_val), val_labels