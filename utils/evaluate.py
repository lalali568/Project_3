import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime

from utils.affiliation.generics import convert_vector_to_events
from utils.affiliation.metrics import pr_from_events
from utils.spot import SPOT


def getThreshold(init_score, test_score, q=1e-2):
    s = SPOT(q=q)
    s.fit(init_score, test_score)
    s.initialize(verbose=False)
    ret = s.run()
    threshold = np.mean(ret['thresholds'])

    return threshold


def getAffiliationMetrics(label, pred):
    events_pred = convert_vector_to_events(pred)
    events_label = convert_vector_to_events(label)
    Trange = (0, len(pred))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    F = 2 * P * R / (P + R)

    return P, R, F


def evaluate(init_score, test_score, test_label=None, q=1e-2):
    res = {
        # 'init_score': init_score,
        # 'test_score': test_score,
        # 'test_label': test_label,
        'q': q,
    }

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    res['time']=current_time
    threshold = getThreshold(init_score, test_score, q=q)
    test_pred = (test_score > threshold).astype(int)
    res['threshold'] = threshold


    if test_label is not None:
        precision, recall, f1_score = getAffiliationMetrics(test_label.copy(), test_pred.copy())
        res['precision'] = precision
        res['recall'] = recall
        res['f1_score'] = f1_score

        # AUC Calculation
        auc_score = roc_auc_score(test_label, test_score)
        res['auc'] = auc_score

        # PRAUC Calculation
        prauc_score = average_precision_score(test_label, test_score)
        res['prauc'] = prauc_score

    return res
