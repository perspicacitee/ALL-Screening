import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

SMOOTH = 1e-9


def save_roc_curve(summaywritter, y_label, y_score, epoch, tag="ROC_curve", exp_name=None):
    fpr, tpr, thresholds = roc_curve(y_label, y_score)
    auc = roc_auc_score(y_label, y_score)
    figure_name = f"ROC Curve, Auc: {auc}"

    fig, ax = plt.subplots()
    ax.set_title(figure_name)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], 'k--')
    summaywritter.add_figure(tag, fig, global_step=epoch, close=True)

    return auc


def plot_prob_dist(y_label, y_score):
    be_indice = np.where(y_label == 0)[0]
    ma_indice = np.where(y_label == 1)[0]
    be_probs = y_score[be_indice]
    ma_probs = y_score[ma_indice]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(be_probs, 50, range=(0, 1), alpha=0.5, color='blue', label="benign samples")
    ax.hist(ma_probs, 50, range=(0, 1), alpha=0.5, color='orange', label="malignant samples")
    ax.set_xlabel("malignant probability")
    ax.set_ylabel("sample count")
    ax.legend()

    return fig


def metrics(y_pred, y_label) -> tuple:
    """calculate the metrics measures
    
    Args:
        y_pred: model predicted label
        y_label: the ground truth
    Returns:
        A tuple contains this metrics
        
        (accuracy, precision, recall, f1_score)
    """
    SMOOTH = 1e-9
    y_pred, y_label = np.array(y_pred), np.array(y_label)
    tp = np.sum(y_pred * y_label)
    fp = np.sum(y_pred * (1 - y_label))
    fn = np.sum((1 - y_pred) * y_label)
    tn = np.sum((1 - y_pred) * (1 - y_label))
    acc = (tp + tn) / (tp + fp + fn + tn + SMOOTH)
    precision = tp / (tp + fp + SMOOTH)
    recall = tp / (tp + fn + SMOOTH)
    f1_score = 2 * tp / (2 * tp + fp + fn + SMOOTH)

    return acc, precision, recall, f1_score

@DeprecationWarning
def metrics_deprecate(y_score, y_label, thershold=0.5) -> tuple:
    """calculate different metric of model

    :returns: (accuracy, precision, recall, f1_score)
    """
    y_score, y_label = np.array(y_score), np.array(y_label)
    tp = np.sum((y_score >= thershold) * y_label)
    fp = np.sum((y_score > thershold) * (1 - y_label))
    fn = np.sum((y_score < thershold) * y_label)
    tn = np.sum((y_score <= thershold) * (1 - y_label))
    acc = (tp + tn) / (tp + fp + fn + tn + SMOOTH)
    precision = tp / (tp + fp + SMOOTH)
    recall = tp / (tp + fn + SMOOTH)
    f1_score = (2 * precision * recall) / (precision + recall + SMOOTH)

    return acc, precision, recall, f1_score


def f_beta(y_score, y_label, thershold, beta):
    y_score, y_label = np.array(y_score), np.array(y_label)
    tp = np.sum((y_score >= thershold) * y_label)
    fp = np.sum((y_score > thershold) * (1 - y_label))
    fn = np.sum((y_score < thershold) * y_label)
    precision = tp / (tp + fp + SMOOTH)
    recall = tp / (tp + fn + SMOOTH)
    f_beta_score = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
    return f_beta_score


def macro_metric(tp: list, fp: list, fn: list, tn: list):
    tp, fp, fn, tn = np.array(tp), np.array(fp), np.array(fn), np.array(tn)
    prec = tp / (tp + fp + SMOOTH)
    rec = tp / (tp + fn + SMOOTH)
    macro_prec = np.mean(prec)
    macro_rec = np.mean(rec)

    return macro_prec, macro_rec


def micro_metric(tp: list, fp: list, fn: list, tn: list):
    tp, fp, fn, tn = np.mean(tp), np.mean(fp), np.mean(fn), np.mean(fn)
    micro_prec = tp / (tp + fp + SMOOTH)
    micro_rec = tp / (tp + fn + SMOOTH)
    return micro_prec, micro_rec


if __name__ == '__main__':
    pass
