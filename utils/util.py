import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

random.seed(a=None, version=2)


def get_curve(gt, pred, target_names, curve="roc"):
    for i in range(len(target_names)):
        if curve == "roc":
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt[:, i], pred[:, i])
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            a, b, _ = curve_function(gt[:, i], pred[:, i])
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], "k--")
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.legend(
                loc="upper center", bbox_to_anchor=(1.3, 1), fancybox=True, ncol=1
            )
        elif curve == "prc":
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            average_precision = average_precision_score(gt[:, i], pred[:, i])
            label = target_names[i] + " Avg.: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where="post", label=label)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(
                loc="upper center", bbox_to_anchor=(1.3, 1), fancybox=True, ncol=1
            )

def print_confidence_intervals(class_labels, statistics):
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df

def get_true_pos(y, pred, th=0.5):
    pred_t = pred > th
    return np.sum((pred_t == True) & (y == 1))


def get_true_neg(y, pred, th=0.5):
    pred_t = pred > th
    return np.sum((pred_t == False) & (y == 0))


def get_false_neg(y, pred, th=0.5):
    pred_t = pred > th
    return np.sum((pred_t == False) & (y == 1))


def get_false_pos(y, pred, th=0.5):
    pred_t = pred > th
    return np.sum((pred_t == True) & (y == 0))


def get_performance_metrics(
    y,
    pred,
    class_labels,
    tp=get_true_pos,
    tn=get_true_neg,
    fp=get_false_pos,
    fn=get_false_neg,
    acc=None,
    prevalence=None,
    spec=None,
    sens=None,
    ppv=None,
    npv=None,
    auc=None,
    f1=None,
    thresholds=[],
):
    if len(thresholds) != len(class_labels):
        thresholds = [0.5] * len(class_labels)

    columns = [
        "",
        "TP",
        "TN",
        "FP",
        "FN",
        "Accuracy",
        "Prevalence",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "AUC",
        "F1",
        "Threshold",
    ]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):
        # df.loc[i] = [""] + [0] * (len(columns) - 1)
        df.loc[i, columns[0]] = class_labels[i]
        df.loc[i, columns[1]] = (
            round(tp(y[:, i], pred[:, i]), 3) if tp != None else "Not Defined"
        )
        df.loc[i, columns[2]] = (
            round(tn(y[:, i], pred[:, i]), 3) if tn != None else "Not Defined"
        )
        df.loc[i, columns[3]] = (
            round(fp(y[:, i], pred[:, i]), 3) if fp != None else "Not Defined"
        )
        df.loc[i, columns[4]] = (
            round(fn(y[:, i], pred[:, i]), 3) if fn != None else "Not Defined"
        )
        df.loc[i, columns[5]] = (
            round(acc(y[:, i], pred[:, i], thresholds[i]), 3)
            if acc != None
            else "Not Defined"
        )
        df.loc[i, columns[6]] = (
            round(prevalence(y[:, i]), 3) if prevalence != None else "Not Defined"
        )
        df.loc[i, columns[7]] = (
            round(sens(y[:, i], pred[:, i], thresholds[i]), 3)
            if sens != None
            else "Not Defined"
        )
        df.loc[i, columns[8]] = (
            round(spec(y[:, i], pred[:, i], thresholds[i]), 3)
            if spec != None
            else "Not Defined"
        )
        df.loc[i, columns[9]] = (
            round(ppv(y[:, i], pred[:, i], thresholds[i]), 3)
            if ppv != None
            else "Not Defined"
        )
        df.loc[i, columns[10]] = (
            round(npv(y[:, i], pred[:, i], thresholds[i]), 3)
            if npv != None
            else "Not Defined"
        )
        df.loc[i, columns[11]] = (
            round(auc(y[:, i], pred[:, i]), 3) if auc != None else "Not Defined"
        )
        df.loc[i, columns[12]] = (
            round(f1(y[:, i], pred[:, i] > thresholds[i]), 3)
            if f1 != None
            else "Not Defined"
        )
        df.loc[i, columns[13]] = round(thresholds[i], 3)

    df = df.set_index("")
    return df


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], "k--")
            plt.plot(
                fpr_rf, tpr_rf, label=labels[i] + " (" + str(round(auc_roc, 3)) + ")"
            )
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("ROC curve")
            plt.legend(loc="best")
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals
