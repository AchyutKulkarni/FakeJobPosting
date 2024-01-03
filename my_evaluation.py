import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below

        self.confusion_matrix = {self.classes_[i]: {"TP":0, "TN": 0, "FP": 0, "FN": 0} for i in range(len(self.classes_))}

        for val in self.classes_:
            for i in range(len(self.actuals)):
                if(val==self.actuals[i] and val==self.predictions[i]):
                    self.confusion_matrix[val]["TP"] += 1
                if (val == self.actuals[i] and val != self.predictions[i]):
                    self.confusion_matrix[val]["FN"] += 1
                if (val != self.actuals[i] and val == self.predictions[i]):
                    self.confusion_matrix[val]["FP"] += 1
                else:
                    self.confusion_matrix[val]["TN"] += 1

        return


    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()

        if(target!=None):
            prec = 0.0
            if (self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FP"])!=0:
                prec = self.confusion_matrix[target]["TP"] / (self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FP"])
        else:
            total = 0
            if average == "macro":
                for val in self.classes_:
                    total += self.precision(val, average)
                prec = total/len(self.classes_)
            elif average == "micro":
                total_nm = 0
                total_dn = 0
                for val in self.classes_:
                    nm = self.confusion_matrix[val]["TP"]
                    dn = self.confusion_matrix[val]["TP"] + self.confusion_matrix[val]["FN"]
                    total_nm += nm
                    total_dn += dn
                prec = total_nm / total_dn
            elif average == "weighted":
                total_dn = 0
                for val in self.classes_:
                    dn = self.confusion_matrix[val]["TP"] + self.confusion_matrix[val]["FN"]
                    total += dn*self.precision(val, average)
                    total_dn += dn
                prec = total / total_dn
            else:
                raise Exception("Unknown average.")

        return prec

    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()

        total = 0
        if (target != None):
            rec = 0.0
            if (self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FN"])!=0:
                rec = self.confusion_matrix[target]["TP"] / (
                        self.confusion_matrix[target]["TP"] + self.confusion_matrix[target]["FN"])
        else:
            total = 0
            if average == "macro":
                for val in self.classes_:
                    total += self.recall(val, average)
                rec = total / len(self.classes_)
            elif average == "micro":
                total_nm = 0
                total_dn = 0

                for val in self.classes_:
                    nm = self.confusion_matrix[val]["TP"]
                    dn = self.confusion_matrix[val]["TP"] + self.confusion_matrix[val]["FN"]
                    total_nm += nm
                    total_dn += dn

                rec = total_nm / total_dn
            elif average == "weighted":
                total_dn = 0
                for val in self.classes_:
                    dn = self.confusion_matrix[val]["TP"] + self.confusion_matrix[val]["FN"]
                    total += dn * self.recall(val, average)
                    total_dn += dn
                rec = total / total_dn
            else:
                raise Exception("Unknown average.")

        return rec

    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        f1_score = 0.0
        if target!=None:
            if (self.precision(target, average) + self.recall(target, average))!=0:
                f1_score = 2 * (self.precision(target, average) * self.recall(target, average))/(self.precision(target, average) + self.recall(target, average))
        else:
            if average == "macro":
                total_nm = 0
                total_dn = 0

                for val in self.classes_:
                    total_nm += self.f1(val, average)
                f1_score = total_nm / len(self.classes_)
            elif average == "micro":
                total_nm = 0
                total_dn = 0

                for val in self.classes_:
                    nm = self.confusion_matrix[val]["TP"]
                    dn = self.confusion_matrix[val]["TP"] + self.confusion_matrix[val]["FN"]
                    total_nm += nm
                    total_dn += dn

                f1_score = total_nm / total_dn
            elif average == "weighted":
                total_dn = 0
                total = 0
                for val in self.classes_:
                    dn = self.confusion_matrix[val]["TP"] + self.confusion_matrix[val]["FN"]
                    total += dn * self.f1(val, average)
                    total_dn += dn
                f1_score = total / total_dn
            else:
                raise Exception("Unknown average.")

        return f1_score


    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if type(self.pred_proba) == type(None):
            return None
        else:
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]
                tp = 0
                fp = 0
                fn = Counter(self.actuals)[target]
                tn = len(self.actuals) - fn
                tpr = 0
                fpr = 0
                auc_target = 0
                for i in order:
                    if self.actuals[i] == target:
                        tp += 1
                        fn -= 1
                        tpr = tp/(tp+fn)
                    else:
                        fp += 1
                        tn -= 1
                        pre_fpr = fpr
                        fpr = fp/(fp+tn)
                        auc_target += tpr * (fpr - pre_fpr)
            else:
                raise Exception("Unknown target class.")

            return auc_target


