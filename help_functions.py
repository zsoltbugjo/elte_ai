import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, auc, make_scorer, accuracy_score, fbeta_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.metrics import geometric_mean_score
import time
from tqdm import tqdm
import itertools

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torchmetrics
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau



def compare_models(model_dict, X, y, savefig_path=None):
    
    start = time.time()
           
    # Cross-validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=16)
    scores_df = pd.DataFrame(index=["auc", "accuracy", "precision", "recall", "geometric mean", "f1 score"])
    
    # ROC-AUC curve and Precision-Recall curve
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,16))
    axes[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", label="no skill", alpha=0.8)
    
    no_skill = len(y[y==1]) / len(y)
    axes[1].plot([0, 1], [no_skill, no_skill], linestyle="--", lw=2, color="k", label="no skill", alpha=0.8)
    for name, model in tqdm(model_dict.items()):
                
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        prs = []
        pr_aucs = []
        mean_recall = np.linspace(0, 1, 100)
        
        # metrics
        accuracies = []
        precisions = []
        recalls = []
        g_means = []
        f1_scores = []
                
        i = 0
        
        for train, test in tqdm(cv.split(X, y), total=30):
            clf = model
            clf.fit(X.iloc[train, :], y.iloc[train])
            probs = clf.predict_proba(X.iloc[test, :])
            preds = clf.predict(X.iloc[test, :])
            
            fpr, tpr, thresholds = roc_curve(y.iloc[test], probs[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            
            prec, rec, thres = precision_recall_curve(y.iloc[test], probs[:, 1])
            prs.append(np.interp(mean_recall, prec, rec))
            pr_auc = auc(rec, prec)
            pr_aucs.append(pr_auc)
            
            accuracy = accuracy_score(y.iloc[test], preds)
            accuracies.append(accuracy)
            precision = precision_score(y.iloc[test], preds)
            precisions.append(precision)
            recall = recall_score(y.iloc[test], preds)
            recalls.append(recall)
            g_mean = geometric_mean_score(y.iloc[test], preds)
            g_means.append(g_mean)
            f1score = f1_score(y.iloc[test], preds)
            f1_scores.append(f1score)
            
            i += 1
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        axes[0].plot(mean_fpr, mean_tpr, label=name + r' Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=3)
        
        mean_precision = np.mean(prs, axis=0)
        mean_pr_auc = auc(mean_recall, mean_precision)
        std_pr_auc = np.std(pr_aucs)
        
        axes[1].plot(mean_precision, mean_recall, label=name + r' Mean (PR AUC = %0.3f $\pm$ %0.2f)' % (mean_pr_auc, std_pr_auc), lw=3)
        
        score_list = [mean_auc,
                      np.mean(accuracies),
                      np.mean(precisions),
                      np.mean(recalls),
                      np.mean(g_means),
                      np.mean(f1_scores)
                     ]
        
        scores_df[name] = score_list
        
        
    axes[0].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    axes[0].set_xlabel("False positive rate", fontsize=15)
    axes[0].set_ylabel("True positive rate", fontsize=15)
    axes[0].set_title("Mean ROC curves of cross validation", fontsize=20, fontweight="bold")
    
    axes[1].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    axes[1].set_xlabel("Recall", fontsize=15)
    axes[1].set_ylabel("Precision", fontsize=15)
    axes[1].set_title("Mean Precision-Recall curves of cross validation", fontsize=20, fontweight="bold")
    
    
    axes[0].legend(prop={"size": 10}, loc="upper left", bbox_to_anchor=(1,1))
    axes[1].legend(prop={"size": 10}, loc="upper left", bbox_to_anchor=(1,1))
    
    if savefig_path is not None:
        plt.savefig(savefig_path, dpi=300)
    
    plt.show()
    
    
    end = time.time()
    print("Runtime: {:.2f} minutes".format((end-start)/60))
    
    display(scores_df.style.highlight_max(color="lightgreen", axis=1).highlight_min(color="red", axis=1))
    
    return scores_df




def model_evaluation(model, X, y, savefig_path=None):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=16)
    scores = pd.DataFrame(index=["accuracy", "precision", "recall", "f1 score"])
    
    model_name = model.__class__.__name__
    
    fig, axes = plt.subplots(figsize=(10,10))
    no_skill = len(y[y==1]) / len(y)
    axes.plot([0, 1], [no_skill, no_skill], linestyle="--", lw=2, color="k", label="no skill", alpha=0.8)
    
    prs = []
    pr_aucs = []
    mean_recall = np.linspace(0, 1, 100)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    pr_thresholds = []
    
    i = 0
    
    clf = model
    
    for train, test in tqdm(cv.split(X, y), total=10):
        clf.fit(X.iloc[train, :], y.iloc[train])
        probs = model.predict_proba(X.iloc[test, :])
        preds = model.predict(X.iloc[test, :])
        
        accuracy = accuracy_score(y.iloc[test], preds)
        accuracies.append(accuracy)
        precision = precision_score(y.iloc[test], preds)
        precisions.append(precision)
        rec = recall_score(y.iloc[test], preds)
        recalls.append(rec)
        f1score = f1_score(y.iloc[test], preds)
        f1_scores.append(f1score)
        
        precision, recall, thresholds = precision_recall_curve(y.iloc[test], probs[:,1])
        fscore = (2 * precision * recall) / (precision + recall)
        
        ix = np.nanargmax(fscore)
        pr_thresholds.append(thresholds[ix])
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        prs.append(np.interp(mean_recall, precision, recall))
        axes.plot(recall, precision, label=r'Fold %d (AUC = %0.2f)' % (i+1, pr_auc))
        
        i += 1
        
    mean_precision = np.mean(prs, axis=0)
    mean_pr_auc = auc(mean_recall, mean_precision)
    std_pr_auc = np.std(pr_aucs)
    std_rec = np.std(prs, axis=0)
            
    axes.plot(mean_precision, mean_recall, color="navy", label=r' Mean (PR AUC = %0.3f $\pm$ %0.2f)' % (mean_pr_auc, std_pr_auc), lw=3)
    
    score_list = [np.mean(accuracies),
                  np.mean(precisions),
                  np.mean(recalls),
                  np.mean(f1_scores)
                 ]
    
    scores[model_name] = score_list
    
    axes.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    axes.set_xlabel("Recall", fontsize=15)
    axes.set_ylabel("Precision", fontsize=15)
    axes.legend(prop={"size": 10}, loc="upper left", bbox_to_anchor=(1,1))
    axes.set_title("Precision-Recall curves of cross-validation", fontsize=20, fontweight="bold")
    
    if savefig_path is not None:
        plt.savefig(savefig_path, dpi=300)
    
    plt.show()
    display(scores)
    
    return scores, pr_thresholds


def model_evaluation_fbetascore(model, X, y, savefig_path=None):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=16)
    scores = pd.DataFrame(index=["accuracy", "precision", "recall", "f1 score", "fbeta_score"])
    beta = 4
    
    model_name = model.__class__.__name__
    
    fig, axes = plt.subplots(figsize=(10,10))
    no_skill = len(y[y==1]) / len(y)
    axes.plot([0, 1], [no_skill, no_skill], linestyle="--", lw=2, color="k", label="no skill", alpha=0.8)
    
    prs = []
    pr_aucs = []
    mean_recall = np.linspace(0, 1, 100)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    fbeta_scores = []
    
    pr_thresholds = []
    
    i = 0
    
    clf = model
    
    for train, test in tqdm(cv.split(X, y), total=10):
        clf.fit(X.iloc[train, :], y.iloc[train])
        probs = model.predict_proba(X.iloc[test, :])
        preds = model.predict(X.iloc[test, :])
        
        accuracy = accuracy_score(y.iloc[test], preds)
        accuracies.append(accuracy)
        precision = precision_score(y.iloc[test], preds)
        precisions.append(precision)
        rec = recall_score(y.iloc[test], preds)
        recalls.append(rec)
        f1score = f1_score(y.iloc[test], preds)
        f1_scores.append(f1score)
        fbetascore = fbeta_score(y.iloc[test], preds, beta=beta)
        fbeta_scores.append(fbetascore)
        
        precision, recall, thresholds = precision_recall_curve(y.iloc[test], probs[:,1])
        f4score = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
        
        ix = np.nanargmax(f4score)
        pr_thresholds.append(thresholds[ix])
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        prs.append(np.interp(mean_recall, precision, recall))
        axes.plot(recall, precision, label=r'Fold %d (AUC = %0.2f)' % (i+1, pr_auc))
        
        i += 1
        
    mean_precision = np.mean(prs, axis=0)
    mean_pr_auc = auc(mean_recall, mean_precision)
    std_pr_auc = np.std(pr_aucs)
    std_rec = np.std(prs, axis=0)
            
    axes.plot(mean_precision, mean_recall, color="navy", label=r' Mean (PR AUC = %0.3f $\pm$ %0.2f)' % (mean_pr_auc, std_pr_auc), lw=3)
    
    score_list = [np.mean(accuracies),
                  np.mean(precisions),
                  np.mean(recalls),
                  np.mean(f1_scores),
                  np.mean(fbeta_scores)
                 ]
    
    scores[model_name] = score_list
    
    axes.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    axes.set_xlabel("Recall", fontsize=15)
    axes.set_ylabel("Precision", fontsize=15)
    axes.legend(prop={"size": 10}, loc="upper left", bbox_to_anchor=(1,1))
    axes.set_title("Precision-Recall curves of cross-validation", fontsize=20, fontweight="bold")
    
    if savefig_path is not None:
        plt.savefig(savefig_path, dpi=300)
    
    plt.show()
    display(scores)
    
    return scores, pr_thresholds






def probs_to_labels(probs, threshold):
        return (probs >= threshold).astype("int")
    
    
# Create custom dataset for neural network
class CustomData(Dataset):
    def __init__(self, X, y):
        self.X = X.to("cuda")
        self.y = y.to("cuda")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    


# Binary classifier neural network with dense layers

class BinaryClassifierNN(nn.Module):
    def __init__(self, input_size, activation_fn=nn.Sigmoid()):
        super(BinaryClassifierNN, self).__init__()
                
        self.layer1 = nn.Linear(in_features=input_size, out_features=128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.layer2 = nn.Linear(in_features=128, out_features=64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.layer3 = nn.Linear(in_features=64, out_features=16)
        self.bn3 = nn.BatchNorm1d(16)
        
        self.final_layer = nn.Linear(in_features=16, out_features=1)
        
        self.activation = activation_fn
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.bn1(y)
        y = self.activation(y)
        
        y = self.layer2(y)
        y = self.bn2(y)
        y = self.activation(y)
        
        y = self.layer3(y)
        y = self.bn3(y)
        y = self.activation(y)
        
        y = self.final_layer(y)
        y = self.sigmoid(y)
        
        return y


def train_nn(model, train_data, valid_data, loss_fn, optimizer, num_epochs=10, lr_scheduler=None, save_name="best_recall_net.pth"):
    train_losses, valid_losses = [], []
    train_precisions, valid_precisions = [], []
    train_recalls, valid_recalls = [], []
    train_f1scores, valid_f1scores = [], []
    
    best_recall = 0
    best_f1score = 0
    
    model.to("cuda")
    for epoch in tqdm(range(1, num_epochs+1)):
        model.train()
        batch_train_loss, batch_val_loss = [], []
        batch_train_precision, batch_valid_precision = [], []
        batch_train_recall, batch_valid_recall = [], []
        batch_train_f1score, batch_valid_f1score = [], []
        
        for x, y in train_data:
            x, y = x.to("cuda"), y.to("cuda")
            
            y = y.unsqueeze(1)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            batch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            y = y.to("cpu").detach().numpy().ravel()
            y_pred = torch.round(y_pred)
            y_pred = y_pred.to("cpu").detach().numpy().ravel()
            
            batch_train_precision.append(precision_score(y, y_pred, average="binary"))
            batch_train_recall.append(recall_score(y, y_pred, average="binary"))
            batch_train_f1score.append(f1_score(y, y_pred, average="binary"))
        
        train_loss = np.mean(batch_train_loss)
        train_losses.append(train_loss)
        train_precision = np.mean(batch_train_precision)
        train_precisions.append(train_precision)
        train_recall = np.mean(batch_train_recall)
        train_recalls.append(train_recall)
        train_f1score = np.mean(batch_train_f1score)
        train_f1scores.append(train_f1score)
        
        model.eval()
        for x, y in valid_data:
            x, y = x.to("cuda"), y.to("cuda")
            
            y = y.unsqueeze(1)
            y_pred = model(x) 
            val_loss = loss_fn(y_pred, y)            
            batch_val_loss.append(val_loss.item())
            
            y = y.to("cpu").detach().numpy().ravel()
            y_pred = torch.round(y_pred)
            y_pred = y_pred.to("cpu").detach().numpy().ravel()
            
            batch_valid_precision.append(precision_score(y, y_pred, average="binary"))
            batch_valid_recall.append(recall_score(y, y_pred, average="binary"))
            batch_valid_f1score.append(f1_score(y, y_pred, average="binary"))
        
        valid_loss = np.mean(batch_val_loss)
        valid_losses.append(valid_loss)
        valid_precision = np.mean(batch_valid_precision)
        valid_precisions.append(valid_precision)
        valid_recall = np.mean(batch_valid_recall)
        valid_recalls.append(valid_recall)
        valid_f1score = np.mean(batch_valid_f1score)
        valid_f1scores.append(valid_f1score)
        
        print(f'Epoch {epoch} | Precision: train: {train_precision:.5f} / validation: {valid_precision:.5f} | Recall: train: {train_recall:.5f} / validation: {valid_recall:.5f}')
        
        if (train_recall > best_recall) & (train_f1score > best_f1score):
            torch.save(model.state_dict(), save_name)
            best_recall = train_recall
            best_f1score = train_f1score        
        
    model.to("cpu")
    
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(train_losses, label="training loss")
    ax.plot(valid_losses, label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    plt.legend()
    plt.title("Losses across epochs")
    plt.show()
    
    return train_precisions, valid_precisions, train_recalls, valid_recalls, train_f1scores, valid_f1scores


def evaluate_nn(model, valid_data, return_labels=False):
    model.to("cuda")
    model.eval()
    predictions, true_values = [], []
    
    for x, y in valid_data:
        x, y = x.to("cuda"), y.to("cuda")
        y = y.unsqueeze(1)
        y_pred = model(x)
        
        y = y.to("cpu").detach().numpy().ravel()
        y_pred = torch.round(y_pred)
        y_pred = y_pred.to("cpu").detach().numpy().ravel()
        
        predictions.append(y_pred)
        true_values.append(y)
        
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions, average="binary")
    recall = recall_score(true_values, predictions, average="binary")
    f1score = f1_score(true_values, predictions, average="binary")
    
    model.to("cpu")
    
    cm = confusion_matrix(true_values, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=["good-cycle", "scrap-cycle"])

    disp.plot(cmap="Blues")
    plt.show()
    
    if return_labels:
        return predictions, true_values, accuracy, precision, recall, f1score
    
    else:
        return accuracy, precision, recall, f1score
    

    


class AutoEncoderData(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs.to("cuda")
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index]

    
class AutoEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(AutoEncoder, self).__init__()
                
        self.encoder = nn.Sequential(nn.Linear(input_size, 128),
                                     nn.BatchNorm1d(128),
                                     nn.Tanh(),
                                     nn.Linear(128, 64),
                                     nn.BatchNorm1d(64),
                                     nn.Tanh(),
                                     nn.Linear(64, latent_size),
                                     nn.BatchNorm1d(latent_size)
                                    )
               
        
        self.decoder = nn.Sequential(nn.Linear(latent_size, 64),
                                     nn.BatchNorm1d(64),
                                     nn.Tanh(),
                                     nn.Linear(64, 128),
                                     nn.BatchNorm1d(128),
                                     nn.Tanh(),
                                     nn.Linear(128, input_size),
                                     nn.BatchNorm1d(input_size),
                                     nn.Tanh()
                                    )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

def train_autoencoder(model, train_data, loss_fn, optimizer, num_epochs = 10, lr_scheduler=None, save_name="autoencoder.pth"):
    train_losses = []
    min_loss = 1000
    
    model.to("cuda")
    for epoch in tqdm(range(1, num_epochs+1)):
        model.train()
        batch_train_losses = []
        
        for inputs in train_data:
            inputs = inputs.to("cuda")
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, inputs)
            batch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
        epoch_train_loss = np.median(batch_train_losses)
        train_losses.append(epoch_train_loss)
        
        if epoch_train_loss < min_loss:
            torch.save(model.state_dict(), save_name)
            min_loss = epoch_train_loss
            best_train_losses = batch_train_losses
        
        print(f'Epoch {epoch} | Train loss: {epoch_train_loss:.5f}')

    model = model.to("cpu")
    return train_losses, best_train_losses

    
    

class AutoEncoder1DConv(nn.Module):
    def __init__(self, input_size, latent_size):
        super(AutoEncoder1DConv, self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(input_size, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(),
                                     nn.Conv1d(64, 16, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm1d(16),
                                     nn.ReLU(),
                                     nn.Conv1d(16, latent_size, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm1d(latent_size)
                                    )
        
        self.decoder = nn.Sequential(nn.Conv1d(latent_size, 16, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm1d(16),
                                     nn.ReLU(),
                                     nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(),
                                     nn.Conv1d(64, input_size, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm1d(input_size),
                                     nn.Sigmoid()
                                    )
            
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
    
        return decoded
    
    

def train_autoencoder_1dconv(model, train_data, loss_fn, optimizer, num_epochs = 10, lr_scheduler=None, save_name="autoencoder_1dconv.pth"):
    train_losses = []
    min_loss = 1000
    
    model.to("cuda")
    for epoch in tqdm(range(1, num_epochs+1)):
        model.train()
        batch_train_losses = []
        
        for inputs in train_data:
            inputs = inputs.to("cuda")
            inputs = inputs.unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, inputs)
            batch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
        epoch_train_loss = np.median(batch_train_losses)
        train_losses.append(epoch_train_loss)
        
        if epoch_train_loss < min_loss:
            torch.save(model.state_dict(), save_name)
            min_loss = epoch_train_loss
            best_train_losses = batch_train_losses
        
        print(f'Epoch {epoch} | Train loss: {epoch_train_loss:.5f}')

        
    model = model.to("cpu")
    return train_losses, best_train_losses



def eval_autoencoder(model, data, y_true, loss_fn, threshold, conv=False):
    model.eval()
    model.to("cuda")
    losses = []
    preds = []
    
    for inputs in data:
        inputs = inputs.to("cuda")
        if conv:
            inputs = inputs.unsqueeze(1)
        
        output = model(inputs)
        loss = loss_fn(output, inputs)
        losses.append(loss.item())
        
        if loss.item() > threshold:
            preds.append(1)
            
        else:
            preds.append(0)
        
    model.to("cpu")
    
    cm = confusion_matrix(y_true=y_true, y_pred=preds)
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=["good", "scrap"])
    cm_disp.plot(cmap="Blues")
    plt.title("Autoencoder confusion matrix")
    plt.show()
    
    return losses, preds