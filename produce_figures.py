import os
import numpy as np
import torch

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import seaborn as sns

from dataset import FireEventDataset
import models
import baseline

# Load data
dataset_name = "spruce_oak_pmma_pur_chipboard"
sample_rate = 32000
hdf5_path = "dataset_{}_sr_{}.hdf5".format(dataset_name, sample_rate)
augment = False

valid_dataset = FireEventDataset(hdf5_path, indice_key='valid_indices', augment=augment)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
test_dataset = FireEventDataset(hdf5_path, indice_key='test_indices', augment=augment)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

loss_function = torch.nn.BCELoss()

# Load model
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000
classes_num = 1

experiment_path = 'experiments/baseline'
model = models.Cnn14(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins,
        fmin=fmin, fmax=fmax, classes_num=classes_num)
model = model.cuda()
model.load_state_dict(torch.load(os.path.join(experiment_path, "best_model.ckpt")))
model.eval()

valid_loss, valid_acc, ys_true_valid, ys_pred_valid = baseline.evaluate(model, valid_loader, loss_function)
test_loss, test_acc, ys_true_test, ys_pred_test = baseline.evaluate(model, test_loader, loss_function)
print("Accuracy: ", valid_acc)
print("Accuracy: ", test_acc)

# Figure 6
fpr, tpr, thresholds = metrics.roc_curve(ys_true_valid, ys_pred_valid)
threshold_idx = int(np.sum(fpr == 0.0))-1

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

scale = 3.54330709/4

fig = plt.figure(figsize=(scale*4, scale*3))
ax = fig.add_subplot(1, 1, 1)

ax.plot(fpr[threshold_idx:], tpr[threshold_idx:], color='k', ls='solid')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve')
plt.savefig("roc_curve_valid.pdf", bbox_inches='tight')
adjusted_thr = thresholds[threshold_idx]

# Figure 5

thr = 0.5
true = ys_true_test == 1
pred = ys_pred_test > thr

cm = confusion_matrix(true, pred, normalize='true')

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

scale = 3.54330709/4

fig = plt.figure(figsize=(scale*4, scale*3))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(cm, ax=ax, annot=True, cmap='Greys')
ax.set_ylabel("True class")
ax.set_xlabel("Predicted class")
ax.set_title('Confusion Matrix')
plt.savefig("confusion_matrix.pdf", bbox_inches='tight')

print("accuracy  : {:.4f}".format(np.mean(true == pred)))
print("f-score   : ", f1_score(true, pred))
print("precision : ", precision_score(true, pred))
print("recall    : ", recall_score(true, pred))

thr = adjusted_thr
true = ys_true_test == 1
pred = ys_pred_test > thr

cm = confusion_matrix(true, pred, normalize='true')

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

scale = 3.54330709/4

fig = plt.figure(figsize=(scale*4, scale*3))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(cm, ax=ax, annot=True, cmap='Greys')
ax.set_ylabel("True class")
ax.set_xlabel("Predicted class")
ax.set_title('Confusion Matrix')
plt.savefig("confusion_matrix_adjusted.pdf", bbox_inches='tight')

print("accuracy: {:.4f}".format(np.mean(true == pred)))
print("f-score   : ", f1_score(true, pred))
print("precision : ", precision_score(true, pred))
print("recall    : ", recall_score(true, pred))
