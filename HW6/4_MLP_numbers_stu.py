"""

    - predict hand-written numbers using a three layer (one hidden) MLP
    - use multi-class prediction by converting strings using one-hot vector method
    - source code for MLP from:
      github.com/rasbt/python-machine-learning-book-3rd-edition
    - data from: http://yann.lecun.com/exdb/mnist/

    -------------How to run this code?------------------------------------
    1) download data file and source code (NeuralNetMLP.py)
    2) create new folder: /src/ copy-paste NeuralNetMLP.py
    3) create new file in /src/ called __init__.py
    4) move data file: mnist_scaled.npz to new folder data/
    5) chose hyperparameters and click run

    see also:
    Raschka, Sebastian, and Vahid Mirjalili. Python Machine Learning - Second Edition.
    Vol. 2nd ed, Packt Publishing, 2017.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn.metrics as skMetric

#----------------------------------------------
from src.NeuralNetMLP import NeuralNetMLP as MLP

np.random.seed( 1234)
#=========================1=============================================================
#              parameters
#=======================================================================================
data_dir = 'data'
file_in  = 'mnist_scaled.npz'
OUTPUT = 'fig'
#---------figure size-------------------
nx,ny    = 28,28
#--------hyper parameters---------------
n_iter   = 100
d_hidden = 100 # number of hidden units
L2_reg   = 0.1 #factor for regularization term
eta      = 0.001 # learning rate
#----------training and testing sizes---
n_minibatch = 100
valid_size  = 0.05 # check model perfomance during testing




#=========================1=============================================================
#                   download and prep data
#=======================================================================================
dData = np.load( f"{data_dir}/{file_in}")
X_train, y_train = dData['X_train'], dData['y_train']
X_test, y_test   = dData['X_test'], dData['y_test']

print('Train: Rows: %d, columns: %d'%(X_train.shape[0], X_train.shape[1]))
print('Test: Rows: %d, columns: %d'%(X_test.shape[0], X_test.shape[1]))
print( 'train targets: ', np.unique( y_train), np.bincount( y_train))
print( 'test targets: ', np.unique( y_test), np.bincount(y_test))

X_train, X_valid, y_train, y_valid = train_test_split( X_train, y_train,
                                                        shuffle=True,
                                                        test_size=valid_size,
                                                        stratify=y_train)

print( 'validation targets: ', np.unique( y_valid), np.bincount( y_valid))


fig1,ax = plt.subplots(nrows=2, ncols=5,
                     sharex=True, sharey=True)
ax = ax.flatten()
for i in range (10):
    img = X_train[y_train == i][0].reshape(nx,ny)
    ax[i].imshow(img,cmap='Greys')
    
ax[0].set_xticks([])
ax[0].set_yticks([])
fig1.suptitle("Train data", fontsize=16) 
plt.tight_layout()
plt.savefig(f"{OUTPUT}/data.png")
plt.show()


random_idx = np.random.choice(X_test.shape[0], size=5, replace=False)

fig2, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))

# Loop over the random indices and plot each image matrix
for i, idx in enumerate(random_idx): 
    
    img = X_test[idx].reshape(nx, ny)    
    axes[i].imshow(img, cmap='Greys')
    axes[i].set_title(f"Index: {idx}")
    axes[i].axis('off')  

fig2.suptitle("Random test data", fontsize=16) 
plt.tight_layout()
plt.savefig(f"{OUTPUT}/Random_test_data.png")
plt.show()



#=========================2=============================================================
#                training and validation
#=======================================================================================

nn_clf = MLP( d_hidden=d_hidden, L2= L2_reg, n_iter=n_iter, 
                eta=eta, shuffle = True,
               minibatch_size=n_minibatch,
               seed=1)
 
nn_clf.fit( X_train = X_train, y_train=y_train,
          X_valid = X_valid, y_valid=y_valid)

y_train_pred = nn_clf.predict( X_train)
fAcc_train = skMetric.accuracy_score( y_train_pred, y_train)*100
print( 'training accuracy: %.2f per.'%( fAcc_train))



#=========================3=============================================================
#               model performance
#=======================================================================================
y_test_pred = nn_clf.predict( X_test)
fAcc_test   = round(skMetric.accuracy_score( y_test_pred, y_test)*100,2)
print( '-----testing--------')
print( 'accuracy: %.2f per.'%( fAcc_test))

f_prec = round(skMetric.precision_score(y_test, y_test_pred, average='macro') * 100,2)
print( 'precision: %.2f per.'%( f_prec))

f_recall = round(skMetric.recall_score(y_test, y_test_pred, average='macro') * 100,2)
print( 'recall: %.2f per.'%( f_recall))

metrics_dict = {
    "train_accuracy": fAcc_train,
    "test_accuracy": fAcc_test,
    "precision": f_prec,
    "recall": f_recall
}

df = pd.DataFrame([metrics_dict])
df.to_csv(f"{OUTPUT}/model_metrics.csv", index=False)


#=========================4=============================================================
#                      plots
#=======================================================================================
#---------------5A performance during training-------------------
fig1, ax = plt.subplots( 2, 1)
ax[0].plot( np.arange(n_iter)+1, 
            np.asarray(nn_clf.dEval['cost'])*1e-3, 
            'ko-', ms = 4,lw = 2,mfc = 'none',label='Cost')
ax[0].set_xlabel( 'Epoch'),ax[0].set_ylabel( 'Cost *1E3')
ax[0].legend( loc='upper right')

ax[1].plot( np.arange(n_iter)+1,  nn_clf.dEval['acc_train'], 
            'bo-', ms = 4,lw = 2,mfc = 'none', label = 'train')
ax[1].plot( np.arange( nn_clf.n_iter)+1, nn_clf.dEval['acc_valid'], 
            'ro-', ms = 4,lw = 2,mfc = 'none', label = 'valid')
ax[1].legend(loc='lower right')
ax[1].set_xlabel( 'Epoch'),ax[1].set_ylabel( 'Accuracy')
plt.savefig(f"{OUTPUT}/performace.png")
plt.show()

#-----------------5B confusion matrix---------------------------
fig2, ax_cm = plt.subplots(figsize=(8, 8))  # Create a clean square figure context

skMetric.ConfusionMatrixDisplay.from_predictions(
    y_true=y_test,
    y_pred=y_test_pred,
    cmap='viridis',                     
    ax=ax_cm                               
)


ax_cm.set_title(f"Confusion Matrix (Test Data)\nAccurancy: {fAcc_test}\nPrecision: {f_prec}\nRecall: {f_recall}"
                , fontsize=14, pad=15)
plt.savefig(f"{OUTPUT}/confusion_matrix.png")
plt.show()

## For answer question which is the most mistake number
#Generate the raw numeric confusion matrix array
cm = skMetric.confusion_matrix(y_test, y_test_pred)

#Find the absolute most mistaken combination (off-diagonal maximum)
cm_off_diagonal = cm.copy()
np.fill_diagonal(cm_off_diagonal, -1)

# Get the indices of the highest value left over
true_digit, pred_digit = np.unravel_index(np.argmax(cm_off_diagonal), cm_off_diagonal.shape)
max_mistakes = cm_off_diagonal[true_digit, pred_digit]

# Find the hardest digit overall (lowest recall / class accuracy)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
print(class_accuracies)
hardest_digit = np.argmin(class_accuracies)
hardest_digit_acc = class_accuracies[hardest_digit] * 100


print(f"Most Mistaken Pairing: True digit '{true_digit}' was mistaken for '{pred_digit}' {max_mistakes} times.")
print(f"Lowest Performing Class: Digit '{hardest_digit}' was the hardest to identify.")





# #---------------5C plot miss-classified numbers-----------------
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab= y_test_pred[y_test != y_test_pred][:25]

fig3, ax = plt.subplots(nrows=5, ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'%( i+1, correct_lab[i], miscl_lab[i]))
ax[0].set_xticks([]),ax[0].set_yticks([])
plt.tight_layout()
plt.savefig(f"{OUTPUT}/miss-classified.png")
plt.show()

