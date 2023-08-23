import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf



def train_kfold(model, inputs, expecteds, num_folds=5, batch_size=32, epochs=10):
    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []


    # define k-fold
    kfold = KFold(n_splits=num_folds, shuffle=True)

    fold_no = 1
    for train, test in kfold.split(inputs, expecteds):

        # train model
        history = model.fit(
            inputs[train], 
            expecteds[train],
            batch_size=batch_size,
            epochs=epochs)
        
        # Generate generalization metrics
        scores = model.evaluate(inputs[test], expecteds[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')