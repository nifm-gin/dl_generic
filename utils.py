import numpy as np
from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred, eps = .001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis = -1)
    return (2. * intersection) / (K.sum(y_true,-1) + K.sum(y_pred,-1))

def dice_loss(y_true, y_pred):
    y_pred = K.round(y_pred)
    return 1 - dice_coef(y_true, y_pred)
    
