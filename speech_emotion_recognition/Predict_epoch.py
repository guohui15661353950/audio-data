#实现的callbacks
from keras.callbacks import Callback
import numpy as np
class PredictEpoch(Callback):
    def __init__(self,validation):
        super(PredictEpoch,self).__init__()
        self.validation = validation
        self.record = []
        self.best = 0
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.best < logs.get('val_acc'):
            self.best = logs.get('val_acc')
            self.record =  np.argmax( self.model.predict(self.validation, verbose=2),axis = 1) 
        else:
            pass