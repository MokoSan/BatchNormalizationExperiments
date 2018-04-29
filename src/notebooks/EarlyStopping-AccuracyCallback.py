from keras.callbacks import Callback

class EarlyStoppingMetric(Callback):
    '''
    max_val: maximum value that a monitored metric can achieve
    monitor: metric to track during training
    
    
    example: In Batch Normalization (https://arxiv.org/pdf/1502.03167.pdf)
             the training is studied by stopping training for each model 
             when a validaition accuracy of 72.2% is achieved. The number of
             steps to achieve this accuracy is studied for multiple variants
             for the same CNN architecture. 
    '''
    
    def __init__(self, max_val, monitor='val_acc'):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = max_value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current < self.value:
            print("Training Stopped at Epoch %05d" % epoch)
            self.model.stop_training = True

