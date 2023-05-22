import tensorflow as tf
import numpy as np
import datetime
import pandas as pd
import argparse
import time

from utils import load_airbnb_data, load_clothing_data, get_metrics

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, f1_score

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-split', default='08')
    parser.add_argument('--modalities', default='all')
    parser.add_argument('--noise', default='strong')
    parser.add_argument('--labelgen-model', default='_lr')
    parser.add_argument('--modality-prop', default='060202')
    parser.add_argument('--embeds-type', default='finetuned')
    parser.add_argument('--label-type', default='synthetic')
    parser.add_argument('--task', default='classification')
    parser.add_argument('--text-cols', default='text')
    parser.add_argument('--dataset')
    parser.add_argument('--outcome-wt', type=float)
    parser.add_argument('--data-dir')
    args = parser.parse_args()

    return args

class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        #note there is only one epsilon were just duplicating it for conformability
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]

def make_aipw(input_dim, reg_l2):

    x = Input(shape=(input_dim,), name='input')
    # representation
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal',name='phi_1')(x)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal',name='phi_2')(phi)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal',name='phi_3')(phi)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y0_hidden_1')(phi)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y1_hidden_1')(phi)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y0_hidden_2')(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y1_hidden_2')(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)

    #propensity prediction
    #Note that the activation is actually sigmoid, but we will squish it in the loss function for numerical stability reasons
    t_prediction = Dense(units=1,activation=None,name='t_prediction')(phi)

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions,t_prediction,phi])
    model = Model(inputs=x, outputs=concat_pred)
    return model


def make_dragonnet(input_dim, reg_l2):

    x = Input(shape=(input_dim,), name='input')
    # representation
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal',name='phi_1')(x)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal',name='phi_2')(phi)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal',name='phi_3')(phi)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y0_hidden_1')(phi)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y1_hidden_1')(phi)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y0_hidden_2')(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2),name='y1_hidden_2')(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)

    #propensity prediction
    #Note that the activation is actually sigmoid, but we will squish it in the loss function for numerical stability reasons
    t_predictions = Dense(units=1,activation=None,name='t_prediction')(phi)
    #Although the epsilon layer takes an input, it really just houses a free parameter. 
    epsilons = EpsilonLayer()(t_predictions)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions,t_predictions,epsilons,phi])
    model = Model(inputs=x, outputs=concat_pred)
    return model

class Base_Loss(Loss):
    #initialize instance attributes
    def __init__(self, alpha=1.0, outcome_wt=1.0, task='classification'):
        super().__init__()
        self.alpha = alpha
        self.name='standard_loss'
        self.outcome_wt = outcome_wt
        self.task = task

    def split_pred(self,concat_pred):
        #generic helper to make sure we dont make mistakes
        preds={}
        if self.task == 'multiclass':
            preds['y0_pred'] = concat_pred[:,:5]
            preds['y1_pred'] = concat_pred[:,5:10]
            preds['t_pred'] = concat_pred[:,10]
            preds['phi'] = concat_pred[:,11:]
        else:
            preds['y0_pred'] = concat_pred[:, 0]
            preds['y1_pred'] = concat_pred[:, 1]
            preds['t_pred'] = concat_pred[:, 2]
            preds['phi'] = concat_pred[:, 3:]
        return preds

    #for logging purposes only
    def treatment_acc(self,concat_true,concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        #Since this isn't used as a loss, I've used tf.reduce_mean for interpretability
        return tf.reduce_mean(binary_accuracy(t_true, tf.math.sigmoid(p['t_pred']), threshold=0.5))

    def treatment_bce(self,concat_true,concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        lossP = tf.reduce_sum(binary_crossentropy(t_true,p['t_pred'],from_logits=True))
        return lossP

    def outcome_acc(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        
        if self.task == 'classification':
            y_pred = p['y0_pred'] * (1 - t_true) + p['y1_pred'] * t_true
            y_pred = (tf.math.sigmoid(y_pred) > 0.5).numpy().astype('float32')
            return tf.reduce_mean(binary_accuracy(y_true, y_pred, threshold=0.5))

        y_pred = p['y0_pred'] * (1 - tf.reshape(t_true, (-1, 1))) + p['y1_pred'] * tf.reshape(t_true, (-1, 1))
        y_pred = (tf.math.sigmoid(y_pred) > 0.5).numpy().astype('float32')
        return tf.reduce_mean(categorical_accuracy(y_true, y_pred))

    def outcome_bce(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        y_pred = p['y0_pred'] * (1 - t_true) + p['y1_pred'] * t_true
        loss = tf.reduce_sum(binary_crossentropy(y_true, y_pred+1e-15, from_logits=True))
        return loss

    def outcome_ce(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        y_pred = p['y0_pred'] * (1 - tf.reshape(t_true, (-1, 1))) + p['y1_pred'] * tf.reshape(t_true, (-1, 1))
        loss = tf.reduce_sum(categorical_crossentropy(tf.one_hot(tf.cast(y_true, tf.int32), depth=5), 
            y_pred+1e-15, from_logits=True))
        return loss

    def regression_loss(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - p['y0_pred']))
        loss1 = tf.reduce_sum(t_true * tf.square(y_true - p['y1_pred']))
        return loss0+loss1

    def ipw_bce(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)

        loss0 = tf.reduce_sum(binary_crossentropy((1-t_true)*y_true/(1-p['t_pred']), (1-t_true)*p['y0_pred']/(1-p['t_pred'])+1e-15))
        loss1 = tf.reduce_sum(binary_crossentropy(t_true*y_true/p['t_pred'], t_true*p['y1_pred']/p['t_pred']+1e-15))
        loss = loss0+loss1

        return loss

    def ipw_ce(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)

        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=5)
        loss0 = tf.reduce_sum(categorical_crossentropy(
            (1-tf.reshape(t_true, (-1, 1)))*y_true_oh/(1-tf.reshape(p['t_pred'], (-1,1))), 
            (1-tf.reshape(t_true, (-1, 1)))*p['y0_pred']/(1-tf.reshape(p['t_pred'], (-1, 1)))+1e-15))
        loss1 = tf.reduce_sum(categorical_crossentropy(
            tf.reshape(t_true, (-1, 1))*y_true_oh/tf.reshape(p['t_pred'], (-1, 1)), 
            tf.reshape(t_true, (-1, 1))*p['y1_pred']/tf.reshape(p['t_pred'], (-1, 1)))+1e-15)
        loss = loss0+loss1

        return loss

    def standard_loss(self,concat_true,concat_pred):
        lossP = self.treatment_bce(concat_true,concat_pred)
        if self.task == 'classification':
            lossR = self.outcome_bce(concat_true,concat_pred)
            lossIPW = self.ipw_bce(concat_true,concat_pred)
            return self.outcome_wt * lossR + lossIPW + self.alpha * lossP
        elif self.task == 'multiclass':
            lossR = self.outcome_ce(concat_true,concat_pred)
            lossIPW = self.ipw_ce(concat_true,concat_pred)
            return self.outcome_wt * lossR + lossIPW + self.alpha * lossP
        elif self.task == 'regression':       
            lossR = self.regression_loss(concat_true,concat_pred)
            return lossR + self.alpha * lossP

    #compute loss
    def call(self, concat_true, concat_pred):        
        return self.standard_loss(concat_true,concat_pred)

class TarReg_Loss(Base_Loss):
    #initialize instance attributes
    def __init__(self, alpha=1,beta=1,task='classification'):
        super().__init__()
        self.alpha = alpha
        self.beta=beta
        self.name='tarreg_loss'
        self.task=task

    def split_pred(self,concat_pred):
        preds={}
        if self.task == 'multiclass':
            preds['y0_pred'] = concat_pred[:,:5]
            preds['y1_pred'] = concat_pred[:,5:10]
            preds['t_pred'] = concat_pred[:,10]
            preds['phi'] = concat_pred[:,11:]
        else:
            preds['y0_pred'] = concat_pred[:, 0]
            preds['y1_pred'] = concat_pred[:, 1]
            preds['t_pred'] = concat_pred[:, 2]
            preds['phi'] = concat_pred[:, 3:]
        return preds

    def calc_hstar(self,concat_true,concat_pred):
        #step 2 above
        p=self.split_pred(concat_pred)
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        t_pred = tf.math.sigmoid(concat_pred[:, 2])
        t_pred = (t_pred + 0.001) / 1.002 # a little numerical stability trick implemented by Shi
        y1_pred = tf.math.sigmoid(p['y1_pred'])
        y0_pred = tf.math.sigmoid(p['y0_pred'])
        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        #calling it cc for "clever covariate" as in SuperLearner TMLE literature
        cc = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        h_star = y_pred + p['epsilon'] * cc
        h_star = (tf.math.sigmoid(h_star) > 0.5).numpy().astype(int)
        return h_star

    def call(self,concat_true,concat_pred):
        y_true = concat_true[:, 0]

        standard_loss=self.standard_loss(concat_true,concat_pred)
        h_star=self.calc_hstar(concat_true,concat_pred)
        #step 3 above
        targeted_regularization = tf.reduce_sum(tf.square(y_true - h_star))

        # final
        loss = standard_loss + self.beta * targeted_regularization
        return loss

def pdist2sq(x,y):
    x2 = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
    y2 = tf.reduce_sum(y ** 2, axis=-1, keepdims=True)
    dist = x2 + tf.transpose(y2, (1, 0)) - 2. * x @ tf.transpose(y, (1, 0))
    return dist

class AIPW_Metrics(Callback):
    def __init__(self,data, verbose=0):   
        super(AIPW_Metrics, self).__init__()
        self.data=data #feed the callback the full dataset
        self.verbose=verbose

        #needed for PEHEnn; Called in self.find_ynn
        self.data['o_idx']=tf.range(self.data['t'].shape[0])
        self.data['c_idx']=self.data['o_idx'][self.data['t'].squeeze()==0] #These are the indices of the control units
        self.data['t_idx']=self.data['o_idx'][self.data['t'].squeeze()==1] #These are the indices of the treated units
    
    def split_pred(self,concat_pred):
        preds={}
        preds['y0_pred'] = tf.math.sigmoid(concat_pred[:, 0])
        preds['y1_pred'] = tf.math.sigmoid(concat_pred[:, 1])
        preds['t_pred'] = concat_pred[:, 2]
        preds['phi'] = concat_pred[:, 3:]
        return preds

    def ATE(self,concat_pred):
        p = self.split_pred(concat_pred)
        return p['y1_pred']-p['y0_pred']


def evaluate(model_output, test_data, task):
    y0_pred = model_output[:, 0].reshape(-1, 1)
    y1_pred = model_output[:, 1].reshape(-1, 1)
    t_pred = model_output[:, 2]
    phi_pred = model_output[:, 3:]

    y_pred = y0_pred * (1 - test_data['t']) + y1_pred * test_data['t']
    if task == 'classification':
        y_pred = (tf.math.sigmoid(y_pred) > 0.5).numpy().astype(int)

    get_metrics(test_data['y'], y_pred, task=task)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import SGD, Adam

start_time = time.time()

args = get_args()
if args.labelgen_model == 'None':
    args.labelgen_model = ''

if args.dataset == 'airbnb':
    load_data = load_airbnb_data
elif args.dataset == 'clothing_review':
    load_data = load_clothing_data

data = load_data(split='train', data='biased', label_split=args.label_split, modalities=args.modalities, representation='embeds', noise=args.noise,
    labelgen_model=args.labelgen_model, modality_prop=args.modality_prop, embeds_type=args.embeds_type, output_for='dragonnet',
    label_type=args.label_type, data_dir=args.data_dir, text_cols=args.text_cols, task=args.task)
print(data['x'].shape)

val_split=0.2
batch_size=64
verbose=1
i = 0
tf.random.set_seed(i)
np.random.seed(i)
yt = np.concatenate([data['y'], data['t']], 1)

# Clear any logs from previous runs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=20, min_delta=0.01),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0),
        tensorboard_callback,
        AIPW_Metrics(data,verbose=verbose)
        ]

sgd_lr = 1e-5
momentum = 0.9

aipw_model=make_aipw(data['x'].shape[1],.01)
aipw_loss=Base_Loss(alpha=1.0, outcome_wt=args.outcome_wt, task=args.task)

if args.task == 'classification' or args.task == 'multiclass':
    aipw_model.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True),
                        loss=aipw_loss,
                        metrics=[aipw_loss,aipw_loss.outcome_acc,aipw_loss.treatment_acc]
                    )
elif args.task == 'regression':
    aipw_model.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True),
                        loss=aipw_loss,
                        metrics=[aipw_loss, aipw_loss.regression_loss, aipw_loss.treatment_acc]
                    )

aipw_model.fit(x=data['x'],y=yt,
                  callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=300,
                  batch_size=batch_size,
                  verbose=verbose)

test_data_biased = load_data(split='test', data='biased', label_split=args.label_split, modalities=args.modalities, representation='embeds', noise=args.noise,
    labelgen_model=args.labelgen_model, modality_prop=args.modality_prop, embeds_type=args.embeds_type, output_for='dragonnet',
    label_type=args.label_type, data_dir=args.data_dir, text_cols=args.text_cols, task=args.task)

test_data_unbiased = load_data(split='test', data='unbiased', label_split=args.label_split, modalities=args.modalities, representation='embeds', noise=args.noise,
    labelgen_model=args.labelgen_model, modality_prop=args.modality_prop, embeds_type=args.embeds_type, output_for='dragonnet',
    label_type=args.label_type, data_dir=args.data_dir, text_cols=args.text_cols, task=args.task)

model_output_biased = aipw_model.predict(test_data_biased['x'])
model_output_unbiased = aipw_model.predict(test_data_unbiased['x'])

evaluate(model_output_biased, test_data_biased, args.task)
evaluate(model_output_unbiased, test_data_unbiased, args.task)

end_time = time.time()

print("Runtime in seconds: {}".format(end_time - start_time))
