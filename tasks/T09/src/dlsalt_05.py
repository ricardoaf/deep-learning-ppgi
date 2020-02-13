# Load standard Python libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Ignore info and warning (gpu) messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load Keras libs
from keras import metrics, regularizers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping

# For reproducibility purposes
seed = 2020
np.random.seed(seed)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
model_name = 'model_05'
lithol = 1
label = 'd24'
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Load data
#------------------------------------------------------------------------------

# Load data from CSV
dataset_org  = pd.read_csv('./labeled_data.csv')

# Select columns
lithol_col = 'lithol'
label_cols = ['d0', 'd24', 'd48']
dataset = pd.DataFrame(dataset_org, columns=[
    'wd', 'ovbd', 'od', 'mud', 'sh', 'lithol', 'interc',
    'd0', 'd24', 'd48'])

# Function to get a data subset
def data_subset (df, lithol, label, lithol_col='lithol', label_cols=['d0', 'd24', 'd48']):
    ignore_cols = [lithol_col] + label_cols
    ignore_cols.remove(label)
    data = df.loc[df.lithol==lithol, :].drop(ignore_cols, axis=1)
    return data

# Select data subset
print('lithol:', lithol,'label:', label)
data = data_subset(dataset, lithol, label)
# print(data.describe())


# Split data for training and validation
#------------------------------------------------------------------------------

# Function to split a range of dataframe into train/validate subranges
def train_validate_split (df, train_part=0.6, validate_part=0.2):
    
    total_size = train_part + validate_part
    train_frac = train_part / total_size

    m = len(df)
    perm = np.random.permutation(df.index)

    train_end = int(train_frac * m)
    train = perm[:train_end]
    validate = perm[train_end:]

    return train, validate

# Split index ranges
train_size, valid_size = (80, 20)
train, valid = train_validate_split (data, train_part=train_size, validate_part=valid_size)

# Extract data for training and validation into x and y vectors
x_train = data.loc[train, :].drop(label, axis=1)
y_train = data.loc[train, [label]]
x_valid = data.loc[valid, :].drop(label, axis=1)
y_valid = data.loc[valid, [label]]


# Prepare data for training and validation of the Keras model
#------------------------------------------------------------------------------

# Function to get statistics about data frame
def norm_stats (d1, d2):
    ds = np.append(d1, d2, axis=0)
    mu = np.mean(ds, axis=0)
    sigma = np.std(ds, axis=0)
    return (mu, sigma)

# Training and validation data arrays
x_train_arr = np.array(x_train)
y_train_arr = np.array(y_train)
x_valid_arr = np.array(x_valid)
y_valid_arr = np.array(y_valid)

# Calculare mean and standard deviation of data
(x_mean, x_std) = norm_stats (x_train_arr, x_valid_arr)
(y_mean, y_std) = norm_stats (y_train_arr, y_valid_arr)

# Normalise training and validation data
xn_train_arr = (x_train_arr - x_mean) / x_std
yn_train_arr = (y_train_arr - y_mean) / y_std
xn_valid_arr = (x_valid_arr - x_mean) / x_std
yn_valid_arr = (y_valid_arr - y_mean) / y_std

print('Training shape: ', xn_train_arr.shape)
print('Training samples: ', xn_train_arr.shape[0])
print('Validation samples: ', xn_valid_arr.shape[0])


# Create Keras model
#------------------------------------------------------------------------------
# Time counter
start = time.time()
nodes = [216, 216, 216, 216, 216]
print('Nodes:',nodes)
activation = ['relu', 'relu', 'relu', 'relu', 'relu']
kernel_initializer = ['normal', 'normal', 'normal', 'normal', 'normal']
kernel_regularizer = [None, None, None, None, None]
bias_regularizer = [None, None, None, None, None]
dropout = [0., 0., 0., 0., 0.]
# optimizer = Adam(learning_rate=0.001)
optimizer = SGD(learning_rate=0.1, momentum=0.8, nesterov=False)

# Define how many epochs of training should be done and what is the batch size
epochs = 100000
# epochs = 100000 ## show this case 
batch_size = 180
print('Epochs: ', epochs)
print('Batch size: ', batch_size)

# Specify Keras callbacks
keras_callbacks = []
# keras_callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)]

# Define model with 4 layers, Nadam optimizer, dropouts and L1/L2 regularisers
def set_model (x_size, y_size):

    model = Sequential()
    
    model.add(Dense(nodes[0],
                    activation=activation[0],
                    kernel_initializer=kernel_initializer[0],
                    kernel_regularizer=kernel_regularizer[0],
                    bias_regularizer=bias_regularizer[0],
                    input_shape=(x_size,)
                    ))
    model.add(Dropout(dropout[0]))
    
    for i in range(1,len(nodes)):
        model.add(Dense(nodes[i],
                    activation=activation[i],
                    kernel_initializer=kernel_initializer[i],
                    kernel_regularizer=kernel_regularizer[i],
                    bias_regularizer=bias_regularizer[i]
                    ))
        model.add(Dropout(dropout[i]))

    model.add(Dense(y_size))

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[metrics.mae])
    return model

# Create the model
model = set_model (xn_train_arr.shape[1], yn_train_arr.shape[1])
# model.summary()


# Fit/Train Keras model
#------------------------------------------------------------------------------

# Fit model and record the history and validation
history = model.fit(xn_train_arr, yn_train_arr,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(xn_valid_arr, yn_valid_arr),
                    callbacks=keras_callbacks)


# Evaluate and report performance of the trained model
#------------------------------------------------------------------------------
train_score = model.evaluate(xn_train_arr, yn_train_arr, verbose=0)
valid_score = model.evaluate(xn_valid_arr, yn_valid_arr, verbose=0)

print('Train MAE: %.2e, Train Loss: %.2e' % (train_score[1], train_score[0]))
print('Valid MAE: %.2e, Valid Loss: %.2e' % (valid_score[1], valid_score[0]))

# Evaluate prediction errors
yn_train_arr_pred = model.predict(xn_train_arr)
y_train_arr_pred = y_mean + yn_train_arr_pred * y_std
train_rel_err = abs(y_train_arr_pred / y_train_arr - 1)

print('Train Minimum Error: %.3f%%' % np.min(train_rel_err*100))
print('Train Maximum Error: %.3f%%' % np.max(train_rel_err*100))
print('Train Mean Error: %.3f%%' % np.mean(train_rel_err*100))

yn_valid_arr_pred = model.predict(xn_valid_arr)
y_valid_arr_pred = y_mean + yn_valid_arr_pred * y_std
valid_rel_err = abs(y_valid_arr_pred / y_valid_arr - 1)

print('Valid Minimum Error: %.3f%%' % np.min(valid_rel_err*100))
print('Valid Maximum Error: %.3f%%' % np.max(valid_rel_err*100))
print('Valid Mean Error: %.3f%%' % np.mean(valid_rel_err*100))

# Elapsed time
dt = time.time() - start
print('Time elapsed: %f sec' % dt)

# This function allows plotting of the training history
def plot_hist (h, xsize=6, ysize=10):

    # Prepare plotting
    fig_size = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Traning vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot it
    plt.draw()
    # plt.show()
    plt_name = f'dlsalt_{model_name}_val.png'
    plt.savefig(plt_name)
    print('Saving', plt_name)
    return

# Now plot the training history
plot_hist(history.history, xsize=8, ysize=12)
