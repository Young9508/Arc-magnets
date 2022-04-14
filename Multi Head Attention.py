import numpy as np
import tensorflow as tf
import pandas as pd
import time
import matplotlib.pyplot as plt
import itertools
import os
import random as rn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTHONHASHSEED'] = '0'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, Layer, GlobalMaxPool1D, Conv1D, \
    MaxPooling1D, Flatten, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from itertools import cycle
from window_slider import Slider
from sklearn import manifold
from sklearn.metrics import classification_report
from numpy import interp
from tensorflow.keras.models import Sequential, load_model

seed = 7
np.random.seed(seed)
rn.seed(seed)
tf.random.set_seed(seed)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

LABELS = ["0",
          "1"]


class Embedding(Layer):

    def __init__(self, vocab_size, model_dim, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        super(Embedding, self).__init__(**kwargs)

    def get_config(self):
        config = {"vocab_size": self._vocab_size, "model_dim": self._model_dim}
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            name="embeddings")
        super(Embedding, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5  # Scale
        return embeddings

    def compute_output_shape(self, input_shape):
        return input_shape + (self._model_dim,)


class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def get_config(self):
        config = {"model_dim": self._model_dim}
        base_config = super(PositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i - i % 2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])  # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])  # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def get_config(self):
        config = {}
        base_config = super(Add, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ScaledDotProductAttention(Layer):

    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2 ** 32 + 1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {"masking": self._masking, "future": self._future, "dropout_rate": self._dropout_rate}
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self._masking_num
        return outputs

    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self._masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        if self._masking:
            scaled_matmul = self.mask(scaled_matmul, masks)  # Mask(opt.)

        if self._future:
            scaled_matmul = self.future_mask(scaled_matmul)

        softmax_out = K.softmax(scaled_matmul)  # SoftMax
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)

        outputs = K.batch_dot(out, values)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {"n_heads": self._n_heads, "head_dim": self._head_dim, "dropout_rate": self._dropout_rate,
                  "masking": self._masking, "future": self._future, "trainable": self._trainable}
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        queries_linear = K.dot(queries, self._weights_queries)
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(
            masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionWiseFeedForward(Layer):

    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")
        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        return outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim


class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def visualize_scatter(data_2d, label_ids, z, figsize=(5.716, 3.578)):
    plt.figure(figsize=figsize)
    # plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth=1,
                    alpha=0.8,
                    label=label_id)
    plt.legend(loc='best')

    if z == 0:
        plt.savefig('./result/t_sne_original-' + file + ".png", dpi=800)
    elif z == 1:
        plt.savefig('./result/t_sne_conv-' + file + ".png", dpi=800)
    elif z == 2:
        plt.savefig('./result/t_sne_LSTM-' + file + ".png", dpi=800)
    elif z == 3:
        plt.savefig('./result/t_sne_Dense-' + file + '.png', dpi=800)
    plt.show()


def t_SNEPLOT(model, x_test, x_test_masks, x_test1, y_test1):
    intermed_tensor_func1 = K.function([model.input], [model.layers[4].output])
    intermed_tensor_func2 = K.function([model.input], [model.layers[8].output])
    intermed_tensor_func3 = K.function([model.input], [model.layers[-1].output])

    intermed_tensor1 = intermed_tensor_func1([x_test, x_test_masks])[0]
    intermed_tensor2 = intermed_tensor_func2([x_test, x_test_masks])[0]
    intermed_tensor3 = intermed_tensor_func3([x_test, x_test_masks])[0]

    print('Data dimension[1] after convolution processing is {}.Data dimension[-1] after convolution processing is {}'
          .format(intermed_tensor1.shape[1], intermed_tensor1.shape[-1]))
    print('Data dimension[1] after LSTM processing is {}.Data dimension[-1] after LSTM processing is {}'
          .format(intermed_tensor2.shape[1], intermed_tensor2.shape[-1]))
    print('Data dimension[0] after fully connected is {}.Data dimension[-1] after fully connected is {}'
          .format(intermed_tensor3.shape[0], intermed_tensor3.shape[-1]))

    intermed_tensor1 = np.reshape(intermed_tensor1, (intermed_tensor1.shape[0], -1))
    intermed_tensor2 = np.reshape(intermed_tensor2, (intermed_tensor2.shape[0], -1))
    intermed_tensor3 = np.reshape(intermed_tensor3, (intermed_tensor3.shape[0], -1))

    # intermed_tensor1 = intermed_tensor1.reshape(N, 3, -1)
    # intermed_tensor1 = intermed_tensor1.sum(axis=1)
    # intermed_tensor2 = intermed_tensor2.reshape(N, 3, -1)
    # intermed_tensor2 = intermed_tensor2.sum(axis=1)
    # intermed_tensor3 = intermed_tensor3.reshape(N, 3, -1)
    # intermed_tensor3 = intermed_tensor3.sum(axis=1)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=7)
    intermed_tsne3 = tsne.fit_transform(intermed_tensor3)
    intermed_tsne2 = tsne.fit_transform(intermed_tensor2)
    intermed_tsne1 = tsne.fit_transform(intermed_tensor1)
    intermed_tsne0 = tsne.fit_transform(x_test1)

    visualize_scatter(intermed_tsne3, y_test1, z=3)
    visualize_scatter(intermed_tsne2, y_test1, z=2)
    visualize_scatter(intermed_tsne1, y_test1, z=1)
    visualize_scatter(intermed_tsne0, y_test1, z=0)


def paintRoc(Y_valid, Y_pred):
    nb_classes = classes
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(5.716, 3.578))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'mediumseagreen', 'violet', 'dodgerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 10,
             }
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel('False Positive Rate', font1)
    plt.ylabel('True Positive Rate', font1)
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./result/ROC_6分类-' + file + ".png", dpi=800)
    plt.show()


def create_sliding_windows(m):
    B = []
    A = np.array(m)
    bucket_size = 3
    overlap_count = 0
    slider = Slider(bucket_size, overlap_count)
    slider.fit(A)
    while True:
        window_data = slider.slide()
        # do your stuff
        # print(window_data)
        B.append(np.mean(window_data).astype(int))
        if slider.reached_end_of_list(): break
    return B


# confusion_matrix_method2
def plot_confuse(x_val, y_val, z, normalize, target_names, cmap):
    cm = confusion_matrix(x_val, y_val)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('summer')

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar(fraction=0.045, pad=0.05)
    # plt.colorbar()

    # Calibration coordinate axis
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        # plt.xticks(tick_marks, target_names, rotation=45)
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 10,
             }
    plt.gcf().subplots_adjust(left=0.15)
    plt.ylabel('True label', font1)
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), font1)
    plt.xlabel('Predicted label\nABC', font1)
    if z == 1:
        # plt.title("Confusion Matrix-Training")
        plt.savefig('./result/CM_Training-' + file + ".png", dpi=800)
    else:
        # plt.title("Confusion Matrix-Testing")
        plt.savefig('./result/CM_Testing-' + file + ".png", dpi=800)
    plt.show()


def fit_model(max_len, model_dim, x_train, x_train_masks, y_train, batch_size, epochs, vocab_size):
    print('Model building ... ')
    inputs = Input(shape=(max_len, model_dim))
    model1 = Conv1D(filters=300, kernel_size=64, activation='relu', padding='same', strides=32)(inputs)
    model2 = MaxPooling1D(2)(model1)
    encodings = PositionEncoding(model_dim)(model2)
    encodings = Add()([model2, encodings])
    masks = Input(shape=((K.int_shape(encodings))[1],), name='masks')
    x1 = MultiHeadAttention(8, 15)([encodings, encodings, encodings, masks])
    x2 = GlobalAveragePooling1D()(x1)
    x3 = Dropout(0.2)(x2)
    outputs = Dense(2, activation='softmax')(x3)
    model = Model(inputs=[inputs, masks], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-09), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, './model/model-Multi-Head Attention' + ".png", show_shapes=True, show_layer_names=True)

    print("Model Training ... ")
    es = EarlyStopping(patience=5)
    history = model.fit([x_train, x_train_masks], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=[es], verbose=1)
    with open('./result/loss-' + file + ".txt", 'a', encoding='utf-8') as f:
        for j in range(len(history.history['loss'])):
            f.write(str(history.history['loss'][j]) + "\n")
    with open('./result/val_loss-' + file + ".txt", 'a', encoding='utf-8') as f:
        for j in range(len(history.history['val_loss'])):
            f.write(str(history.history['val_loss'][j]) + "\n")
    with open('./result/accuracy-' + file + ".txt", 'a', encoding='utf-8') as f:
        for j in range(len(history.history['accuracy'])):
            f.write(str(history.history['accuracy'][j]) + "\n")
    with open('./result/val_accuracy-' + file + ".txt", 'a', encoding='utf-8') as f:
        for j in range(len(history.history['val_accuracy'])):
            f.write(str(history.history['val_accuracy'][j]) + "\n")
    # model.save_weights(r'./result/model_weights-'+file+".h5")
    # model.save(r'./result/model-'+file+".h5")
    # model = load_model('./result/model-'+file+".h5")
    return model, history


def experiment(train, test, batch_size, epochs, vocab_size, max_len, model_dim):
    index = []
    x_train1, y_train1 = train[:, :-1], train[:, -1]
    x_test1, y_test1 = test[:, :-1], test[:, -1]

    x_train = x_train1.reshape(x_train1.shape[0], 16000, 1)
    x_test = x_test1.reshape(x_test1.shape[0], 16000, 1)

    # 普通情况，训练集为80，测试集为40
    a = np.ones((3360, 250, 300))
    b = np.ones((240, 250, 300))

    x_train_masks = tf.equal(a[:, :, 0], 20000)
    x_test_masks = tf.equal(b[:, :, 0], 20000)

    y_train = to_categorical(y_train1)
    # y_test = to_categorical(y_test1)

    starttime = time.time()
    model, history = fit_model(max_len, model_dim, x_train, x_train_masks, y_train, batch_size, epochs, vocab_size)
    endtime = time.time()
    dtime = endtime - starttime

    # forecast the entire training dataset to build up state for forecasting
    print('Forecasting Training Data')
    y_trainpredictions = model.predict([x_train, x_train_masks], batch_size=batch_size)
    predictions_train = np.array(y_trainpredictions)
    predictions_train1 = np.argmax(predictions_train, axis=1)
    index0 = np.arange(0, train.shape[0])
    index1 = index0[predictions_train1 != y_train1]
    for i in range(len(index1)):
        j = index1[i]
        expected = float(y_train1[j])
        p = predictions_train1[j]
        print('Sammple=%d, Predicted=%f, Expected=%f' % (j + 1, p, expected))
    # show_confusion_matrix(labels_train, predictions_train, 1)
    plot_confuse(y_train1, predictions_train1, z=1, normalize=False, target_names=LABELS, cmap=plt.cm.summer)

    # forecast the test data
    print('Forecasting Testing Data')
    labels_test = y_test1.astype(int)
    labels_test2 = create_sliding_windows(labels_test)
    labels_test3 = to_categorical(labels_test2)
    starttime = time.time()
    y_test = model.predict([x_test, x_test_masks], batch_size=1)
    predictions_test = np.array(y_test)
    predictions_test = predictions_test.reshape(N, 3, 2)
    predictions_test2 = predictions_test.sum(axis=1)
    predictions_test3 = np.argmax(predictions_test2, axis=1)
    endtime = time.time()
    ttime = (endtime - starttime) / predictions_test3.shape[0]
    index2 = np.arange(0, labels_test3.shape[0])
    index3 = index2[predictions_test3 != labels_test2]
    for i in range(len(index3)):
        j = index3[i]
        expected = float(labels_test2[j])
        p = predictions_test3[j]
        print('Sammple=%d, Predicted=%f, Expected=%f' % (j + 1, p, expected))
    # show_confusion_matrix(labels_test2, predictions_test3, 2)
    plot_confuse(labels_test2, predictions_test3, z=2, normalize=False, target_names=LABELS, cmap=plt.cm.summer)
    predictions_test4 = to_categorical(predictions_test3)
    paintRoc(labels_test3, predictions_test4)
    t_SNEPLOT(model, x_test, x_test_masks, x_test1, y_test1)

    # report performance
    accuracy_train = accuracy_score(y_train1, predictions_train1)
    print('Train Accuracy: %.3f' % accuracy_train)
    accuracy_test = accuracy_score(labels_test2, predictions_test3)
    print('Test Accuracy: %.3f' % accuracy_test)
    print("模型训练运行时间：%.8f s" % dtime)
    print("单个样本测试时间：%.8f s" % ttime)
    index.append(accuracy_train)
    index.append(accuracy_test)
    index.append(dtime)
    index.append(ttime)
    with open(r'./result/index-' + file + ".txt", 'a', encoding='utf-8') as f:
        for j in range(len(index)):
            f.write(str(index[j]) + "\n")
    predictions = np.concatenate((predictions_train1, predictions_test3), axis=0)
    with open(r'./result/prediction_data-' + file + ".txt", 'a', encoding='utf-8') as f:
        for k in range(len(predictions)):
            f.write(str(predictions[k]) + "\n")

    plt.figure(1, figsize=(8, 5))
    x_train = np.arange(1, y_train1.shape[0] + 1, 1)
    plt.scatter(x_train, y_train1, label='true', marker='o', s=20, color='blue')
    plt.scatter(x_train, predictions_train1, label='estimation', marker='x', s=20, color='red')
    plt.legend()
    plt.title('Classify_train')
    plt.savefig('./result/SOC_train-' + file + ".png", dpi=800)
    plt.show()

    print("\n--- Classification report for train data ---\n")
    print(classification_report(y_train1, predictions_train1, digits=2))

    plt.figure(2, figsize=(8, 5))
    x_test = np.arange(1, len(labels_test2) + 1, 1)
    plt.scatter(x_test, labels_test2, label='true', marker='o', s=20, color='blue')
    plt.scatter(x_test, predictions_test3, label='estimation', marker='x', s=20, color='red')
    plt.legend()
    plt.title('Classify_test')
    plt.savefig('./result/SOC_test-' + file + ".png", dpi=800)
    plt.show()

    print("\n--- Classification report for test data ---\n")
    print(classification_report(labels_test2, predictions_test3, digits=2))

    return history


def run():
    global file, N, classes, L, Q
    N = 80
    classes = 2
    L = 16000
    Q = 1

    vocab_size = 10080
    max_len = L
    model_dim = Q
    batch_size = 32
    epochs = 10

    file_name_train = './data/CtrainO_80_E162.csv'
    file_name_test = './data/CtestO_40_E162.csv'
    file = file_name_train[7:-4]

    train = pd.read_csv(file_name_train, header=None)
    test = pd.read_csv(file_name_test, header=None)

    train = train.values
    test = test.values
    history = experiment(train, test, batch_size, epochs, vocab_size, max_len, model_dim)

    return history


history = run()
plt.figure(3, figsize=(6, 4))
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.savefig('./result/Loss-' + file + ".png", dpi=800)
plt.show()

plt.figure(4, figsize=(6, 4))
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.savefig('./result/Accuracy-' + file + ".png", dpi=800)
plt.show()
