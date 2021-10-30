import keras
import argparse
import numpy as np
from sklearn import metrics

from psl.dataset import load_data
from psl.utils import preprocess_input

import os
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
tf.compat.v1.Session(config=config)


def nn_preprocess(data):
    for i in range(data.shape[0]):
        data[i, :, :, :] = (data[i, :, :, :]-np.min(data[i, :, :, :]))/(np.max(data[i, :, :, :])-np.min(data[i, :, :, :]))
    data = np.reshape(data, (len(data), 64, 64, 2))
    return data


def parse_args():
    available_models = ['deepyeast', 'resnet', 'mobilenet', 'densenet']
    parser = argparse.ArgumentParser(description="Evaluate model's performance.")
    parser.add_argument('model', help="Specification of the model's structure",
                        choices=available_models)
    parser.add_argument('weights', help="Path to weights file.")
    parser.add_argument('--split', nargs='+', help='', default=['test', 'val', 'train'])
    args = parser.parse_args()
    return args


def evaluate(split):
    print('Loading {} set...'.format(split))
    x, y_true = load_data(split)
    x = preprocess_input(x)
    y_pred = model.predict(x)


    print(y_pred.shape)
    # np.savetxt("/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/M1.txt", y_pred)

    # np.savetxt("/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/4cSEnew_gap2.txt", y_pred)
    y_pred = y_pred.argmax(axis=1)

    print("{} set statistics:".format(split))
    print("Top-1-accuracy: {:.4f}".format(np.mean(y_true == y_pred)))
    print(metrics.classification_report(y_true, y_pred, digits=4))
    # Cmo = confusion_matrix(y_true, y_pred)
    # np.savetxt("/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/c1.txt", Cmo)



def load_model(model_name):
    if model_name == 'resnet':
        from psl.models import ResNet34
        model = ResNet34()
    elif model_name == 'mobilenet':
        from psl.models import MobileNet
        model = MobileNet()
    elif model_name == 'densenet':
        from psl.models import DenseNet40_BC
        model = DenseNet40_BC()
    return model


if __name__ == '__main__':
    num_classes = 12

    args = parse_args()
    print(args)
    print('Loading model...')
    model = load_model(args.model)
    model.load_weights(args.weights)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy', 'accuracy'])

    if 'test' in args.split:
        evaluate('test')
