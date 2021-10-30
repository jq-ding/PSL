from psl..dataset import load_data
from psl..utils import preprocess_input, create_feature_extractor
from psl..models import DeepYeast, ResNet34
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from deepyeast.models import DenseNet40_BC
# from deepyeast.models import ResNet50
# from deepyeast.models import MobileNet
from collections import Counter

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import keras
# config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess =tf.compat.v1.Session(config=config)

x_test, y_test = load_data('test')
# y_test = y_test.astype(np.uint8)
# np.savetxt("/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/test_labels.txt", y_test)

x = x_test
# x = x_test[[0]]
x = preprocess_input(x)

# res = Counter(y_test)
# print("count:", res)

model = ResNet34()
# weights_path = '/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/weights/2gap1-weights-47-1.000.hdf5'
weights_path = '/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/weights/4cSEnew-weights-21-0.971.hdf5'

model.load_weights(weights_path)
# model = DenseNet40_BC()
model.summary()  # see feature names
gap_extractor = create_feature_extractor(model, layer_name="global_average_pooling2d_20")
gap_features = gap_extractor.predict(x)
# gap_features = gap_features.reshape(12500, 13120)

print("feature_dimen:", gap_features.shape)
np.savetxt("/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/gap4_features.txt", gap_features)
print("save over")

#  begin

# y = TSNE(n_components=2, perplexity=50.0, early_exaggeration=15, init='pca')
# x_tsne = y.fit_transform(gap_features)
# print("x_tsne:", x_tsne.shape)
# #
# n_0 = int(res[0])
# n_1 = int(res[1])
# n_2 = int(res[2])
# n_3 = int(res[3])
# n_4 = int(res[4])
# n_5 = int(res[5])
# n_6 = int(res[6])
# n_7 = int(res[7])
# n_8 = int(res[8])
# n_9 = int(res[9])
# n_10 = int(res[10])
# n_11 = int(res[11])
#
# # x_tsne_0 = np.zeros((n_0, 3))
# # x_tsne_1 = np.zeros((n_1, 3))
# # x_tsne_2 = np.zeros((n_2, 3))
# # x_tsne_3 = np.zeros((n_3, 3))
# # x_tsne_4 = np.zeros((n_4, 3))
# # x_tsne_5 = np.zeros((n_5, 3))
# # x_tsne_6 = np.zeros((n_6, 3))
# # x_tsne_7 = np.zeros((n_7, 3))
# # x_tsne_8 = np.zeros((n_8, 3))
# # x_tsne_9 = np.zeros((n_9, 3))
# # x_tsne_10 = np.zeros((n_10, 3))
# # x_tsne_11 = np.zeros((n_11, 3))
#
# x_tsne_0 = np.zeros((n_0, 2))
# x_tsne_1 = np.zeros((n_1, 2))
# x_tsne_2 = np.zeros((n_2, 2))
# x_tsne_3 = np.zeros((n_3, 2))
# x_tsne_4 = np.zeros((n_4, 2))
# x_tsne_5 = np.zeros((n_5, 2))
# x_tsne_6 = np.zeros((n_6, 2))
# x_tsne_7 = np.zeros((n_7, 2))
# x_tsne_8 = np.zeros((n_8, 2))
# x_tsne_9 = np.zeros((n_9, 2))
# x_tsne_10 = np.zeros((n_10, 2))
# x_tsne_11 = np.zeros((n_11, 2))
#
# a, b, c, d, e, f, g, h, m, j, k, l = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# for i in range(len(y_test)):
#     if y_test[i] == 0:
#         x_tsne_0[a] = x_tsne[i]
#         a = a + 1
#     if y_test[i] == 1:
#         x_tsne_1[b] = x_tsne[i]
#         b = b + 1
#     if y_test[i] == 2:
#         x_tsne_2[c] = x_tsne[i]
#         c = c + 1
#     if y_test[i] == 3:
#         x_tsne_3[d] = x_tsne[i]
#         d = d + 1
#     if y_test[i] == 4:
#         x_tsne_4[e] = x_tsne[i]
#         e = e + 1
#     if y_test[i] == 5:
#         x_tsne_5[f] = x_tsne[i]
#         f = f + 1
#     if y_test[i] == 6:
#         x_tsne_6[g] = x_tsne[i]
#         g = g + 1
#     if y_test[i] == 7:
#         x_tsne_7[h] = x_tsne[i]
#         h = h + 1
#     if y_test[i] == 8:
#         x_tsne_8[m] = x_tsne[i]
#         m = m + 1
#     if y_test[i] == 9:
#         x_tsne_9[j] = x_tsne[i]
#         j = j + 1
#     if y_test[i] == 10:
#         x_tsne_10[k] = x_tsne[i]
#         k = k + 1
#     if y_test[i] == 11:
#         x_tsne_11[l] = x_tsne[i]
#         l = l + 1
#
# plt.scatter(x_tsne_0[:, 0], x_tsne_0[:, 1], c='b', marker='.', s=8, label='0')
# plt.scatter(x_tsne_1[:, 0], x_tsne_1[:, 1], c='r', marker='.', s=8, label='1')
# plt.scatter(x_tsne_2[:, 0], x_tsne_2[:, 1], c='g', marker='.', s=8, label='2')
# plt.scatter(x_tsne_3[:, 0], x_tsne_3[:, 1], c='y', marker='.', s=8, label='3')
# plt.scatter(x_tsne_4[:, 0], x_tsne_4[:, 1], c='c', marker='.', s=8, label='4')
# plt.scatter(x_tsne_5[:, 0], x_tsne_5[:, 1], c='k', marker='.', s=8, label='5')
#
# plt.scatter(x_tsne_6[:, 0], x_tsne_6[:, 1], c='b', marker='.', s=8, label='6')
# plt.scatter(x_tsne_7[:, 0], x_tsne_7[:, 1], c='r', marker='.', s=8, label='7')
# plt.scatter(x_tsne_8[:, 0], x_tsne_8[:, 1], c='g', marker='.', s=8, label='8')
# plt.scatter(x_tsne_9[:, 0], x_tsne_9[:, 1], c='y', marker='.', s=8, label='9')
# plt.scatter(x_tsne_10[:, 0], x_tsne_10[:, 1], c='c', marker='.', s=8, label='10')
# plt.scatter(x_tsne_11[:, 0], x_tsne_11[:, 1], c='k', marker='.', s=8, label='11')
#
# # fig = plt.figure()
# # ax = Axes3D(fig)
# #
# # ax.scatter(x_tsne_0[:, 0], x_tsne_0[:, 1], x_tsne_0[:, 2], c='b', marker='.', label='0')
# # ax.scatter(x_tsne_1[:, 0], x_tsne_1[:, 1], x_tsne_1[:, 2], c='r', marker='.', label='1')
# # ax.scatter(x_tsne_2[:, 0], x_tsne_2[:, 1], x_tsne_2[:, 2], c='g', marker='.', label='2')
# # ax.scatter(x_tsne_3[:, 0], x_tsne_3[:, 1], x_tsne_3[:, 2], c='y', marker='.', label='3')
# # ax.scatter(x_tsne_4[:, 0], x_tsne_4[:, 1], x_tsne_4[:, 2], c='c', marker='.', label='4')
# # ax.scatter(x_tsne_5[:, 0], x_tsne_5[:, 1], x_tsne_5[:, 2], c='k', marker='.', label='5')
# #
# # ax.scatter(x_tsne_6[:, 0], x_tsne_6[:, 1], x_tsne_6[:, 2], c='b', marker='.', label='6')
# # ax.scatter(x_tsne_7[:, 0], x_tsne_7[:, 1], x_tsne_7[:, 2], c='r', marker='.', label='7')
# # ax.scatter(x_tsne_8[:, 0], x_tsne_8[:, 1], x_tsne_8[:, 2], c='g', marker='.', label='8')
# # ax.scatter(x_tsne_9[:, 0], x_tsne_9[:, 1], x_tsne_9[:, 2], c='y', marker='.', label='9')
# # ax.scatter(x_tsne_10[:, 0], x_tsne_10[:, 1], x_tsne_10[:, 2], c='c', marker='.', label='10')
# # ax.scatter(x_tsne_11[:, 0], x_tsne_11[:, 1], x_tsne_11[:, 2], c='k', marker='.', label='11')
#
#
# plt.legend()
# plt.savefig("/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/pictures/12gap1_tsne6.png")
