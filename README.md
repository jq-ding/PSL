# PSL
Implementation of  Protein Subcellular Localization

### A Multi-scale Multi-model Deep Neural Network via Ensemble Strategy on High-throughput Microscopy Image for Protein Subcellular Localization

![flow](https://user-images.githubusercontent.com/93422935/139533266-24f62fe7-c401-4440-b94a-c45ae9382477.png)


In this study, we propose a multi-scale multi-model deep neural network via ensemble strategy for protein subcellular localization on single-cell high-throughput images. First of all, we employ a deep convolutional neural network as multi-scale feature extractor and use global average pooling to map extracted features at different stages into feature vectors, then concatenate these multi-scale features to form a multi-model structure for image classification. In addition, we add Squeeze-and-Excitation Blocks to the network to emphasize more informative features. What’s more, we use an ensemble method to fuse the classification results from the multi-model structure to obtain the final sub-cellular location of each single-cell image. Experiments show the validity and effectiveness of our method on yeast cell images, it can significantly improve the accuracy of high-throughput microscopy image-based protein subcellular localization, and we achieve the classification accuracy of 0.9098 on the high-throughput microscopy images of yeast cells. In the work of protein subcellular localization, our method provides a framework for processing and classifying microscope images, and further lays the foundation for the study of protein and gene functions.

### Data

T. Pärnamaa, L. Parts, Accurate classification of protein subcellular localization from high-throughput microscopy images using deep learning, G3: Genes, Genomes, Genetics 7 (5) (2017) 1385–1392. arXiv:https: //www.g3journal.org/content/7/5/1385.full.pdf, doi:10.1534/g3. 810 116.033654.

### Data pre-processing

```
dataset.py 
```

### Training

```
python training.py

model = ResNet34()
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

metric = 'val_accuracy'
filepath="/home/dingjiaqi/Program/deepyeast-master/deepyeast-master/deepyeast/weights/A-{accuracy:.3f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor=metric, verbose=1, save_best_only=True, mode='max')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, cooldown=0, min_lr=1e-5)
callbacks_list = [checkpoint, reduce_lr]

batch_size = 32
epochs = 50
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          validation_split=0.1,
          # validation_data=(x_val, y_val)
          callbacks=callbacks_list)
```

### Predict

```
# predict and output the performance on each class

python predict.py resnet weights/weight_name.hdf5
```

### Feature_extraction

```
# extract feature and visualize it by t-SNE

python feature_extration
```

### Results

#### Performance on different backbone networks

<img src="https://user-images.githubusercontent.com/93422935/139533290-82c4b137-cf98-4646-bae0-9d51dd4952f7.png" alt="radar" width="50%" height="50%">

#### Verify Effectiveness in SE Block

<img src="https://user-images.githubusercontent.com/93422935/139533294-df337f5e-8388-4e60-bc71-38eb787bf164.png" alt="withSE" width="50%" height="50%">

##### F-values of features with or without SE

<img src="https://user-images.githubusercontent.com/93422935/139533310-9a76b075-9ed0-4a31-bee0-182beed4baab.png" alt="sebar" width="50%" height="50%">

#### Feature distributions of Model 4 visualized by t-SNE

<img src="https://user-images.githubusercontent.com/93422935/139533313-d4c389ba-e3af-4c89-94d9-99b985c6bc95.png" alt="tsne" width="50%" height="50%">

####  Performance on various models in our network

<img src="https://user-images.githubusercontent.com/93422935/139533377-a0ab4965-1265-450c-8cce-a5484cdc5d6f.png" alt="ms" width="50%" height="50%">

#### Compare with others

<img src="https://user-images.githubusercontent.com/93422935/139533330-403ff750-00f1-45a5-be06-606ff40599c7.png" width="50%" height="50%"/>

![image](https://user-images.githubusercontent.com/93422935/139533511-ed54cfe3-2af3-48d0-bbb2-34322b70c289.png)
