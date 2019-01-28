---
title: "Bayesian Active Learning"
mathjax: "true"
---

# Active Learning

A  big  challenge  in  many  machine  learning  applications is obtaining labelled data.  This can be a long, laborious and costly process. In active learning, a model is trained on a small amount of data (the initial training set), and an acquisition function(often based on the model’s uncertainty) decides which data points to ask an external oracle (Typically a Human expert ) for a label.  The acquisition function selects one or more points from a pool of un-labelled data points, with the pool points lying outside of the training set. The oracle labels the selected data points, these are added to the training set and a new model is trained on the updated training set. This process is then repeated, with the training set increasing in size over time.  The advantage of such systems is that they often result in dramatic reductions in the amount of labelling required to train an ML system.

A major challenge in active learning is its lack of scalability to high-dimensional data. As a result most approaches to active learning have focused on low dimensional problems.

# Bayesian approaches to deep learning

Recent advances in deep learning depend on large amounts of data. Second, many AL acquisition functions rely on model uncertainty. But in deep learning we rarely represent such model uncertainty. To resolve this we use approximate Bayesian inference and use the uncertainty in the acquisition function to do active learning on MNIST dataset.

Bayesian CNNs are CNNs with prior probability distributions placed over a setof model parameters  
$$\omega = \left\{ W _ { 1 } , \ldots , W _ { L } \right\}$$

$$p ( y = c | \mathbf { x } , \boldsymbol { \omega } ) = \operatorname { softmax } \left( \mathbf { f } ^ { \omega } ( \mathbf { x } ) \right)$$


To perform approximate inference in the Bayesian CNN model we make use of stochastic regularization techniques such as dropout which was shown to be a Bayesian approximation
by Gal Et al(2016). Inference is done by training a model with dropout before every weight layer, and by performing dropout attest time as well to sample from the approximate posterior(stochastic forward passes, referred to as MC dropout).

More formally, this approach is equivalent to performingapproximate variational inference where we find a distribution $q _ { 0 } ^ { * } ( \omega )$ in  a  tractable  family  which  minimizes  the Kullback-Leibler (KL) divergence to the true model posterior.

$$p ( y = c | \mathbf { x } , \mathcal { D } _ { \text { train } } ) = \int p ( y = c | \mathbf { x } , \boldsymbol { \omega } ) p ( \boldsymbol { \omega } | \mathcal { D } _ { \text { train } } ) \mathrm { d } \boldsymbol { \omega }$$  

$$\approx \int p ( y = c | \mathbf { x } , \omega ) q _ { \theta } ^ { * } ( \omega ) \mathrm { d } \omega$$  

$$\approx \frac { 1 } { T } \sum _ { t = 1 } ^ { T } p ( y = c | \mathbf { x } , \widehat { \omega } _ { t } )$$  

with $\widehat { \omega } _ { t } \sim q _ { 0 } ^ { * } ( \omega )$ where $q _ { 0 } ( \omega )$ is the dropout distribution.

#  Acquisition Functions  

Acquisition function is a function of x that the AL system uses to decide where to query next:

$$x ^ { * } = \operatorname { argmax } _ { x \in \mathcal { D } _ { \text { pool } } } a ( x , \mathcal { M } )$$

Here we choose pool points that maximize the predictive Shannon Entropy  
$$\begin{array} { l } { \mathbb { H } [ y | \mathbf { x } , \mathcal { D } _ { \text { train } } ] : = } \\ { \quad - \sum p ( y = c | \mathbf { x } , \mathcal { D } _ { \text { train } } ) \log p ( y = c | \mathbf { x } , \mathcal { D } _ { \text { train } } ) } \end{array}$$

In MNIST we are able to get 95% accuracy after querying only 450 samples.


<p align="center">
<img src="https://imgur.com/Elz8Sdg.jpg">

</p>

<center>
Active learning with Dropout
</center>



```python
from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K  
import random
random.seed(2001)
import scipy.io
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2


Experiments = 3

batch_size = 128
nb_classes = 10

#use a large number of epochs
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 5
# convolution kernel size
nb_conv = 3


score=0
all_accuracy = 0
acquisition_iterations = 98

#use a large number of dropout iterations
dropout_iterations = 100

Queries = 10


Experiments_All_Accuracy = np.zeros(shape=(acquisition_iterations+1))

for e in range(Experiments):

	print('Experiment Number ', e)


	# the data, shuffled and split between tran and test sets
	(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

	X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

	random_split = np.asarray(random.sample(range(0,X_train_All.shape[0]), X_train_All.shape[0]))

	X_train_All = X_train_All[random_split, :, :, :]
	y_train_All = y_train_All[random_split]

	X_valid = X_train_All[10000:15000, :, :, :]
	y_valid = y_train_All[10000:15000]

	X_Pool = X_train_All[20000:60000, :, :, :]
	y_Pool = y_train_All[20000:60000]


	X_train_All = X_train_All[0:10000, :, :, :]
	y_train_All = y_train_All[0:10000]

	#training data to have equal distribution of classes
	idx_0 = np.array( np.where(y_train_All==0)  ).T
	idx_0 = idx_0[0:2,0]
	X_0 = X_train_All[idx_0, :, :, :]
	y_0 = y_train_All[idx_0]

	idx_1 = np.array( np.where(y_train_All==1)  ).T
	idx_1 = idx_1[0:2,0]
	X_1 = X_train_All[idx_1, :, :, :]
	y_1 = y_train_All[idx_1]

	idx_2 = np.array( np.where(y_train_All==2)  ).T
	idx_2 = idx_2[0:2,0]
	X_2 = X_train_All[idx_2, :, :, :]
	y_2 = y_train_All[idx_2]

	idx_3 = np.array( np.where(y_train_All==3)  ).T
	idx_3 = idx_3[0:2,0]
	X_3 = X_train_All[idx_3, :, :, :]
	y_3 = y_train_All[idx_3]

	idx_4 = np.array( np.where(y_train_All==4)  ).T
	idx_4 = idx_4[0:2,0]
	X_4 = X_train_All[idx_4, :, :, :]
	y_4 = y_train_All[idx_4]

	idx_5 = np.array( np.where(y_train_All==5)  ).T
	idx_5 = idx_5[0:2,0]
	X_5 = X_train_All[idx_5, :, :, :]
	y_5 = y_train_All[idx_5]

	idx_6 = np.array( np.where(y_train_All==6)  ).T
	idx_6 = idx_6[0:2,0]
	X_6 = X_train_All[idx_6, :, :, :]
	y_6 = y_train_All[idx_6]

	idx_7 = np.array( np.where(y_train_All==7)  ).T
	idx_7 = idx_7[0:2,0]
	X_7 = X_train_All[idx_7, :, :, :]
	y_7 = y_train_All[idx_7]

	idx_8 = np.array( np.where(y_train_All==8)  ).T
	idx_8 = idx_8[0:2,0]
	X_8 = X_train_All[idx_8, :, :, :]
	y_8 = y_train_All[idx_8]

	idx_9 = np.array( np.where(y_train_All==9)  ).T
	idx_9 = idx_9[0:2,0]
	X_9 = X_train_All[idx_9, :, :, :]
	y_9 = y_train_All[idx_9]

	X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
	y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')

	print('Distribution of Training Classes:', np.bincount(y_train))


	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')
	X_Pool = X_Pool.astype('float32')
	X_train /= 255
	X_valid /= 255
	X_Pool /= 255
	X_test /= 255

	Y_test = np_utils.to_categorical(y_test, nb_classes)
	Y_valid = np_utils.to_categorical(y_valid, nb_classes)
	Y_Pool = np_utils.to_categorical(y_Pool, nb_classes)


	#loss values in each experiment
	Pool_Valid_Loss = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Loss = np.zeros(shape=(nb_epoch, 1))
	Pool_Valid_Acc = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Acc = np.zeros(shape=(nb_epoch, 1))
	x_pool_All = np.zeros(shape=(1))

	Y_train = np_utils.to_categorical(y_train, nb_classes)

	print('Training Model Without Acquisitions in Experiment', e)



	model = Sequential()
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.25))



	c = 3.5
	Weight_Decay = c / float(X_train.shape[0])
	model.add(Flatten())
	model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))


	model.compile(loss='categorical_crossentropy', optimizer='adam')
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
	Train_Result_Optimizer = hist.history
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T
	Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
	Valid_Loss = np.asarray([Valid_Loss]).T
	Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
	Train_Acc = np.array([Train_Acc]).T
	Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
	Valid_Acc = np.asarray([Valid_Acc]).T


	Pool_Train_Loss = Train_Loss
	Pool_Valid_Loss = Valid_Loss
	Pool_Train_Acc = Train_Acc
	Pool_Valid_Acc = Valid_Acc



	print('Evaluating Test Accuracy Without Acquisition')
	score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

	all_accuracy = acc

	print('Starting Active Learning in Experiment ', e)


	for i in range(acquisition_iterations):
		print('POOLING ITERATION', i)


		pool_subset = 2000
		pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset))
		X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]
		y_Pool_Dropout = y_Pool[pool_subset_dropout]

		score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))

		for d in range(dropout_iterations):
			print ('Dropout Iteration', d)
			dropout_score = model.predict_stochastic(X_Pool_Dropout,batch_size=batch_size, verbose=1)
			#np.save(''+'Dropout_Score_'+str(d)+'Experiment_' + str(e)+'.npy',dropout_score)
			score_All = score_All + dropout_score

		Avg_Pi = np.divide(score_All, dropout_iterations)
		Log_Avg_Pi = np.log2(Avg_Pi)
		Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
		Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

		U_X = Entropy_Average_Pi

		# THIS FINDS THE MINIMUM INDEX
		# a_1d = U_X.flatten()
		# x_pool_index = a_1d.argsort()[-Queries:]

		a_1d = U_X.flatten()
		x_pool_index = U_X.argsort()[-Queries:][::-1]

		x_pool_All = np.append(x_pool_All, x_pool_index)

			#saving pooled images
		# for im in range(x_pool_index[0:2].shape[0]):
		# 	Image = X_Pool[x_pool_index[im], :, :, :]
		# 	img = Image.reshape((28,28))
			#sp.misc.imsave('/home/ri258/Documents/Project/Active-Learning-Deep-Convolutional-Neural-Networks/ConvNets/Cluster_Experiments/Dropout_Max_Entropy/Pooled_Images/'+ 'Exp_'+str(e) + 'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

		Pooled_X = X_Pool_Dropout[x_pool_index, 0:3,0:32,0:32]
		Pooled_Y = y_Pool_Dropout[x_pool_index]

		#first delete the random subset used for test time dropout from X_Pool
		#Delete the pooled point from this pool set (this random subset)
		#then add back the random pool subset with pooled points deleted back to the X_Pool set
		delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
		delete_Pool_Y = np.delete(y_Pool, (pool_subset_dropout), axis=0)

		delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (x_pool_index), axis=0)
		delete_Pool_Y_Dropout = np.delete(y_Pool_Dropout, (x_pool_index), axis=0)

		X_Pool = np.concatenate((X_Pool, X_Pool_Dropout), axis=0)
		y_Pool = np.concatenate((y_Pool, y_Pool_Dropout), axis=0)

		print('Acquised Points added to training set')

		X_train = np.concatenate((X_train, Pooled_X), axis=0)
		y_train = np.concatenate((y_train, Pooled_Y), axis=0)

		print('Train Model with pooled points')

		# convert class vectors to binary class matrices
		Y_train = np_utils.to_categorical(y_train, nb_classes)


		model = Sequential()
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
		model.add(Dropout(0.25))


		c = 3.5
		Weight_Decay = c / float(X_train.shape[0])
		model.add(Flatten())
		model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))


		model.compile(loss='categorical_crossentropy', optimizer='adam')
		hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
		Train_Result_Optimizer = hist.history
		Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
		Train_Loss = np.array([Train_Loss]).T
		Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
		Valid_Loss = np.asarray([Valid_Loss]).T
		Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
		Train_Acc = np.array([Train_Acc]).T
		Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
		Valid_Acc = np.asarray([Valid_Acc]).T

		#Accumulate the training and validation/test loss after every pooling iteration - for plotting
		Pool_Valid_Loss = np.append(Pool_Valid_Loss, Valid_Loss, axis=1)
		Pool_Train_Loss = np.append(Pool_Train_Loss, Train_Loss, axis=1)
		Pool_Valid_Acc = np.append(Pool_Valid_Acc, Valid_Acc, axis=1)
		Pool_Train_Acc = np.append(Pool_Train_Acc, Train_Acc, axis=1)



```
