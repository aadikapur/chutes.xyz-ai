#!/usr/bin/python

import tensorflowjs as tfjs
from keras.models import load_model

model = load_model('./mymodel.model')
tfjs.converters.save_keras_model(model, 'tfjs_dir')
