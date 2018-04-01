from keras.models import Sequential
from keras.layers import Dense
import numpy

from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from keras import backend as K
from sklearn.cross_validation import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("./pima-indians-diabetes.data", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y,epochs=50,batch_size=10,validation_split=0.13)
predictions = model.predict(X)


tahmin=numpy.array([1,89,66,23,94,28.1,0.167,21]).reshape(1,8)
print(model.predict_classes(tahmin))

def export_model(saver, model, input_node_names, output_node_name,MODEL_NAME):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
                         MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
                              False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
                              "save/restore_all", "save/Const:0", \
                              'out/modelim' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/modelim' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

    return

X_train, X_test, y_train, y_test = train_test_split(X, Y)
estimators = []
estimator = KerasRegressor(build_fn=model, epochs=20, batch_size=50, verbose=2)

export_model(tf.train.Saver(), estimator, ["dense_1_input"], "dense_3/Sigmoid","test")