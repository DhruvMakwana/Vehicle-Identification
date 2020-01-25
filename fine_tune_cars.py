#importing libraries
#To run this file, execute folloing command
#python fine_tune_cars.py --vgg vgg16/vgg16 --checkpoints checkpoints --prefix vggnet
from config import car_config as config
import mxnet as mx
import argparse
import  logging
import os

#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--vgg", required=True,
	help="path to pre-trained VGGNet for fine-tuning")
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
	help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG,
	filename="training_{}.log".format(args["start_epoch"]),
	filemode="w")

#determine the batch
batchSize = config.BATCH_SIZE * config.NUM_DEVICES

trainIter = mx.io.ImageRecordIter(
	path_imgrec=config.TRAIN_MX_REC,
	data_shape=(3, 224, 224),
	batch_size=batchSize,
	rand_crop=True,
	rand_mirror=True,
	rotate=15,
	max_shear_ratio=0.1,
	mean_r=config.R_MEAN,
	mean_g=config.G_MEAN,
	mean_b=config.B_MEAN,
	preprocess_threads=config.NUM_DEVICES * 2)

valIter = mx.io.ImageRecordIter(
	path_imgrec=config.VAL_MX_REC,
	data_shape=(3, 224, 224),
	batch_size=batchSize,
	mean_r=config.R_MEAN,
	mean_g=config.G_MEAN,
	mean_b=config.B_MEAN)

opt = mx.optimizer.SGD(learning_rate=1e-4, momentum=0.9, wd=0.0005,
	rescale_grad=1.0 / batchSize)
#ctx = [mx.gpu(3)]

checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
argParams = None
auxParams = None
allowMissing = False

if args["start_epoch"] <= 0:
	print("[INFO] loading pre-trained model...")
	(symbol, argParams, auxParams) = mx.model.load_checkpoint(args["vgg"], 0)
	allowMissing = True

	#grab the layers from the pre-trained model, then find the dropout layer *prior* to 
	#the final FC layer (i.e., the layer that contains the number of class labels)
	#HINT: you can find layer names like this: 
	#for layer in layers:
		#print(layer.name)
	layers = symbol.get_internals()
	net = layers["drop7_output"]
	net = mx.sym.FullyConnected(data=net, num_hidden=config.NUM_CLASSES, name="fc8")
	net = mx.sym.SoftmaxOutput(data=net, name="softmax")
	#construct a new set of network arguments, removing any previous arguments pertaining 
	#to FC8 (this will allow us to train the final layer)
	argParams = dict({k:argParams[k] for k in argParams
		if "fc8" not in k})
	#delete any parameter entries for fc8, the FC layer we just surgically removed 
	#from the network. The problem is that argParams does not contain any information 
	#regarding ournew FC head, which is exactly why we set allowMissing to True earlier 
	#in the code.

else:
	#load the checkpoint from disk
	print("[INFO] loading epoch {}...".format(args["start_epoch"]))
	(net, argParams, auxParams) = mx.model.load_checkpoint(
		checkpointsPath, args["start_epoch"])

batchEndCBs = [mx.callback.Speedometer(batchSize, 50)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5),
	mx.metric.CrossEntropy()]

print("[INFO] training network...")
model = mx.mod.Module(symbol=net)
model.fit(
	trainIter,
	eval_data=valIter,
	num_epoch=65,
	begin_epoch=args["start_epoch"],
	initializer=mx.initializer.Xavier(),
	arg_params=argParams,
	aux_params=auxParams,
	optimizer=opt,
	allow_missing=allowMissing,
	eval_metric=metrics,
	batch_end_callback=batchEndCBs,
	epoch_end_callback=epochEndCBs)
