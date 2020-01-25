# Vehicle-Identification

# Dataset: 
Here we have used the Stanford cars dataset is a collection of 16,185 images of 196 cars

You can download dataset from <a href="http://pyimg.co/9s9mx">here</a>

After downloading dataset unzip it and put `car_ims` folder inside `dataset/cars` directory

# Files:
1. car_config.py: This file will contain all necessary configurations to build the Stanford Cars Dataset and fine-tune VGG16 on it
2. build_dataset.py: This file will input our image paths and class labels and then generate a .lst file for each of the training, validation, and testing splits, respectively
3. fine_tune_cars.py: This file will be responsible for fine-tuning VGG16 on our dataset
4. test_cars.py: This file will be used to check performance of trained model
5. vis_classification.py: This script will load an input image from disk, pre-process it, pass it through our fine-tuned VGG16, and display the output predictions

# How to Run?
To run `build_dataset.py`, execute following command

`python build_dataset.py`

After this command you will have three files named `train.lst` , `test.lst` and `val.lst` inside `dataset/cars/lists` directory.

After getting these three files, execute following commands to generete `.rec` files

`python mxnet/tools/im2rec.py dataset/cars/lists/train.lst ""  --resize 256 --encoding ".jpg"` <br>
`python mxnet/tools/im2rec.py dataset/cars/lists/test.lst ""  --resize 256 --encoding ".jpg"` <br>
`python mxnet/tools/im2rec.py dataset/cars/lists/val.lst ""  --resize 256 --encoding ".jpg"`

Note: Make sure you have copied MXNET installation folder in root path which contais im2rec file

After generating these three files move them inside `dataset/cars/rec` directory
<hr>

Before generating `fine_tune_cars.py`, you need weights for vgg16

you can download weights from <a href="http://data.dmlc.ml/models/imagenet/vgg/">here</a>

After downloading you will have two files named `vgg16-0000.params` and `vgg16-symbol.json` move them to `vgg16` directory

To run `fine_tune_cars.py`, execute following command

`python fine_tune_cars.py --vgg vgg16/vgg16 --checkpoints checkpoints --prefix vggnet`

<hr>

To run `test_cars.py`, execute following command

`python test_cars.py --checkpoints checkpoints --prefix vggnet --epoch 65`

<hr>

To run `vis_classification.py`, execute following command

`python vis_classification.py --checkpoints checkpoints --prefix vggnet --epoch 65`
