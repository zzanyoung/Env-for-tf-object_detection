# Env-for-tf-object_detection

This repository documents supplementary materials for training object detection models in TensorFlow. The object detection API used is based on the TensorFlow Model Garden repository (https://github.com/tensorflow/models).<br><br>

## Detail

#### * 1_ImgObjDetection_saveModel.py <br> * 2_WebcamObjDetection_savedModel.py
This is a Python test script that inputs an image or webcam frame into a saved model and outputs object recognition results. <br><br>

#### * 3_TfliteConverter.py
This is a Python script that converts SavedModel to tflite format. the most basic code provided in the official TensorFlow documentation.<br>
[https://www.tensorflow.org/lite/models/convert/convert_models?hl=ko](https://www.tensorflow.org/lite/models/convert/convert_models?hl=ko#savedmodel_%EB%B3%80%ED%99%98%EA%B6%8C%EC%9E%A5)
<br><br>

#### * builder.py
**This file is subject to the Google Protocol Buffer License.**<br>
The protobuf 3.16 package is missing builder.py. This is necessary to resolve errors that occur during the process of installing and checking the object detection API. After installing Protocol Buffer version 22.0, follow the solution with the most votes at the address below. 

```
# builder.py path
venv(or conda)dir/Lib/site-packages/google/protobuf/internal
```

https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal<br><br>

#### * generate_tfrecord.py 
deprecated.
<br>now using RoboFlow<br>
https://universe.roboflow.com<br><br>

#### * requirements.txt 
This is information on the packages installed within my Python virtual environment.<br>
Run the command below in the folder where the file is located.
```
pip install .
```
<br>


## PIP LIST (package/version)
Package                       Version
absl-py                       1.4.0<br>
apache-beam                   2.46.0<br>
astor                         0.8.1<br>
astunparse                    1.6.3<br>
avro-python3                  1.10.2<br>
bleach                        6.0.0<br>
cachetools                    4.2.4<br>
certifi                       2024.2.2<br>
charset-normalizer            3.3.2<br>
click                         8.1.7<br>
cloudpickle                   2.2.1<br>
colorama                      0.4.6<br>
contextlib2                   21.6.0<br>
crcmod                        1.7<br>
cycler                        0.11.0<br>
Cython                        3.0.10<br>
dill                          0.3.1.1<br>
dm-tree                       0.1.8<br>
dnspython                     2.3.0<br>
docopt                        0.6.2<br>
etils                         0.9.0<br>
fastavro                      1.8.0<br>
fasteners                     0.19<br>
flatbuffers                   24.3.25<br>
fonttools                     4.38.0<br>
gast                          0.3.3<br>
gin-config                    0.5.0<br>
google-api-core               2.19.0<br>
google-api-python-client      2.128.0<br>
google-auth                   2.30.0<br>
google-auth-httplib2          0.2.0<br>
google-auth-oauthlib          0.4.6<br>
google-pasta                  0.2.0<br>
googleapis-common-protos      1.63.0<br>
grpcio                        1.62.2<br>
h5py                          3.8.0<br>
hdfs                          2.7.3<br>
httplib2                      0.21.0<br>
idna                          3.7<br>
immutabledict                 2.2.5<br>
importlib-metadata            6.7.0<br>
importlib-resources           5.12.0<br>
joblib                        1.3.2<br>
kaggle                        1.6.12<br>
keras                         2.10.0<br>
Keras-Applications            1.0.8<br>
Keras-Preprocessing           1.1.2<br>
kiwisolver                    1.4.5<br>
labelImg                      1.8.6<br>
libclang                      18.1.1<br>
lvis                          0.5.3<br>
lxml                          5.2.1<br>
Markdown                      3.4.4<br>
MarkupSafe                    2.1.5<br>
matplotlib                    3.5.3<br>
numpy                         1.21.6<br>
oauth2client                  4.1.3<br>
oauthlib                      3.2.2<br>
object-detection              0.1<br>
objsize                       0.6.1<br>
opencv-python                 4.9.0.80<br>
opencv-python-headless        4.9.0.80<br>
opt-einsum                    3.3.0<br>
orjson                        3.9.7<br>
packaging                     24.0<br>
pandas                        1.1.5<br>
Pillow                        9.5.0<br>
pip                           24.0<br>
portalocker                   2.7.0<br>
promise                       2.3<br>
proto-plus                    1.23.0<br>
protobuf                      3.19.6<br>
psutil                        5.9.8<br>
py-cpuinfo                    9.0.0<br>
pyarrow                       9.0.0<br>
pyasn1                        0.5.1<br>
pyasn1-modules                0.3.0<br>
pycocotools                   2.0<br>
pydot                         1.4.2<br>
pymongo                       3.13.0<br>
pyparsing                     2.4.7<br>
PyQt5                         5.15.10<br>
PyQt5-Qt5                     5.15.2<br>
PyQt5-sip                     12.13.0<br>
python-dateutil               2.9.0.post0<br>
python-slugify                8.0.4<br>
pytz                          2024.1<br>
pywin32                       306<br>
PyYAML                        5.4.1<br>
regex                         2024.4.16<br>
requests                      2.31.0<br>
requests-oauthlib             2.0.0<br>
rsa                           4.9<br>
sacrebleu                     2.2.0<br>
scikit-learn                  1.0.2<br>
scipy                         1.4.1<br>
sentencepiece                 0.2.0<br>
seqeval                       1.2.2<br>
setuptools                    68.0.0<br>
six                           1.16.0<br>
tabulate                      0.9.0<br>
tensorboard                   2.11.2<br>
tensorboard-data-server       0.6.1<br>
tensorboard-plugin-wit        1.8.1<br>
tensorflow                    2.10.1<br>
tensorflow-addons             0.19.0<br>
tensorflow-datasets           4.8.2<br>
tensorflow-estimator          2.10.0<br>
tensorflow-gpu                2.10.1<br>
tensorflow-gpu-estimator      2.1.0<br>
tensorflow-hub                0.16.0<br>
tensorflow-io                 0.31.0<br>
tensorflow-io-gcs-filesystem  0.31.0<br>
tensorflow-metadata           1.12.0<br>
tensorflow-model-optimization 0.7.3<br>
tensorflow-text               2.10.0<br>
termcolor                     2.3.0<br>
text-unidecode                1.3<br>
tf-models-official            2.10.1<br>
tf-slim                       1.1.0<br>
threadpoolctl                 3.1.0<br>
toml                          0.10.2<br>
tqdm                          4.66.4<br>
typeguard                     2.13.3<br>
typing_extensions             4.7.1<br>
uritemplate                   4.1.1<br>
urllib3                       1.26.7<br>
webencodings                  0.5.1<br>
Werkzeug                      2.2.3<br>
wheel                         0.42.0<br>
wrapt                         1.16.0<br>
zipp                          3.15.0<br>
zstandard                     0.21.0<br>
