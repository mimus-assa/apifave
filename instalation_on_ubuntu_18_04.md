Install

what we need:

ubuntu 18.04 fresh install cuda_10.1.105_418.39_linux.run cudnn-10.1-linux-x64-v7.6.5.32.tgz Anaconda3-2020.02-Linux-x86_64.sh

in the terminal:

$ sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy python-six python3-six build-essential python-pip python3-pip python-virtualenv swig python-wheel python3-wheel libcurl3-dev libcupti-dev

$ sudo apt-get install libcurl4-openssl-dev

$ sudo add-apt-repository ppa:graphics-drivers/ppa

$ sudo apt update

$ sudo apt upgrade

$ ubuntu-drivers devices

$ sudo ubuntu-drivers autoinstall

$ sudo reboot

after reboot

$ nvidia-smi

$ cd Downloads/

$ sudo sh cuda_10.1.105_418.39_linux.run --override --silent --toolkit

$ tar -xzvf cudnn-10.1-linux-x64-v7.6.5.32.tgz

$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include

$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64

$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

$ gedit ~/.bashrc

and we copy this on the opened notepad at the end

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64" export CUDA_HOME=/usr/local/cuda

again in the same terminal

$ source ~/.bashrc

$ cd /usr/local/cuda/lib64/

$ sudo rm libcudnn.so.7

$ echo $CUDA_HOME

$ sudo ldconfig

$ cd

and now we are going to install anaconda

$ sh Anaconda3-2020.02-Linux-x86_64.sh

$ source ~/.bashrc

now we need to install cmake and other stuff

$ sudo apt-get install build-essential cmake pkg-config

$ sudo apt-get install libx11-dev libatlas-base-dev

$ sudo apt-get install libgtk-3-dev libboost-python-dev

$ sudo apt-get install python-dev python-pip python3-dev python3-pip

$ sudo -H pip2 install -U pip numpy

$ sudo -H pip3 install -U pip numpy

here we create the the conda enviorement

$ conda create -n apiais python=3.6

$ source activate apiais

and install jupyter labs and ipywidgets

$ conda install -c conda-forge jupyterlab

$ conda install -n base -c conda-forge widgetsnbextension

$ conda install -n apiais -c conda-forge ipywidgets

$ conda install -c conda-forge nodejs

$ jupyter labextension install @jupyter-widgets/jupyterlab-manager

we close the terminal an open another one to install g++ and gcc

$ sudo apt-get update &&
sudo apt-get install build-essential software-properties-common -y &&
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y &&
sudo apt-get update &&
sudo apt-get install gcc-6 g++-6 -y &&
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6 &&
gcc -v

here we install the dlib library with cuda enable

$ wget http://dlib.net/files/dlib-19.17.tar.bz2

$ tar xvf dlib-19.17.tar.bz2

$ cd dlib-19.17/

$ mkdir build

$ cd build

$ cmake ..

$ cmake --build . --config Release

$ sudo make install

$ sudo ldconfig

$ cd

$ pkg-config --libs --cflags dlib-1

here we install the dlib on the apiais env

$ source activate apiais

$ cd dlib-19.17

$ python setup.py install

and delete this files for the next install

$ rm -rf dist

$ rm -rf tools/python/build

$ rm python_examples/dlib.so

$ pip install dlib

here we install the base libraries for the deep learning

$ conda install
tensorflow-gpu==1.14
cudatoolkit=10.1
cudnn=7.6.5
keras-gpu==2.2.4
h5py
pillow

and finaly we install al the libs

$ pip3 install face_recognition

$ pip3 install flask

$ pip3 install Flask-Bootstrap

$ pip3 install -U Flask-SQLAlchemy

$ pip3 install keras_vggface

$ pip3 install matplotlib

$ pip3 install pandas

$ pip3 install opencv-python

$ pip3 install mtcnn

$ pip install fuzzywuzzy[speedup]

$ pip3 install imutils

$ pip3 install sklearn

$ pip3 install scikit-image

$ pip3 install simplejson

now we need to create another env for the training on jupyter notebooks, so we close the terminal and in a new one

$ conda create -n tf2 python=3.6

$ source activate tf2

$ conda install -c conda-forge jupyterlab

$ conda install -n tf2 -c conda-forge ipywidgets

$ conda install -c conda-forge nodejs

$ jupyter labextension install @jupyter-widgets/jupyterlab-manager

$ conda install
tensorflow
cudatoolkit=10.1
cudnn=7.6.5
h5py
pillow

$ pip3 install matplotlib

$ pip3 install pandas

$ pip3 install opencv-python

$ pip install fuzzywuzzy[speedup]

$ pip install -q git+https://github.com/tensorflow/examples.git

$ pip3 install -q git+https://github.com/tensorflow/examples.git

$ pip install -q -U tfds-nightly

$ pip3 install -q -U tfds-nightly
