wget -O ClassifierModel.pt --no-check-certificate --content-disposition https://github.com/berkgultekin/CGWIN/blob/master/ClassifierModel.pt?raw=true
wget -O GeneratorModel.pt --no-check-certificate --content-disposition https://github.com/berkgultekin/CGWIN/blob/master/GeneratorModel.pt?raw=true
wget -O DiscriminatorModel.pt --no-check-certificate --content-disposition https://github.com/berkgultekin/CGWIN/blob/master/DiscriminatorModel.pt?raw=true

if [ ! -d train-images-idx3-ubyte.gz ]; then
    wget --no-check-certificate --content-disposition http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    gunzip train-images-idx3-ubyte.gz
    echo "MNIST data set (1) downloaded!"
fi

if [ ! -d train-labels-idx1-ubyte.gz ]; then
    wget --no-check-certificate --content-disposition http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gunzip train-labels-idx1-ubyte.gz
    echo "MNIST data set (2) downloaded!"
fi

if [ ! -d t10k-images-idx3-ubyte.gz ]; then
    wget --no-check-certificate --content-disposition http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    gunzip t10k-images-idx3-ubyte.gz
    echo "MNIST data set (3) downloaded!"
fi
fi

if [ ! -d t10k-labels-idx1-ubyte.gz ]; then
    wget --no-check-certificate --content-disposition http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gunzip t10k-labels-idx1-ubyte.gz
    echo "MNIST data set (4) downloaded!"
fi