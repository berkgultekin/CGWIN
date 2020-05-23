wget -O ClassifierModel.pt --no-check-certificate --content-disposition https://github.com/berkgultekin/CGWIN/blob/master/ClassifierModel.pt?raw=true
wget -O GeneratorModel.pt --no-check-certificate --content-disposition https://github.com/berkgultekin/CGWIN/blob/master/GeneratorModel.pt?raw=true
wget -O DiscriminatorModel.pt --no-check-certificate --content-disposition https://github.com/berkgultekin/CGWIN/blob/master/DiscriminatorModel.pt?raw=true
wget -O paramstore.out --no-check-certificate --content-disposition https://github.com/berkgultekin/CGWIN/blob/master/paramstore.out?raw=true

if [["$OSTYPE" == "linux*"]]; then
    PATH="MNIST\\raw\\"
fi
if [["$OSTYPE" == "darwin*"]]; then
    PATH="MNIST\\raw\\"
fi
if [["$OSTYPE" == "msys*"]]; then
    PATH="MNIST/raw/"
fi

TRAIN1="train-images-idx3-ubyte.gz"
TRAIN2="train-labels-idx1-ubyte.gz"
TEST1="t10k-images-idx3-ubyte.gz"
TEST2="t10k-labels-idx1-ubyte.gz"

if [ ! -d MNIST ]; then

    mkdir MNIST
    cd MNIST
    if [ ! -d raw ]; then
        mkdir raw
        cd raw

        wget --no-check-certificate --content-disposition http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
        gunzip train-labels-idx1-ubyte.gz -k
        echo "MNIST data set (2) downloaded!"

        wget --no-check-certificate --content-disposition http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
        gunzip train-images-idx3-ubyte.gz -k
        echo "MNIST data set (1) downloaded!"

        wget --no-check-certificate --content-disposition http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
        gunzip t10k-images-idx3-ubyte.gz -k
        echo "MNIST data set (3) downloaded!"

        wget --no-check-certificate --content-disposition http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
        gunzip t10k-labels-idx1-ubyte.gz -k
        echo "MNIST data set (4) downloaded!"
    fi
fi
