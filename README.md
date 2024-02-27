# Efficient Split_Federated Learning
Combining split learning and federated learning

# pytorch-vgg-cifar10
This is the PyTorch implementation of VGG network trained on CIFAR10 dataset 

### Requirements. 
[PyTorch] (https://github.com/pytorch/pytorch)

[torchvision] (https://github.com/pytorch/vision)


### Evaluation 
	
	# CUDA
	wget http://www.cs.unc.edu/~cyfu/cifar10/model_best.pth.tar
	python main.py --resume=./model_best.pth.tar -e
	# or use CPU version
	wget http://www.cs.unc.edu/~cyfu/cifar10/model_best_cpu.pth.tar
	python main.py --resume=./model_best_cpu.pth.tar -e --cpu
### Training in Federated learning mode
    python FL_train.py --arch=vgg19 --b=32 --lr=0.01 --epochs=1500 --local_epochs=10 --lrdecay=600 --uplr=1 --data_path=./data_split_user/train_100_iid_ns --process=5 --user=10  --save_dir=...
### Parser Meaning
    arch: model name (VGG.py)
    b: batch_size
    lr: local learning rate
    epochs: Global training rounds
    local_epochs: lcoal training epochs
    lrdecay: learning rate decay
    uplr: Global learning rate
    data_path: Distributed training dataset path
    process: The number of local training processes
    user: The number of selected users
    save_dir: Model saving path
	




