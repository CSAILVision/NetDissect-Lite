# Network Dissection Lite in PyTorch

## Introduction
This repository is a light version of [NetDissect](https://github.com/CSAILVision/NetDissect), which contains the demo code for the work [Network Dissection: Quantifying Interpretability of Deep Visual Representations](http://netdissect.csail.mit.edu). This code is written in pytorch and python3.6, tested on Ubuntu 16.04. The processing speed is greatly improved compared to the original version: It only takes about 20 mins for netdissecting the Resnet18, and about 2 hours for DenseNet161, and no complex shell commands. Note that the dissection result will be slightly different to the original version due to the faster upsampling function used. Please install [Pytorch](http://pytorch.org/) in python36 and [Torchvision](https://github.com/pytorch/vision) first.


## Download
* Clone the code of Network Dissection Lite from github
```
    git clone https://github.com/CSAILVision/NetDissect-Lite
    cd NetDissect-Lite
```
* Download the Broden dataset (~1GB space) and the example pretrained model. If you already download this, you can create a symbolic link to your original dataset.
```
    ./script/dlbroden.sh
    ./script/dlzoo_example.sh
```

Note that AlexNet models work with 227x227 image input, while VGG, ResNet, GoogLeNet works with 224x224 image input.

## Run NetDissect in PyTorch

* Please install [PyTorch](http://pytorch.org/) and [Torchvision](https://github.com/pytorch/vision) first. You can configure `settings.py` to load your own model, or change the default parameters.

* Run NetDissect 

```
    python main.py
```


## NetDissect Result

* At the end of the dissection script, a report will be generated inside `result` folder that summarizes the interpretable units of the tested network. These are, respectively, the HTML-formatted report, the semantics of the units of the layer summarized as a bar graph, visualizations of all the units of the layer (using zero-indexed unit numbers), and a CSV file containing raw scores of the top matching semantic concepts in each category for each unit of the layer.


## Reference
If you find the codes useful, please cite this paper
```
@inproceedings{netdissect2017,
  title={Network Dissection: Quantifying Interpretability of Deep Visual Representations},
  author={Bau, David and Zhou, Bolei and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}
```
