# Network Dissection (Lite)

## Introduction
This repository is a light and portable version of [NetDissect](https://github.com/CSAILVision/NetDissect), which contains the demo code for the [CVPR'17 paper](http://netdissect.csail.mit.edu/final-network-dissection.pdf) Network Dissection: Quantifying Interpretability of Deep Visual Representations. This code is written on pytorch of python3.6, tested on Ubuntu 16.04. There are dissection results for several networks at the [project page](http://netdissect.csail.mit.edu/).

To run the code, you may have to do following steps.

* Download the model and dataset via given scripts or provide your own.
* Edit the settings.py file as all global settings is there.
* Simply run `python3 main.py `.
* Check the result at "result" folder.


## Download
* Clone the code of Network Dissection from github
```
    git clone https://github.com/sunyiyou/NetDissect-Lite
    cd NetDissect-Lite
```
* Download the Broden dataset (~1GB space) and the example pretrained models.
```
    script/dlbroden.sh
    script/dlzoo_example.sh
```

Note that AlexNet models work with 227x227 image input, while VGG, ResNet, GoogLeNet works with 224x224 image input.

## Run in PyTorch

* Run Network Dissection in PyTorch. Please install [PyTorch](http://pytorch.org/) and [Torchvision](https://github.com/pytorch/vision) first.
You can change the `settings.py` change the global settings as you like. Then simply run:

```
    python3 main.py
```

## Report
* At the end of the dissection script, a report will be generated at `result` folder that summarizes the semantics of the networks. These are, respectively, the HTML-formatted report, the semantics of the units of the layer summarized as a bar graph, visualizations of all the units of the layer (using zero-indexed unit numbers), and a CSV file containing raw scores of the top matching semantic concepts in each category for each unit of the layer.


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
