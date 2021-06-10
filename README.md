<div align="center">    
 
# A preliminary analysis on software frameworks for the development of spiking neural networks

[![Conference](https://img.shields.io/badge/Hais-2021-informational)]()  

 ----
</div>
 
## Description   
In this repo the source code of the experimental analysis presented is available. 

## How to run   
Each tool is executed using its own Python file, therefore, the file `ANN.py` contains the Keras model, `annarchy.py` the SNN in ANNarchy, and so on.

First, you might want to run the experiments in a Docker container which contains all the specified tools.

Here we provide a Docker container to run all frameworks, except NEST. For running NEST (`NEST.ipynb`) it is require to build the NEST-simulator docker container images, which contains the NEST Kernel and all the required dependencies. Please, visit: https://github.com/nest/nest-docker in order to install this container. We strongly recommend the use of the notebook version container.

Next, clone this repo and run any of the files to obtain the results:
```bash
# clone project   
git clone https://github.com/agvico/prelim_snn_frameworks

# run BindsNet SNN
python binds.py
```

### Citation  

Please, cite this work as:
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
