## UWB-Radar-Pedestrian-Tracking 
The codes are assoicating with a preprint paper https://arxiv.org/pdf/2109.12856.pdf  
If you use or refer to the codes, please cite the following paper,  
```bash
@misc{li2021multistatic,  
      title={Multi-Static UWB Radar-based Passive Human Tracking Using COTS Devices},   
      author={Chenglong Li and Emmeric Tanghe and Jaron Fontaine and Luc Martens and Jac Romme and Gaurav Singh and Eli De Poorter and Wout Joseph},  
      year={2021},  
      eprint={2109.12856},  
      archivePrefix={arXiv},  
      primaryClass={eess.SP}  
} 
```
## Description
The whole dataset were collected along a long single trajectory. The codes file contains the three-case consecutive trajectory partitioning for training and predicting. The baisc idea of this work is to estimated the reflected ToFs from the accumulated channel impulse response (CIR) or the corresponding variance based on two proposed CNN models. Then a particle filter algorithm is adopted for the pedestrian tracking. Get access to more technical details via the [paper](https://arxiv.org/pdf/2109.12856.pdf).  
This work is implemented based on the open-access dataset of a published paper, you can find more details about the dataset via the paper or the link below  

```bash
A. Ledergerber and R. D’Andrea, “A multi-static radar network with ultra-wideband radio-equipped devices,” Sensors, vol. 20, no. 6, pp. 1–20, Mar 2020  
```

The labrotory-based experimental data is avalialable
https://www.research-collection.ethz.ch/handle/20.500.11850/397625  
In the code file, you can find the generated (resampled) CIR and variance series of the background and dynamic scenarios and the well-trained CNN models.  
Note that the required 'Dyn_CIR_VAR.mat' is not included in the folder as it is quite large. If you are interested in this work and need the dataset, please contact me  

```bash
Chenglong[dot]Li[at]UGent[dot]be  
```

You are also encouraged to process the original dataset and develop your own algorithms.

 
