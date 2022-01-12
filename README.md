## UWB-Radar-Pedestrian-Tracking 
The codes are assoicating with the paper [published](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9677997) or [preprint](https://arxiv.org/pdf/2109.12856.pdf)  
If you use or refer to the codes, please cite the following paper,  
```bash
@ARTICLE{Li2022UWBradar,
  author={Li, Chenglong and Tanghe, Emmeric and Fontaine, Jaron and Martens, Luc and Romme, Jac and Singh, Gaurav and De Poorter, Eli and Joseph, Wout},
  journal={IEEE Antennas and Wireless Propagation Letters}, 
  title={Multi-Static UWB Radar-based Passive Human Tracking Using COTS Devices}, 
  year={2022},
  pages={1-5},
  doi={10.1109/LAWP.2022.3141869}}
```
## Description
The whole dataset were collected along a long single trajectory. The codes file contains the three-case consecutive trajectory partitioning for training and predicting. The baisc idea of this work is to estimated the reflected ToFs from the accumulated channel impulse response (CIR) or the corresponding variance based on two proposed CNN models. Then a particle filter algorithm is adopted for the pedestrian tracking. Get access to more technical details via the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9677997).  
This work is implemented based on the open-access dataset of a published paper, you can find more details about the dataset via the paper or the link below  

```bash
A. Ledergerber and R. D’Andrea, “A multi-static radar network with ultra-wideband radio-equipped devices,” Sensors, vol. 20, no. 6, pp. 1–20, Mar 2020  
```

The labrotory-based experimental data is avalialable
https://www.research-collection.ethz.ch/handle/20.500.11850/397625  
In the code file, you can find the generated (resampled) CIR and variance series of the background and dynamic scenarios and the well-trained CNN models.  
Note that the required 'Dyn_CIR_VAR.mat' is not included in the folder but [here](https://drive.google.com/file/d/1jo-PErF5nnqWJ8UUdzZv_OpWcDMesgxB/view). If you are interested in this work and have any question, please contact me  

```bash
Chenglong[dot]Li[at]UGent[dot]be  
```

You are also encouraged to process the original dataset and develop your own algorithms.

 
