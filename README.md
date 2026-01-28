# Generalizable Implicit Neural Representations for Improved OCT Interpolation and Segmentation
The PyTorch implementation of the work [Bridging Gaps in Retinal Imaging - Fusing OCT and SLO Information with Implicit Neural Representations for Improved Interpolation and Segmentation](https://link.springer.com/chapter/10.1007/978-3-658-47422-5_24). 

| ![](images/overview.png) |
|:------------------------:|
|   *Schematic overview*   |

This work uses generalizable implicit neural representations (INRs) to improve interpolation and retinal layer 
segmentation in optical coherence tomography (OCT). Usually, OCT scans have large slices distances, which can 
potentially miss small retinal structures and reduce measurement accuracy. Existing interpolation methods struggle 
with shape accuracy and missing data. By using population-based training, generalizable INRs improve shape 
representation with minimal annotated slices. The method also integrates scanning laser ophthalmoscopy (SLO) 
to enhance inter-slice information. Additionally, the generalizable INR adapts to new images, enabling segmentation of 
unseen cases.

## Prerequisites
- Linux 20.04
- Python 3.9
- NVIDIA GPU + CUDA CuDNN

### Packages
- PyTorch (2.1)
- numpy
- json
- SimpleITK
- SciPy
- ruamel
- matplotlib
- lpips
- scikit-image

## Getting Started
### Installation
- Clone this repository
- Install the necessary requirements

### Dataset
The dataset used for INteRp-OCT consists of 100 OCT and SLO images from 50 healthy volunteers (right and left eye) 
acquired with a Heidelberg Spectralis scanner. OCT and SLO images have a resolution of 496x512x512 (dense sampling in OCTA setting) and 768x768.

### Preprocessing
Pre-processing of the OCT images consists of the following steps:
- Flattening at the Bruch's membrane
- B-scan-wise normalization to \[0,1\]
- Cropping of B-scans to size 230x496 
- Subsampling to 16 equidistant B-scans

&rarr; Resolution of OCT images used for training is 230x496x16 


SLO pre-processing:
- Subsampling to B-Scan positions (of original OCT volume)
- Normalization to \[0,1\]
- Cropping and subsampling

&rarr; Resolution of SLO images used for training is 496x16 


Data format:
For each fold, we organized our training and test images in .npz-files named
'dataset_flat_subsmpl32_train_fold{config.SETTINGS.FOLD}.npz'. For evaluation purposes, we used images 
sampled to 31 B-scans which are stored as 'dataset_flat_subsmpl16_test_fold{config.SETTINGS.FOLD}.npz'.
For our code to work, each .npz-container must have the following components:
- 'oct_vols': The training/test images in format \[num_samples, H, W, D\]
- 'seg_vols': The corresponding segmentation images in format \[num_samples, H, W, D\]
- 'slo_imgs': The corresponding SLO images in format \[num_samples, W, D\]
- 'subject_names': A list containing identifiers of your data

If you want to use a different data format, the following lines of code need to be adapted:
- In train_test_geninr.py: Lines 66 - 70, 217 - 235, 246
- In training.py: Lines 188 - 193
- In evaluation.py: Lines 165 - 171


## Usage
We provide two main scripts, one for the training and subsequent fitting to test images of the 
proposed generalizable INR and one to fit an INR on individual OCT plus SLO images. Both scripts
additionally perform an evaluation of the models. 

To run the scripts use the following commands:
```sh
python train_test_geninr.py ../configs/geninr_config.yaml --gpu 0
```
```sh
python train_test_single.py ../configs/singleinr_config.yaml --gpu 0
```

For both the generalizable and the single INR, we provide the configuration used in the paper in the form of YAML files.

## Qualitative results
### Learning of position-dependent shap of the retina
![](images/result1.png)


### SLO incorporation allows enhanced inter-slice representation enabling reconstruction of small structures like vessels
![](images/result1.png)


## Citation
This work has been accepted for the German Conference on Medical Image Computing 2025. If you use this code, please cite as follows:
```
@inproceedings{kepp2025interp_oct,
  title={Bridging gaps in retinal imaging},
  booktitle ={Bildverarbeitung f{\"u}r die Medizin 2025},
  author={Kepp, Timo and Andresen, Julia and Falta, Fenja and Handels, Heinz},
  publisher={Springer Fachmedien Wiesbaden},
  editor={Palm, Christoph and Breininger, Katharina and Deserno, Thomas and Handels, Heinz and Maier, Andreas and Maier-Hein, Klaus H. and Tolxdorff, Thomas M.},
  pages={107--112},
  year={2025},
  adress={Wiesbaden} 
  doi = {https://doi.org/10.1007/978-3-658-47422-5_24},
}
```
