# globa_cpis_codes
 
Here are some demo codes for the detection of CPIS in Sentinel-2 images. 

## Installation Requirements

- Operating system:  Ubuntu Linux (20.04)

- Python version: python v3.7

- CUDA version: 10.1

- Dependent libraries: gdal(3.2.0) +  PyTorch(1.6.0) + MMDetection(v2.7.0) + mmcv(1.2.4) + shapely(1.7.1)

- This project is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Therefore the installation is the same as original MMDetection.


  
## How to run this demo

Once the installation is done, you can follow the steps below to run this demo.

- Unzip the model file (in the subdirectory `model/`) into the same path.
- Go to the root directory of this project in terminal and activate the corresponding virtual environment.
- Run

  ```
  python demo.py 
  ```

- You will obtain the detection results of the sample images (in the subdirectory `imgs/`) in a new created subdirectory (`result/`).
   
   Note that the sample images are in GeoTiff format with four bands (R-G-B-NIR).


