# MaskNet

MaskNet is an app for running live tests on a set of convolutional neural networks that work together to detect how a person is using their face mask (correctly, incorrectly or not wearing it at all) and if they are using it wrong suggest them how to correct its use. 

## Motivation

The COVID-19 pandemic has made the use of face masks a crucial preventive measure to reduce the transmission of the virus. However, many people do not wear masks properly, either exposing their nose or mouth, or wearing them loosely. This reduces the effectiveness of the masks and increases the risk of infection. Therefore, it is important to monitor and correct the use of masks in public places, such as schools, workplaces, or transportation hubs.

MaskNet is a computer vision system that aims to address this problem by automatically detecting and classifying the use of masks in real time. MaskNet consists of three modules: a face detector, a mask classifier, and a mask corrector. The face detector locates the faces in the input image or video stream. The mask classifier determines if each face is wearing a mask correctly, incorrectly, or not at all. The mask corrector generates suggestions on how to improve the use of masks for those who are wearing them incorrectly.

## Installation

To run MaskNet, you need to have Python 3.9 and conda installed on your system. You also need to create a conda environment with the required packages and dependencies. You can follow these steps to set up the environment:

- Clone this repository to your local machine:

```
bash git clone https://github.com/alexkalen/MaskNet.git
```

- Navigate to the repository directory:

```
cd MaskNet
```

- Create a conda environment from the environment.yml file:

```
conda env create -f environment.yml
```

- Activate the conda environment:

```
conda activate masknet
```

## Usage

To run MaskNet as a Flask app, you can use the following steps:

- Navigate to the repository directory:

```
bash cd MaskNet
```

- Run the app.py script:

```
python MaskNet.py
```

- Open a web browser and go to http://localhost:5000

- Upload an image file that contains one or more faces and click on the “Submit” button.

- The app will display the uploaded image with bounding boxes around each detected face and labels indicating their mask status (correct, incorrect, or none). If a face is wearing a mask incorrectly, the app will also display a suggestion on how to correct it.

## Research papers that were based on this project

- Kalen Targa, A. et al., (2021). Mask-net: Detection of Correct Use of Masks Through Computer Vision [Oral Presentation]. International Conference on Machine Learning Conference: LatinX in AI (LXAI) Research Workshop 2021, Virtual. https://doi.org/10.52591/lxai202107247

- Kalen-Targa, A., Landi-Cortinas, A., Araque-Volk, N., & Van-Grieken, A. M.  (2021). Mask-net: Identificación del uso correcto de mascarilla mediante visión por computador. Research in Computing Science, 150(5), 5-16. https://rcs.cic.ipn.mx/2021_150_5/Mask-net_%20Identificacion%20del%20uso%20correcto%20de%20mascarilla%20mediante%20vision%20por%20computador.pdf