# About
Codes pertaining to the results in our paper *Predictive Modeling of Evoked Intracranial EEG Response to Medial Temporal Lobe Stimulation in Patients with Epilepsy*.

Authors: Gagan Acharya<sup>1</sup>, Kathryn A. Davis<sup>2</sup>, Erfan Nozari<sup>1</sup> 

<sup>1</sup> University of California, Riverside \
<sup>2</sup> Hospital of the University of Pennsylvania

---
# Requirement
Supported environment:
- Python 3.7
## Downloading the repository
    git clone https://github.com/nozarilab/2023Acharya_evoked_ieeg.git
## Installing the dependencies
    cd 2023Acharya_evoked_ieeg
    conda create --name <env name> --file requirements.txt
## RAM (Restoring Active Memory) data

Go to UPenn's [Data Request](https://memory.psych.upenn.edu/Data_Request) page and fill in the required details to access the data.
Download and expand the 19AUG2020a release using 7-Zip or Winzip. 

---
# Getting started
## Configure data path
Edit the *ROOTDIR* variable in the **config.py** file to reflect the location of the dataset in your machine. 

## Run experiment
Make sure you are within the created virtual environment:

    conda activate <env name>

Make sure you are working in the *2023Acharya_evoked_ieeg* directory and run the experiment.py code to generate the results shown in the paper: 

    python experiment.py

The results will be stored in the path indicated by the *OUTPUT_PATH* variable

## Plotting results
Once the results are generated in the *OUTPUT_PATH* folder, you can plot them by calling the routines in **plotter.py**. For example, the plots in figure 2 of the paper can be plotted using the following commands:

    python -i plotter.py
    >>> plot_fig2()

This should plot the following figure:



    



