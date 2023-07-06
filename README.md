# IR-Data-Analyzer

* [General](#general-info)
* [Purpose](#purpose)
* [Instalation](#installation)
* [How-to](#how-to)
* [Comments](#Comments)


## General

IR-Data-Analyser is a python package to be used within the Jupyter environment for 
analysing and processing IR spectroscopy data.
Currently IR-Data-Analyser is used at the section for catalysis at
the University of Oslo.

## Purpose

IR-Data-Analyser was created to improve data analysis workflow and has been designed
to be easily extended with more advanced analysis, both quantitive and qualitive analysis. 

## Installation


The following python packages are needed:

<ul>
    <li>python (>= 3.8.8)
    <li>jupyter (>= 1.0.0)
    <li>jupyterlab (>= 2.2.6)
    <li>pandas (>= 1.1.3) 
    <li>matplotlib (>= 3.3.1)
    <li>ipympl (>= 0.5.8)
    <li>numpy (>= 1.19.1)
    <li>ipywidgets (>= 7.5.1)
    <li>ipysheet (>= 0.4.0)
    <li>xlsxwriter (>=1.3.7)
    <li>natsort (>= 7.0.1)
    <li>colour (>= 0.1.5)
    <li>lmfit (>= 1.0.2)
    <li>adjusttext (>= 0.7.3.1)
    <li>scipy (>= 1.6.1)
    <li>asyncio (>= 1.6.1)
    
</ul>

The following JupyterLab extensions should be installed: <br>

<ul>
    <li>@jupyter-widgets/jupyterlab-manager (Required)
    <li>jupyter-matplotlib (Required)  
    <li>@epi2melabs/jupyterlab-autorun-cells (Optional)
    <li>@aquirdturtle/collapsible_headings (Optional)
    <li>@jupyterlab/toc (Optional)
</ul>



## How-to

Download the notebook and the python script and open in Jupyter lab. It is important to change the directory for the raw data files to the correct directory. 
The program will read every folder in the give directory and can analyse several spectra at once.

```
def __init__(self,*args,**kwargs):

    
        directory= r'CHANGE THE DIRECTORY FOR THE CORRECT LOCATION OF THE RAW DATA FOLDERS'
        together = list()
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in dirs:
                joint = ('{}'.format(name), os.path.join(root, name))
                together.append(joint)
        ...
```
and 
```
def Reshape_file_names(self,**kwargs):
        directory = r'CHANGE THE DIRECTORY FOR THE CORRECT LOCATION OF THE RAW DATA FOLDERS'
        together = list()
        for root, dirs, files in os.walk(directory, topdown=False):
                for name in dirs:
                    joint = ('{}'.format(name), os.path.join(root, name))
                    together.append(joint)
                    
def rehape(self,**kwargs):
        entry = kwargs.get('entry', 0)
        directory = kwargs.get('directory', r'CHANGE THE DIRECTORY FOR THE CORRECT LOCATION OF THE RAW DATA FOLDERS')
        for file in os.listdir(directory): #apend all the IR spectro files into a list
        ...
```


To navigate the the program, use the `obj.help()` method.


## Comments

Annotation in the `IR_class.py` have been added to more easily navigate the code.

For any questions please email **b.g.solemsli@smn.uio.no**



