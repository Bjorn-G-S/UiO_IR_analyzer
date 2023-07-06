'''
Written by: Bjørn Gading Solemsli
Email: b.g.solemsli@smn.uio.no

Date: 01.10.2021

Affiliation: iCSI, SMN Catalysis, UiO

'''


import os
import pandas as pd
import numpy as np
from decimal import Decimal
import random

import ipywidgets as widgets
from IPython.display import Javascript, display, HTML
from ipywidgets import widgets, interact, interact_manual, HBox, VBox 
from scipy import sparse
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 100
from matplotlib.widgets import SpanSelector
from scipy import trapz


import matplotlib.colors as colors
import scipy.signal as spsign
import scipy.integrate as sp# trapz, Simps, cumtrapz, romb
import ipywidgets as widgets

import asyncio
import mplcursors

from colour import Color
from IPython.display import display, clear_output
from ipywidgets import interact, interactive, fixed, interact_manual
from adjustText import adjust_text
from lmfit import models



class IR_analysis():
    
       
    
    
############################################################################################
#                                                                                          #
#                                   Initiating the class.                                  #
#         by doing so, a directory for the IR failes to be analysed will be needed         #
#                                                                                          #
############################################################################################

    def __init__(self,*args,**kwargs):

        #will need to change the directory to a directory containing the raw IR files
        directory= r'C:\Users\bjorngso\OneDrive - Universitetet i Oslo\01 Results\IR\Rawdata\water compensated'
        together = list()
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in dirs:
                joint = ('{}'.format(name), os.path.join(root, name))
                together.append(joint)
        
        direct= widgets.Dropdown(options=together, description='Eksperiment:', disabled=False,layout=widgets.Layout(width='90%'),style = {'description_width': 'initial'})
        x_s = widgets.IntSlider(min=0, max=500,step=1,layout=widgets.Layout(width='75%'),description='Files to skip from start',style = {'description_width': 'initial'})
        x_e = widgets.IntSlider(min=0, max=500,step=1,layout=widgets.Layout(width='75%'), description='Files to skip from end',style = {'description_width': 'initial'})
        
        w = widgets.interactive(self.getting_IR_files,directory= direct,x_start=x_s,x_end=x_e)
        display(w)
        
        
    def help(self):
        print('''
        
        


                                              THIS CLASS IS USED FOR ANALYZING IR SPECTRA USING JUPYTER NOTEBOOK
                                              use the following commands in the IR_analysis:


        obj.help()                    -        Will give this list over commands, and what their function is.






                                              IR:



        obj.Reshape_file_names        -        Used to reshape files so that they are .01->.99 or .001->.999. ONLY USE THIS ONA COPY OF THE RAW DATA. NO BACKSIES POSSIBLE
        
        obj.getting_IR_files()        -        Used to read the IR file/files from a given directory. Will needed to be called prior to do any other method is called.

        obj.IR_spectrum_read_to_df()  -        Used to load all data into a pandas dataframe. Also useful so se what files are included in the dataframe
        
        obj.plot()                    -        Used to plot the raw (not normalized) data and used to choose the point to normalize to
 
        obj.plot_norm()               -        Used to plot the normalized data
        
        obj.plot_subtracted()         -        Used for plot a subrtacted spectra. used first and last form plot_norm(). Can also smooth using the Savitzky-Golay filter

        obj.peak_finder()             -        Used to guess the peaks of the soothed subtractedplot form plot_subtracted()

        obj.Choose_one_spectrum()     -        Used to chose one spectrum for ALS_peak fitter()

        obj.ALS_peak_fitter()         -        Used to deconvolute peaks using differnt curve models using lmfit 

        obj.selective_subtract()      -        Used to subtract two specified specra after normalizing the data

        obj.Area_tool()               -        Used to integrate peaks in a single spectrum for quantification. Will need file to be manually read in the .csv form. 



        
        
        
        
        ''')  
        
############################################################################################
#                                                                                          #
#                     Interactive method for chooding the files.                           #
#         x_start and x_end is to spesify form what fiels in the list to what files        #
#                                                                                          #
############################################################################################

    def Reshape_file_names(self,**kwargs):
        directory = r'C:\Users\bjorngso\OneDrive - Universitetet i Oslo\01 Results\IR\Rawdata\water compensated'
        together = list()
        for root, dirs, files in os.walk(directory, topdown=False):
                for name in dirs:
                    joint = ('{}'.format(name), os.path.join(root, name))
                    together.append(joint)
        print('This function is used to alter the naming of files so that they are .01->.99 or .001->.999')

        options_entry= (('Pass (no alteration)',0),('>100 scans in a row',10),('>1000 scans in a row',100))

        direct = widgets.Dropdown(options=together, description='Eksperiment:', disabled=False,layout=widgets.Layout(width='90%'),style = {'description_width': 'initial'})
        entry_ =  widgets.Dropdown(options=options_entry, description='Alteration option:',value= 0,disabled=False,layout=widgets.Layout(width='90%'),style = {'description_width': 'initial'})
        w = widgets.interactive(self.rehape,directory= direct,entry=entry_)
        display(w)

    def rehape(self,**kwargs):
        entry = kwargs.get('entry', 0)
        directory = kwargs.get('directory', r'C:\Users\bjorngso\Desktop\Rawdata\IR\water compensated\210419_0.15CuMOR6.5_Slow_ramp')
        for file in os.listdir(directory): #apend all the IR spectro files into a list
        
            strings = file.split('.')

            if entry == 100:
                
                if len(strings[-1]) < 2:
                    new_name = strings[0] + '.00' + strings[-1]
                    os.rename(file,new_name)
                    
                elif len(strings[-1]) >= 2 and len(strings[-1]) < 3:
                    new_name = strings[0] + '.0' + strings[-1]
                    os.rename(file,new_name)
                    
                    
            elif entry == 10:
                
                if len(strings[-1]) < 2:
                    new_name = strings[0] + '.0' + strings[-1]
                    os.rename(file,new_name)
                    
                    
            else:
                pass




    def getting_IR_files(self,**kwargs):
        directory = kwargs.get('directory', r'C:\Users\bjorngso\Desktop\Rawdata\IR\water compensated\210419_0.15CuMOR6.5_Slow_ramp')
        sub_folder = []

        for file in os.listdir(directory): #apend all the IR spectro files into a list
            sub_folder.append(file)


        x_start = kwargs.get('x_start', 0)
        x_end = kwargs.get('x_end', 0)


        del sub_folder[:x_start]
        x = len(sub_folder)-x_end
        del sub_folder[x:]
        self.sub_folder = sub_folder
        self.directory = directory
#         display(sub_folder)
        return 
        
    
    
#############################################################################################
#                                                                                           #
#                          Loading the files into a dataframe.                              #
#                                                                                           #
#                                                                                           #
#############################################################################################

    
    def IR_spectrum_read_to_df(self,**kwargs):
        IR_data = {}
        for file in self.sub_folder:
            IR_data["%s" %file] = pd.read_csv(r'{}\{}'.format(self.directory,file), skiprows=None,  decimal='.', header=None, names=['cm-1','Absorbance'], usecols = ['cm-1','Absorbance'])  
            # print(file)
        self.IR_data = IR_data
        print(len(self.IR_data))
        display(self.IR_data)
        
        
        return
    
#############################################################################################
#                                                                                           #
#                            Plotting where to normalize to.                                #
#                                                                                           #
#                                                                                           #
#############################################################################################    
    
    
    
    def plotting(self,**kwargs):
        
        number_of_spectra_plottet = kwargs.get('number_of_spectra_plottet',(1,1))
        freq_collected = kwargs.get('freq_collected', 30)
        x_to_norm = kwargs.get('x_to_norm',1)
        y_to_norm = kwargs.get('y_to_norm',1)
        ax = kwargs.get('ax','')
        fig = kwargs.get('fig','')
        ax.clear()
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        self.freq_collected = freq_collected
        
        print(number_of_spectra_plottet)
        
        first_specter = number_of_spectra_plottet[0]
        last_specter = number_of_spectra_plottet[1]
        
        x=0
        
        for file in self.IR_data:
            
            
            if x < first_specter:
                pass
            elif x > last_specter:
                pass
            else:
                cm = np.array(self.IR_data[file]['cm-1'])
                Absorbance = np.array(self.IR_data[file]['Absorbance'])
#                 print(cm, Absorbance)

                ax.plot(cm, Absorbance, label='time_{} min'.format(x*freq_collected), linewidth=0.5)
                ax.grid(color='gray', linestyle='--', linewidth=0.5)
                ax.set_title('IR-Spectra (set where to normalize to)')
                ax.set_ylabel('Absorbance (a.u.)')
                ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
                ax.invert_xaxis()
                ax.set_xlim(np.max(cm),np.min(cm))
           
                if x == y_to_norm:
                    norm_index = np.where((cm > x_to_norm) & (cm < (x_to_norm+1)))
            
                    self.norm_index = norm_index
                    self.norm_to_absorbance = Absorbance
                    self.norm_to_cm = cm
                    
                    print(Absorbance[norm_index])
                    ax.plot(cm[norm_index],Absorbance[norm_index],'o',label='Point to normalize to',markersize=2.5,c='tab:red')
                    star_stop_line = [-0.5,4]
                    cm_line = [cm[norm_index],cm[norm_index]]
                    ax.plot(cm_line,star_stop_line, linewidth=0.5,linestyle='--', c='tab:red')
#                     ax.legend()
             
            x = x+1 
        return 
                    

            
            
       
    def plot(self,**kwargs):
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=False)
        w = widgets.interactive(self.plotting,
                                number_of_spectra_plottet= widgets.IntRangeSlider(values=[0,len(self.IR_data.keys())], min=0,max=len(self.IR_data.keys()),step=1,description='No. of spectra displayed:', disabled=False,layout=widgets.Layout(width='80%'),style = {'description_width': 'initial'}),
                                freq_collected = widgets.BoundedIntText(value=30,min=1, max=300000,step=1,description='Frequency collected:', disabled=False,layout=widgets.Layout(width='20%'),style = {'description_width': 'initial'}),
                                x_to_norm = widgets.FloatSlider(value='1855',min='1700', max='2000', step=1, description='Normalize to (x-axis):', disabled=False,layout=widgets.Layout(width='40%'),style = {'description_width': 'initial'}),
                                y_to_norm = widgets.IntSlider(value=1,min=0,max=len(self.IR_data.keys()), step=1, description='Normalize to (y-axis):', disabled=False,layout=widgets.Layout(width='40%'),style = {'description_width': 'initial'}),
                                ax = widgets.fixed(ax),
                                fig = widgets.fixed(fig),

                               )
        plt.show()
        return display(w)
            
        
        
#############################################################################################
#                                                                                           #
#                            Plotting normalized spectra.                                   #
#                                                                                           #
#                                                                                           #
#############################################################################################         
        
        
        
        
    def plotting_norm(self,**kwargs):
        number_of_spectra_plottet = kwargs.get('number_of_spectra_plottet',(1,1))
        name_fig = kwargs.get('name_fig', '--FILLER--')
        color_from = kwargs.get('color_from','black')
        color_to = kwargs.get('color_to','black')
        ax = kwargs.get('ax','')
        fig = kwargs.get('fig','')
        ax.clear()
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        
        norm_IR_data = {}
        for file in self.IR_data:
            Absorbance = np.array(self.IR_data[file]['Absorbance'])
            Y = Absorbance
            Y_input = Absorbance[self.norm_index]
            Y_normalize_to = self.norm_to_absorbance[self.norm_index]
            
            DELTA = (float(Y_normalize_to)/float(Y_input))

            Y_norm = Y * np.array(DELTA)
            norm_IR_data["%s" %file] = Y_norm
        
        self.norm_IR_data = norm_IR_data
        
        first_specter = number_of_spectra_plottet[0]
        last_specter = number_of_spectra_plottet[1]
        self.fist_specter = first_specter
        self.last_specter = last_specter
        
        
        colorf = Color(color_from)
        colors = list(colorf.range_to(Color(color_to),len(self.norm_IR_data)))  
        
        x=0
        
        for file in self.norm_IR_data:
            self.cm_norm = np.array(self.IR_data[file]['cm-1'])
            
            if x < first_specter:
                pass
            elif x > last_specter:
                pass
            else:              
                cm = np.array(self.IR_data[file]['cm-1'])
                Absorbance = np.array(self.norm_IR_data[file])
                color = colors[x]
               
                ax.plot(cm, Absorbance, label='time_{} min'.format(x*self.freq_collected), linewidth=0.5,c='{}'.format(color))
                ax.grid(color='gray', linestyle='--', linewidth=0.5)
                ax.set_title('{}'.format(name_fig))
                ax.set_ylabel('Absorbance (a.u.)')
                ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
                ax.invert_xaxis()
                ax.set_xlim(np.max(cm),np.min(cm)) 
            x = x+1 
            
        return 
    
    
    
    
    
    def plot_norm(self,**kwargs):
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=False)
        w = widgets.interactive(self.plotting_norm,
                                number_of_spectra_plottet= widgets.IntRangeSlider(value=[0,len(self.IR_data.keys())], min=0,max=len(self.IR_data.keys()),step=1,description='No. of spectra displayed:', disabled=False,layout=widgets.Layout(width='80%'),style = {'description_width': 'initial'}),
                                name_fig = widgets.Text(value= '- - - FILLER - - -',description='Name of figure:', disabled=False,layout=widgets.Layout(width='40%'),style = {'description_width': 'initial'}),
                                color_from = widgets.ColorPicker(concise=False, value = 'red', description='Pick color from:',style = {'description_width': 'initial'}),
                                color_to = widgets.ColorPicker(concise=False, value = 'darkgreen', description='Pick color to:',style = {'description_width': 'initial'}),
                                ax = widgets.fixed(ax),
                                fig = widgets.fixed(fig),

                               )
        plt.show()
        return display(w)

    
    
    
    
    
#############################################################################################
#                                                                                           #
#                              Plotting subtracted specter.                                 #
#                                                                                           #
#                                                                                           #
#############################################################################################     
    
    
    
    
    

    def subtracted(self,**kwargs):
        name_fig = kwargs.get('name_fig', '--FILLER--')
        color = kwargs.get('color','black')
        show_1deriv = kwargs.get('show_1deriv', False)
        show_2deriv = kwargs.get('show_2deriv', False)
        ax = kwargs.get('ax','')
        fig = kwargs.get('fig','')
        smother_y_n = kwargs.get('smother_y_n', False)
        window_length = kwargs.get('window_length',50)
        polyorder = kwargs.get('polyorder',1)
        x_limit = kwargs.get('x_limit','default')
        happy = kwargs.get('happy', False)
        first_last = kwargs.get('first_last',True)
        last_first = kwargs.get('last_first',False)
        ax.clear()
        
        x = 0
        for file in self.norm_IR_data:
            
            if x == self.fist_specter:
                cm = np.array(self.IR_data[file]['cm-1'])
                self.cm = cm
                Absorbance_first = np.array(self.norm_IR_data[file])
            
            elif x == self.last_specter-1:
                Absorbance_last = np.array(self.norm_IR_data[file])
            else:
                pass
            x=x+1
            
        if first_last is True:
            sub_Absorbnbance = self.subtracting(Absorbance_first,Absorbance_last)
            self.sub_Absorbnbance = sub_Absorbnbance
        elif last_first is True:
            sub_Absorbnbance = self.subtracting(Absorbance_last,Absorbance_first)
            self.sub_Absorbnbance = sub_Absorbnbance    
        else:
            raise TypeError('Subtraction not possible')

        
        if smother_y_n is True:
            sub_Absorbnbance = self.smooth_data_array(sub_Absorbnbance,window_length,polyorder)
            
            
            if happy is True:
                norm_abs = sub_Absorbnbance
                self.norm_abs = norm_abs
        else: pass
        
        ax.plot(cm, sub_Absorbnbance, label='Subtracted plot', linewidth=0.5,c='{}'.format(color))
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.set_title('{}'.format(name_fig))
        ax.set_ylabel('Absorbance (a.u.)')
        ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
        ax.invert_xaxis()
        if x_limit == 'default':
            ax.set_xlim(np.max(cm),np.min(cm))
        else:
            ax.set_xlim(x_limit[-1],x_limit[0])
        x=0.5
        ax.set_ylim(-x,x) 
        ax.legend()
        
        if show_1deriv is True:
            cm1 = cm[:-1]
            derivated = self.derive(sub_Absorbnbance,1) 
            
            ax.plot(cm1, derivated, label='1st derivative', linewidth=0.5,c='{}'.format('red'))
            ax.legend()
            
        if show_2deriv is True:
            cm2 = cm[:-2]
            derivated = self.derive(sub_Absorbnbance,2)
            ax.plot(cm2, derivated, label='2nd derivative', linewidth=0.5,c='{}'.format('blue'))
            ax.legend()
            

            
    def plot_subtracted(self,**kwargs):
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=False)
        w = widgets.interactive(self.subtracted,
                                name_fig = widgets.Text(value='- - - - Filler - - - - -',description='Name of figure:', disabled=False,layout=widgets.Layout(width='40%'),style = {'description_width': 'initial'}),
                                color = widgets.ColorPicker(concise=False, value = 'black', description='Pick color:',style = {'description_width': 'initial'}),
                                first_last = widgets.Checkbox(value= True, description='First specter - last specter'),
                                last_first = widgets.Checkbox(value= False, description='Last specter - first specter'),
                                show_1deriv = widgets.Checkbox(value= False, description='1nd derivative'),
                                show_2deriv = widgets.Checkbox(value= False, description='2nd derivative'),
                                ax = widgets.fixed(ax),
                                fig = widgets.fixed(fig),
                                smother_y_n = widgets.Checkbox(value= False, description='Smooth (Savitzky-Golay filter)',style = {'description_width': 'initial'}),
                                window_length = widgets.IntSlider(min=3,max=100,step=2,description='Window_length:',style = {'description_width': 'initial'},layout=widgets.Layout(width='30%')),
                                polyorder = widgets.IntSlider(min=0, max=11,step=1,description='Polyorder:',style = {'description_width': 'initial'},layout=widgets.Layout(width='30%')),
                                x_limit = widgets.IntRangeSlider(value=[1200,4000],min=200, max=4000,step=1, description='Limit of the x-axis:',style = {'description_width': 'initial'},layout=widgets.Layout(width='40%')),
                                happy = widgets.Checkbox(value= False, description='Happy with the smoothing?',style = {'description_width': 'initial'})
                               )
        plt.show()
        return display(w)    
  

    
    
    
#############################################################################################
#                                                                                           #
#                        Math: Subtracting, smoothing, derivating                           #
#                                                                                           #
#                                                                                           #
#############################################################################################            
   
    
            
          
    def subtracting(self,first_spectrum,last_spectrum):
        sub = first_spectrum-last_spectrum
        self.sub = sub
        
        return self.sub

    
    
    def derive(self,y,N):
        if N == 1:
            Y_deriv = np.diff(y,n=N)*10
            
        elif N == 2:
            Y_deriv = np.diff(y,n=N)*100
            
        return Y_deriv 
    
    
    
    def smooth_data_array(self,sub_Absorbnbance,window_length,polyorder):
        #using the function savgol_filter from the scipy.signal package, based on the digital signal filter method ny Savitzky and Golay
        smoothed_array = spsign.savgol_filter(x = sub_Absorbnbance, window_length = window_length, polyorder = polyorder)
        
        return smoothed_array

    
    
#############################################################################################
#                                                                                           #
#                                    Finding peaks                                          #
#                                                                                           #
#                                                                                           #
#############################################################################################      
    
    
    def Peak_finder(self,**kwargs):
        fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=False)
        options_color = ['black','red','darkred','darkgreen','green','orange','yellow', 'blue','darkblue','pink','brown','lightbrown']
        w = widgets.interactive(self.find_signals_prog,
                                name_fig = widgets.Text(value='- - - - Filler - - - - -',description='Name of figure:', disabled=False,layout=widgets.Layout(width='40%'),style = {'description_width': 'initial'}),
                                color = widgets.Dropdown(options=options_color, description='Color:'),
                                peaks_find = widgets.Checkbox(valeu=False,description='Find peaks'),
                                peaks_find_fro_derive = widgets.Checkbox(valeu=False,description='Find peaks from 2nd derivative'),
                                ax = widgets.fixed(ax),
                                fig = widgets.fixed(fig),
                                minimum_height = widgets.FloatSlider(min=0,max=10,step=0.01,description='Minimum height of peak:',style = {'description_width': 'initial'},layout=widgets.Layout(width='30%')),
                                distance = widgets.FloatSlider(min=1,max=100,step=1,description='Distance between peaks:',style = {'description_width': 'initial'},layout=widgets.Layout(width='30%')),
                                Width = widgets.FloatSlider(min=0.01,max=50,step=0.1,description='Minimum width of the peaks:',style = {'description_width': 'initial'},layout=widgets.Layout(width='30%')),
                                x_limit = widgets.IntRangeSlider(value=[200,4000],min=200, max=4000,step=1, description='Limit of the x-axis:',style = {'description_width': 'initial'},layout=widgets.Layout(width='40%')),
                                
                               )
        plt.show()
        return display(w)    
    
    
    
    def find_signals_prog(self,**kwargs):
        name_fig = kwargs.get('name_fig','- - - - Filler - - - -')
        ax = kwargs.get('ax','')
        fig = kwargs.get('fig','')
        x_limit = kwargs.get('x_limit',[4000,0])
        minimum_height = kwargs.get('minimum_height',0.01)
        distance = kwargs.get('distance',1)
        color = kwargs.get('color','')
        peaks_find = kwargs.get('peaks_find',False)
        Width = kwargs.get('Width',0)
        peaks_find_fro_derive = kwargs.get('peaks_find_fro_derive',False)
        

        
        ax.clear()
#         ax.plot(self.cm, (self.norm_abs+1), label='Subtracted plot', linewidth=0.5,c='{}'.format(color))
        main_specter = ax.plot(self.cm, (self.norm_abs+1), label='Subtracted plot', linewidth=0.5,c='{}'.format(color))
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.set_title('{}'.format(name_fig))
        ax.set_ylabel('Absorbance (a.u.)')
        ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
        ax.invert_xaxis()
        ax.set_xlim(x_limit[-1],x_limit[0])
        
        x=0.4
    
        ax.legend()

        
        
        if peaks_find is True:
            
            signal = 1+self.norm_abs
            peaks = spsign.find_peaks(x=signal,height=minimum_height, distance=distance, width=Width)
            ax.scatter(self.cm[peaks[0]],(peaks[1]['peak_heights']), color = 'r', s = 15, marker = 'D', label='Peaks')
            ax.legend()
            
            
            
            for i in range(len(self.cm)):
                i=i-1
                x = self.cm[peaks[0]][i]
                y = peaks[1]['peak_heights'][i]
                text = '{}'.format(self.cm[peaks[0][i]])
                
               
                ax.annotate(text, xy=(x, y), xytext=(x+120, y+0.2), arrowprops=dict(arrowstyle='->'))

                
   

        elif peaks_find_fro_derive is True:
            ax.clear()
            ax.plot(self.cm, (self.norm_abs), label='Subtracted plot', linewidth=0.5,c='{}'.format('orange'))
            
            cm2 = self.cm[:-2]
            derivated = self.smooth_data_array(-1*self.derive(self.norm_abs,2),11,1)
                    
            ax.plot(cm2, derivated, label='2nd derivative (inverted)', linewidth=0.5,c='{}'.format('blue'))
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_title('{}'.format(name_fig))
            ax.set_ylabel('Absorbance (a.u.)')
            ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
            ax.invert_xaxis()
            ax.set_ylim(-x,x) 
            
            ax.set_xlim(x_limit[-1],x_limit[0])

            
            signal = derivated
            peaks = spsign.find_peaks(x=signal,height=(np.max(signal)*minimum_height), distance=distance, width=Width)
            ax.scatter(cm2[peaks[0]],(peaks[1]['peak_heights']), color = 'r', s = 15, marker = 'D', label='Peaks')
            ax.legend()
                    
            for i in range(len(cm2)):
                i=i-1
                X = cm2[peaks[0]][i]
                Y = peaks[1]['peak_heights'][i]
                text = '{}'.format(self.cm[peaks[0][i]])

                ax.annotate(text, xy=(X, Y), xytext=(X+120, Y+0.2), arrowprops=dict(arrowstyle='->'))

        else: pass





#############################################################################################
#                                                                                           #
#                                       ALS method                                          #
#                                                                                           #
#                                                                                           #
#############################################################################################   




    
    def Choose_one_spectrum(self, **kwargs):
        fig, (ax1, ax2)= plt.subplots(1,2,figsize=(18,6), constrained_layout=False)


        
        
        X = []
        Y = []
        ax1.invert_xaxis()
        ax2.invert_xaxis()


        line1, = ax1.plot(X,Y,c='tab:red',label= r'Spectrum')
        ax1.grid(color='gray', linestyle='--', linewidth=0.5)


        line2, = ax1.plot(X,Y,c='green',label= r'Background')
        ax2.grid(color='gray', linestyle='--', linewidth=0.5)

        line3, = ax2.plot(X,Y,c='black',label= r'Background corrected spectrum')
        ax2.grid(color='gray', linestyle='--', linewidth=0.5)

        ax1.legend()
        ax2.legend()



        w = widgets.interactive(self.Choosing_one_spectrum,
                                x = widgets.Dropdown(options= self.sub_folder),
                                ax1 = widgets.fixed(ax1),
                                ax2 = widgets.fixed(ax2), 
                                norm_IR_data = widgets.fixed(self.norm_IR_data),
                                IR_data = widgets.fixed(self.IR_data),
                                lam = widgets.FloatSlider(min=0,max=100,step=0.01, description='Lambda:',disabled=False,indent=False,style = {'description_width': 'initial'},layout=widgets.Layout(width='30%')),
                                p = widgets.FloatSlider(min=0,max=100,step=0.01, description='p:',disabled=False,indent=False,style = {'description_width': 'initial'},layout=widgets.Layout(width='30%')),
                                niter = widgets.IntSlider(value=10,min=1,max=100,step=1, description='niter:',disabled=False,indent=False,style = {'description_width': 'initial'},layout=widgets.Layout(width='30%')),
                                line1=widgets.fixed(line1),line2=widgets.fixed(line2),line3=widgets.fixed(line3),
                                x_limit = widgets.IntRangeSlider(value=[400,3000],min=0, max=(len(self.cm_norm)-1),step=1, description='Limit of the x-axis:',style = {'description_width': 'initial'},layout=widgets.Layout(width='40%')),
                                y_limit = widgets.FloatRangeSlider(value=[-1.5,1.5],min=-5, max=5,step=0.1, description='Limit of the y-axis:',style = {'description_width': 'initial'},layout=widgets.Layout(width='40%')),
                                fig = widgets.fixed(fig), cm =widgets.fixed(self.cm_norm),
                                )
        plt.show()
        return display(w)


 


    def Choosing_one_spectrum(self,x,ax1,ax2,norm_IR_data,lam,p,niter,line1,line2,line3,cm, x_limit, fig, y_limit):
        
            
        ax1.set_xlim(cm[x_limit[-1]],cm[x_limit[0]])
        ax2.set_xlim(cm[x_limit[-1]],cm[x_limit[0]])
        
        ax1.set_ylim(y_limit[0],y_limit[-1])
        ax2.set_ylim(y_limit[0],y_limit[-1]-0.9)
        for file in norm_IR_data:
            if file == x:
                cm = np.array(cm)
                
                
                index_1 = int(np.where(cm == cm[x_limit[0]])[0])
                index_2 = int(np.where(cm == cm[x_limit[-1]])[0])
              
                cm = np.array(cm)[index_1:index_2]
                Absorbance = np.array(norm_IR_data[file])[index_1:index_2]
                line1.set_xdata(cm)
                line2.set_xdata(cm)
                line3.set_xdata(cm)
                line1.set_ydata(Absorbance)

                y = Absorbance.reshape(-1)
                L = len(y)
                D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
                D = lam * D.dot(D.transpose())
                w = np.ones(L)
                W = sparse.spdiags(w, 0, L, L)

                for i in range(niter):
                    W.setdiag(w)
                    Z = W + D
                    z = spsolve(Z, w*y)
                    w = p * (y > z) + (1-p) * (y < z)
                line2.set_ydata(z)

                subtracted = []
                for i in range(len(z)):
                    sub = y[i]-z[i]
                    subtracted.append(sub)



                line3.set_ydata(subtracted)
                self.single_subtracted = subtracted
                self.short_cm = cm
                
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        return 

        

    
    def generate_model(self,spec):
        composite_model = None
        params = None
        x = spec['x']
        y = spec['y']
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min
        y_max = np.max(y)
        for i, basis_func in enumerate(spec['model']):
            prefix = f'm{i}_'
            model = getattr(models, basis_func['type'])(prefix=prefix)
            if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
                model.set_param_hint('sigma', min=1e-6, max=x_range)
                model.set_param_hint('center', min=x_min, max=x_max)
                model.set_param_hint('height', min=1e-6, max=1.1*y_max)
                model.set_param_hint('amplitude', min=1e-6)
                # default guess is horrible!! do not use guess()
                default_params = {
                    prefix+'center': x_min + x_range * random.random(),
                    prefix+'height': y_max * random.random(),
                    prefix+'sigma': x_range * random.random()
                }
            else:
                raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
            if 'help' in basis_func:  # allow override of settings in parameter
                for param, options in basis_func['help'].items():
                    model.set_param_hint(param, **options)
            model_params = model.make_params(**default_params, **basis_func.get('params', {}))
            if params is None:
                params = model_params
            else:
                params.update(model_params)
            if composite_model is None:
                composite_model = model
            else:
                composite_model = composite_model + model
        return composite_model, params

    
    

    
    
    def ALS_peak_fitter(self,**kwargs):
        
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,6))
        cm = self.short_cm
        absorbance = self.single_subtracted
        
        ax1.set_xlim(np.min(cm),np.max(cm))
        ax1.invert_xaxis()
        ax1.grid(color='gray', linestyle='--', linewidth=0.5)
        ax2.set_xlim(np.min(cm),np.max(cm))
        ax2.invert_xaxis()
        ax2.grid(color='gray', linestyle='--', linewidth=0.5)
        ax3.set_xlim(np.min(cm),np.max(cm))
        ax3.invert_xaxis()
        ax3.grid(color='gray', linestyle='--', linewidth=0.5)
        
        ax1.scatter(cm,absorbance,label='Bkg-subtracted plot',s=1,c='black')
        ax1.legend()
        
        style_output = {'description_width': 'initial'}
        layout_ouput = widgets.Layout(width='500px', height='30px')

        self.i = 0
        self.curve = {}
        self.center = {}
        self.height = {}
        self.sigma = {}
        
        def list_of_widgets():
            i = self.i
            if i == 0:
                list_of_wg=[]
            else:
                list_of_wg=[widgets.HTML(value = r'<b>Peak   {}</b>'.format(i)),\
                            widgets.Dropdown(options=[('Gaussian','GaussianModel'), ('Voigt','VoigtModel'), ('Lorentzian','LorentzianModel')], value='VoigtModel',
                                                    description=r'Curve type for peak {}'.format(i),
                                                    disabled=False, style=style_output, layout=layout_ouput,continuous_update=True),\
                            widgets.FloatText(min=np.min(cm), max=np.max(cm),step = np.diff(cm)[0],
                                                    description=r'Center of peak {}'.format(i),
                                                    disabled=False, style=style_output, layout=layout_ouput,continuous_update=True),\
                            widgets.FloatText(min=0,max=1,step=0.0001,
                                                    description=r'Height of peak {}'.format(i),
                                                    disabled=False, style=style_output, layout=layout_ouput,continuous_update=True),\
                            widgets.FloatText(min=0, max=1, step=0.01,
                                                    description=r'S.D. of peak {}'.format(i),
                                                    disabled=False, style=style_output, layout=layout_ouput,continuous_update=True)]
            return list_of_wg
            
            
        
        
        #Button 1 ----- Add peak
        def on_bttn_clicked_1(b): 
            i = self.i
            i = i + 1
            self.i = i
            vb.children=tuple(list(vb.children) + list_of_widgets())      
            self.values = vb.children
            
        #Button 2 ----- See peak estimate
        def on_bttn_clicked_2(change):
            
            ax1.clear()
            extr = 0.5*np.max(np.diff(absorbance))
            ax1.set_xlim((np.min(cm)),(np.max(cm)))
            ax1.invert_xaxis()
            ax1.grid(True)
            ax1.scatter(cm,absorbance,label='Bkg-subtracted plot',s=1,c='black')
            i = self.i
            line = ((np.min(absorbance)-extr),(np.max(absorbance)+extr))
            peak = {}
            curve = {}
            center = {}
            height = {}
            sigma = {}
            for n in range(i):
                L = {}
                peak[n+1] = self.values[n*5:5+(n*5)][0].value
                curve[n+1] = self.values[n*5:5+(n*5)][1].value
                center[n+1] = self.values[n*5:5+(n*5)][2].value
                height[n+1] = self.values[n*5:5+(n*5)][3].value
                sigma[n+1] = self.values[n*5:5+(n*5)][4].value

                ax1.plot((center[n+1],center[n+1]), line, linestyle=':', color='r', label=r'Guessed peak {}'.format(n+1),linewidth=1.2)

                L = {       'type':curve[n+1],
                            'params': {'center':center[n+1], 'height': height[n+1], 'sigma': sigma[n+1]},
                            'help': {'center':{'min': (center[n+1]-5), 'max': (center[n+1]+5)}}
                                }
            ax1.legend()
   
        #Button 3 ------ Clear all
        def on_bttn_clicked_3(b):
            output1.clear_output()
            output2.clear_output()
            clear_output()
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax1.set_xlim(np.min(cm),np.max(cm))
            ax1.invert_xaxis()
            ax1.grid(color='gray', linestyle='--', linewidth=0.5)
            ax2.set_xlim(np.min(cm),np.max(cm))
            ax2.invert_xaxis()
            ax2.grid(color='gray', linestyle='--', linewidth=0.5)
            ax3.set_xlim(np.min(cm),np.max(cm))
            ax3.invert_xaxis()
            ax3.grid(color='gray', linestyle='--', linewidth=0.5)
            ax1.scatter(cm,absorbance,label='Bkg-subtracted plot',s=1,c='black')
            ax1.legend()
            
                
                
        #Button 4 ------ deconvolute based on lmfit
        def on_bttn_clicked_plot(change):
            output2.clear_output()
            ax2.clear()
            ax3.clear()
            i = self.i
            peak = {}
            curve = {}
            center = {}
            height = {}
            sigma = {}
            model = []
            for n in range(i):
                L = {}
                peak[n+1] = self.values[n*5:5+(n*5)][0].value
                curve[n+1] = self.values[n*5:5+(n*5)][1].value
                center[n+1] = self.values[n*5:5+(n*5)][2].value
                height[n+1] = self.values[n*5:5+(n*5)][3].value
                sigma[n+1] = self.values[n*5:5+(n*5)][4].value


                L = {       'type':curve[n+1],
                            'params': {'center':center[n+1], 'height': height[n+1], 'sigma': sigma[n+1]}
                                }

                model.append(L) 

            spec = {
                    'x': cm,
                    'y': absorbance,
                    'model': model
                    }
            
            
            with output2:
                
                ax2.clear
                ax3.clear
                ax2.set_xlim(np.min(cm),np.max(cm))
                ax2.invert_xaxis()
                ax2.grid(color='gray', linestyle='--', linewidth=0.5)
                ax3.set_xlim(np.min(cm),np.max(cm))
                ax3.invert_xaxis()
                ax3.grid(color='gray', linestyle='--', linewidth=0.5)

                ax2.scatter(cm,absorbance,label='Bkg-subtracted plot',s=1,c='black')
                model, params = self.generate_model(spec)
                output = model.fit(spec['y'], params, x=spec['x'])
                def average(lst):
                    return sum(lst) / len(lst)
                
                r_sqrd = (1- (average(output.best_fit)/average(absorbance)))
                ax2.plot(spec['x'], output.best_fit, 'r-', label=r'best fit (R$^{:}$: {:.3f})'.format(2,abs(r_sqrd)))
                
                
                
                ax3.scatter(spec['x'], spec['y'], s=1, c='black',label='Bkg-subtracted plot')
                components = output.eval_components(x=spec['x'])
                for i, model in enumerate(spec['model']):
                    index = int(np.where(components[f'm{i}_'] == np.max(components[f'm{i}_']))[0])
                    
                    ax3.plot(spec['x'], components[f'm{i}_'], label=r'Peak {:} at {:.0f}cm$^{:}$$^{:}$'.format(i+1,spec['x'][index],'-',1))
                    ax3.fill_between(spec['x'], components[f'm{i}_'].min(), components[f'm{i}_'], alpha=0.5)
                ax2.legend()
                ax3.legend()
                print(output.fit_report())
                

            
        output1 = widgets.Output()
        output2 = widgets.Output() 
              
        vb =  widgets.VBox(list_of_widgets())
        
        btn = widgets.Button(description = 'Add peaks',style=style_output) 
        btn.on_click(on_bttn_clicked_1)
        
        btn1 = widgets.Button(description = 'Show estimated peaks',style=style_output) 
        btn1.on_click(on_bttn_clicked_2)
        
        btn2 = widgets.Button(description = 'Clear',style=style_output) 
        btn2.on_click(on_bttn_clicked_3)
        
        btn3 = widgets.Button(description = 'Calcualte',style=style_output)
        btn3.on_click(on_bttn_clicked_plot)
        
        
        buttons = widgets.HBox([btn,btn1,btn2,btn3])
        display(buttons,vb,output1,output2)
        
    plt.savefig(r'C:\Users\bjorngso\OneDrive - Universitetet i Oslo\01 Results\Graphs\test_or.svg')
    plt.show()





    def selective_subtract(self,**kwargs):
        fig, (ax1, ax2)= plt.subplots(1,2,figsize=(18,6), constrained_layout=False)
        
        
        X = []
        Y = []
        ax1.invert_xaxis()
        ax2.invert_xaxis()


        line1, = ax1.plot(X,Y,c='tab:red',label= r'Spectrum 1')
        ax1.grid(color='gray', linestyle='--', linewidth=0.5)


        line2, = ax1.plot(X,Y,c='green',label= r'Spectrum 2')
        ax2.grid(color='gray', linestyle='--', linewidth=0.5)

        line3, = ax2.plot(X,Y,c='black',label= r'Subtracted spectrum')
        ax2.grid(color='gray', linestyle='--', linewidth=0.5)

        ax1.legend()
        ax2.legend()
        
        w = widgets.interactive(self.Choosing_sub,
                                x1 = widgets.Dropdown(options= self.sub_folder, value=self.sub_folder[0]),
                                x2 = widgets.Dropdown(options= self.sub_folder, value=self.sub_folder[-1]),
                                ax1 = widgets.fixed(ax1),
                                ax2 = widgets.fixed(ax2), 
                                norm_IR_data = widgets.fixed(self.norm_IR_data),
                                IR_data = widgets.fixed(self.IR_data),
                                line1=widgets.fixed(line1),line2=widgets.fixed(line2),line3=widgets.fixed(line3),
                                x_limit = widgets.IntRangeSlider(value=[400,3000],min=0, max=(len(self.cm_norm)-1),step=1, description='Limit of the x-axis:',style = {'description_width': 'initial'},layout=widgets.Layout(width='40%')),
                                y_limit = widgets.FloatRangeSlider(value=[-1.5,1.5],min=-5, max=5,step=0.1, description='Limit of the y-axis:',style = {'description_width': 'initial'},layout=widgets.Layout(width='40%')),
                                fig = widgets.fixed(fig), cm =widgets.fixed(self.cm_norm),w1 = widgets.ToggleButton(valuse=False, description='Save differance plot',disabled=False,button_style='', tooltip='Description'),
                                filename = widgets.Text(description='Filename for the differance plot:',placeholder='Type file name',disabled=False,style = {'description_width': 'initial'},layout=widgets.Layout(width='40%'))
                                )
        plt.show()
        return display(w)

    def Choosing_sub(self,x1,x2,ax1,ax2,norm_IR_data,line1,line2,line3,cm, x_limit, fig, y_limit,w1,filename):

            
        ax1.set_xlim(cm[x_limit[-1]],cm[x_limit[0]])
        ax2.set_xlim(cm[x_limit[-1]],cm[x_limit[0]])

        ax1.set_ylim(y_limit[0],y_limit[-1])
        ax2.set_ylim(y_limit[0],y_limit[-1])
        
        cm = np.array(cm)
        
        
        index_1 = int(np.where(cm == cm[x_limit[0]])[0])
        index_2 = int(np.where(cm == cm[x_limit[-1]])[0])
        
        cm = np.array(cm)[index_1:index_2]
        self.Absorbance1 = np.array(norm_IR_data[x1])[index_1:index_2]
        self.Absorbance2 = np.array(norm_IR_data[x2])[index_1:index_2]
        line1.set_xdata(cm)
        line2.set_xdata(cm)
        line3.set_xdata(cm)

        line1.set_ydata(self.Absorbance1)
        line2.set_ydata(self.Absorbance2)

        x3 = self.Absorbance1 - self.Absorbance2

        if w1 == True:
            data = {'x': cm, 'y': x3}
            df = pd.DataFrame(data,columns=['x','y'])
            
            if filename != '':
                df.to_csv(r'{}.csv'.format(filename))
                print('')
                print('')
                print('')
                print((r'{}.csv'.format(filename)))
                print('')
                print(df)
                print('')
                print('')

        line3.set_ydata(x3)

        ax1.invert_xaxis()
        ax2.invert_xaxis()


        return 









    def Area_tool(self):
        print('')
        csv_name = input("Enter file name:")
        df =pd.read_csv(r'{}'.format(csv_name))
        print('')
        print('')
        print('')

        
        # Integration

        # Global Variables
        area = 0

        output_area = widgets.Output()

        with output_area:
            fig_area, ax_area = plt.subplots(figsize=(12, 5))
            
        x_data = np.flip(df['x'].to_numpy())
        y_data = np.flip(df['y'].to_numpy())

        box_placement_y = np.max(y_data) - 0.25*np.max(y_data)
        box_placement_x = np.min(x_data) + 0.05*np.min(x_data)

        ax_area.plot(x_data,y_data, label='Differnace spectrum')
        ax_area.text(np.max(box_placement_x), np.max(box_placement_y),  'Area: 0', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='yellow', alpha=0.5))
        ax_area.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax_area.set_xlim(min(x_data), max(x_data))
        ax_area.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=12)
        ax_area.set_ylabel(r'Absortion (a.u)', fontsize=12)
        ax_area.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax_area.legend(title=r'{}'.format(csv_name), bbox_to_anchor=(1.0, 0.7), fontsize=8)
        ax_area.legend()
        ax_area.set_title('Peak integration\nPress left mouse button and drag to select integration region', fontsize=10)
        ax_area.fill_between(x_data, 0, y_data, color='white')
        ax_area.invert_xaxis()
        ax_area.grid(color='gray', linestyle='--', linewidth=0.5)


        line2 = ax_area.plot(0, 0, c='r')
        line_straigt = ax_area.plot(0, 0, c='r')
        y_selected = 0

        def onselect_area(xmin, xmax):
            global area
            global mean_base
            global y_selected
            
            if xmin == xmax:
                return
            
            ax_area.collections.clear()
            
            xmin_l = xmin
            xmax_l = xmax
            
            indmin, indmax = np.searchsorted(x_data, (xmin, xmax))
            indmax = min(len(x_data) - 1, indmax)

            thisx = x_data[indmin:indmax]
            thisy = y_data[indmin:indmax]
            
            coef = np.polyfit([thisx[0], thisx[-1]], [thisy[0], thisy[-1]], 1)
            straight_line = np.poly1d(coef)
            y_fitted = straight_line(thisx)
            
            area = trapz(thisy-y_fitted, thisx, 0.001)
            
            std_y = np.std(thisy)
            mean_val = 'Area: {:.3E}'.format(Decimal(area))
            
            
            ax_area.texts[0].set_text(mean_val)
            line2[0].set_data(thisx, thisy)
            line_straigt[0].set_data(thisx, straight_line(thisx))
            ax_area.fill_between(thisx, y_fitted, thisy, color='grey', alpha=0.5)
            print('Area: {:.3E}'.format(Decimal(area)))
            return

        
        self.span_area = SpanSelector(ax_area, onselect_area, 'horizontal', useblit=True,  span_stays=True, 
                            rectprops=dict(alpha=0.5, facecolor='yellow'))


        

        Main = widgets.VBox([output_area])
        return Main

















