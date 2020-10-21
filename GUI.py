from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
import os
import tkinter
from keras.preprocessing.image import ImageDataGenerator
import keras
import os.path
import numpy as np
import sklearn.metrics as metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

root = Tk()
root.title('Cutting Speed Classifier')
root.iconbitmap('C:/MUSP_Local/GUI/GUI.ico')
#root.geometry('400x400')
root.resizable(width=True, height=True)
    
    
def openfn():
    filename = filedialog.askopenfilenames(title='Select Image(s)')
    return filename

def open_img():
    
    global img_names
    img_names = openfn()
    i = 0
    
 
    for img_name in img_names:
       
        #print('Name of image is ' , img_name)
        img = Image.open(img_name)
        img = img.resize((400, 400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        image_label = Label(root, image=img)
        image_label.grid( row = 2,  column = i*10 ,columnspan = 10)
        image_label.image = img
        image_address = Label(root , text = 'Selected image name ' +  os.path.basename(img_name))
        image_address.grid( row = 3,  column = i*10 ,columnspan = 10)
        i = i + 1
    btn2 = Button(root, text ='Predict class', command = pred_img )
    btn2.grid( row = 4,  column =0, columnspan = 1,padx = 5, pady = 5)
   
    return 

def pred_img():
    
    i = 0
    global y_pred_label
    y_pred_label = [] 
    for img_name in img_names:

        image = keras.preprocessing.image.load_img(img_name , color_mode="grayscale" ,target_size = (512,512))
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        model = keras.models.load_model('C:/MUSP_Local/Keras_models/Hi_Lo_CNN')
        predictions = model.predict(input_arr)
        y_pred_labels = np.argmax(predictions, axis=1)
        y_pred_label.append(y_pred_labels)  # only necessary if output has one-hot-encoding, shape=(n_samples)
        if (y_pred_labels == 0):
            predict_label = Label(root , text = 'Predicted class is High Speed')
            predict_label.grid( row = 5,  column = i*10 ,columnspan = 10 )
            
            if(str(os.path.basename(img_name)) in High_Speed):
                true_label = Label(root , text = 'True class is High Speed ' )
                true_label.grid( row = 6,  column = i*10 ,columnspan = 10)
            else:
                true_label = Label(root , text = 'True class is Low Speed!!! ', fg='red')
                true_label.grid( row = 6,  column = i*10 ,columnspan = 10)
                #popup()
        else:
            predict_label = Label(root , text = 'Predicted class is Low Speed')
            predict_label.grid( row = 5,  column = i*10 ,columnspan = 10)
            
            if(str(os.path.basename(img_name)) in Low_Speed):
                true_label = Label(root , text = 'True class is Low Speed ')
                true_label.grid( row = 6,  column = i*10 ,columnspan = 10)
            else:
                true_label = Label(root , text = 'True class is High Speed!!! ',  fg='red' )
                true_label.grid( row = 6,  column = i*10 ,columnspan = 10)
                #popup()
                
        i = i + 1
        
    if(len(img_names) > 1):
        btn_cm  = Button(root, text ='Confusion matrix', command = confusion_matrix1)
        btn_cm.grid( row = 8,  column =0 , columnspan =1 ,padx = 5, pady = 5)
    else:
        return
    return

def confusion_matrix1():
    
    global y_pred_label
    i = 0
    y_true_labels = []   ############## prdictions is single value check
    for img in img_names:
        if(str(os.path.basename(img)) in High_Speed):
            y_true_labels.append(int(0)) 
        else:
            y_true_labels.append(int(1)) 
    print('Predicted labels' , y_pred_label)
    y_true_labels = np.array(y_true_labels)
    y_true_labels = y_true_labels.ravel()
    y_pred_labels = np.array(y_pred_label)
    y_pred_labels = y_pred_labels.ravel()
    print('Shape True labels' , y_true_labels)
    print('Shape Pred labels' , y_pred_labels)  
    cm = confusion_matrix(y_true_labels ,y_pred_labels) 
    disp = ConfusionMatrixDisplay(cm , display_labels = ['Low_Speed'])
    
    #fig = Figure(figsize=(5, 4), dpi=100)
    #disp.plot(fig , cmap=plt.cm.Blues)
    canvas = FigureCanvasTkAgg(disp.figure_, master = root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().grid(row = 9,  column =0 , columnspan =1 ,padx = 5, pady = 5)
    return


def popup():
    error = messagebox.askyesno('Wrong prediction' , 'Prediction number is wrong')
    


img = Image.open('C:/MUSP_Local/GUI/default_image.png')
img = img.resize((400, 400), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
image_label = Label(root, image = img)
image_label.grid( row = 2,  column = 0 ,columnspan = 1)
image_label.image = img



btn1 = Button(root,text ='Open Image(s)'   , command = open_img)
btn1.grid( row = 0,  column = 0 ,columnspan = 1 ,padx = 5, pady = 5)


btn2 = Button(root, text ='Predict class', command = pred_img , state= DISABLED)
btn2.grid( row =4 ,  column =0, columnspan = 1,padx = 5, pady = 5)

btn_quit = btn2 = Button(root, text ='Exit', command = root.destroy)
btn_quit.grid( row = 1,  column =0 , columnspan =1)
#label1 = Label(root , text = '')


#scale_h = Scale(root, from_ = 0 , to = 200 , orient = HORIZONTAL)
#scale_h.grid(row = 10 ,  column =0 , columnspan = 1)
#btn_forward =  Button(root, text ='>>', command = root.destroy)

#btn_backward = Button(root, text ='<<', command = root.destroy)


Low_Speed = ['A2.tif','A4.tif','B1.tif','B4.tif','C1.tif','C2.tif','D2.tif','D4.tif',
             'E1.tif','E4.tif','F2.tif','F4.tif','G3.tif','G4.tif','H3.tif','H4.tif','I1.tif','L4.tif']
High_Speed = ['A1.tif','A3.tif','B2.tif','B3.tif','C3.tif','C4.tif','D1.tif','D3.tif','E2.tif',
              'E3.tif','F1.tif','F3.tif','G1.tif','G2.tif','H1.tif','H2.tif','I3.tif.tif','I4.tif' ,'L2.tif','L3.tif','I2.tif','L1.tif']


root.mainloop()