from tkinter import * 
from tkinter.ttk import *
from tkinter import filedialog
import tkinter as tk
import os
from PIL import Image, ImageTk
import numpy as np
import cv2
from snic import SNIC
import matplotlib.pyplot as plt
from slic_centroidx import SLICProcessor

def browseImage() :
    filename = filedialog.askopenfilename(initialdir="./", title="Select An Image", filetypes=(("jpg files", "*.jpg"), ("jpeg files", "*.jpeg"), ("PNG files", "*.png"), ("webp files", "*.webp")))
    
    global img
    global faces
    global img_clean_copy
    global snic

    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_clean_copy = np.copy(img)
    temp_img = Image.fromarray(img)
    width, height = temp_img.size
    width += 80
    height += 100

    temp_img = ImageTk.PhotoImage(temp_img)
    lbl.configure(image=temp_img)
    lbl.image = temp_img


def select_mode():
    ip1 = inputtxt1.get("1.0", "end-1c")
    ip2 = inputtxt1.get("1.0", "end-1c")
    global snic

    text = clicked.get()
    if text == "SNIC" or text == "Centroid SNIC" or text == "SNIC Edge":
        snic = SNIC(img, ip1, ip2)
        snic.mode(text)
    elif text == "SLIC":
        img = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
        slic = SLICProcessor(img,ip1,ip2,"slic")
        slic.iterations()
    else:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
        slic = SLICProcessor(img,ip1,ip2,"slic_centroid")
        slic.iterations()
root = Tk()


lbl = Label(root)

lbl.grid(row = 0,column = 0,pady = 2,columnspan = 4)

error_msg = Label(root)

btn = Button(root, text="Browse Image", command=browseImage)
btn.grid(row = 1,column = 0,pady = 2)

btn_quit = Button(root, text="Exit", command= root.quit)
btn_quit.grid(row = 2,column = 3,pady = 2)


options = [
    "select mode",
    "SLIC",
    "Centroid SLIC",
    "SNIC",
    "Centroid SNIC",
    "SNIC Edge"
]
  
clicked = StringVar()
  
clicked.set( "select mode" )
  
drop = OptionMenu( root , clicked , *options )
drop.grid(row = 1,column = 2,pady = 2)

inputtxt1 = Text(root, height = 1,
                width = 10)
inputtxt1.grid(row = 2,column = 0,pady = 2)
inputtxt1 = Text(root, height = 1,
                width = 10)
inputtxt1.grid(row = 2,column = 1,pady = 2)
  
btn_select = Button( root , text = "select" , command = select_mode )
btn_select.grid(row = 1,column = 3,pady = 2)
  
root.mainloop()