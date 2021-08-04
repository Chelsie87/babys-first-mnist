#Hand writting recognition program
#pip install keras
#pip install tensorflow
#pip install np_utils
import tensorflow
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from keras.models import load_model
from PIL import ImageGrab, Image
from tkinter import *
import tkinter as tk
import win32gui
import matplotlib.pyplot as plt
import numpy as np
#%%
#Load the train data and split it into train and test sets
#X_train = train image     Y_Train = train labels  
#X_test = images to test      Y_test = labels to test
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
#%%
#Get image shape
print(X_train.shape)
print(X_test.shape)

#%%
#Look at first image, look at training dataset
print(X_train[0])

#Print image label
print(Y_train[0])
#%%
#Show image as picture
plt.imshow(X_train[0])
#%%
#Reshape data to fit model
#(rows, pixel, pixel, greyscale)
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)   
#%%
#One-Hot Encoding
Y_train_one_hot = to_categorical(Y_train)
Y_test_one_hot = to_categorical(Y_test)

#Print new label
print(Y_train_one_hot[0])
#print(Y_test_one_hot[0])
#%%
#Build the CNN model
model = Sequential()
#Add model layers
#Convolution layer to extract features from the input image 
#32 channels, 3x3 kernel, activation layer record rectifier linear unit, images 28x28 depth 1  
model.add(Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = (28, 28, 1)))
#2nd Conv layer 64 channels
model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
#Flatten layer - takes images and flattens them to turn them into 1D array/vector so it can connect to NN
model.add(Flatten())
#10 neurons 
model.add(Dense(10, activation = 'softmax'))

#Compile the model
#optimizer, loss function - classes that are greater than 2 (10 total classes), metrics accuracy of model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#%%
#Train the model
#fit = train, Y_train_one_hot is formatted, validation data = tets set, epochs = number of iterations through the NN (more means higher accuracy)
hist = model.fit(X_train, Y_train_one_hot, validation_data=(X_test, Y_test_one_hot), epochs = 2)
model.save('mnist2.h5')
#%%
#Graph to show accuracy
#plt.plot(hist.history['accuracy'])
#plt.plot(hist.history['val_accuracy'])
#plt.title('Total Accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Val'], loc = 'upper left')
#plt.show()

#%%
#Show predictions as probablitiles for first 4 images in test set
#precition array = [prob of being 0, prob of 1, 2, 3..., 9]
predictions = model.predict(X_test[:4])
predictions
#%%
#Predict as number labels for first 4 images
#array of max value from predictions on horizontal axis (1)
print(np.argmax(predictions, axis = 1))
#Print the actual labels to compare
print(Y_test[:4])

#Show first 4 images as pictures
for i in range(0,4):
    image = X_test[i]
    image = np.array(image, dtype = 'float')
    pixels = image.reshape ((28,28))
    plt.imshow(pixels, cmap = 'gray')
    plt.show()
    

#%%    
#GUI Creation
#Used from (link below):
#https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/
#We wanted to add this to showcase actually writing the digits
model = load_model('mnist2.h5')
def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)
   
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "black", cursor="dot")
        self.label = tk.Label(self, text="Write a digit", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Guess", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        #self.x1,y1 = (event.x-1), (event.y-1)
        #self.x2,y2 = (event.x+1), (event.y+1)
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='white', outline = 'white')
        #self.canvas.create_oval(self.x1,y1,self.x2,y2, fill ='white')
app = App()
mainloop()
