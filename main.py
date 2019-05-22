# Written by bdelta for Python 3.7
# https://github.com/bdelta
# Draw a number and use the neural network to predict
# Select a model by importing it

from tkinter import *
# from nn_784_50_10 import *
# from nntf_784_800_10 import *
# from nntf_784_800_800_10 import *
from cnntf_784_5_5_10 import *
from extract_data import normalizeImage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Pick correct weights according to the model
# path_weights = "nn_784_50_10.p"
# path_weights = "./nntf_784_800_10/nntf_784_800_10.ckpt"
# path_weights = "./nntf_784_800_800_10/nntf_784_800_800_10.ckpt"
path_weights = "./cnntf_784_5_5_10/cnntf_784_5_5_10.ckpt"

width = 420
height = 420

class PaintApp(object):

    def __init__(self, root, nnModel):
        self.left_but = "up"
        self.x_pos, self.y_pos = None, None
        self.x1, self.y1 = None, None

        self.canvas = Canvas(root, bg='black')
        self.canvas.config(width=width, height=height)
        self.canvas.pack(expand=YES, fill=BOTH)

        # Buttons
        predict = Button(text='Predict', command=self.predict)
        clear = Button(text='Clear', command=self.clear)
        self.v = StringVar()
        self.pre = Label(root, text="-", textvariable=self.v, font=("Helvetica", 20))
        self.v.set("Draw and press predict!")
        self.pre.pack()
        predict.pack()
        clear.pack()

        # User Input
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<ButtonPress-1>", self.left_but_down)
        self.canvas.bind("<ButtonRelease-1>", self.left_but_up)

        # Pillow
        self.image = Image.new("RGB", (width, height), (0,0,0))
        self.draw = ImageDraw.Draw(self.image)

        # Neural network model
        self.nnModel = nnModel

    def left_but_down(self, event=None):
        self.left_but = "down"

        self.x1 = event.x
        self.y1 = event.y

    def left_but_up(self, event=None):
        self.left_but = "up"

        self.x_pos = None
        self.y_pos = None

    def motion(self, event=None):
        if self.left_but == "down":
            if self.x_pos and self.y_pos:
                event.widget.create_line(self.x_pos, self.y_pos, event.x, event.y, \
                 smooth=True, width=35, fill='white', capstyle=ROUND)

                self.draw.line([self.x_pos, self.y_pos, event.x, event.y], \
                    fill="white",width=30)

            self.x_pos = event.x
            self.y_pos = event.y

    def clear(self):
        self.canvas.delete(ALL)
        self.draw.rectangle([0, 0, width, height], fill=(0,0,0,0))

    def predict(self):
        blurred = self.image.filter(ImageFilter.BoxBlur(radius=1))
        resized = blurred.convert(mode="L").resize((28,28))
        
        arr = np.array(resized)

        # Check the image is preprocessed correctly 
        # plt.imshow(arr)
        # plt.show()
        if self.nnModel:
            arr = normalizeImage(arr)
            arr = arr.reshape(1, -1)
            self.v.set("Predicted: " + str(self.nnModel.predict(arr)[0]))


# Ensure the correct weights are loaded
# Weight loading for nn from scratch
# weights = pickle.load(open(path_weights, "rb"))
# nn = NNModel(0, weights)

# Weights for tensorflow nn's
nn = NNModel(path_weights)

# Initialize app
root = Tk()
paint_app = PaintApp(root, nn)
root.mainloop()
nn.closeSession()