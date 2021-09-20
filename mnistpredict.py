import pathlib
from tensorflow.keras import models
import numpy as np
import tkinter as tk
from mnistdraw import MnistDrawer

# path to model
path = pathlib.Path('models/mnist_model_2021-05-13_15-44-02').absolute()

# load model
# TODO figure out a lighter weight way to get the model ready for prediction
#  - maybe only saving weights, and creating a new model with them
model = models.load_model(path)


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.add_widgets()
        self.pack()

    def add_widgets(self):
        # TODO - make GUI look better
        self.label = tk.Label(text="MNIST NUMBER DRAWING")
        self.button = tk.Button(text="Predict", command=self.evaluate)
        self.canvas = MnistDrawer(self, width=140, height=140, bg="green")
        self.reset = tk.Button(text="Reset", command=self.canvas.reset_image)
        self.prediction = tk.Label(text="Prediction: ")
        self.label.pack()
        self.button.pack()
        self.reset.pack()
        self.prediction.pack()
        self.canvas.pack()

    def evaluate(self):
        pred = np.argmax(model.predict((self.canvas.get_image() / 255).reshape((1, 784))), axis=-1)[0]
        self.prediction.config(text="Prediction: " + str(pred))


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    app.mainloop()
