import tkinter as tk
import numpy as np

# "#rrggbb" - 0 to 255 for each color
palette = list()
for i in range(256):
    palette.append('#' + 3 * hex(i)[2:])


class MnistDrawer(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # todo figure out how to parse kwargs for width and height, store these in variables here
        # todo figure out why canvas is a few pixels greater in size in both directions after setting width and height (winfo_width/height)
        self.width = 140
        self.height = 140

        # for MNIST, images are 28 x 28 pixels
        self.pixel_width = 140 / 28
        self.pixel_height = 140 / 28
        self.pixels = dict()
        self.image = np.zeros((28, 28), dtype='uint8')
        self.add_pixels()
        self.bind("<Button-1>", self.stroke)
        self.bind("<ButtonRelease-1>", self.refresh_display)
        self.bind("<B1-Motion>", self.stroke)

        # brush size probably 3 pixels wide, grid below shows how much each pixel should light up when brush passes by
        # little  some  little
        # some    lot   some
        # little  some  little
        self.brush = np.array([[1, 42, 1],
                               [42, 96, 42],
                               [1, 42, 1]])

    def add_pixels(self):
        for i in range(28):
            for j in range(28):
                start_x = j * self.pixel_width
                start_y = i * self.pixel_height
                pixel = self.create_rectangle(start_x, start_y, start_x + self.pixel_width, start_y + self.pixel_height,
                                              fill=palette[0])
                self.pixels[(i, j)] = pixel

    # TODO figure out a better way to draw so that a fast-moving cursor doesn't skip when moving quickly
    #  - maybe taking a path of points and then calling stroke on every point intersecting with the lines created by the path points
    def stroke(self, event):
        j = int(event.x // self.pixel_width)
        i = int(event.y // self.pixel_height)
        # the following condition ensures that all 9 pixels should be able to be updated
        if 1 <= j <= 26 and 1 <= i <= 26:
            top = i - 1
            bottom = i + 2
            left = j - 1
            right = j + 2
            self.image[top:bottom, left:right] = np.minimum(self.image[top:bottom, left:right] + self.brush, 255)

    # TODO implement a different system for display; maybe only updating the pixels that have been modified
    #  maybe also figure out how to constantly refresh the display so the user can see what's drawn immediately
    def refresh_display(self, event):
        # update pixel display
        for i in range(28):
            for j in range(28):
                color = palette[self.image[i, j]]
                self.itemconfigure(self.pixels[(i, j)], fill=color, outline=color)

    def get_image(self):
        return self.image

    def reset_image(self):
        self.image = np.zeros((28, 28), dtype='uint8')
        self.refresh_display(None)
