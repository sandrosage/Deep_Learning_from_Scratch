import numpy as np
import matplotlib.pyplot as plt 

class Checker:
    def __init__(self, resolution, tile_size) -> None:
        self.resolution = resolution
        self.tile_size = tile_size
        if ((self.resolution % (2*self.tile_size))!=0):
            print("Resolution has to be evenly divided by 2*tile_size")
            raise ValueError
    
    def draw(self):
        first_row = np.concatenate((np.zeros(self.tile_size), np.ones(self.tile_size)), axis=None)
        inverted_row = first_row[::-1]
        grid_row = np.tile(first_row, (self.tile_size, int(self.resolution/(2*self.tile_size))))
        grid_inverted = np.tile(inverted_row, (self.tile_size, int(self.resolution/(2*self.tile_size))))
        grids = np.concatenate((grid_row,grid_inverted),axis=0)
        self.output = np.tile(grids, (int(self.resolution/(2*self.tile_size)),1))
        print(self.output.shape)
        return self.output.copy()


    def show(self, title):
        plt.imshow(self.output, cmap="gray")
        plt.axis("off")
        plt.title(title)
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position) -> None:
        self.resolution = resolution
        self.radius = radius
        self.position = position
    
    def draw(self):
        (x_shift, y_shift)  = self.position
        x_shift = x_shift - int(self.resolution/2)
        y_shift = - y_shift + int(self.resolution/2)
        xx, yy = np.meshgrid(np.linspace(-((self.resolution-1)/2) - x_shift, ((self.resolution-1)/2) - x_shift, self.resolution) , np.linspace(-((self.resolution-1)/2) + y_shift, ((self.resolution-1)/2)+ y_shift, self.resolution))
        distance = np.sqrt(xx**2 + yy**2)
        circle = np.zeros((self.resolution, self.resolution))
        circle[distance <= self.radius] = 1
        self.output = circle
        print(self.output.shape)
        return self.output.copy()

    def show(self,title):
        plt.imshow(self.output, cmap="gray")
        # plt.axis("off")
        plt.title(title)
        plt.show()

class Spectrum:
    def __init__(self, resolution) -> None:
        self.resolution = resolution

    def draw(self):
        blank_image =np.zeros([self.resolution,self.resolution, 3])
        blank_image[:,:,0] = np.linspace(0,1,self.resolution) # first channel: red -> from right to left: 0-1
        blank_image[:,:,1] = np.linspace(0,1,self.resolution).reshape (self.resolution, 1) # second channel: green -> reshape: from bottom to top
        blank_image[:,:,2] = np.linspace(1,0,self.resolution) # third channel: blue -> from left to right: 1-0
        self.output = blank_image
        print(self.output.shape)
        return self.output.copy()

    def show(self,title):
        plt.axis("off")
        plt.title(title)
        plt.imshow(self.output)
        plt.show()