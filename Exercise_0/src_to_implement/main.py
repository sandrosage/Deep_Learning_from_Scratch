from pattern import Checker,Circle,Spectrum
from generator import ImageGenerator
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os

# if not os.path.exists("data"):
#     with zipfile.ZipFile("data.zip", 'r') as zip_ref:
#         zip_ref.extractall("./")


# resolution = 20
# tile_size = 2
# checker = Checker(resolution,tile_size)
# test = checker.draw()
# checker.show("Checkerboard with res: " + str(resolution) + " and tile_size: " + str(tile_size))

# resolution = 1024
# radius = 200
# circle = Circle(resolution,radius,(512, 256))
# test = circle.draw()
# circle.show("Circle with res: " + str(resolution) + " r: " + str(radius))

# resolution = 100
# spectrum = Spectrum(resolution)
# spectrum.draw()
# spectrum.show("Spektrum with res: " + str(resolution))

# generator = ImageGenerator("exercise_data", "Labels.json", 25, [50, 50, 3])
gen = ImageGenerator("exercise_data", "Labels.json", 10, [32, 32, 3], rotation=True, mirroring=False,
                             shuffle=False)
gen.show(3)
