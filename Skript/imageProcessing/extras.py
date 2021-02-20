import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import numpy as np
from numpy.fft import fftshift
import PIL
from PIL import ImageDraw

def plotFrequencySpectrum(scoefficient):
	"""
	Diese Methode zeigt das Amplituden Spektrum des Fourierspektrums.
	Input ist ein numpy 2d Array
	"""
	y, x = scoefficient.shape
	fig = plt.figure()
	plt.subplot(1, 1, 1)
	h = plt.imshow(np.abs(scoefficient), cmap="gray", interpolation='nearest', norm=LogNorm(vmin=0.1, vmax=np.abs(scoefficient).max()+0.2))
	plt.colorbar(h, shrink=1, aspect=5)
	plt.xlim(x, 0)
	plt.ylim(y, 0)
	plt.show()

def plotShiftedSpectrum(coefficient):
	"""
	Diese Methode zeigt das zentralisierte Amplituden Spektrum des Fourierspektrums
	Parameter ist ein numpy 2d Array
	"""
	y, x = coefficient.shape
	coefficient = fftshift(coefficient)
	plt.imshow(np.abs(coefficient), cmap="gray", interpolation='nearest', norm=LogNorm(vmin=0.1, vmax=(np.abs(coefficient).max())))
	plt.xlim(x, 0)
	plt.ylim(y, 0)
	plt.show()
	
def createImage(simageArr):
	"""
	Diese Methode zeigt eine 2-dimensionale Matrix als Luminanzbild
	Parameter ist ein numpy 2d Array
	"""

	plt.subplot(1, 1, 1)
	f = plt.imshow(simageArr, cmap="gray")
	plt.colorbar(f, shrink=1, aspect=5)
	plt.show()

def exportImage(simageArr):
	"""
	Diese Methode exportiert eine 2-dimensionale Matrix als Luminanzbild unter dem namen export.png.
	Falls export.png bereits existiert wird es überschrieben!
	Parameter ist ein numpy 2d Array
	"""
	N, M = simageArr.shape
	image = PIL.Image.new("L", (M, N), color=0)
	image.save("export.png")
	draw = ImageDraw.Draw(image)
	for m in range(M):
		for n in range(N):
			draw.point((m, n), int(simageArr[n][m].real))
	image.save("export.png")
