import PIL.Image as Image
from PIL import ImageDraw
import numpy as np
import numpy.fft as fft
from imageProcessing.extra import *
from PIL import ImageFilter

def crop(image, amount):
	""" Die crop methode ist für die Ausführung des SeamCarving Algrotihmus verantwortlich. amount gibt dabei an um wie viel Prozent
	das Bild skaliert werden soll. (0-1)"""

	luminanz_image = np.array(image.convert("L")) #erstelle ein luminanz array von dem Bild
	#importance = apply_highpass(luminanz_image, 9)
	img = np.array((image.convert("L")).filter(ImageFilter.FIND_EDGES)) #Kantenerkennung mit Kernel
	importance = np.array(img) #Erstelle eine Mtrix von dem Bild. Diese Matrix gibt an wie wichtig der Pixel ist. Pixel einer Kantes sind am wichtigsten.
	#importance = np.array([[1,3,1,1],[4,1,1,1],[1,1,1,6]])

	energy, direction = calculate_energy2(importance)
	#exportImage(energy)

	rows, columns = luminanz_image.shape
	image_size = int(columns * amount)

	for i in range(image_size + 1):
		print(i)
		luminanz_image,importance = cut_and_merge(luminanz_image, importance)
		#print(energy.shape)

	#print(luminanz_image.shape)
	exportImage(np.array(luminanz_image))

def find_index_of_lowest_energy(energy):
	""" findet den Index mit dem nierigsten Energie wert in der obersten Reihe der Matrix"""
	y_max, x_max = energy.shape
	lowest_value = energy[0][0]
	lowest_index = 0
	for x in range(x_max):
		if energy[0][x] < lowest_value:
			lowest_value = energy[0][x]
			lowest_index = x
	return lowest_index


def cut_and_merge(image, importance):
	"""bekommt das zu skalierende Bild und dessen importacne Matrix Übergeben.
	Daraus wird dann die energy und direction Matrix berechnet.
	Danach wird jeweils der weg mit niedrigster energie aus dem Bild gelöscht, und alle Pixel "rutschen" einen auf."""
	#print(energy)
	energy, direction = calculate_energy2(importance)
	y_max, x_max = energy.shape
	x_max -= 1

	index = find_index_of_lowest_energy(energy)
	image = image.tolist() #konvertiere numpy ndarray zu python list objekt.
	#Ich mache dies weil python list objekte keine festgelegte länge haben. Wenn ich ein element mit list.pop()
	#entferne dann rutschen alle elemente einen auf. Alternativ könnte ich den das "aufrutschen" selber implementieren und dann die
	#numpy.ndarray.resize() methode verwenden. Dies könnte vorteile für die Laufzeit erbringen.
	importance = importance.tolist()
	for y in range(y_max):
		#energy[y][index] = 0
		#removed_image = np.delete(image,index)
		#print(index)

		#finder heraus ob nach below_left below_right oder below gegangen werden muss.
		if direction[y][index] == -1:
			index = index - 1
		elif direction[y][index] == 1:
			index = index + 1

		image[y].pop(index) #lösche den Pixel an der stelle
		importance[y].pop(index) #lösche den importance wert, damit die energie MAtrix neu berechnet werden kann.
		#TODO: eventuell nur direction amtrix neu berechnen und nicht Energie Matrix

	#print(energy)
	image = np.array(image) #konvertiere das python list pbejtk zurück zu numpy ndarray.
	importance = np.array(importance)
	return image, importance

def find_direction(energy):
	index = find_index_of_lowest_energy(energy)
	y_max, x_may = energy.shape
	for y in range(1, y_max):
		energy_below = energy[y][index]
		energy_below_right = energy[y][index + 1]
		energy_below_left = energy[y][index - 1]

def calculate_energy2(importance):
	"""Berechnet die Energie von Jedem Feld des Bildes. Prinzipiell wird die Energie berechnet, indem man die importance matrix
	von Unten nach Oben durchläuft. Die Energie der Untersten Reihe ist dabei identisch zu der Wichtigkeit. Für die nächst höhere Reihe
	werden die direkten unteren Narbarn (energy_below_right, energy_below_left, energy_below) betrachtet. Und es wird die Energie des Nachbarns
	mit der geringsten Energie zu der Energie des betrachteten Feldes addiert. Gleichzeitig wird in die direction Matrix eingetragen in welche,
	welcher nachbar die niedrigste Energie besitzt (-1 = below_left 0 = below 1 = below_right)

	TODO: Im moment wird die energy jedes mal neu berechnet wenn eine reihe von Pixeln entfernt wird, da es sontzt zu grafik Fehlern
	im skalierten Bild kommt (siehe carveImage1.py). Eventuell reicht es aber NUR die Richtungs Matrix jedes mal neu zu berechnen.
	Dies könnt edie Laufzeit deutlich verkürzen"""

	y_max, x_max = importance.shape
	y_max -= 1
	x_max -= 1
	energy = np.zeros(importance.shape) #erstelle energy Matrix mit gleicher dimension der importance Matrix
	n_energy = np.zeros(importance.shape)
	direction = np.zeros(importance.shape)
	current_importance = 0
	energy_below = 0
	energy_below_left = 0
	energy_below_right = 0
	for y in range(y_max,-1,-1): #für jede Reihe von Unten nach Oben wiederhole:
		for x in range(x_max + 1):
			current_importance = importance[y][x]

			if y == y_max: #falls wir in der untersten Reihe sind.
				energy[y][x] = current_importance
				direction[y][x] = 0

			elif x == 0: #falls wir ganz links in der Reihe sind. Notwendig, weil es dann keine energy_below_left Nachbar gibt.
				#TO DO: Eventuel könnte dieser Verzweiungsbaum entfernt werden durch die Verwendung von try: und catch: Welche Laufzeit ist besser?
				energy_below = energy[y + 1][x] #Feld eine Zeile unter dem derzeitigen Feld
				energy_below_right = energy[y + 1][x + 1] # Felde eine Zeile unter dme derzeitigen Feld und einen nach rechts

				#findet den Nachbarn mit der Niedrigsten Energy
				if energy_below > energy_below_right:
					energy[y][x] = current_importance + energy_below_right
					direction[y][x] = 1
				else:
					energy[y][x] = current_importance + energy_below
					direction[y][x] = 0

			elif x == x_max: #genauso wie vorheriger edge case. Diesmal gibt es keine energy_below_right nachbarn
				energy_below = energy[y + 1][x]
				energy_below_left = energy[y + 1][x - 1]

				if energy_below > energy_below_left:
					energy[y][x] = current_importance + energy_below_left
					direction[y][x] = -1

				else:
					energy[y][x] = current_importance + energy_below
					direction[y][x] = 0

			else: #kein sonderfall alle nachabrn existieren
				energy_below_left = energy[y + 1][x - 1]
				energy_below_right = energy[y + 1][x + 1]
				energy_below = energy[y + 1][x]

				if energy_below < energy_below_left and energy_below < energy_below_right:
					energy[y][x] = current_importance + energy_below
					direction[y][x] = 0

				elif energy_below_left < energy_below and energy_below_left < energy_below_right:
					energy[y][x] = current_importance + energy_below_left
					direction[y][x] = -1
				else:
					energy[y][x] = current_importance + energy_below_right
					direction[y][x] = 1

			n_energy[y][x] = (energy[y][x] / (y_max + 1 - y) ) * 6

	#print(direction)
	#print(energy)
	return energy, direction


def calculate_energy(importance):
	y_max, x_max = importance.shape
	y_max = y_max - 1
	x_max = x_max - 1
	energy = np.zeros(importance.shape)
	current_importance = 0
	current_coordinate = 0, 0
	energy_below = 0
	energy_below_left = 0
	energy_below_right = 0
	print(x_max - 1)
	for y in range(y_max + 1):
		for x in range(x_max + 1):

			current_importance = importance[y_max - y][x]
			if y == 0:
				#print("y 0")
				energy[y_max - y][x] = current_importance
			elif x == 0:
				#print("x 1")
				energy_below = energy[y_max - y - 1][x]
				energy_below_right = energy[y_max - y - 1][x + 1]

				if energy_below > energy_below_right:
					energy[y_max - y][x] = current_importance + energy_below_right
				else:
					energy[y_max - y][x] = current_importance + energy_below

			elif x == x_max - 1:
				#print("x = " + str(x))
				#print("x max")
				energy_below = energy[y_max - y - 1][x]
				energy_below_left = energy[y_max - y - 1][x - 1]

				if energy_below > energy_below_left:
					energy[y_max - y][x] = current_importance + energy_below_left
				else:
					energy[y_max - y][x] = current_importance + energy_below

			else:
				#print("else")
				energy_below = energy[y_max - y - 1][x]
				energy_below_left = energy[y_max - y - 1][x - 1]
				#print(x_max)
				energy_below_right = energy[y_max - y - 1][x + 1]

				if energy_below < energy_below_left and energy_below < energy_below_right:
					energy[y_max - y][x] = current_importance + energy_below

				elif energy_below_left < energy_below and energy_below_left < energy_below_right:
					energy[y_max - y][x] = current_importance + energy_below_left
				else:
					energy[y_max - y][x] = current_importance + energy_below_right
	print(energy)
	return energy


crop(Image.open("salvadore_deli.jpeg"),0.4)
