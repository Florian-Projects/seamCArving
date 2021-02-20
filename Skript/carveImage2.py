import PIL.Image as Image
from Skript.imageProcessing.extras import *
from PIL import ImageFilter

def crop(image, amount):
	"""
	The crop method is responsible for executinng the seamcarving algorithm. amount is a vlaue beetween 0-1 which determines by how many percent the image should be cropped
	"""

	luminanz_image = np.array(image.convert("L")) #create luminanz array from image
	#importance = apply_highpass(luminanz_image, 9)
	img = np.array((image.convert("L")).filter(ImageFilter.FIND_EDGES)) #edge detection with kernel
	importance = np.array(img) #create matrix of edge detected image. Pixel on edges are more important.
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
	"""find the index with the lowest energy value in the top most row of the matrix"""
	y_max, x_max = energy.shape
	lowest_value = energy[0][0]
	lowest_index = 0
	for x in range(x_max):
		if energy[0][x] < lowest_value:
			lowest_value = energy[0][x]
			lowest_index = x
	return lowest_index


def cut_and_merge(image, importance):
	"""
	Takes the image and its importance matrix as arguments and calculates the energy and direction matrix
	The seam with the lowest energy is deleted from the image
	"""
	#print(energy)
	energy, direction = calculate_energy2(importance)
	y_max, x_max = energy.shape
	x_max -= 1

	index = find_index_of_lowest_energy(energy)
	image = image.tolist() #convert numpy ndarray to python lit object
	# we are doing this because python lists are not fixed size. If an element is removed by using list.pop() the gap will also be closed automatically.
	#alternatively we could implement the gap closing and than use numpy.ndarray.resize(). This could improve runtime.
	importance = importance.tolist()
	for y in range(y_max):
		#energy[y][index] = 0
		#removed_image = np.delete(image,index)
		#print(index)

		#figures out the index of the next pixel in the seam
		if direction[y][index] == -1:
			index = index - 1
		elif direction[y][index] == 1:
			index = index + 1

		image[y].pop(index) #delete the pixel at the index
		importance[y].pop(index) #delete the importance value in order recalculate the energy matrix
		#TODO: Maybe only the direction Matrix needs to be recalculated and not the energy maatrix

	#print(energy)
	image = np.array(image) #convert python lsit object back to numpy ndarray.
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
	"""
	Calculates the energy of each pixel in the image.
	Energy is calculated by going through the image bottom to top. The energy in the bottom row is identical to the importance. For the next higher row
	we will look at its direct neighbours (energy_below_right, energy_below_left, energy_below). The neighbour with the lowest energy value is than added
	to the importacne of the current pixel. This value is the energy of that pixel.
	Depending on which neighbour has the lowest energy a value is entered in the direction matrix (-1 = below_left 0 = below 1 = below_right). We can later follow 
	that matrix to create the seam.
	
	TODO: at the moment the energy is recalculated with each time a seam is removed from the image. If this is not done there will be graphical glitches.
	However it might be enough to only recalculate the direction matrix. THis could drastically decrease runtime.
	"""

	y_max, x_max = importance.shape
	y_max -= 1
	x_max -= 1
	energy = np.zeros(importance.shape) #create energy matrix with the same dimensions as the importance matrix
	n_energy = np.zeros(importance.shape)
	direction = np.zeros(importance.shape)
	current_importance = 0
	energy_below = 0
	energy_below_left = 0
	energy_below_right = 0
	for y in range(y_max,-1,-1): #for each row top to bottom repeat:
		for x in range(x_max + 1):
			current_importance = importance[y][x]

			if y == y_max: #if we are at the lowest row
				energy[y][x] = current_importance
				direction[y][x] = 0

			elif x == 0: #if we are at the left most column. This is necessary because ther is no energy_below_left neighbour
				#TODO: This decision tree could be removed by using try: and cath:. Which runtime is better?
				energy_below = energy[y + 1][x] #field underneath the current field
				energy_below_right = energy[y + 1][x + 1] #field below and to the right of the current field

				#find neighbour of lowest energy
				if energy_below > energy_below_right:
					energy[y][x] = current_importance + energy_below_right
					direction[y][x] = 1
				else:
					energy[y][x] = current_importance + energy_below
					direction[y][x] = 0

			elif x == x_max: #the same as previous edge case. Missing energy_below_right neighbour
				energy_below = energy[y + 1][x]
				energy_below_left = energy[y + 1][x - 1]

				if energy_below > energy_below_left:
					energy[y][x] = current_importance + energy_below_left
					direction[y][x] = -1

				else:
					energy[y][x] = current_importance + energy_below
					direction[y][x] = 0

			else: #no edge case all neighbours exist
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

if __name__ == '__main__':
	crop(Image.open("salvadore_deli.jpeg"),0.4)
