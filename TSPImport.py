import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx

#  We use a regex here to clean characters and keep only numerics

cities_set = []
cities_tups = []
#  we open the TSP file and put each line cleaned of spaces
#  and newline characters in a list 
def read_tsp_data(tsp_name):
	tsp_name = tsp_name
	with open(tsp_name) as f:
		content = f.read().splitlines()
		cleaned = [x.lstrip() for x in content if x != ""]
		return cleaned
"""
We return a list like 
['NAME: ulysses16.tsp',
'TYPE: TSP',
'COMMENT: Odyssey of Ulysses (Groetschel/Padberg)',
'DIMENSION: 16',
'EDGE_WEIGHT_TYPE: GEO',
'DISPLAY_DATA_TYPE: COORD_DISPLAY',
'NODE_COORD_SECTION',
'1 38.24 20.42',
'2 39.57 26.15',
'3 40.56 25.32',
................
'EOF']
"""

"""
Check for the line DIMENSION in the file and keeps the numeric value
"""
def detect_dimension(in_list):
	non_numeric = re.compile(r'[^\d]+')
	for element in in_list:
		if element.startswith("DIMENSION"):
			return non_numeric.sub("",element)

"""
Iterate through the list of line from the file
if the line starts with a numeric value within the 
range of the dimension , we keep the rest which are
the coordinates of each city
1 33.00 44.00 results to 33.00 44.00
"""
def get_cities(list,dimension):
	dimension = int(dimension)
	for item in list:
		for num in range(1, dimension + 1):
			if item.startswith(str(num)):
				index, space, rest = item.partition(' ')
				if rest not in cities_set:
					cities_set.append(rest)
	return cities_set


"""
Brake each coordinate 33.00 44.00 to a tuple ('33.00','44.00')
"""
def city_tup(list):
	G=nx.Graph()
	for item in list:
		first_coord, space, second_coord = item.partition(' ')
		cities_tups.append((float(first_coord.strip()), float(second_coord.strip())))
	weights = squareform(pdist(cities_tups))
	#print(weights)
	G = nx.from_numpy_matrix(weights)
	print(nx.info(G))
	print(nx.nodes(G))
	#G.nodes(data=True)
	plt.show()
	return G


"""
Putting it all together
"""
def produce_final(file="ulysses22.tsp"):
	data = read_tsp_data(file)
	dimension = detect_dimension(data)
	cities_set = get_cities(data,dimension)
	cities_tups = city_tup(cities_set)


if __name__ == '__main__':
	produce_final()
	# or produce_final("berlin52.tsp") or whatever filename you wish to have
	

