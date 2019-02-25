import numpy as np
import math
import networkx as nx
import json
from operator import itemgetter


def import_graph(file):
	with open(file) as f:
		data_dict = json.load(f)
	G = nx.Graph()
	for node_id in data_dict:
		for neighbor in data_dict[node_id]:
			G.add_edge(node_id, neighbor)
	return G

def write_seeds(seed_list):
	file = open("output_file.txt","w") 
	for i in range(50):
		for node in seed_list:
			file.write(node+"\n")
	file.close()


def highest_degree(num_players, num_seeds, G):
	node_degrees = G.degree()
	sorted_node_degrees = np.array(sorted(node_degrees,key=itemgetter(1), reverse = True))
	return sorted_node_degrees[:num_seeds, 0]


def betweenness_centrality(num_players, num_seeds, G):
	betweenness = nx.betweenness_centrality(G)
	sorted_betweenness = np.array(sorted(betweenness.items(),key=itemgetter(1), reverse = True))
	return sorted_betweenness[:num_seeds, 0]

def current_flow_betweenness_centrality(num_players, num_seeds, G):
	betweenness = nx.current_flow_betweenness_centrality(G)
	sorted_betweenness = np.array(sorted(betweenness.items(),key=itemgetter(1), reverse = True))
	return sorted_betweenness[:num_seeds, 0]


def degree_centrality(num_players, num_seeds, G):
	degree = nx.degree_centrality(G)
	sorted_degree = np.array(sorted(degree.items(),key=itemgetter(1), reverse = True))
	return sorted_degree[:num_seeds, 0]

def closeness_centrality(num_players, num_seeds, G):
	closeness = nx.closeness_centrality(G)
	sorted_closeness = np.array(sorted(closeness.items(),key=itemgetter(1), reverse = True))
	return sorted_closeness[:num_seeds, 0]

def current_flow_closeness_centrality(num_players, num_seeds, G):
	closeness = nx.current_flow_closeness_centrality(G)
	sorted_closeness = np.array(sorted(closeness.items(),key=itemgetter(1), reverse = True))
	return sorted_closeness[:num_seeds, 0]

def eigenvector_centrality(num_players, num_seeds, G):
	closeness = nx.eigenvector_centrality(G)
	sorted_closeness = np.array(sorted(closeness.items(),key=itemgetter(1), reverse = True))
	return sorted_closeness[:num_seeds, 0]

def katz_centrality(num_players, num_seeds, G):
	closeness = nx.katz_centrality(G)
	sorted_closeness = np.array(sorted(closeness.items(),key=itemgetter(1), reverse = True))
	return sorted_closeness[:num_seeds, 0]

def communicability_centrality(num_players, num_seeds, G):
	closeness = nx.communicability_centrality(G)
	sorted_closeness = np.array(sorted(closeness.items(),key=itemgetter(1), reverse = True))
	return sorted_closeness[:num_seeds, 0]

def load_centrality(num_players, num_seeds, G):
	closeness = nx.load_centrality(G)
	sorted_closeness = np.array(sorted(closeness.items(),key=itemgetter(1), reverse = True))
	return sorted_closeness[:num_seeds, 0]


def main(filename, num_players, num_seeds, strategy = 'highest_degree'):
	G = import_graph(filename)
	if strategy == 'highest_degree':
		seeds = highest_degree(num_players, num_seeds, G)
	elif strategy == 'betweenness_centrality':
		seeds = betweenness_centrality(num_players, num_seeds, G)
	elif strategy == 'current_flow_betweenness_centrality':
		seeds = current_flow_betweenness_centrality(num_players, num_seeds, G)
	elif strategy == 'degree_centrality':
		seeds = betweenness_centrality(num_players, num_seeds, G)
	elif strategy == 'closeness_centrality':
		seeds = closeness_centrality(num_players, num_seeds, G)
	elif strategy == 'current_flow_closeness_centrality':
		seeds = current_flow_closeness_centrality(num_players, num_seeds, G)
	elif strategy == 'eigenvector_centrality':
		seeds = eigenvector_centrality(num_players, num_seeds, G)
	elif strategy == 'katz_centrality':
		seeds = katz_centrality(num_players, num_seeds, G)
	elif strategy == 'communicability_centrality':
		seeds = communicability_centrality(num_players, num_seeds, G)
	elif strategy == 'load_centrality':
		seeds = load_centrality(num_players, num_seeds, G)

	write_seeds(seeds)


 
if __name__ == '__main__':
	num_players = 1
	num_seeds = 10
	filename = '2.10.20.json'
	main(filename, num_players, num_seeds, 'betweenness_centrality')



