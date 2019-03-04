import sim
#sim.run([graph], [dict with keys as names and values as a list of nodes])
import numpy as np
import math
import networkx as nx
import json
from operator import itemgetter
from networkx.algorithms.approximation import max_clique
from networkx.algorithms import find_cliques
from networkx.algorithms.approximation import min_weighted_dominating_set, min_weighted_vertex_cover
import random
from sklearn.cluster import SpectralClustering

import operator
def import_graph(file):
	with open(file) as f:
		data_dict = json.load(f)
	G = nx.Graph()
	for node_id in data_dict:
		# s = np.random.uniform(0.0,1.0,1)
		# if s > 0.6:
		for neighbor in data_dict[node_id]:
			G.add_edge(node_id, neighbor)
	return G

def write_seeds(seed_list, outfile):
	file = open(outfile,"w") 
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


def load_centrality(num_players, num_seeds, G):
	closeness = nx.load_centrality(G)
	sorted_closeness = np.array(sorted(closeness.items(),key=itemgetter(1), reverse = True))
	return sorted_closeness[:num_seeds, 0]


# Choose num_seeds nodes from the maximal clique of the graph.
def only_max_clique(num_seeds, G):
	max_cliq = max_clique(G)
	print("GOT CLIQUE")
	seeds = random.sample(max_cliq, min(num_seeds, len(max_cliq)))
	print("GOT FIRST SAMPLE")

	if len(seeds)<num_seeds:
		seeds.extend(random.sample(max_cliq, num_seeds- len(seeds)))
		# seeds = list(set(seeds))
	print("GOT SECOND SAMPLE")
	return seeds




# Choose num_seeds nodes from the min weighted dominating set of the graph.
def dominating_set(num_seeds, G):
	max_set = min_weighted_dominating_set(G)
	seeds = random.sample(max_set, min(num_seeds, len(max_set)))
	if len(seeds) < num_seeds:
		seeds.extend(random.sample(set(G.nodes())-set(seeds), num_seeds-len(seeds)))
		# seeds = list(set(seeds))
	return seeds

# Choose num_seeds nodes from the min weighted vertex cover of the graph.
def vertex_cover(num_seeds, G):
	max_set = min_weighted_vertex_cover(G)
	seeds = random.sample(max_set, min(num_seeds, len(max_set)))

	if len(seeds) < num_seeds:
		seeds.extend(random.sample(set(G.nodes())-set(seeds), num_seeds-len(seeds)))
		# seeds = list(set(seeds))
	return seeds



# Choose num_seeds nodes from the top-K cliques of the graph.
def top_cliques(num_seeds, G):
	seeds = []
	top_cliques = list(find_cliques(G))
	num_cliq = len(top_cliques)


	size_per_clique = int(num_seeds/(num_cliq-1))
	last_clique = num_seeds - (size_per_clique * (num_cliq-1))
	for k in range(num_cliq-1):
		
		seeds.extend(random.sample(top_cliques[k], min(size_per_clique, len(top_cliques[k]))))
		

	seeds.extend(random.sample(top_cliques[-1], min(last_clique, len(top_cliques[-1]))))

	# seeds = list(set(seeds))
	# while len(seeds)!= num_seeds:
	# 	seeds.extend(random.sample(G.nodes(), num_seeds-len(seeds)))
	# 	seeds = list(set(seeds))
		
	
	return seeds



# Choose nodes adjacent to top degree nodes (with probability p).
def adjacent_to_max_degree(num_seeds, G, prob):
	node_degrees = G.degree()
	sorted_node_degrees = np.array(sorted(node_degrees,key=itemgetter(1), reverse = True))
	sorted_node_degrees = sorted_node_degrees[:num_seeds, 0]
	for i in range(len(sorted_node_degrees)):
		s = np.random.uniform(0.0,1.0,1)
		if s <= prob:
			sorted_node_degrees[i] = random.choice(list(G.neighbors(sorted_node_degrees[i])))
	return sorted_node_degrees



def clustering(num_seeds, G):
	adj_matrix = nx.to_numpy_matrix(G)
	cluster_labels = SpectralClustering(n_clusters=6, assign_labels="discretize", random_state=0).fit_predict(adj_matrix)
	#print("ADJACENCY: ", adj_matrix)
	clusters_dict = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
	nodes_clusters_dict = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
	#print("LABELS:", cluster_labels)
	#nodes = G.nodes()
	degrees = list(G.degree())
	# print("NODES", nodes)
	# print("DEGREES", degrees)
	#print("TEST", G.nodes[100].degree())
	# print("len", len(degrees))
	# print("lem2", len(cluster_labels))
	# for c in range(len(degrees)):
	# 	#print("WHAT", nodes[0])
	# 	print(c)
	# 	degree = degrees[c][1]
	# 	print("VALUE", str(cluster_labels[c]))
	# 	clusters_dict[str(cluster_labels[c])].append(degree)
	for i in range(len(cluster_labels)):
		degree = degrees[i][1]
		#print(str(cluster_labels[i]))
		clusters_dict[str(cluster_labels[i])].append(degree)
		nodes_clusters_dict[str(cluster_labels[i])].append(degrees[i][0])

	# for elem in range(len(degrees)):
	# 	print(degrees[elem][1])
	

	for cluster in clusters_dict:
		clusters_dict[cluster] = np.mean(clusters_dict[cluster])
	#print(clusters_dict)
	#print(max(clusters_dict.items(), key=operator.itemgetter(1))[0])
	#top_nodes_list = nodes_clusters_dict[max(clusters_dict.items(), key=operator.itemgetter(1))[0]]
	#print("TOP", top_nodes_list)

	top_three_keys = sorted(clusters_dict, key=clusters_dict.get, reverse=True)[:3]
	#top_nodes = []
	if len(nodes_clusters_dict[top_three_keys[0]]) >= len(nodes_clusters_dict[top_three_keys[1]]) and\
	 len(nodes_clusters_dict[top_three_keys[0]]) >= len(nodes_clusters_dict[top_three_keys[2]]):
	 top_nodes = nodes_clusters_dict[top_three_keys[0]]
	elif len(nodes_clusters_dict[top_three_keys[1]]) >= len(nodes_clusters_dict[top_three_keys[0]]) and\
	 len(nodes_clusters_dict[top_three_keys[1]]) >= len(nodes_clusters_dict[top_three_keys[2]]):
	 top_nodes = nodes_clusters_dict[top_three_keys[1]]
	else:
		top_nodes = nodes_clusters_dict[top_three_keys[2]]

	node_degrees = G.degree(top_nodes)
	sorted_node_degrees = np.array(sorted(node_degrees,key=itemgetter(1), reverse = True))
	return sorted_node_degrees[:num_seeds, 0]

	



# Choose nodes adjacent to top degree nodes (with probability p).
def ensemble_three(num_seeds, num_players, G):
	# num_seeds_partial = int(num_seeds/3)
	# num_seeds_last = num_seeds-(num_seeds_partial*2)
	seeds = []
	seeds.extend(adjacent_to_max_degree(3, G, 0.5))
	seeds.extend(dominating_set(3, G))
	seeds.extend(betweenness_centrality(num_players, 4, G))

	# seeds = list(set(seeds))
	# while len(seeds)<= num_seeds:
	# 	seeds.extend(random.sample(G.nodes(), num_seeds-len(seeds)))
	# 	seeds = list(set(seeds))
	return seeds

def ensemble_two(num_seeds, num_players, G):
	# num_seeds_partial = int(num_seeds/3)
	# num_seeds_last = num_seeds-(num_seeds_partial*2)
	seeds = []
	seeds.extend(adjacent_to_max_degree(3, G, 0.5))
	seeds.extend(vertex_cover(3, G))
	seeds.extend(closeness_centrality(num_players, 4, G))

	# seeds = list(set(seeds))
	# while len(seeds)<= num_seeds:
	# 	seeds.extend(random.sample(G.nodes(), num_seeds-len(seeds)))
	# 	seeds = list(set(seeds))
	return seeds

def ensemble_one(num_seeds, num_players, G):
	# num_seeds_partial = int(num_seeds/3)
	# num_seeds_last = num_seeds-(num_seeds_partial*2)
	seeds = []
	seeds.extend(adjacent_to_max_degree(3, G, 0.5))
	seeds.extend(vertex_cover(3, G))
	seeds.extend(dominating_set(4, G))

	# seeds = list(set(seeds))
	# while len(seeds)<= num_seeds:
	# 	seeds.extend(random.sample(G.nodes(), num_seeds-len(seeds)))
	# 	seeds = list(set(seeds))
	return seeds



def main(G, filename, num_players, num_seeds, strategy = 'highest_degree'):
	
	print("FINISHED IMPORTING GRAPH")
	if strategy == 'highest_degree':
		seeds = highest_degree(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_highest_degree.txt")

	elif strategy == 'betweenness_centrality':
		seeds = betweenness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_betweenness_centrality.txt")

	elif strategy == 'current_flow_betweenness_centrality':
		seeds = current_flow_betweenness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_current_flow_betweenness_centrality.txt")

	elif strategy == 'degree_centrality':
		seeds = betweenness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_betweenness_centrality.txt")

	elif strategy == 'closeness_centrality':
		seeds = closeness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_closeness_centrality.txt")

	elif strategy == 'current_flow_closeness_centrality':
		seeds = current_flow_closeness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_current_flow_closeness_centrality.txt")

	elif strategy == 'eigenvector_centrality':
		seeds = eigenvector_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_eigenvector_centrality.txt")

	elif strategy == 'load_centrality':
		seeds = load_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_load_centrality.txt")

	elif strategy == 'only_max_clique':
		seeds = only_max_clique(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_only_max_clique.txt")

	elif strategy == 'top_cliques':
		seeds = top_cliques(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_top_cliques.txt")

	elif strategy == 'dom_set':
		seeds = dominating_set(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_dom_set.txt")

	elif strategy == 'vertex_cover':
		seeds = vertex_cover(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_vertex_cover.txt")

	elif strategy == 'adjacent_to_max':
		prob = 0.5
		seeds = adjacent_to_max_degree(num_seeds, G, prob)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_adjacent_to_max.txt")

	elif strategy == 'emsemble_three':
		seeds = ensemble_three(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_emsemble_three.txt")

	elif strategy == 'emsemble_two':
		seeds = ensemble_two(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_emsemble_two.txt")

	elif strategy == 'emsemble_one':
		seeds = ensemble_one(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_emsemble_one.txt")
	elif strategy == 'clustering':
		seeds = clustering(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		write_seeds(seeds, "output_clustering.txt")

def main_test(G, filename, num_players, num_seeds, strategy = 'highest_degree'):
	
	print("FINISHED IMPORTING GRAPH")
	if strategy == 'highest_degree':
		seeds = highest_degree(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'betweenness_centrality':
		seeds = betweenness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'current_flow_betweenness_centrality':
		seeds = current_flow_betweenness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'degree_centrality':
		seeds = betweenness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'closeness_centrality':
		seeds = closeness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'current_flow_closeness_centrality':
		seeds = current_flow_closeness_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'eigenvector_centrality':
		seeds = eigenvector_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'load_centrality':
		seeds = load_centrality(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'only_max_clique':
		seeds = only_max_clique(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'top_cliques':
		seeds = top_cliques(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'dom_set':
		seeds = dominating_set(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'vertex_cover':
		seeds = vertex_cover(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'adjacent_to_max':
		prob = 0.5
		seeds = adjacent_to_max_degree(num_seeds, G, prob)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'emsemble_three':
		seeds = ensemble_three(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'emsemble_two':
		seeds = ensemble_two(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'emsemble_one':
		seeds = ensemble_one(num_players, num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds

	elif strategy == 'clustering':
		seeds = clustering(num_seeds, G)
		print("STRATEGY:"+ strategy + "SEEDS: ", seeds)
		return seeds
	

 
def run_output():
	filename = 'pandemaniac_sim/6.10.7.json'
	G = import_graph(filename)
	num_players = 1
	num_seeds = 10
	
	all_strats = ['highest_degree','clustering','adjacent_to_max','emsemble_three','emsemble_two', 'emsemble_one', 'top_cliques', 'vertex_cover', \
	'dom_set', 'betweenness_centrality', \
		'degree_centrality', 'closeness_centrality', \
		'eigenvector_centrality', 'load_centrality', 'only_max_clique']
	#all_strats = ['emsemble_three',  'top_cliques']
	for strat in all_strats:
		main(G, filename, num_players, num_seeds, strat)

def half_half(num_seeds, G):
	nodes = []
	nodes.extend(only_max_clique(math.floor(num_seeds/2), G))
	nodes.extend(clustering(math.ceil(num_seeds/2), G))
	return nodes


# def run_test():
# 	filename = 'pandemaniac_sim/27.10.2.json'
# 	G = import_graph(filename)
# 	dict_G = nx.to_dict_of_lists(G)
# 	num_players = 2
# 	num_seeds = 0

# 	TA_degree_nodes = main_test(G, filename, num_players, num_seeds, "highest_degree")
# 	my_nodes = main_test(G, filename, num_players, num_seeds, 'only_max_clique')
# 	#my_nodes = half_half(num_seeds, G)
# 	nodes = {"TA": TA_degree_nodes, "mine": my_nodes}
# 	print("NODES", nodes)
# 	outcome = sim.run(dict_G, nodes)
# 	print(outcome)




if __name__ == '__main__':
	run_output()
