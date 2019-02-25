import json 
from pprint import pprint

NUM_SEEDS = 5
NUM_ROUNDS = 50
topNcentralities = [0] * NUM_SEEDS
degCentralityDict = {} # <key> = degCentrality, <val> = [nodeId]

with open('f.json') as f: 
	data = json.load(f)

n = len(data)
for i in data: 
	deg = len(data[i])
	degCentrality = float(deg) / (n-1)
	if degCentrality > topNcentralities[0]:
		if degCentrality in degCentralityDict: 
			degCentralityDict[degCentrality].append(i)
		else: 
			degCentralityDict[degCentrality] = [i]
		topNcentralities.append(degCentrality)
		topNcentralities = sorted(topNcentralities)[1:]
		
usedNodes = set()
topNNodes = []
for degCentrality in topNcentralities: 
	for node in degCentralityDict[degCentrality]:
		if node not in usedNodes: 
			usedNodes.add(node)
			topNNodes.append(node)


output = open("output.txt", "w+")
for i in range(NUM_ROUNDS): 
	for j in range(NUM_SEEDS):
		output.write(str(topNNodes[j]) + "\n")
