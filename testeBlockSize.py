import os
import matplotlib.pyplot as plt
import numpy as np

def get_time():
	f = open("saida.txt")

	for line in f:
		if "Total time" in line:
			value = line.split()[3]
			break

	f.close()
	return float(value)


def teste(n, bx, by, qtd = 1, executavel="./testeBlockSize"):
	value = 0.0

	for i in range(qtd):
		os.system("%s %d %d %d > saida.txt" % (executavel, n, bx, by))
		value += get_time()

	return value/qtd


N = 2000
threads = [128, 160, 192, 224, 256]
resultados = []
resultados2 = []


for i in threads:
	tempo = teste(N, i, 1, 5, "./testeBlockSize")
	resultados.append(tempo)
	print "Threads %d -> %f" % (i, tempo)

	tempo = teste(N, i, 1, 5, "./testeBlockSizeTrans")
	resultados2.append(tempo)
	print "Threads %d -> %f" % (i, tempo)

plt.plot(threads, resultados)
plt.plot(threads, resultados2)
plt.show()






