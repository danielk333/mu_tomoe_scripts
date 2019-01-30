import numpy as n
import matplotlib.pyplot as plt

data = n.genfromtxt('data/tomoe_part.csv')
data_tomoe = n.genfromtxt('data/simultaneous_event_candidates_mu+tomoe.csv', delimiter=',', skip_header=1)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(0.5*data[:,7] + 0.5*data[:,8])
ax.set_xlabel('Trajectory mean displacement')
ax.set_ylabel('Frequency')

plt.show()
