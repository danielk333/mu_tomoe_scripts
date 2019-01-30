import numpy as n
import matplotlib.pyplot as plt

def plot_random_meteor_example():
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)

    R = 3000

    px = 512

    data = n.random.randn(R,2)*50.0 + float(px/2)
    data = data[data[:,0] > 0,:]
    data = data[data[:,1] > 0,:]
    data = data[data[:,0] < px,:]
    data = data[data[:,1] < px,:]
    data = n.array(data,dtype=n.int)
    
    ax.plot(data[:,0],data[:,1],'.b',alpha=0.25)
    ax.set(xlabel='X pixel', ylabel='Y pixel', title='Random meteor detection?')
    plt.show()
    


def tomoe_plots():

    in_file = 'data/simultaneous_event_candidates_table_20181115.csv'

    data = n.genfromtxt(in_file,delimiter=',',skip_header=1)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)

    R = data.shape[0]

    data = data.T
    delta=5
    for i in range(R):
        ax.plot([data[9+delta,i],data[12+delta,i]],[data[10+delta,i],data[13+delta,i]],'-b',alpha=0.05)
        ax.plot(data[18+delta,i],data[19+delta,i],'.r',alpha=0.05)
        ax.plot(data[15+delta,i],data[16+delta,i],'.g',alpha=0.05)
    
    fig.savefig('img/concurrent_plot',bbox_inches='tight')
    plt.show()

if __name__=='__main__':

    tomoe_plots()
