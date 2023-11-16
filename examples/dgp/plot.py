import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np

def plot(X, Y, Sigma, rec, drec, title, d2rec=None, d3rec=None ):

    x = np.linspace(0,2.5,1000)
    y = 70*np.sqrt(0.3*(1+x)**3+0.7)

    #plt.subplot(221)
    plt.xlim(0, 2.5)
    plt.fill_between(rec[:, 0], rec[:, 1] + rec[:, 2], rec[:, 1] - rec[:, 2],
                     facecolor='lightblue')
    plt.plot(rec[:, 0], rec[:, 1], label='Reconstrução')
    plt.plot(x,y, linestyle="dashed", label="LCDM", color="black")
    plt.errorbar(X, Y, Sigma, color='red', fmt='_', label="Observações")
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('H(z)')

    # plt.subplot(222)
    # plt.xlim(0, 2.5)
    # plt.fill_between(drec[:, 0], drec[:, 1] + drec[:, 2], 
    #                  drec[:, 1] - drec[:, 2], facecolor='lightblue')
    # plt.plot(drec[:, 0], drec[:, 1])
    # plt.xlabel('x')
    # plt.ylabel("f'(x)")

    # plt.subplot(223)
    # plt.xlim(0, 2.5)
    # plt.fill_between(d2rec[:, 0], d2rec[:, 1] + d2rec[:, 2], 
    #                  d2rec[:, 1] - d2rec[:, 2], facecolor='lightblue')
    # plt.plot(d2rec[:, 0], d2rec[:, 1])
    # plt.xlabel('x')
    # plt.ylabel("f''(x)")

    # plt.subplot(224)
    # plt.xlim(0, 2.5)
    # plt.fill_between(d3rec[:, 0], d3rec[:, 1] + d3rec[:, 2], 
    #                  d3rec[:, 1] - d3rec[:, 2], facecolor='lightblue')
    # plt.plot(d3rec[:, 0], d3rec[:, 1])
    # plt.xlabel('x')
    # plt.ylabel("f'''(x)")
    
    plt.savefig(f'{title}')

if __name__=="__main__":
    (X,Y,Sigma) = loadtxt("../inputdata.txt", unpack='True')
    rec = loadtxt("f.txt")
    drec = loadtxt("df.txt")
    # d2rec = loadtxt("d2f.txt")
    # d3rec = loadtxt("d3f.txt")

    plot(X, Y, Sigma, rec, drec)
