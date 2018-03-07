
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py


def plot_hit_rate(hdf5,epoch=None,figname=None,irmsd_thr = 4.0,inverse = False):

    '''Plot the hit rate of the different training/valid/test sets found in a dictionnary

    The hit rate is defined as:
        the percentage of positive decoys that are included among the top m decoys.
        a positive decoy is a native-like one with a i-rmsd < 4A

    Args:
        hdf5 (str): name of the data.hdf5 file
        epcoch (int): index of the epoch
        figname (str): filename for the plot
        irmsd_thr (float, optional): threshold for 'good' models
        inverse (bool, optional): Must be true if score is inverse to ranking (e.g. for IRMSD)

    '''

    color_plot = {'train':'red','valid':'blue','test':'green'}
    labels = ['train','valid','test']

    h5 = h5py.File(hdf5,'r')
    if epoch is None:
        keys = list(h5.keys())
        last_epoch_key = list(filter(lambda x: 'epoch_' in x,keys))[-1]
    else:
        last_epoch_key = 'epoch_%04d' %epoch
        if last_epoch_key not in h5:
            print('Incorrect epcoh name\n Possible options are: ' + ' '.join(list(h5.keys())))
            h5.close()
            return
    data = h5[last_epoch_key]

    fig,ax = plt.subplots()

    for l in labels:

        if l in data:

            # get the target values
            out = data[l]['outputs']

            # get the irmsd
            irmsd = []
            for fname,mol in data[l]['mol']:
                if not os.path.isfile(fname):
                    raise FileNotFoundError('File %s  not found' %fname)

                f5 = h5py.File(fname,'r')
                irmsd.append(f5[mol+'/targets/IRMSD'].value)
                f5.close()

            # sort the data
            ind_sort = np.argsort(out)
            if not inverse:
                ind_sort = ind_sort[::-1]
            irmsd = np.array(irmsd)[ind_sort]

            # compute the hit rate
            npos = len(irmsd[irmsd<irmsd_thr])
            if npos == 0:
                npos = len(irmsd)
                print('Warning : Non positive decoys found in %s for hitrate plot' % l)
            hit = np.cumsum(irmsd<irmsd_thr)/ npos

            # plot
            plt.plot(hit,c = color_plot[l],label=l)

    legend = ax.legend(loc='upper left')
    ax.set_xlabel('Top M')
    ax.set_ylabel('Hit Rate')
    if figname is not None:
    	fig.savefig(figname)
    	plt.close()
    else:
    	plt.show()
    h5.close()

if __name__ == '__main__':
    hdf5 = 'data.hdf5'
    plot_hit_rate(hdf5,epoch=None,figname=None,irmsd_thr = 400.0,inverse = False)