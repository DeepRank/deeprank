import numpy as np 
import subprocess as sp
from deeprank.tools import StructureSimilarity
import matplotlib.pyplot as plt
import time


def test_rmsd():

	# specify wich data to us
	MOL = './1AK4/'
	decoys = MOL + '/decoys/'
	ref    = MOL + '/native/1AK4.pdb'
	data   = MOL + '/haddock_data/' 



	# options
	verbose = True
	plot = False
	save = False

	# get the list of decoy names
	decoy_list = sp.check_output('ls %s/*.pdb' %decoys,shell=True).decode('utf-8').split()

	# reference data used to compare ours
	haddock_data = {}
	haddock_files =  [data+'1AK4.Fnat',data+'1AK4.lrmsd',data+'1AK4.irmsd']

	# extract the data from the haddock files
	for i,fname in enumerate(haddock_files):
		
		# read the file
		f = open(fname,'r')
		data = f.readlines()
		data = [d.split() for d in data if not d.startswith('#')]
		f.close()

		# extract/store the data
		for line in data:
			mol_name = line[0].split('.')[0]
			if i == 0:
				haddock_data[mol_name] = np.zeros(3)
			haddock_data[mol_name][i] = float(line[1])

			
	# init all the data handlers
	nconf = len(haddock_data)
	deep = np.zeros((nconf,3))
	hdk = np.zeros((nconf,3))

	# compute the data with deeprank
	deep_data = {}
	t0 = time.time()
	for i,decoy in enumerate(decoy_list):

		print('\n-->' + decoy)

		sim = StructureSimilarity(decoy,ref)
		lrmsd = sim.compute_lrmsd_fast(method='svd',lzone='1AK4.lzone')
		irmsd = sim.compute_irmsd_fast(method='svd',izone='1AK4.izone')
		fnat = sim.compute_Fnat_fast(ref_pairs='1AK4.refpairs')
		
		
		mol_name = decoy.split('/')[-1].split('.')[0]
		deep_data[mol_name] =  [fnat,lrmsd,irmsd]
		deep[i,:] = deep_data[mol_name]
		hdk[i,:] = haddock_data[mol_name]

		if verbose:
			print("HADDOCK : fnat = %1.6f\tlrmsd = %2.7f\tirmsd = %2.7f" %(haddock_data[mol_name][0],haddock_data[mol_name][1],haddock_data[mol_name][2]))
			print("DEEP    : fnat = %1.6f\tlrmsd = %2.7f\tirmsd = %2.7f" %(deep_data[mol_name][0],deep_data[mol_name][1],deep_data[mol_name][2]))
		
	t1=time.time()-t0

	# save the data	
	if save:
		np.savetxt('deep.dat',deep)
		np.savetxt('hdk.dat',hdk)
	
	# remove the data
	#sp.call('rm 1AK4.lzone 1AK4.izone 1AK4.refpairs') 

	# plot
	if plot:
		plt.subplot(3,1,1)
		plt.scatter(hdk[:,0],deep[:,0],label='Fnat')
		mini = np.min(deep[:,0])
		maxi = np.max(deep[:,0])
		plt.plot(  [mini,maxi],[mini,maxi],'--',color='black' )
		plt.legend(loc=4)
		plt.xlabel('PROFIT')
		plt.ylabel('DEEP')

		plt.subplot(3,1,2)
		plt.scatter(hdk[:,1],deep[:,1],label='l-rmsd')
		mini = np.min(deep[:,1])
		maxi = np.max(deep[:,1])
		plt.plot(  [mini,maxi],[mini,maxi],'--',color='black' )  
		plt.legend(loc=4)
		plt.xlabel('PROFIT')
		plt.ylabel('DEEP')

		plt.subplot(3,1,3)
		plt.scatter(hdk[:,2],deep[:,2],label='i-rmsd')
		mini = np.min(deep[:,2])
		maxi = np.max(deep[:,2])
		plt.plot(  [mini,maxi],[mini,maxi],'--',color='black' )
		plt.legend(loc=4)
		plt.xlabel('PROFIT')
		plt.ylabel('DEEP')
		plt.savefig('figure.png')


	# print the deltas
	delta = np.max(np.abs(deep-hdk),0)

	# assert the data
	try:
		assert np.all(delta<[1E-3,1,1E-3])
		print('\n')
		print('OK : %d molecules tested in %f sec.' %(len(decoy_list),t1))
		print('   : Maximum Fnat  deviation %1.3e' %(delta[0]))
		print('   : Maximum LRMSD deviation %1.3e' %(delta[1]))
		print('   : Maximum IRMSD deviation %1.3e' %(delta[2]))

	except AssertionError:
		print('\n')
		print('Failed : %d molecules tested in %f sec.' %(len(decoy_list),t1))
		print('       : Maximum Fnat  deviation %1.3e' %(delta[0]))
		print('       : Maximum LRMSD deviation %1.3e' %(delta[1]))
		print('       : Maximum IRMSD deviation %1.3e' %(delta[2]))	

if __name__ == '__main__':
	test_rmsd()

