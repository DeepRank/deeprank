import numpy as np 
import subprocess as sp 
import numpy as np 
import sys




def write_newfile(names_oldfile,name_newfile):

	chainID = {0:'A',1:'B'}
	resconv = {
		'A' : 'ALA',
		'R' : 'ARG',
		'N'	: 'ASN',
		'D' : 'ASP',
		'C' : 'CYS',
		'E' : 'GLU',
		'Q' : 'GLN',
		'G' : 'GLY',
		'H' : 'HIS',
		'I' : 'ILE',
		'L' : 'LEU',
		'K' : 'LYS',
		'M' : 'MET',
		'F' : 'PHE',
		'P' : 'PRO',
		'S' : 'SER',
		'T' : 'THR',
		'W' : 'TRP',
		'Y' : 'TYR',
		'V' : 'VAL'
	}

	# write the new file
	new_file = open(name_newfile,'w')


	for ifile,f in enumerate(names_oldfile):

		# read the file
		f = open(f,'r')
		data = f.readlines()[4:-6]
		f.close()

		# write the new file
		for l in data:
			l = l.split()
			if len(l)>0:
				
				chain = chainID[ifile]
				feat = '{:>4}'.format(chain)

				resNum = l[0]
				feat += '{:>10}'.format(resNum)

				resName1 = l[2]
				resName3 = resconv[resName1]
				feat += '{:>10}'.format(resName3)

				feat += '\t'
				values = float(l[-2])
				feat += '\t{:>10}'.format(values)

				feat+= '\n'
				new_file.write(feat)

	new_file.close()



oldfile_dir = '../PSSM/'
oldfiles = sp.check_output('ls %s/*PSSM' %(oldfile_dir),shell=True).decode('utf-8').split()

nfile = len(oldfiles)
oldfiles = np.array(oldfiles).reshape(int(nfile/2),2).tolist()


for filenames in oldfiles:

	print('process files\n\t%s\n\t%s' %(filenames[0],filenames[1]))
	cplx_name = []
	cplx_name.append(filenames[0].split('/')[-1])
	cplx_name.append(filenames[1].split('/')[-1])
	cplx_name = list(set([cplx_name[0][:4],cplx_name[1][:4]]))
	print(cplx_name)
	if len(cplx_name)>1:
		print('error' + cplx_name)
		sys.exit()

	name_newfile = './'+cplx_name[0]+'.PSSM_IC'
	print('\nexport to \t%s\n' %(name_newfile))
	write_newfile(filenames,name_newfile)





