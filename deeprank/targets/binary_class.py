import numpy as np

def __compute_target__(decoy,targrp):

	# fet the mol group
	molgrp = targrp.parent
	molname = molgrp.name

	# if we have a ref
	if '_' not in molname:
		targrp.create_dataset('BIN_CLASS',data=np.array(1.0))

	# or it's a decoy
	else:
		targrp.create_dataset('BIN_CLASS',data=np.array(0.0))
