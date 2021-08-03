import warnings

from pdb2sql import pdb2sql, interface
from deeprank.features import FeatureClass
from copy import deepcopy
from io import StringIO
import numpy as np

try:
    from Bio.PDB import PDBParser, SASA

except ImportError:
    warnings.warn('Biopython module not found. Make sure you have Biopython 1.79 or later')
    raise


class Edesolv(FeatureClass):

    # init the class
    def __init__(self, pdb_data):
        """Compute the energy of desolvation feature.
        Desolvation energy calculation method taken from
        Dominguez et al. JACS 125:1731 (2003)
        Biopython 1.79 or later is required for this feature to calculate SASA.

        Energy of desolvation for each atom is calculated as:
        Edesolv = (Cplx desolvation energy) - (Unbound desolvation energy)

        Complex (Cplx) and Unbound desolvation energy are calculated as:
        CplxEdesolv = Cplx SASA * Atomic Desolvation Parameter
        UnboundEdesolv = Unbound SASA * Atomic Desolvation Parameter        

        Atomic solvation parameters are taken from 
        Fernandez-Recio et al. JMB 335:843 (2004)
        Unbound SASA is calculated by simply moving away the two chains 
        and calculating the resulting SASA.

        Args:
            pdb_data (list(byte) or str): pdb data

        Example:
            >>> edesolv = Edesolv('1AK4.pdb')
            >>> edesolv.get_feature()
        """
        self.pdb = pdb_data
        self.feature_data = {}
        self.feature_data_xyz = {}
    
    # the feature extractor
    def compute_feature(self, chain1='A', chain2='B'):
        """ Computes the actual Edesolv feature"""
        
        esolcpx = 0.0
        esolfree = 0.0
        
        # create a sql database
        pdb_db = pdb2sql(self.pdb)
        self.db = interface(pdb_db)
        
        #Create a temporary pdb-like string to calculate complex and free Edesolv
        temp_pdb = StringIO(('\n').join(pdb_db.sql2pdb()))
        temp_pdb.seek(0)

        # Parse structure in Bio.PDB
        p = PDBParser(QUIET=1) 
        struct = p.get_structure('temp', temp_pdb)
        
        # Make free_structure fake object and translate the chains away from each other
        # TODO: use pdb2sql.translation for this
        free_struct = deepcopy(struct)
        for i, chain in enumerate(chains):
            for residue in free_struct[0][chain]:
                for atom in residue:
                    if i == 0:
                        atom.coord += 300
                    elif i == 0:
                        atom.coord -= 300
        
        
        # get the contact atoms
        indA,indB = list(self.db.get_contact_atoms(chain1=chain1, chain2=chain2).values())
        contact = indA + indB
        
        # extract the atom keys and xyz of the contact CA atoms
        keys = self.db.get('serial,chainID,resName,resSeq,name',rowID=contact)
        xyz = self.db.get('x,y,z',rowID=contact)
        
        #Make SASA class
        sr = SASA.ShrakeRupley()
        #Compute complex structue SASA
        sr.compute(struct, level='A') 
        #Compute free structure SASA
        sr.compute(free_struct, level='A')


        # Get disulfide bonds (important to determine sulfide atoms edesolv values)
        disulfides_cys = get_disulfide_bonds(self.db)

        
        # create the dictionary of human readable and xyz-val data
        self.edesolv_data = {}
        self.edesolv_data_xyz = {}

        for key, coords in zip(keys, xyz):
            brk_flag = False
            disulf_bond = False
            atom = Atom(key)

            # Calculate complex sasa
            try:
                atom.cpx_sasa = struct[0][atom.chainID][atom.resid][atom.name].sasa
            except KeyError: #Handling alternative residues error
                print('Alternative residue found at:', key)
                brk_flag = True
                pass 
            
            # Calculate unbound (translated) sasa
            try:
                atom.free_sasa = free_struct[0][atom.chainID][atom.resid][atom.name].sasa
            except KeyError: #Handling alternative residues error
                print('Alternative residue found at:', key)
                brk_flag = True
                pass 
            
            if not brk_flag:
                
                # check if the atom is a sulfide involved in a disulfide bond
                for cys in disulfides_cys:
                    if cys == (atom.chainID, atom.resid):
                        disulf_bond = True
                        
                # Assign solvation parameters
                Assign_solv_param(atom, disulf_bond)
                
                #Calculate atom complex edesolv, free edesolv and final edesolv (cpx - free)
                atom.esolcpx = atom.cpx_sasa * atom.solv
                atom.esolfree = atom.free_sasa * atom.solv
                atom.edesolv = atom.esolcpx - atom.esolfree
                esolcpx += atom.esolcpx
                esolfree += atom.esolfree
                
                # xyz-val
                # { (0|1,x,y,z) : [val] }
                chain = [{chain1:0,chain2:1}[key[1]]]
                k = tuple(chain + coords)

                #Human readable
                self.edesolv_data[(key[1], key[3], key[2], key[4])] = [atom.edesolv]
                #XYZ data
                self.edesolv_data_xyz[k] = [atom.edesolv]

        self.feature_data['Edesolv'] = self.edesolv_data
        self.feature_data_xyz['Edesolv'] = self.edesolv_data_xyz

##################################################################

##################################################################

def __compute_feature__(pdb_data, featgrp, featgrp_raw):
    """Main function called in deeprank for the feature calculations.

    Args:
        pdb_data (list(bytes)): pdb information
        featgrp (str): name of the group where to save xyz-val data
        featgrp_raw (str): name of the group where to save human readable data
    """

    edesolv_feat = Edesolv(pdb_data)
    edesolv_feat.compute_feature()

    # export in the hdf5 file
    edesolv_feat.export_dataxyz_hdf5(featgrp)
    edesolv_feat.export_data_hdf5(featgrp_raw)

    # close
    edesolv_feat.db._close()        
        
class Atom():

    def __init__(self,atom_specs):
        """Atom class necessary to calculate atomic-level surface area,
        solvation parameters and desolvation energy.
        
        Args:
            atom_specs("""

        serial = atom_specs[0]
        chainID = atom_specs[1]
        resn = atom_specs[2]
        resid = atom_specs[3]
        atom = atom_specs[4]

        self.serial = serial
        self.chainID = chainID
        self.resn = resn
        self.resid = resid
        self.name = atom
        if atom in ['C', 'O', 'N', 'CA']:
            self.position = 'BB'
        else:
            self.position = 'SC'


def Assign_solv_param(atom, disulf_bond=False):
    """Assigns solvation parameter to an Atom object

    Args:
        atom (Atom): Edesolv.Atom object.
        disulf_bond (bool, optional): Disulfide bond flag. If true, the atom is a sulfide involved in a disulfie bond. Defaults to False.

    Returns:
        None.

    """
    
    arofac = 6.26
    alifac = 1.27
    polfac = 2.30


    if atom.position == "BB" or atom.position=="SC": #Check meaning and position of this line
        atom.solv = 0.0000

    if (((atom.name.startswith('CG')) or (atom.name.startswith('CD')) or (atom.name.startswith('CE')) or (atom.name.startswith('CH')) or (atom.name.startswith('CZ')))
        and (atom.resn=='PHE' or atom.resn=='TYR' or atom.resn=='HIS' or atom.resn=='TRP')):
        atom.solv = 0.0176 * arofac
    elif atom.name.startswith('C'):
        atom.solv = 0.0151 * alifac
    elif atom.name.startswith('NH') and atom.resn=='ARG':
        atom.solv = -0.0273 * polfac
    elif atom.name.startswith('NT') or (atom.name.startswith('NZ') and atom.resn=='LYS'):
        atom.solv = -0.0548 * polfac
    elif atom.name.startswith('N'):
        atom.solv = -0.0170 * polfac
    elif (atom.name.startswith('OD') and atom.resn=='ASP') or (atom.name.startswith('OE') and atom.resn=='GLU'):
        atom.solv = -0.0299 * polfac
    elif atom.name.startswith('OG') or atom.name=='OH':
        atom.solv = -0.0185 * polfac
    elif atom.name.startswith('O'):
        atom.solv = -0.0136 * polfac
    elif (atom.name.startswith('SD') and atom.resn=='MET') or (atom.name == 'SG' and disulf_bond == True): ## If S in CYS disulfide bond
        atom.solv = 0.0022 * polfac
    elif atom.name.startswith('S'):
        atom.solv = 0.0112 * polfac
    elif atom.name == 'SHA':
        atom.solv = 0.0000

    '''
    # Corarse-Grained model from HADDOCK code for eventual future implementation
    if atom.position == "BB" and atom.resn=="ALA":
        atom.solv = -0.0107
    elif atom.position == "BB" and atom.resn=="GLY":
        atom.solv = -0.0089
    elif atom.position == "BB" and atom.resn=="ILE":
        atom.solv = -0.0153
    elif atom.position == "BB" and atom.resn=="VAL":
        atom.solv = -0.0158
    elif atom.position == "BB" and atom.resn=="PRO":
        atom.solv = -0.0046
    elif atom.position == "BB" and atom.resn=="ASN":
        atom.solv = -0.0137
    elif atom.position == "BB" and atom.resn=="GLN":
        atom.solv = -0.0147
    elif atom.position == "BB" and atom.resn=="THR":
        atom.solv = -0.0165
    elif atom.position == "BB" and atom.resn=="SER":
        atom.solv = -0.0154
    elif atom.position == "BB" and atom.resn=="MET":
        atom.solv = -0.0130
    elif atom.position == "BB" and atom.resn=="CYS":
        atom.solv = -0.0167
    elif atom.position == "BB" and atom.resn=="PHE":
        atom.solv = -0.0126
    elif atom.position == "BB" and atom.resn=="TYR":
        atom.solv = -0.0134
    elif atom.position == "BB" and atom.resn=="TRP":
        atom.solv = -0.0134
    elif atom.position == "BB" and atom.resn=="ASP":
        atom.solv = -0.0169
    elif atom.position == "BB" and atom.resn=="GLU":
        atom.solv = -0.0150
    elif atom.position == "BB" and atom.resn=="HIS":
        atom.solv = -0.0155
    elif atom.position == "BB" and atom.resn=="LYS":
        atom.solv = -0.0163
    elif atom.position == "BB" and atom.resn=="ARG":
        atom.solv = -0.0162
    elif atom.position == "SC" and atom.resn=="ILE":
        atom.solv = 0.0255
    elif atom.position == "SC" and atom.resn=="VAL":
        atom.solv = 0.0222
    elif atom.position == "SC" and atom.resn=="PRO":
        atom.solv = 0.0230
    elif atom.position == "SC" and atom.resn=="ASN":
        atom.solv = -0.0192
    elif atom.position == "SC" and atom.resn=="GLN":
        atom.solv = -0.0135
    elif atom.position == "SC" and atom.resn=="THR":
        atom.solv = -0.0009
    elif atom.position == "SC" and atom.resn=="SER":
        atom.solv = -0.0056
    elif atom.position == "SC" and atom.resn=="MET":
        atom.solv = 0.0202
    elif atom.position == "SC" and atom.resn=="CYS":
        atom.solv = 0.0201
    elif atom.position == "SC" and atom.resn=="PHE":
        atom.solv = 0.1005
    elif atom.position == "SC" and atom.resn=="TYR":
        atom.solv = 0.0669
    elif atom.position == "SC" and atom.resn=="TRP":
        atom.solv = 0.0872
    elif atom.position == "SC" and atom.resn=="ASP":
        atom.solv = -0.0360
    elif atom.position == "SC" and atom.resn=="GLU":
        atom.solv = -0.0301
    elif atom.position == "SC" and atom.resn=="HIS":
        atom.solv = 0.0501
    elif atom.position == "SC" and atom.resn=="LYS":
        atom.solv = -0.0210
    elif atom.position == "SC" and atom.resn=="ARG":
        atom.solv = -0.0229
    elif atom.position == "SCD" and atom.resn=="ASN":
        atom.solv = 0.0
    elif atom.position == "SCD" and atom.resn=="GLN":
        atom.solv = 0.0
    elif atom.position == "SCD" and atom.resn=="SER":
        atom.solv = 0.0
    elif atom.position == "SCD" and atom.resn=="THR":
        atom.solv = 0.0
    elif atom.position == "SCD" and atom.resn=="ARG":
        atom.solv = 0.0
    elif atom.position == "SCD" and atom.resn=="LYS":
        atom.solv = 0.0
    elif atom.position == "SCD" and atom.resn=="GLU":
        atom.solv = 0.0
    elif atom.position == "SCD" and atom.resn=="ASP":
        atom.solv = 0.0
    '''
    
def get_disulfide_bonds(pdb_data):
    '''Gets all the cysteine pairs that have the SG around 1.80 and 2.05 Angstron and the CB-SG-SG-CB angle around 
       +- 90°

    Args:
        pdb_data (pdb2sqlcore): pdb2sqlcore object 

    Returns:
        disulfide_cys (list(tuple(str, int))): returns a list of chain IDs and resSeq numbers of cysteines bound in disulfide bonds

    '''
    cys_atoms = pdb_data.get('serial,chainID,resName,resSeq,name,x,y,z', resName='CYS')
    cys_dict = {}
    for atom in cys_atoms:
        try:
            cys_dict[(atom[1],atom[3])][atom[4]] = np.asarray(atom[5:8])
        except KeyError:
            cys_dict[(atom[1],atom[3])] = { atom[4] : np.asarray(atom[5:8]) }
            
    disulfide_cys = []
    for first_cys in cys_dict:
        for second_cys in cys_dict:
            if second_cys != first_cys:
                # calculate distance
                dist = np.linalg.norm(cys_dict[first_cys]['SG']-cys_dict[second_cys]['SG'])
                #if distance 1.80~2.05
                if dist >= 1.80 and dist <= 2.10:
                    #calc dihedral
                    selection = [cys_dict[first_cys]['CB'], cys_dict[first_cys]['SG'],
                                 cys_dict[second_cys]['CB'], cys_dict[second_cys]['SG']]
                    dihedral = get_dihedral(selection)
                    #if dihedral = ~+- 90
                    if (-91.0 <= dihedral <= -89.0) or (89.0 <= dihedral <= 91.0):
                        disulfide_cys.extend([first_cys, second_cys])
                    #append to list to return
    
    disulfide_cys = list(set(disulfide_cys))
    return disulfide_cys
        
def get_dihedral(p):
    """Get dihedral angle from four points (atoms) 3D coordinates.
    
    Args:
        p (list): list of four points 3D coordinates. Each element must be a list of floats.
        
    Returns:
        dihedral (float): Dihedral angle.
        """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    dihedral = np.degrees(np.arctan2(y, x))
    return dihedral
