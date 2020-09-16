import freesasa
import pdb2sql
from Bio.PDB import PDBParser



class Edesolv(FeatureClass)

    def __init__(self,pdbfile):
        super().__init__('Atomic')
            self.pdb = pdb

        # the feature extractor
        def get_feature(self):

            # create a sql database
            db = pdb2sql.pdb2sql(self.pdb)
            i_db = pdb2sql.interface(db)

            # get the contact atoms
            contact_dict = i_db.get_contact_atoms()
            contact = []
            for chainID in contact_dict.keys():
                contact.extend(db.get('resName,resSeq,name',chainID=chainID,rowID=contact_dict[chainID]))

            # add edesolv params
            FS_structure = freesasa.Structure(self.pdb)
            SASA = freesasa.calc(FS_structure)

            arofac = 6.26
            alifac = 1.27
            polfac = 2.30


            for atomID in contact
                if ((atom.name().startswith('CG')) or (atom.name().startswith('CD')) or (atom.name().startswith('CE')) or (atom.name().startswith('CH')) or (atom.name().startswith('CZ'))) and (atom.resn()=='PHE' or atom.resn()=='TYR' or atom.resn()=='HIS' or atom.resn()=='TRP')
                    atom.solv() = 0.0176 * arofac
                elif atom.name().startswith('C'):
                    atom.solv() = 0.0151 * alifac
                elif atom.name().startswith('NH') and atom.resn()=='ARG':
                    atom.solv() = -0.0273 * polfac
                elif atom.name().startswith('NT') or (atom.name().startswith('NZ') and atom.resn()=='LYS'):
                    atom.solv() = -0.0548 * polfac
                elif atom.name().startswith('N'):
                    atom.solv() = -0.0170 * polfac
                elif (atom.name().startswith('OD') and atom.resn()=='ASP') or (atom.name().startswith('OE') and atom.resn()=='GLU'):
                    atom.solv() = -0.0299 * polfac
                elif atom.name().startswith('OG') or atom.name()=='OH'
                    atom.solv() = -0.0185 * polfac
                elif atom.name().startswith('O'):
                    atom.solv() = -0.0136 * polfac
                elif (atom.name().startswith('S') and atom.charge()== -0.3) or (atom.name().startswith('SD') and atom.resn()=='MET'):
                    atom.solv() = 0.0022 * polfac
                elif atom.name().startswith('S'):
                    atom.solv() = 0.0112 * polfac
                elif atom.name() == 'SHA':
                    atom.solv() = 0.0000
                                                                                                                                                           1,1           Top
                elif atom.name() == "BB" or atom.name().startswith("SC"):
                    atom.solv() = 0.0000)
                elif atom.name() == "BB" and atom.resn()=="ALA":
                    atom.solv() = -0.0107)
                elif atom.name() == "BB" and atom.resn()=="GLY":
                    atom.solv() = -0.0089)
                elif atom.name() == "BB" and atom.resn()=="ILE":
                    atom.solv() = -0.0153)
                elif atom.name() == "BB" and atom.resn()=="VAL":
                    atom.solv() = -0.0158)
                elif atom.name() == "BB" and atom.resn()=="PRO":
                    atom.solv() = -0.0046)
                elif atom.name() == "BB" and atom.resn()=="ASN":
                    atom.solv() = -0.0137)
                elif atom.name() == "BB" and atom.resn()=="GLN":
                    atom.solv() = -0.0147)
                elif atom.name() == "BB" and atom.resn()=="THR":
                    atom.solv() = -0.0165)
                elif atom.name() == "BB" and atom.resn()=="SER":
                    atom.solv() = -0.0154)
                elif atom.name() == "BB" and atom.resn()=="MET":
                    atom.solv() = -0.0130)
                elif atom.name() == "BB" and atom.resn()=="CYS":
                    atom.solv() = -0.0167)
                elif atom.name() == "BB" and atom.resn()=="PHE":
                    atom.solv() = -0.0126)
                elif atom.name() == "BB" and atom.resn()=="TYR":
                    atom.solv() = -0.0134)
                elif atom.name() == "BB" and atom.resn()=="TRP":
                    atom.solv() = -0.0134)
                elif atom.name() == "BB" and atom.resn()=="ASP":
                    atom.solv() = -0.0169)
                elif atom.name() == "BB" and atom.resn()=="GLU":
                    atom.solv() = -0.0150)
                elif atom.name() == "BB" and atom.resn()=="HIS":
                    atom.solv() = -0.0155)
                elif atom.name() == "BB" and atom.resn()=="LYS":
                    atom.solv() = -0.0163)
                elif atom.name() == "BB" and atom.resn()=="ARG":
                    atom.solv() = -0.0162)
                elif atom.name().startswith("SC*") and atom.resn()=="ILE":
                    atom.solv() = 0.0255)
                elif atom.name().startswith("SC*") and atom.resn()=="VAL":
                    atom.solv() = 0.0222)
                elif atom.name().startswith("SC*") and atom.resn()=="PRO":
                    atom.solv() = 0.0230)
                elif atom.name().startswith("SC*") and atom.resn()=="ASN":
                    atom.solv() = -0.0192)
                elif atom.name().startswith("SC*") and atom.resn()=="GLN":
                    atom.solv() = -0.0135)
                elif atom.name().startswith("SC*") and atom.resn()=="THR":
                    atom.solv() = -0.0009)
                elif atom.name().startswith("SC*") and atom.resn()=="SER":
                    atom.solv() = -0.0056)
                elif atom.name().startswith("SC*") and atom.resn()=="MET":
                    atom.solv() = 0.0202)
                elif atom.name().startswith("SC*") and atom.resn()=="CYS":
                    atom.solv() = 0.0201)
                elif atom.name().startswith("SC*") and atom.resn()=="PHE":
                    atom.solv() = 0.1005)
                elif atom.name().startswith("SC*") and atom.resn()=="TYR":
                    atom.solv() = 0.0669)
                elif atom.name().startswith("SC*") and atom.resn()=="TRP":
                    atom.solv() = 0.0872)
                elif atom.name().startswith("SC*") and atom.resn()=="ASP":
                    atom.solv() = -0.0360)
                elif atom.name().startswith("SC*") and atom.resn()=="GLU":
                    atom.solv() = -0.0301)
                elif atom.name().startswith("SC*") and atom.resn()=="HIS":
                    atom.solv() = 0.0501)
                elif atom.name().startswith("SC*") and atom.resn()=="LYS":
                    atom.solv() = -0.0210)
                elif atom.name().startswith("SC*") and atom.resn()=="ARG":
                    atom.solv() = -0.0229)
                elif atom.name().startswith("SCD*") and atom.resn()=="ASN":
                    atom.solv() = 0.0)
                elif atom.name().startswith("SCD*") and atom.resn()=="GLN":
                    atom.solv() = 0.0)
                elif atom.name().startswith("SCD*") and atom.resn()=="SER":
                    atom.solv() = 0.0)
                elif atom.name().startswith("SCD*") and atom.resn()=="THR":
                    atom.solv() = 0.0)
                elif atom.name().startswith("SCD*") and atom.resn()=="ARG":
                    atom.solv() = 0.0)
                elif atom.name().startswith("SCD*") and atom.resn()=="LYS":
                    atom.solv() = 0.0)
                elif atom.name().startswith("SCD*") and atom.resn()=="GLU":
                    atom.solv() = 0.0)
                elif atom.name().startswith("SCD*") and atom.resn()=="ASP":
                    atom.solv() = 0.0)


            self.feature_data['Edesolv']
