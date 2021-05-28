import logging
import pdb2sql
import re
import os

import numpy

from deeprank.models.pair import Pair
from deeprank.operate.pdb import get_residue_contact_atom_pairs, get_distance
from deeprank.features.FeatureClass import FeatureClass
from deeprank.parse.param import ParamParser
from deeprank.parse.top import TopParser
from deeprank.parse.patch import PatchParser
from deeprank.models.patch import PatchActionType

_log = logging.getLogger(__name__)


class ResidueSynonymCriteria:
    """The ResidueSynonymCriteria is an object that holds the criteria
       for a residue to have a certain synonym. It does not hold the synonym string itself however.
    """

    def __init__(self, residue_name, atoms_present, atoms_absent):
        """Build new criteria

        Args:
            residue_name (string): the name of the residue
            atoms_present (list of strings): the names of the atoms that should be present in the residue
            atoms_absent (list of strings) the names of the atoms that should be absent in the residue
        """

        self.residue_name = residue_name
        self.atoms_present = atoms_present
        self.atoms_absent = atoms_absent

    def matches(self, residue_name, atom_names):
        """Check whether the given residue matches this set of criteria

        Args:
            residue_name (string): the name of the residue to match
            atom_names (list of strings): the names of the atoms in the residue
        """

        if self.residue_name != 'all' and residue_name != self.residue_name:
            return False

        for atom_name in self.atoms_present:
            if atom_name not in atom_names:
                return False

        for atom_name in self.atoms_absent:
            if atom_name in atom_names:
                return False

        return True


def wrap_values_in_lists(dict_):
    """Wrap the dictionary's values in lists. This
       appears to be necessary for the exported features to work.

    Args:
        dict_(dictionary): the dictionary that should be converted
    """

    return {key: [value] for key,value in dict_.items()}

class _PhysicsStorage:
    "A helper object that holds the physics values while summing them"

    ATOM_KEY = ["chainID", "resSeq", "resName", "name"]

    EPSILON0 = 1.0
    COULOMB_CONSTANT = 332.0636

    VANDERWAALS_DISTANCE_OFF = 8.5
    VANDERWAALS_DISTANCE_ON = 6.5

    SQUARED_VANDERWAALS_DISTANCE_OFF = numpy.square(VANDERWAALS_DISTANCE_OFF)
    SQUARED_VANDERWAALS_DISTANCE_ON = numpy.square(VANDERWAALS_DISTANCE_ON)

    @staticmethod
    def _sum_up(dict_, key, value_to_add):
        """A helper function to sum the values of a dictionary.

        Args:
            dict_(dictionary): the dictionary to store the value in
            key(hashable object): the key under which the value should be stored in dict_
            value_to_add: the value to add onto the dictionary value
        """

        if key not in dict_:
            dict_[key] = 0.0

        dict_[key] += value_to_add

    @staticmethod
    def get_vanderwaals_energy(epsilon1, sigma1, epsilon2, sigma2, distance):
        """The formula to calculate the vanderwaals energy for two atoms (atom 1 and atom 2)

        Args:
            epsilon1 (float): the vanderwaals epsilon parameter of atom 1
            sigma1 (float): the vanderwaals sigma parameter of atom 1
            epsilon2 (float): the vanderwaals epsilon parameter of atom 2
            sigma2 (float): the vanderwaals sigma parameter of atom 2
            distance (float): the vanderwaals distance between atom 1 and atom 2
        """

        average_epsilon = numpy.sqrt(epsilon1 * epsilon2)
        average_sigma = 0.5 * (sigma1 + sigma2)

        squared_distance = numpy.square(distance)
        prefactor = (pow(_PhysicsStorage.SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distance, 2) *
                     (_PhysicsStorage.SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distance - 3 * (_PhysicsStorage.SQUARED_VANDERWAALS_DISTANCE_ON - squared_distance)) /
                     pow(_PhysicsStorage.SQUARED_VANDERWAALS_DISTANCE_OFF - _PhysicsStorage.SQUARED_VANDERWAALS_DISTANCE_ON, 3))

        if distance > _PhysicsStorage.VANDERWAALS_DISTANCE_OFF:
            prefactor = 0.0

        elif distance < _PhysicsStorage.VANDERWAALS_DISTANCE_ON:
            prefactor = 1.0

        return 4.0 * average_epsilon * (pow(average_sigma / distance, 12) - pow(average_sigma / distance, 6)) * prefactor

    @staticmethod
    def get_coulomb_energy(charge1, charge2, distance, max_distance):
        """The formula to calculate the coulomb energy for two atoms (atom 1 and atom 2)

        Args:
            charge1 (float): the charge of atom 1
            charge2 (float): the charge of atom 2
            distance (float): the vanderwaals distance between atom 1 and atom 2
            max_distance (float): the max distance that was used to find atoms 1 and 2
        """

        return (charge1 * charge2 * _PhysicsStorage.COULOMB_CONSTANT /
                (_PhysicsStorage.EPSILON0 * distance) * pow(1 - pow(distance / max_distance, 2), 2))

    def __init__(self, sqldb):
        """Build a new set of physical parameters

        Args:
            sqldb (pdb2sql): interface to the contents of a PDB file, with charges and vanderwaals parameters included.

        Raises:
            RuntimeError: if features are missing from sqldb
        """

        self._vanderwaals_parameters = sqldb.get('inter_epsilon,inter_sigma,intra_epsilon,intra_sigma')
        self._charges = sqldb.get('CHARGE')
        self._atom_info = sqldb.get(",".join(_PhysicsStorage.ATOM_KEY))

        if len(self._vanderwaals_parameters) == 0:
            raise RuntimeError("vanderwaals parameters are empty, please run '_assign_parameters' first")

        if len(self._charges) == 0:
            raise RuntimeError("vanderwaals parameters are empty, please run '_assign_parameters' first")

        if len(self._atom_info) == 0:
            raise RuntimeError("atom info is empty, please create a AtomicContacts object first")

        self._vanderwaals_per_atom = {}
        self._vanderwaals_per_position = {}
        self._charge_per_atom = {}
        self._charge_per_position = {}
        self._coulomb_per_atom = {}
        self._coulomb_per_position = {}

    def include_pair(self, atom1, atom2, max_distance):
        """Add a pair of atoms to the sum

        Args:
            atom1 (int): number of atom 1
            atom2 (int): number of atom 2
            max_distance (float): the max distance that was used to find the atoms

        Raises:
            ValueError: if atom1 and atom2 are at the same position
        """

        position1 = atom1.position
        position2 = atom2.position

        # Which epsilon and sigma we take from the atoms, depends on whether the contact
        # is inter- or intra-chain.
        if atom1.chain_id == atom2.chain_id:
            epsilon1, sigma1 = self._vanderwaals_parameters[atom1.id][2:]
            epsilon2, sigma2 = self._vanderwaals_parameters[atom2.id][2:]
        else:
            epsilon1, sigma1 = self._vanderwaals_parameters[atom1.id][:2]
            epsilon2, sigma2 = self._vanderwaals_parameters[atom2.id][:2]

        charge1 = self._charges[atom1.id]
        charge2 = self._charges[atom2.id]

        distance = get_distance(position1, position2)
        if distance == 0.0:
            raise ValueError("encountered two atoms {} and {} with distance zero".format(atom1, atom2))

        vanderwaals_energy = _PhysicsStorage.get_vanderwaals_energy(epsilon1, sigma1, epsilon2, sigma2, distance)
        coulomb_energy = _PhysicsStorage.get_coulomb_energy(charge1, charge2, distance, max_distance)

        atom1_key = tuple(self._atom_info[atom1.id])
        atom2_key = tuple(self._atom_info[atom2.id])

        position1 = tuple(position1)
        position2 = tuple(position2)

        self._charge_per_atom[atom1_key] = charge1
        self._charge_per_atom[atom2_key] = charge2

        self._charge_per_position[position1] = charge1
        self._charge_per_position[position2] = charge2

        _PhysicsStorage._sum_up(self._vanderwaals_per_atom, atom1_key, vanderwaals_energy)
        _PhysicsStorage._sum_up(self._vanderwaals_per_atom, atom2_key, vanderwaals_energy)

        _PhysicsStorage._sum_up(self._coulomb_per_atom, atom1_key, coulomb_energy)
        _PhysicsStorage._sum_up(self._coulomb_per_atom, atom2_key, coulomb_energy)

        _PhysicsStorage._sum_up(self._vanderwaals_per_position, position1, vanderwaals_energy)
        _PhysicsStorage._sum_up(self._vanderwaals_per_position, position2, vanderwaals_energy)

        _PhysicsStorage._sum_up(self._coulomb_per_position, position1, coulomb_energy)
        _PhysicsStorage._sum_up(self._coulomb_per_position, position2, coulomb_energy)

    def add_to_features(self, feature_data, feature_data_xyz):
        """Convert the summed interactions to deeprank features and store them in the corresponding dictionaries

        Args:
            feature_data (dictionary): where the per atom features should be stored
            feature_data_xyz (dictionary): where the per position features should be stored
        """

        feature_data['vdwaals'] = wrap_values_in_lists(self._vanderwaals_per_atom)
        feature_data_xyz['vdwaals'] = wrap_values_in_lists(self._vanderwaals_per_position)

        feature_data['coulomb'] = wrap_values_in_lists(self._coulomb_per_atom)
        feature_data_xyz['coulomb'] = wrap_values_in_lists(self._coulomb_per_position)

        feature_data['charge'] = wrap_values_in_lists(self._charge_per_atom)
        feature_data_xyz['charge'] = wrap_values_in_lists(self._charge_per_position)


class AtomicContacts(FeatureClass):
    "A class that collects features that involve contacts between a residue and its surrounding atoms"

    # This dictionary holds the data used to find residue alternative names:
    RESIDUE_SYNONYMS = {'PROP': ResidueSynonymCriteria('PRO', ['HT1', 'HT2'], []),
                        'NTER': ResidueSynonymCriteria('all', ['HT1', 'HT2', 'HT3'], []),
                        'CTER': ResidueSynonymCriteria('all', ['OXT'], []),
                        'CTN': ResidueSynonymCriteria('all', ['NT', 'HT1', 'HT2'], []),
                        'CYNH': ResidueSynonymCriteria('CYS', ['1SG'], ['2SG']),
                        'DISU': ResidueSynonymCriteria('CYS', ['1SG', '2SG'], []),
                        'HISE': ResidueSynonymCriteria('HIS', ['ND1', 'CE1', 'CD2', 'NE2', 'HE2'], ['HD1']),
                        'HISD': ResidueSynonymCriteria('HIS', ['ND1', 'CE1', 'CD2', 'NE2', 'HD1'], ['HE2'])}

    @staticmethod
    def get_alternative_residue_name(residue_name, atom_names):
        """Get the alternative residue name, according to the static dictionary in this class

        Args:
            residue_name (string): the name of the residue
            atom_names (list of strings): the names of the atoms in the residue
        """

        for name, crit in AtomicContacts.RESIDUE_SYNONYMS.items():
            if crit.matches(residue_name, atom_names):
                return name

        return None

    def __init__(self, mutant,
                 top_path, param_path, patch_path,
                 max_contact_distance=8.5):
        """Build a new residue contacts feature object

        Args:
            pdb_path (string): where the pdb file is located on disk
            chain_id (string): identifier of the residue's protein chain within the pdb file
            residue_number (int): identifier of the residue within the protein chain
            top_path (string): location of the top file on disk
            param_path (string): location of the param file on disk
            patch_path (string): location of the patch file on disk
            max_contact_distance (float): the maximum distance allowed for two atoms to be considered a contact pair
        """

        super().__init__("Atomic")

        self.mutant = mutant
        self.max_contact_distance = max_contact_distance

        self.top_path = top_path
        self.param_path = param_path
        self.patch_path = patch_path

    def __enter__(self):
        "open the with-clause"

        self.sqldb = pdb2sql.interface(self.mutant.pdb_path)
        return self

    def __exit__(self, exc_type, exc, tb):
        "close the with-clause"

        self.sqldb._close()

    def _read_top(self):
        "read the top file and store its data in memory"

        self._residue_charges = {}
        self._residue_atom_types = {}
        self._valid_residue_names = set([])

        with open(self.top_path, 'rt') as f:
            for obj in TopParser.parse(f):

                # store the charge
                self._residue_charges[(obj.residue_name, obj.atom_name)] = obj.kwargs['charge']

                # put the resname in a list so far
                self._valid_residue_names.add(obj.residue_name)

                # dictionary for conversion name/type
                self._residue_atom_types[(obj.residue_name, obj.atom_name)] = obj.kwargs['type']

    def _read_param(self):
        "read the param file and store its data in memory"

        with open(self.param_path, 'rt') as f:
            self._vanderwaals_parameters = ParamParser.parse(f)

    def _read_patch(self):
        "read the patch file and store its data in memory"

        self._patch_charge = {}
        self._patch_type = {}

        with open(self.patch_path, 'rt') as f:
            for action in PatchParser.parse(f):

                # get the new charge
                self._patch_charge[(action.selection.residue_type, action.selection.atom_name)] = action.kwargs['CHARGE']

                # get the new type if any
                if 'TYPE' in action.kwargs:
                    self._patch_type[(action.selection.residue_type, action.selection.atom_name)] = action.kwargs['TYPE']

    def _find_contact_atoms(self):
        "find out which atoms of the pdb file lie within the max distance of the residue"

        self._contact_atom_pairs = get_residue_contact_atom_pairs(self.sqldb,
                                                                  self.mutant.chain_id,
                                                                  self.mutant.residue_number,
                                                                  self.max_contact_distance)

    def _extend_contact_to_residues(self):
        "find out of which residues the contact atoms are a part, then include their atoms"

        for atom1, atom2 in set(self._contact_atom_pairs):
            for atom in atom1.residue.atoms:
                self._contact_atom_pairs.add(Pair(atom, atom2))

            for atom in atom2.residue.atoms:
                self._contact_atom_pairs.add(Pair(atom1, atom))

    def _get_atom_type(self, residue_name, alternative_residue_name, atom_name):
        """Find the type name of the given atom, according to top and patch data

        Args:
            residue_name (string): the name of the residue that the atom is in
            alternative_residue_name (string): the name of the residue, outputted from 'get_alternative_residue_name'
            atom_name (string): the name of the atom itself
        """

        if (alternative_residue_name, atom_name) in self._patch_type:
            return self._patch_type[(alternative_residue_name, atom_name)]

        elif (residue_name, atom_name) in self._residue_atom_types:
            return self._residue_atom_types[(residue_name, atom_name)]

        else:
            return None

    def _get_charge(self, residue_name, alternative_residue_name, atom_name):
        """Find the charge of the atom, according to top and patch data

        Args:
            residue_name (string): the name of the residue that the atom is in
            alternative_residue_name (string): the name of the residue, outputted from 'get_alternative_residue_name'
            atom_name (string): the name of the atom itself
        """

        if residue_name not in self._valid_residue_names:
            return 0.0

        if (alternative_residue_name, atom_name) in self._patch_charge:
            return self._patch_charge[(alternative_residue_name, atom_name)]

        elif (residue_name, atom_name) in self._residue_charges:
            return self._residue_charges[(residue_name, atom_name)]

        else:
            _log.warn("Atom type {} not found for {}/{}, set charge to 0.0"
                      .format(atom_name, residue_name, alternative_residue_name))

            return 0.0

    def _get_vanderwaals_parameters(self, residue_name, alternative_residue_name, atom_name, atom_type):
        """Find the vanderwaals parameters of the atom, according to param data

        Args:
            residue_name (string): the name of the residue that the atom is in
            alternative_residue_name (string): the name of the residue, outputted from 'get_alternative_residue_name'
            atom_name (string): the name of the atom itself
            atom_type (string): output from '_get_atom_type'
        """

        if residue_name not in self._valid_residue_names:
            return VanderwaalsParam(0.0, 0.0, 0.0, 0.0)

        if atom_type in self._vanderwaals_parameters:
            o = self._vanderwaals_parameters[atom_type]
            return o
        else:
            return VanderwaalsParam(0.0, 0.0, 0.0, 0.0)

    def _assign_parameters(self):
        "Get parameters from top, param and patch data and put them in the pdb2sql database"

        atomic_data = self.sqldb.get("rowID,name,chainID,resSeq,resName")
        count_atoms = len(atomic_data)

        atomic_charges = numpy.zeros(count_atoms)
        atomic_inter_epsilon = numpy.zeros(count_atoms)
        atomic_inter_sigma = numpy.zeros(count_atoms)
        atomic_intra_epsilon = numpy.zeros(count_atoms)
        atomic_intra_sigma = numpy.zeros(count_atoms)

        atomic_types = numpy.zeros(count_atoms, dtype='<U5')
        atomic_alternative_residue_names = numpy.zeros(count_atoms, dtype='<U5')

        # here, we map the atom names per residue
        residue_atom_names = {}
        for atom_nr, atom_name, chain_id, residue_number, residue_name in atomic_data:
            key = (chain_id, residue_number)
            if key not in residue_atom_names:
                residue_atom_names[key] = set([])
            residue_atom_names[key].add(atom_name)

        # loop over all atoms
        for atom_nr, atom_name, chain_id, residue_number, residue_name in atomic_data:
            atoms_in_residue = residue_atom_names[(chain_id, residue_number)]

            alternative_residue_name = AtomicContacts.get_alternative_residue_name(residue_name, atoms_in_residue)
            atomic_alternative_residue_names[atom_nr] = alternative_residue_name

            atom_type = self._get_atom_type(residue_name, alternative_residue_name, atom_name)
            atomic_types[atom_nr] = atom_type

            atomic_charges[atom_nr] = self._get_charge(residue_name, alternative_residue_name, atom_name)

            params = self._get_vanderwaals_parameters(residue_name, alternative_residue_name, atom_name, atom_type)
            atomic_inter_epsilon[atom_nr] = params.inter_epsilon
            atomic_inter_sigma[atom_nr] = params.inter_sigma
            atomic_intra_epsilon[atom_nr] = params.intra_epsilon
            atomic_intra_sigma[atom_nr] = params.intra_sigma

        # put in sql
        self.sqldb.add_column('CHARGE')
        self.sqldb.update_column('CHARGE', atomic_charges)

        self.sqldb.add_column('inter_epsilon')
        self.sqldb.update_column('inter_epsilon', atomic_inter_epsilon)

        self.sqldb.add_column('inter_sigma')
        self.sqldb.update_column('inter_sigma', atomic_inter_sigma)

        self.sqldb.add_column('intra_epsilon')
        self.sqldb.update_column('intra_epsilon', atomic_intra_epsilon)

        self.sqldb.add_column('intra_sigma')
        self.sqldb.update_column('intra_sigma', atomic_intra_sigma)

        self.sqldb.add_column('type', 'TEXT')
        self.sqldb.update_column('type', atomic_types)

        self.sqldb.add_column('altRes', 'TEXT')
        self.sqldb.update_column('altRes', atomic_alternative_residue_names)

    def _evaluate_physics(self):
        """From the top, param and patch data,
           calculate energies and charges per residue atom and surrounding atoms
           and add them to the feature dictionaries"""

        physics_storage = _PhysicsStorage(self.sqldb)

        for atom1, atom2 in self._contact_atom_pairs:  # loop over atoms pairs that involve the residue of interest

            physics_storage.include_pair(atom1, atom2, self.max_contact_distance)

        physics_storage.add_to_features(self.feature_data, self.feature_data_xyz)

    def evaluate(self):
        "collect the features before calling 'export_dataxyz_hdf5' and 'export_data_hdf5' on this object"

        self._read_top()
        self._read_param()
        self._read_patch()
        self._assign_parameters()

        self._find_contact_atoms()
        self._extend_contact_to_residues()

        self._evaluate_physics()


def __compute_feature__(pdb_path, feature_group, raw_feature_group, mutant):

    forcefield_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'forcefield')
    top_path = os.path.join(forcefield_path, 'protein-allhdg5-4_new.top')
    param_path = os.path.join(forcefield_path, 'protein-allhdg5-4_new.param')
    patch_path = os.path.join(forcefield_path, 'patch.top')

    with AtomicContacts(mutant, top_path, param_path, patch_path) as feature_object:

        feature_object.evaluate()

        # export in the hdf5 file
        feature_object.export_dataxyz_hdf5(feature_group)
        feature_object.export_data_hdf5(raw_feature_group)
