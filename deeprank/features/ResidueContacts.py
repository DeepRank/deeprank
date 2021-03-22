import logging
import pdb2sql
import re
import os

import numpy

from deeprank.features.FeatureClass import FeatureClass
from deeprank.parse.param import ParamParser
from deeprank.parse.top import TopParser
from deeprank.parse.patch import PatchParser
from deeprank.models.patch import PatchActionType

_log = logging.getLogger(__name__)


class ResidueSynonymCriteria:
    def __init__(self, residue_name, atoms_present, atoms_absent):
        self.residue_name = residue_name
        self.atoms_present = atoms_present
        self.atoms_absent = atoms_absent

    def matches(self, residue_name, atom_names):
        if self.residue_name != 'all' and residue_name != self.residue_name:
            return False

        for atom_name in self.atoms_present:
            if atom_name not in atom_names:
                return False

        for atom_name in self.atoms_absent:
            if atom_name in atom_names:
                return False

        return True


def get_squared_distance(pos1, pos2):
    return numpy.sum([numpy.square(pos1[i] - pos2[i]) for i in range(3)])

def get_distance(pos1, pos2):
    return numpy.sqrt(get_squared_distance(pos1, pos2))

def wrap_values_in_lists(dict_):
    return {key: [value] for key,value in dict_.items()}

class _PhysicsStorage:
    ATOM_KEY = ["chainID", "resSeq", "resName", "name"]

    EPSILON0 = 1.0
    COULOMB_CONSTANT = 332.0636

    VANDERWAALS_DISTANCE_OFF = 8.5
    VANDERWAALS_DISTANCE_ON = 6.5

    SQUARED_VANDERWAALS_DISTANCE_OFF = numpy.square(VANDERWAALS_DISTANCE_OFF)
    SQUARED_VANDERWAALS_DISTANCE_ON = numpy.square(VANDERWAALS_DISTANCE_ON)

    @staticmethod
    def _sum_up(dict_, key, value_to_add):
        if key not in dict_:
            dict_[key] = 0.0

        dict_[key] += value_to_add

    @staticmethod
    def get_vanderwaals_energy(epsilon1, sigma1, epsilon2, sigma2, distance):
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
        return (charge1 * charge2 * _PhysicsStorage.COULOMB_CONSTANT /
                (_PhysicsStorage.EPSILON0 * distance) * pow(1 - pow(distance / max_distance, 2), 2))

    def __init__(self, sqldb, max_distance):

        self._vanderwaals_parameters = sqldb.get('eps,sig')
        self._charges = sqldb.get('CHARGE')
        self._atom_info = sqldb.get(",".join(_PhysicsStorage.ATOM_KEY))
        self._positions = sqldb.get('x,y,z')

        self._max_distance = max_distance

        self._vanderwaals_per_atom = {}
        self._vanderwaals_per_position = {}
        self._charge_per_atom = {}
        self._charge_per_position = {}
        self._coulomb_per_atom = {}
        self._coulomb_per_position = {}

    def include_pair(self, atom1, atom2):
        position1 = tuple(self._positions[atom1])
        position2 = tuple(self._positions[atom2])

        epsilon1, sigma1 = self._vanderwaals_parameters[atom1]
        epsilon2, sigma2 = self._vanderwaals_parameters[atom2]

        charge1 = self._charges[atom1]
        charge2 = self._charges[atom2]

        distance = get_distance(position1, position2)
        if distance == 0.0:
            distance = 3.0

        vanderwaals_energy = _PhysicsStorage.get_vanderwaals_energy(epsilon1, sigma1, epsilon2, sigma2, distance)
        coulomb_energy = _PhysicsStorage.get_coulomb_energy(charge1, charge2, distance, self._max_distance)

        atom1_key = tuple(self._atom_info[atom1])
        atom2_key = tuple(self._atom_info[atom2])

        self._charge_per_atom[atom1_key] = charge1
        self._charge_per_atom[atom2_key] = charge2

        _PhysicsStorage._sum_up(self._vanderwaals_per_atom, atom1_key, vanderwaals_energy)
        _PhysicsStorage._sum_up(self._vanderwaals_per_atom, atom2_key, vanderwaals_energy)

        _PhysicsStorage._sum_up(self._coulomb_per_atom, atom1_key, coulomb_energy)
        _PhysicsStorage._sum_up(self._coulomb_per_atom, atom2_key, coulomb_energy)

        _PhysicsStorage._sum_up(self._vanderwaals_per_position, position1, vanderwaals_energy)
        _PhysicsStorage._sum_up(self._vanderwaals_per_position, position2, vanderwaals_energy)

        _PhysicsStorage._sum_up(self._coulomb_per_position, position1, coulomb_energy)
        _PhysicsStorage._sum_up(self._coulomb_per_position, position2, coulomb_energy)

    def add_to_features(self, feature_data, feature_data_xyz):
        feature_data['vdwaals'] = wrap_values_in_lists(self._vanderwaals_per_atom)
        feature_data_xyz['vdwaals'] = wrap_values_in_lists(self._vanderwaals_per_position)
        feature_data['coulomb'] = wrap_values_in_lists(self._coulomb_per_atom)
        feature_data_xyz['coulomb'] = wrap_values_in_lists(self._coulomb_per_position)
        feature_data['charge'] = wrap_values_in_lists(self._charge_per_atom)
        feature_data_xyz['charge'] = wrap_values_in_lists(self._charge_per_position)


class ResidueContacts(FeatureClass):

    RESIDUE_SYNONYMS = {'PROP': ResidueSynonymCriteria('PRO', ['HT1', 'HT2'], []),
                        'NTER': ResidueSynonymCriteria('all', ['HT1', 'HT2', 'HT3'], []),
                        'CTER': ResidueSynonymCriteria('all', ['OXT'], []),
                        'CTN': ResidueSynonymCriteria('all', ['NT', 'HT1', 'HT2'], []),
                        'CYNH': ResidueSynonymCriteria('CYS', ['1SG'], ['2SG']),
                        'DISU': ResidueSynonymCriteria('CYS', ['1SG', '2SG'], []),
                        'HISE': ResidueSynonymCriteria('HIS', ['ND1', 'CE1', 'CD2', 'NE2', 'HE2'], ['HD1']),
                        'HISD': ResidueSynonymCriteria('HIS', ['ND1', 'CE1', 'CD2', 'NE2', 'HD1'], ['HE2'])}

    RESIDUE_KEY = ["chainID", "resSeq", "resName"]

    @staticmethod
    def get_alternative_residue_name(residue_name, atom_names):
        for name, crit in ResidueContacts.RESIDUE_SYNONYMS.items():
            if crit.matches(residue_name, atom_names):
                return name

        return None

    def __init__(self, pdb_path, chain_id, residue_number,
                 top_path, param_path, patch_path,
                 max_contact_distance=8.5):

        super().__init__("Atomic")

        self.pdb_path = pdb_path
        self.chain_id = chain_id
        self.residue_number = residue_number
        self.max_contact_distance = max_contact_distance

        self.top_path = top_path
        self.param_path = param_path
        self.patch_path = patch_path

    def __enter__(self):
        self.sqldb = pdb2sql.interface(self.pdb_path)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.sqldb._close()

    def _read_top(self):
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
        with open(self.param_path, 'rt') as f:
            self._vanderwaals_parameters = ParamParser.parse(f)

    def _read_patch(self):
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
        self._residue_atoms = set(self.sqldb.get("rowID", chainID=self.chain_id, resSeq=self.residue_number))

        atomic_postions = {r[0]: numpy.array([r[1], r[2], r[3]]) for r in self.sqldb.get('rowID,x,y,z')}

        squared_max_distance = numpy.square(self.max_contact_distance)

        self._contact_atoms = set()
        for atom, position in atomic_postions.items():
            if atom in self._residue_atoms:
                continue  # we don't pair a residue with itself

            for residue_atom in self._residue_atoms:
                residue_atom_position = atomic_postions[residue_atom]
                if get_squared_distance(position, residue_atom_position) < squared_max_distance:
                    self._contact_atoms.add(atom)
                    break

    def _extend_contact_to_residues(self):
        per_chain = {}
        for chain_id, residue_number in self.sqldb.get("chainID,resSeq", rowID=list(self._contact_atoms)):
            if chain_id not in per_chain:
                per_chain[chain_id] = set([])
            per_chain[chain_id].add(residue_number)

        # list the new contact atoms, per residue
        self._contact_atoms = set([])
        for chain_id, residue_numbers in per_chain.items():
            for atom in self.sqldb.get("rowID", chainID=chain_id, resSeq=list(residue_numbers)):
                self._contact_atoms.add(atom)

    def _get_atom_type(self, residue_name, alternative_residue_name, atom_name):
        if (alternative_residue_name, atom_name) in self._patch_type:
            return self._patch_type[(alternative_residue_name, atom_name)]

        elif (residue_name, atom_name) in self._residue_atom_types:
            return self._residue_atom_types[(residue_name, atom_name)]

        else:
            return None

    def _get_charge(self, residue_name, alternative_residue_name, atom_name):
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
        if residue_name not in self._valid_residue_names:
            return (0.0, 0.0)

        if atom_type in self._vanderwaals_parameters:
            o = self._vanderwaals_parameters[atom_type]
            return (o.epsilon, o.sigma)
        else:
            return (0.0, 0.0)

    def _assign_parameters(self):
        atomic_data = self.sqldb.get("rowID,name,chainID,resSeq,resName")
        count_atoms = len(atomic_data)

        atomic_charges = numpy.zeros(count_atoms)
        atomic_epsilon = numpy.zeros(count_atoms)
        atomic_sigma = numpy.zeros(count_atoms)

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

            alternative_residue_name = ResidueContacts.get_alternative_residue_name(residue_name, atoms_in_residue)
            atomic_alternative_residue_names[atom_nr] = alternative_residue_name

            atom_type = self._get_atom_type(residue_name, alternative_residue_name, atom_name)
            atomic_types[atom_nr] = atom_type

            atomic_charges[atom_nr] = self._get_charge(residue_name, alternative_residue_name, atom_name)

            epsilon, sigma = self._get_vanderwaals_parameters(residue_name, alternative_residue_name, atom_name, atom_type)
            atomic_epsilon[atom_nr] = epsilon
            atomic_sigma[atom_nr] = sigma

        # put in sql
        self.sqldb.add_column('CHARGE')
        self.sqldb.update_column('CHARGE', atomic_charges)

        self.sqldb.add_column('eps')
        self.sqldb.update_column('eps', atomic_epsilon)

        self.sqldb.add_column('sig')
        self.sqldb.update_column('sig', atomic_sigma)

        self.sqldb.add_column('type', 'TEXT')
        self.sqldb.update_column('type', atomic_types)

        self.sqldb.add_column('altRes', 'TEXT')
        self.sqldb.update_column('altRes', atomic_alternative_residue_names)

    def _evaluate_physics(self):
        physics_storage = _PhysicsStorage(self.sqldb, self.max_contact_distance)

        for contact_atom in self._contact_atoms:  # loop over atoms that contact the residue of interest

            for residue_atom in self._residue_atoms:  # loop over atoms in the residue of iterest

                physics_storage.include_pair(contact_atom, residue_atom)

        physics_storage.add_to_features(self.feature_data, self.feature_data_xyz)

    def evaluate(self):
        self._read_top()
        self._read_param()
        self._read_patch()
        self._assign_parameters()

        self._find_contact_atoms()
        self._extend_contact_to_residues()

        self._evaluate_physics()


def __compute_feature__(pdb_path, feature_group, raw_feature_group, chain_id, residue_number):

    forcefield_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'forcefield')
    top_path = os.path.join(forcefield_path, 'protein-allhdg5-4_new.top')
    param_path = os.path.join(forcefield_path, 'protein-allhdg5-4_new.param')
    patch_path = os.path.join(forcefield_path, 'patch.top')

    with ResidueContacts(pdb_path, chain_id, residue_number,
                         top_path, param_path, patch_path) as feature_object:

        feature_object.evaluate()

        # export in the hdf5 file
        feature_object.export_dataxyz_hdf5(feature_group)
        feature_object.export_data_hdf5(raw_feature_group)


