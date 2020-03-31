import numpy as np
import pdb2sql


class SASA(object):

    def __init__(self, pdbfile):
        """Simple class that computes Surface Accessible Solvent Area.

        The method follows some of the approaches presented in :

        Solvent accessible surface area approximations for rapid and
        accurate protein structure prediction
        https://link.springer.com/article/10.1007%2Fs00894-009-0454-9

        Example:
            >>> sasa = SASA('1AK4_1w.pdb')
            >>> NV = sasa.neighbor_vector()
            >>> print(NV)

        Args:
            pdbfile (str): PDB file of the conformation
        """

        self.pdbfile = pdbfile

    def get_center(self, chainA='A', chainB='B', center='cb'):
        """Get the center of the resiudes.

        Args:
            chainA (str, optional): Name of the first chain
            chainB (str, optional): Name of the second chain
            center (str, optional): Specify the center.
                'cb': the center locates on carbon beta of each residue
                'center': average position of all atoms of the residue
        Raises:
            ValueError: If the center is not recpgnized
        """

        if center == 'center':
            self.get_residue_center(chainA=chainA, chainB=chainB)
        elif center == 'cb':
            self.get_residue_carbon_beta(chainA=chainA, chainB=chainB)
        else:
            raise ValueError(
                'Options %s not recognized in SASA.get_center' %
                center)

    def get_residue_center(self, chainA='A', chainB='B'):
        """Compute the average position of all the residues.

        Args:
            chainA (str, optional): Name of the first chain
            chainB (str, optional): Name of the second chain
        """

        sql = pdb2sql.pdb2sql(self.pdbfile)
        resA = np.array(sql.get('resSeq,resName', chainID=chainA))
        resB = np.array(sql.get('resSeq,resName', chainID=chainB))

        resSeqA = np.unique(resA[:, 0].astype(np.int))
        resSeqB = np.unique(resB[:, 0].astype(np.int))

        self.xyz = {}

        self.xyz[chainA] = []
        for r in resSeqA:
            xyz = sql.get('x,y,z', chainID=chainA, resSeq=str(r))
            self.xyz[chainA].append(np.mean(xyz))

        self.xyz[chainB] = []
        for r in resSeqB:
            xyz = sql.get('x,y,z', chainID=chainB, resSeq=str(r))
            self.xyz[chainA].append(np.mean(xyz))

        self.resinfo = {}
        self.resinfo[chainA] = []
        for r in resA[:, :2]:
            if tuple(r) not in self.resinfo[chainA]:
                self.resinfo[chainA].append(tuple(r))

        self.resinfo[chainB] = []
        for r in resB[:, :2]:
            if tuple(r) not in self.resinfo[chainB]:
                self.resinfo[chainB].append(tuple(r))
        sql._close()

    def get_residue_carbon_beta(self, chainA='A', chainB='B'):
        """Extract the position of the carbon beta of each residue.

        Args:
            chainA (str, optional): Name of the first chain
            chainB (str, optional): Name of the second chain
        """

        sql = pdb2sql.pdb2sql(self.pdbfile)
        resA = np.array(
            sql.get(
                'resSeq,resName,x,y,z',
                name='CB',
                chainID=chainA))
        resB = np.array(
            sql.get(
                'resSeq,resName,x,y,z',
                name='CB',
                chainID=chainB))
        sql._close()

        assert len(resA[:, 0].astype(np.int).tolist()) == len(
            np.unique(resA[:, 0].astype(np.int)).tolist())
        assert len(resB[:, 0].astype(np.int).tolist()) == len(
            np.unique(resB[:, 0].astype(np.int)).tolist())

        self.xyz = {}
        self.xyz[chainA] = resA[:, 2:].astype(np.float)
        self.xyz[chainB] = resB[:, 2:].astype(np.float)

        self.resinfo = {}
        self.resinfo[chainA] = resA[:, :2]
        self.resinfo[chainB] = resB[:, :2]

    def neighbor_vector(
            self,
            lbound=3.3,
            ubound=11.1,
            chainA='A',
            chainB='B',
            center='cb'):
        """Compute teh SASA folowing the neighbour vector approach.

        The method is based on Eq on page 1097 of
        https://link.springer.com/article/10.1007%2Fs00894-009-0454-9

        Args:
            lbound (float, optional): lower boubd
            ubound (float, optional): upper bound
            chainA (str, optional): name of the first chain
            chainB (str, optional): name of the second chain
            center (str, optional): specify the center
                (see get_residue_center)

        Returns:
            dict: neighbouring vectors
        """

        # get the center
        self.get_center(chainA=chainA, chainB=chainB, center=center)

        NV = {}

        for chain in [chainA, chainB]:

            for i, xyz in enumerate(self.xyz[chain]):

                vect = self.xyz[chain] - xyz
                dist = np.sqrt(np.sum((self.xyz[chain] - xyz)**2, 1))

                dist = np.delete(dist, i, 0)
                vect = np.delete(vect, i, 0)

                vect /= np.linalg.norm(vect, axis=1).reshape(-1, 1)

                weight = self.neighbor_weight(
                    dist, lbound=lbound, ubound=ubound).reshape(-1, 1)
                vect *= weight

                vect = np.sum(vect, 0)
                vect /= np.sum(weight)

                resSeq, resName = self.resinfo[chain][i].tolist()
                key = tuple([chain, int(resSeq), resName])
                value = np.linalg.norm(vect)
                NV[key] = value

        return NV

    def neighbor_count(
            self,
            lbound=4.0,
            ubound=11.4,
            chainA='A',
            chainB='B',
            center='cb'):
        """Compute the neighbourhood count of each residue.

        The method is based on Eq on page 1097 of
        https://link.springer.com/article/10.1007%2Fs00894-009-0454-9

        Args:
            lbound (float, optional): lower boubd
            ubound (float, optional): upper bound
            chainA (str, optional): name of the first chain
            chainB (str, optional): name of the second chain
            center (str, optional): specify the center
            (see get_residue_center)

        Returns:
            dict: Neighborhood count
        """

        # get the center
        self.get_center(chainA=chainA, chainB=chainB, center=center)

        # dict of NC
        NC = {}

        for chain in [chainA, chainB]:

            for i, xyz in enumerate(self.xyz[chain]):
                dist = np.sqrt(np.sum((self.xyz[chain] - xyz)**2, 1))
                resSeq, resName = self.resinfo[chain][i].tolist()
                key = tuple([chain, int(resSeq), resName])
                value = np.sum(self.neighbor_weight(dist, lbound, ubound))
                NC[key] = value

        return NC

    @staticmethod
    def neighbor_weight(dist, lbound, ubound):
        """Neighboor weight.

        Args:
            dist (np.array): distance from neighboors
            lbound (float): lower bound
            ubound (float): upper bound

        Returns:
            float: distance
        """
        ind = np.argwhere((dist > lbound) & (dist < ubound))
        dist[ind] = 0.5 * \
            (np.cos(np.pi * (dist[ind] - lbound) / (ubound - lbound)) + 1)
        dist[dist <= lbound] = 1
        dist[dist >= ubound] = 0
        return dist


if __name__ == '__main__':

    sasa = SASA('1AK4_1w.pdb')
    NV = sasa.neighbor_vector()
    print(NV)
