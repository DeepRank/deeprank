import numpy as np
import pdb2sql


class SASA(object):

    def __init__(self, pdbfile):
        """Simple class that computes Surface Accessible Solvent Area.

        The method follows some of the approaches presented in:

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

    def get_center(self, chain1='A', chain2='B', center='cb'):
        """Get the center of the resiudes.

        Args:
            chain1 (str, optional): Name of the first chain
            chain2 (str, optional): Name of the second chain
            center (str, optional): Specify the center.
                'cb': the center locates on carbon beta of each residue
                'center': average position of all atoms of the residue
        Raises:
            ValueError: If the center is not recpgnized
        """

        if center == 'center':
            self.get_residue_center(chain1=chain1, chain2=chain2)
        elif center == 'cb':
            self.get_residue_carbon_beta(chain1=chain1, chain2=chain2)
        else:
            raise ValueError(
                'Options %s not recognized in SASA.get_center' %
                center)

    def get_residue_center(self, chain1='A', chain2='B'):
        """Compute the average position of all the residues.

        Args:
            chain1 (str, optional): Name of the first chain
            chain2 (str, optional): Name of the second chain
        """

        sql = pdb2sql.pdb2sql(self.pdbfile)
        resA = np.array(sql.get('resSeq,resName', chainID=chain1))
        resB = np.array(sql.get('resSeq,resName', chainID=chain2))

        resSeqA = np.unique(resA[:, 0].astype(np.int))
        resSeqB = np.unique(resB[:, 0].astype(np.int))

        self.xyz = {}

        self.xyz[chain1] = []
        for r in resSeqA:
            xyz = sql.get('x,y,z', chainID=chain1, resSeq=str(r))
            self.xyz[chain1].append(np.mean(xyz))

        self.xyz[chain2] = []
        for r in resSeqB:
            xyz = sql.get('x,y,z', chainID=chain2, resSeq=str(r))
            self.xyz[chain1].append(np.mean(xyz))

        self.resinfo = {}
        self.resinfo[chain1] = []
        for r in resA[:, :2]:
            if tuple(r) not in self.resinfo[chain1]:
                self.resinfo[chain1].append(tuple(r))

        self.resinfo[chain2] = []
        for r in resB[:, :2]:
            if tuple(r) not in self.resinfo[chain2]:
                self.resinfo[chain2].append(tuple(r))
        sql._close()

    def get_residue_carbon_beta(self, chain1='A', chain2='B'):
        """Extract the position of the carbon beta of each residue.

        Args:
            chain1 (str, optional): Name of the first chain
            chain2 (str, optional): Name of the second chain
        """

        sql = pdb2sql.pdb2sql(self.pdbfile)
        resA = np.array(
            sql.get(
                'resSeq,resName,x,y,z',
                name='CB',
                chainID=chain1))
        resB = np.array(
            sql.get(
                'resSeq,resName,x,y,z',
                name='CB',
                chainID=chain2))
        sql._close()

        assert len(resA[:, 0].astype(np.int).tolist()) == len(
            np.unique(resA[:, 0].astype(np.int)).tolist())
        assert len(resB[:, 0].astype(np.int).tolist()) == len(
            np.unique(resB[:, 0].astype(np.int)).tolist())

        self.xyz = {}
        self.xyz[chain1] = resA[:, 2:].astype(np.float)
        self.xyz[chain2] = resB[:, 2:].astype(np.float)

        self.resinfo = {}
        self.resinfo[chain1] = resA[:, :2]
        self.resinfo[chain2] = resB[:, :2]

    def neighbor_vector(
            self,
            lbound=3.3,
            ubound=11.1,
            chain1='A',
            chain2='B',
            center='cb'):
        """Compute teh SASA folowing the neighbour vector approach.

        The method is based on Eq on page 1097 of
        https://link.springer.com/article/10.1007%2Fs00894-009-0454-9

        Args:
            lbound (float, optional): lower boubd
            ubound (float, optional): upper bound
            chain1 (str, optional): name of the first chain
            chain2 (str, optional): name of the second chain
            center (str, optional): specify the center
                (see get_residue_center)

        Returns:
            dict: neighbouring vectors
        """

        # get the center
        self.get_center(chain1=chain1, chain2=chain2, center=center)

        NV = {}

        for chain in [chain1, chain2]:

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
            chain1='A',
            chain2='B',
            center='cb'):
        """Compute the neighbourhood count of each residue.

        The method is based on Eq on page 1097 of
        https://link.springer.com/article/10.1007%2Fs00894-009-0454-9

        Args:
            lbound (float, optional): lower boubd
            ubound (float, optional): upper bound
            chain1 (str, optional): name of the first chain
            chain2 (str, optional): name of the second chain
            center (str, optional): specify the center
            (see get_residue_center)

        Returns:
            dict: Neighborhood count
        """

        # get the center
        self.get_center(chain1=chain1, chain2=chain2, center=center)

        # dict of NC
        NC = {}

        for chain in [chain1, chain2]:

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
