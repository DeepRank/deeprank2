import os


class Environment:
    "holds directory paths and device settings"

    def __init__(self, pdb_root=None, pssm_root=None, conservation_root=None, device="cpu"):
        self.pdb_root = pdb_root
        self.pssm_root = pssm_root
        self.conservation_root = conservation_root
        self.device = device

    def get_pdb_path(self, pdb_ac):
        """ Finds a pdb file

            Args:
                pdb_ac(str): the pdb accession code

            Returns(str): the path of an existing pdb file
            Throws(FileNotFoundError): if no such pdb file can be found
        """

        for path in [os.path.join(self.pdb_root, "{}.pdb".format(pdb_ac.lower())),
                     os.path.join(self.pdb_root, "{}.PDB".format(pdb_ac.upper())),
                     os.path.join(self.pdb_root, "{}/{}.pdb".format(pdb_ac.upper(), pdb_ac.upper())),
                     os.path.join(self.pdb_root, "pdb{}.ent".format(pdb_ac.lower()))]:

            if os.path.isfile(path):

                return path

        raise FileNotFoundError("No pdb file found for {} under {}".format(pdb_ac, self.pdb_root))

    def get_pssm_path(self, pdb_ac, chain_id):
        """ Finds a pssm file

            Args:
                pdb_ac(str): the pdb accession code
                chain_id(str): the pdb chain, that the pssm file should represent

            Returns(str): the path of an existing pssm file
            Throws(FileNotFoundError): if no such pssm file can be found
        """

        for path in [os.path.join(self.pssm_root, "{}/{}.{}.pdb.pssm".format(pdb_ac, pdb_ac, chain_id))]:

            if os.path.isfile(path):
                return path

        raise FileNotFoundError("No pssm file found for {} {} under {}".format(pdb_ac, chain_id, self.pdb_root))
