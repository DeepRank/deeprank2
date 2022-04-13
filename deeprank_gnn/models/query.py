import logging
import os
from typing import Dict, List, Iterator, Optional
import freesasa
import numpy
import pdb2sql
from deeprank_gnn.domain.amino_acid import (
    alanine,
    arginine,
    asparagine,
    aspartate,
    cysteine,
    glutamate,
    glutamine,
    glycine,
    histidine,
    isoleucine,
    leucine,
    lysine,
    methionine,
    phenylalanine,
    proline,
    serine,
    threonine,
    tryptophan,
    tyrosine,
    valine
    )
from deeprank_gnn.domain.feature import (
    FEATURENAME_POSITION,
    FEATURENAME_AMINOACID,
    FEATURENAME_VARIANTAMINOACID,
    FEATURENAME_CHAIN,
    FEATURENAME_CHARGE,
    FEATURENAME_POLARITY,
    FEATURENAME_SIZE,
    FEATURENAME_BURIEDSURFACEAREA,
    FEATURENAME_HALFSPHEREEXPOSURE,
    FEATURENAME_PSSM,
    FEATURENAME_CONSERVATION,
    FEATURENAME_INFORMATIONCONTENT,
    FEATURENAME_RESIDUEDEPTH,
    FEATURENAME_PSSMDIFFERENCE,
    FEATURENAME_PSSMWILDTYPE,
    FEATURENAME_PSSMVARIANT,
    FEATURENAME_SASA,
    FEATURENAME_SIZEDIFFERENCE,
    FEATURENAME_POLARITYDIFFERENCE,
    FEATURENAME_EDGECOULOMB,
    FEATURENAME_EDGEVANDERWAALS,
    FEATURENAME_EDGEDISTANCE,
    FEATURENAME_EDGETYPE,
    EDGETYPE_INTERNAL,
    EDGETYPE_INTERFACE
    )
from deeprank_gnn.domain.forcefield import atomic_forcefield
from deeprank_gnn.domain.graph import EDGETYPE_INTERNAL, EDGETYPE_INTERFACE
from deeprank_gnn.models.graph import Graph, Edge, Node
from deeprank_gnn.models.amino_acid import AminoAcid
from deeprank_gnn.models.contact import Contact
from deeprank_gnn.models.structure import Residue, Atom
from deeprank_gnn.tools import BioWrappers, BSA
from deeprank_gnn.tools.pdb import (
    get_residue_contact_pairs,
    get_surrounding_residues,
    get_structure,
    get_atomic_contacts,
    get_residue_contacts
)
from deeprank_gnn.tools.pssm import parse_pssm

_log = logging.getLogger(__name__)


class Query:
    """Represents one entity of interest, like a single residue variant or a protein-protein interface.

    Query objects are used to generate graphs from structures.
    objects of this class should be created before any model is loaded
    """

    def __init__(self, model_id: str, targets: Dict[str, float] = None):
        """
        Args:
            model_id(str): the id of the model to load, usually a pdb accession code
            targets(dict, optional): target values associated with this query
        """

        self._model_id = model_id

        if targets is None:
            self._targets = {}
        else:
            self._targets = targets

    @staticmethod
    def _build_graph_from_contacts(graph_id: str, contacts: List[Contact], distance_cutoff: float) -> Graph:
        """ The most basic method to build a graph.
            It simply connects all contacts as edges.
            The contacted atoms/residues become nodes.
        """

        graph = Graph(graph_id)
        for contact in contacts:
            if contact.distance < distance_cutoff:

                graph.add_edge(Edge(contact))

                # add the nodes, as the edge doesn't add them automatically
                graph.add_node(Node(contact.item1))
                graph.add_node(Node(contact.item2))

        return graph

    def _set_graph_targets(self, graph: Graph):
        "simply copies target data from query to graph"

        for target_name, target_data in self._targets.items():
            graph.targets[target_name] = target_data

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def targets(self) -> Dict[str, float]:
        return self._targets

    def __repr__(self):
        return f"{type(self)}({self.get_query_id()})"


class SingleResidueVariantResidueQuery(Query):
    "creates a residue graph from a single residue variant in a pdb file"
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        pdb_path: str,
        chain_id: str,
        residue_number: int,
        insertion_code: str,
        wildtype_amino_acid: AminoAcid,
        variant_amino_acid: AminoAcid,
        pssm_paths: Optional[Dict[str, str]] = None,
        wildtype_conservation: Optional[float] = None,
        variant_conservation: Optional[float] = None,
        radius: Optional[float] = 10.0,
        external_distance_cutoff: Optional[float] = 4.5,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file

            chain_id(str): the pdb chain identifier of the variant residue

            residue_number(int): the number of the variant residue

            insertion_code(str): the insertion code of the variant residue, set to None
            if not applicable

            wildtype_amino_acid(deeprank amino acid object): the wildtype amino acid

            variant_amino_acid(deeprank amino acid object): the variant amino acid

            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier

            wildtype_conservation(float): conservation value for the wildtype

            variant_conservation(float): conservation value for the variant

            radius(float): in Ångström, determines how many residues will be included in the graph

            external_distance_cutoff(float): max distance in Ångström between a pair of atoms
            to consider them as an external edge in the graph

            targets(dict(str,float)): named target values associated with this query
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths
        self._wildtype_conservation = wildtype_conservation
        self._variant_conservation = variant_conservation

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

        self._radius = radius
        self._external_distance_cutoff = external_distance_cutoff

    @property
    def residue_id(self) -> str:
        "residue identifier within chain"

        if self._insertion_code is not None:

            return f"{self._residue_number}{self._insertion_code}"
        
        return str(self._residue_number)

    def get_query_id(self) -> str:
        return f"residue-graph-{self.model_id}:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}"

    @staticmethod
    def _get_residue_node_key(residue: Residue) -> str:
        "produce an unique node key, given the residue"

        return str(residue)

    @staticmethod
    def _is_next_residue_number(residue1: Residue, residue2: Residue) -> bool:
        if residue1.number == residue2.number:
            if (
                residue1.insertion_code is not None
                and residue2.insertion_code is not None
            ):
                return (ord(residue1.insertion_code) + 1) == ord(
                    residue2.insertion_code
                )

        elif (residue1.number + 1) == residue2.number:
            return True

        return False

    @staticmethod
    def _is_covalent_bond(atom1: Atom, atom2: Atom, distance: float) -> bool:
        if distance < 2.3:

            # peptide bonds
            if (
                atom1.name == "C"
                and atom2.name == "N"
                and SingleResidueVariantResidueQuery._is_next_residue_number(
                    atom1.residue, atom2.residue
                )
            ):
                return True

            if (
                atom2.name == "C"
                and atom1.name == "N"
                and SingleResidueVariantResidueQuery._is_next_residue_number(
                    atom2.residue, atom1.residue
                )
            ):
                return True

            # disulfid bonds
            if atom1.name == "SG" and atom2.name == "SG":
                return True

        return False

    @staticmethod
    def _set_sasa(graph: Graph, pdb_path: str):

        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)

        for node in graph.nodes:
            residue = node.id

            select_str = (
                f"residue, (resi {residue.number_string}) and (chain {residue.chain.id})"
            )

            area = freesasa.selectArea(select_str, structure, result)["residue"]

            if numpy.isnan(area):
                raise ValueError(f"freesasa returned {area} for {pdb_path}:{residue}")

            node.features[FEATURENAME_SASA] = area

    @staticmethod
    def _set_amino_acid_properties(graph: Graph, variant_residue: Residue,
                                   wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid):

        for node in graph.nodes:
            residue = node.id

            node.features[FEATURENAME_POSITION] = numpy.mean([atom.position for atom in residue.atoms], axis=0)
            node.features[FEATURENAME_POLARITY] = residue.amino_acid.polarity.onehot
            node.features[FEATURENAME_SIZE] = residue.amino_acid.size

            if residue == variant_residue:

                node.features[FEATURENAME_AMINOACID] = wildtype_amino_acid.onehot
                node.features[FEATURENAME_VARIANTAMINOACID] = variant_amino_acid.onehot
                node.features[FEATURENAME_SIZEDIFFERENCE] = variant_amino_acid.size - wildtype_amino_acid.size
                node.features[FEATURENAME_POLARITYDIFFERENCE] = variant_amino_acid.polarity.onehot - wildtype_amino_acid.polarity.onehot
            else:
                node.features[FEATURENAME_AMINOACID] = residue.amino_acid.onehot
                node.features[FEATURENAME_VARIANTAMINOACID] = numpy.zeros(len(residue.amino_acid.onehot))
                node.features[FEATURENAME_SIZEDIFFERENCE] = 0
                node.features[FEATURENAME_POLARITYDIFFERENCE] = numpy.zeros(len(residue.amino_acid.polarity.onehot))

    amino_acid_order = [alanine, arginine, asparagine, aspartate, cysteine, glutamine, glutamate, glycine, histidine, isoleucine,
                        leucine, lysine, methionine, phenylalanine, proline, serine, threonine, tryptophan, tyrosine, valine]

    @staticmethod
    def _set_pssm(graph: Graph, variant_residue: Residue,
                  wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid):

        for node in graph.nodes:
            residue = node.id

            pssm_row = residue.get_pssm()

            pssm_value = [
                pssm_row.conservations[amino_acid]
                for amino_acid in SingleResidueVariantResidueQuery.amino_acid_order
            ]

            if residue == variant_residue:

                node.features[FEATURENAME_PSSMDIFFERENCE] = pssm_row.get_conservation(variant_amino_acid) - \
                                                                     pssm_row.get_conservation(wildtype_amino_acid)

                node.features[FEATURENAME_PSSMWILDTYPE] = pssm_row.get_conservation(wildtype_amino_acid)
                node.features[FEATURENAME_PSSMVARIANT] = pssm_row.get_conservation(variant_amino_acid)
            else:
                node.features[FEATURENAME_PSSMDIFFERENCE] = 0.0
                node.features[FEATURENAME_PSSMWILDTYPE] = pssm_row.get_conservation(residue.amino_acid)
                node.features[FEATURENAME_PSSMVARIANT] = pssm_row.get_conservation(residue.amino_acid)

            node.features[FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content
            node.features[FEATURENAME_PSSM] = pssm_value

    def build_graph(self) -> Graph:
        # pylint: disable=too-many-locals
        # pylint: disable=protected-access
        # load pdb strucure
        pdb = pdb2sql.pdb2sql(self._pdb_path)

        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close()

        # read the pssm
        if self._pssm_paths is not None:
            for chain in structure.chains:
                if chain.id in self._pssm_paths:
                    pssm_path = self._pssm_paths[chain.id]

                    with open(pssm_path, "rt", encoding = "utf-8") as f:
                        chain.pssm = parse_pssm(f, chain)

        # find the variant residue
        variant_residues = [
            r
            for r in structure.get_chain(self._chain_id).residues
            if r.number == self._residue_number
            and r.insertion_code == self._insertion_code
        ]
        if len(variant_residues) == 0:
            raise ValueError(
                "Residue {self._chain_id}:{self.residue_id} not found in {self._pdb_path}"
            )
        variant_residue = variant_residues[0]

        # get the residues and atoms involved
        residues = get_surrounding_residues(structure, variant_residue, self._radius)
        residues.add(variant_residue)
        atoms = []
        for residue in residues:
            if residue.amino_acid is not None:
                atoms.extend(residue.atoms)

        # get the contacts
        contacts = get_residue_contacts(residues)

        # build the graph
        graph = self._build_graph_from_contacts(self.get_query_id(), contacts, self._external_distance_cutoff)
        self._set_graph_targets(graph)

        # add edge features
        for edge in graph.edges:

            contact = edge.id

            edge.features[FEATURENAME_EDGEDISTANCE] = contact.distance
            edge.features[FEATURENAME_EDGEVANDERWAALS] = contact.vanderwaals_potential
            edge.features[FEATURENAME_EDGECOULOMB] = contact.electrostatic_potential

        # set the node features
        self._set_amino_acid_properties(graph, variant_residue, self._wildtype_amino_acid, self._variant_amino_acid)
        self._set_sasa(graph, self._pdb_path)
        self._set_pssm(graph, variant_residue, self._wildtype_amino_acid, self._variant_amino_acid)

        return graph


class SingleResidueVariantAtomicQuery(Query):
    "creates an atomic graph for a single residue variant in a pdb file"

    def __init__(self, pdb_path: str, chain_id: str, residue_number: int, insertion_code: str,
                 wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid,
                 pssm_paths: Optional[Dict[str, str]] = None,
                 radius: Optional[float] = 10.0,
                 external_distance_cutoff: Optional[float] = 4.5,
                 internal_distance_cutoff: Optional[float] = 3.0, targets: Optional[Dict[str, float]] = None):
        """
            Args:
                pdb_path(str): the path to the pdb file

                chain_id(str): the pdb chain identifier of the variant residue

                residue_number(int): the number of the variant residue

                insertion_code(str): the insertion code of the variant residue, set to None if not applicable

                wildtype_amino_acid(deeprank amino acid object): the wildtype amino acid

                variant_amino_acid(deeprank amino acid object): the variant amino acid

                pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier

                radius(float): in Ångström, determines how many residues will be included in the graph

                external_distance_cutoff(float): max distance in Ångström between a pair of atoms to
                consider them as an external edge in the graph

                internal_distance_cutoff(float): max distance in Ångström between a pair of atoms to consider
                them as an internal edge in the graph (must be shorter than external)

                targets(dict(str,float)): named target values associated with this query
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

        self._radius = radius

        if external_distance_cutoff < internal_distance_cutoff:
            raise ValueError(
                "this query is not supported with internal distance cutoff shorter than external distance cutoff" # noqa: pycodestyle
            )

        self._external_distance_cutoff = external_distance_cutoff
        self._internal_distance_cutoff = internal_distance_cutoff

    @property
    def residue_id(self) -> str:
        "string representation of the residue number and insertion code"

        if self._insertion_code is not None:
            return f"{self._residue_number}{self._insertion_code}"
        
        return str(self._residue_number)

    def get_query_id(self):
        return f"{self.model_id}:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}"

    def __eq__(self, other):
        return (
            isinstance(self, type(other))
            and self.model_id == other.model_id
            and self._chain_id == other._chain_id
            and self.residue_id == other.residue_id
            and self._wildtype_amino_acid == other._wildtype_amino_acid
            and self._variant_amino_acid == other._variant_amino_acid
        )

    def __hash__(self):
        return hash(
            (
                self.model_id,
                self._chain_id,
                self.residue_id,
                self._wildtype_amino_acid,
                self._variant_amino_acid,
            )
        )

    @staticmethod
    def _get_atom_node_key(atom):
        """Pickle has problems serializing the graph when the nodes are atoms,
        so use this function to generate an unique key for the atom"""

        # This should include the model, chain, residue and atom
        return str(atom)

    def build_graph(self) -> Graph:
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        # pylint: disable=protected-access

        # load pdb strucure
        pdb = pdb2sql.pdb2sql(self._pdb_path)

        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close()

        # read the pssm
        if self._pssm_paths is not None:
            for chain in structure.chains:
                if chain.id in self._pssm_paths:
                    pssm_path = self._pssm_paths[chain.id]

                    with open(pssm_path, "rt", encoding = "utf-8") as f:
                        chain.pssm = parse_pssm(f, chain)

        # find the variant residue
        variant_residues = [
            r
            for r in structure.get_chain(self._chain_id).residues
            if r.number == self._residue_number
            and r.insertion_code == self._insertion_code
        ]
        if len(variant_residues) == 0:
            raise ValueError(
                "Residue {self._chain_id}:{self.residue_id} not found in {self._pdb_path}"
            )
        variant_residue = variant_residues[0]

        # get the residues and atoms involved
        residues = get_surrounding_residues(structure, variant_residue, self._radius)
        residues.add(variant_residue)
        atoms = []
        for residue in residues:
            if residue.amino_acid is not None:
                atoms.extend(residue.atoms)

        # get the contacts
        contacts = get_atomic_contacts(atoms)

        # build the graph
        graph = self._build_graph_from_contacts(self.get_query_id(), contacts, self._external_distance_cutoff)
        self._set_graph_targets(graph)

        # set edge features
        for edge in graph.edges:
            contact = edge.id

            edge.features[FEATURENAME_EDGEDISTANCE] = contact.distance
            edge.features[FEATURENAME_EDGEVANDERWAALS] = contact.vanderwaals_potential
            edge.features[FEATURENAME_EDGECOULOMB] = contact.electrostatic_potential

        # set node features
        SingleResidueVariantAtomicQuery._set_pssm(graph, variant_residue,
                                                  self._wildtype_amino_acid, self._variant_amino_acid)

        SingleResidueVariantAtomicQuery._set_sasa(graph, self._pdb_path)

        SingleResidueVariantAtomicQuery._set_amino_acid(
            graph, variant_residue, self._wildtype_amino_acid, self._variant_amino_acid)

        return graph

    amino_acid_order = [
        alanine,
        arginine,
        asparagine,
        aspartate,
        cysteine,
        glutamine,
        glutamate,
        glycine,
        histidine,
        isoleucine,
        leucine,
        lysine,
        methionine,
        phenylalanine,
        proline,
        serine,
        threonine,
        tryptophan,
        tyrosine,
        valine,
    ]

    @staticmethod
    def _set_amino_acid(graph: Graph, variant_residue: Residue,
                        wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid):

        for node in graph.nodes:
            atom = node.id

            node.features[FEATURENAME_POSITION] = atom.position

            if atom.residue == variant_residue:

                node.features[FEATURENAME_AMINOACID] = wildtype_amino_acid.onehot
                node.features[FEATURENAME_VARIANTAMINOACID] = variant_amino_acid.onehot
                node.features[FEATURENAME_SIZEDIFFERENCE] = variant_amino_acid.size - wildtype_amino_acid.size
                node.features[FEATURENAME_POLARITYDIFFERENCE] = variant_amino_acid.polarity.onehot - wildtype_amino_acid.polarity.onehot
            else:
                node.features[FEATURENAME_AMINOACID] = atom.residue.amino_acid.onehot
                node.features[FEATURENAME_VARIANTAMINOACID] = numpy.zeros(len(atom.residue.amino_acid.onehot))
                node.features[FEATURENAME_SIZEDIFFERENCE] = 0
                node.features[FEATURENAME_POLARITYDIFFERENCE] = numpy.zeros(len(atom.residue.amino_acid.polarity.onehot))

    @staticmethod
    def _set_pssm(graph: Graph, variant_residue: Residue,
                  wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid):

        for node in graph.nodes:
            atom = node.id

            pssm_row = atom.residue.get_pssm()
            pssm_value = [pssm_row.conservations[amino_acid]
                          for amino_acid in SingleResidueVariantAtomicQuery.amino_acid_order]

            if atom.residue == variant_residue:

                node.features[FEATURENAME_PSSMDIFFERENCE] = pssm_row.get_conservation(variant_amino_acid) - \
                                                            pssm_row.get_conservation(wildtype_amino_acid)

                node.features[FEATURENAME_PSSMWILDTYPE] = pssm_row.get_conservation(wildtype_amino_acid)
                node.features[FEATURENAME_PSSMVARIANT] = pssm_row.get_conservation(variant_amino_acid)
            else:
                node.features[FEATURENAME_PSSMDIFFERENCE] = 0.0
                node.features[FEATURENAME_PSSMWILDTYPE] = pssm_row.get_conservation(atom.residue.amino_acid)
                node.features[FEATURENAME_PSSMVARIANT] = pssm_row.get_conservation(atom.residue.amino_acid)

            node.features[FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content
            node.features[FEATURENAME_PSSM] = pssm_value

    @staticmethod
    def _set_sasa(graph: Graph, pdb_path: str):

        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)

        for node in graph.nodes:
            atom = node.id

            if atom.element == "H":  # freeSASA doesn't have these
                area = 0.0
            else:
                select_str = (
                    f"atom, (name {atom.name}) and (resi {atom.residue.number_string}) and (chain {atom.residue.chain.id})"
                )
                area = freesasa.selectArea(select_str, structure, result)["atom"]

            if numpy.isnan(area):
                raise ValueError(f"freesasa returned {area} for {pdb_path}:{atom}")

            node.features[FEATURENAME_SASA] = area


class ProteinProteinInterfaceAtomicQuery(Query):
    "a query that builds atom-based graphs, using the residues at a protein-protein interface"
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        pdb_path: str,
        chain_id1: str,
        chain_id2: str,
        pssm_paths: Optional[Dict[str, str]] = None,
        interface_distance_cutoff: Optional[float] = 8.5,
        internal_distance_cutoff: Optional[float] = 3.0,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file

            chain_id1(str): the pdb chain identifier of the first protein of interest

            chain_id2(str): the pdb chain identifier of the second protein of interest

            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
            
            interface_distance_cutoff(float): max distance in Ångström between two interacting
            residues of the two proteins

            internal_distance_cutoff(float): max distance in Ångström between two interacting
            residues within the same protein

            targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff
        self._internal_distance_cutoff = internal_distance_cutoff

    def get_query_id(self) -> str:
        return f"atom-ppi-{self.model_id}:{self._chain_id1}-{self._chain_id2}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self.model_id == other.model_id
            and {self._chain_id1, self._chain_id2}
            == {other._chain_id1, other._chain_id2}
        )

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    @staticmethod
    def _residue_is_valid(residue: Residue) -> bool:
        if residue.amino_acid is None:
            return False

        if residue not in residue.chain.pssm:
            _log.debug("%s not in pssm", residue)
            return False

        return True

    def build_graph(self) -> Graph:
        """Builds the residue graph.

        Returns(deeprank graph object): the resulting graph
        """

        # get residues from the pdb
        interface_pairs = get_residue_contact_pairs(
            self._pdb_path,
            self.model_id,
            self._chain_id1,
            self._chain_id2,
            self._interface_distance_cutoff,
        )
        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        # get all atoms in the selection
        atoms_selected = set([])
        for residue1, residue2 in interface_pairs:
            atoms_selected |= set(residue1.atoms)
            atoms_selected |= set(residue2.atoms)
        atoms_selected = list(atoms_selected)

        # get the contacts between the atoms
        contacts = get_atomic_contacts(atoms_selected)

        # build the graph
        graph = self._build_graph_from_contacts(self.get_query_id(), contacts, self._interface_distance_cutoff)
        self._set_graph_targets(graph)

        model = contacts[0].atom1.residue.chain.model
        chain1 = model.get_chain(self._chain_id1)
        chain2 = model.get_chain(self._chain_id2)

        # read the pssm
        if self._pssm_paths is not None:
            for chain in (chain1, chain2):
                pssm_path = self._pssm_paths[chain.id]

                with open(pssm_path, "rt") as f:
                    chain.pssm = parse_pssm(f, chain)

        # build sasa structures
        sasa_structures = {chain1: freesasa.Structure(), chain2: freesasa.Structure()}
        sasa_structure_both = freesasa.Structure()
        for residue in chain1.residues:
            for atom in residue.atoms:
                sasa_structures[chain1].addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                                atom.residue.number, atom.residue.chain.id,
                                                atom.position[0], atom.position[1], atom.position[2])
                sasa_structure_both.addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                            atom.residue.number, atom.residue.chain.id,
                                            atom.position[0], atom.position[1], atom.position[2])

        for residue in chain2.residues:
            for atom in residue.atoms:
                sasa_structures[chain2].addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                                atom.residue.number, atom.residue.chain.id,
                                                atom.position[0], atom.position[1], atom.position[2])
                sasa_structure_both.addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                            atom.residue.number, atom.residue.chain.id,
                                            atom.position[0], atom.position[1], atom.position[2])

        sasa_results = {chain1: freesasa.calc(sasa_structures[chain1]),
                        chain2: freesasa.calc(sasa_structures[chain2])}
        sasa_result_both = freesasa.calc(sasa_structure_both)

        # give each chain a numerical value
        chain_codes = {chain1: 0.0, chain2: 1.0}

        # add edge features
        for edge in graph.edges:
            contact = edge.id
            if contact.atom1.residue.chain == contact.atom2.residue.chain:
                edge.features[FEATURENAME_EDGETYPE] = EDGETYPE_INTERNAL
            else:
                edge.features[FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE

            edge.features[FEATURENAME_EDGEDISTANCE] = contact.distance

        # add the node features
        for node in graph.nodes:
            atom = node.id
            residue = atom.residue

            pssm_row = residue.get_pssm()
            pssm_value = [
                pssm_row.conservations[amino_acid]
                for amino_acid in self.amino_acid_order
            ]

            node.features[FEATURENAME_CHAIN] = chain_codes[residue.chain]
            node.features[FEATURENAME_POSITION] = atom.position
            node.features[FEATURENAME_AMINOACID] = residue.amino_acid.onehot
            node.features[FEATURENAME_CHARGE] = atomic_forcefield.get_charge(atom)
            node.features[FEATURENAME_POLARITY] = residue.amino_acid.polarity.onehot

            select_str = ('atom, (name %s) and (resi %s) and (chain %s)' % (atom.name, residue.number_string, residue.chain.id),)
            area_unbound = freesasa.selectArea(select_str, sasa_structures[residue.chain], sasa_results[residue.chain])['atom']
            area_bound = freesasa.selectArea(select_str, sasa_structure_both, sasa_result_both)['atom']
            node.features[FEATURENAME_BURIEDSURFACEAREA] = area_unbound - area_bound

            if self._pssm_paths is not None:
                node.features[FEATURENAME_PSSM] = pssm_value
                node.features[FEATURENAME_CONSERVATION] = pssm_row.conservations[residue.amino_acid]
                node.features[FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content

        return graph

    amino_acid_order = [
        alanine,
        arginine,
        asparagine,
        aspartate,
        cysteine,
        glutamine,
        glutamate,
        glycine,
        histidine,
        isoleucine,
        leucine,
        lysine,
        methionine,
        phenylalanine,
        proline,
        serine,
        threonine,
        tryptophan,
        tyrosine,
        valine,
    ]


class ProteinProteinInterfaceResidueQuery(Query):
    "a query that builds residue-based graphs, using the residues at a protein-protein interface" # noqa: pycodestyle
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        pdb_path: str,
        chain_id1: str,
        chain_id2: str,
        pssm_paths: Optional[Dict[str, str]] = None,
        interface_distance_cutoff: float = 8.5,
        internal_distance_cutoff: float = 3.0,
        use_biopython: bool = False,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file

            chain_id1(str): the pdb chain identifier of the first protein of interest

            chain_id2(str): the pdb chain identifier of the second protein of interest

            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier

            interface_distance_cutoff(float): max distance in Ångström between two interacting
            residues of the two proteins

            internal_distance_cutoff(float): max distance in Ångström between two interacting
            residues within the same protein

            use_biopython(bool): whether or not to use biopython tools

            targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff
        self._internal_distance_cutoff = internal_distance_cutoff

        self._use_biopython = use_biopython

    def get_query_id(self) -> str:
        return f"residue-ppi-{self.model_id}:{self._chain_id1}-{self._chain_id2}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self.model_id == other.model_id
            and {self._chain_id1, self._chain_id2}
            == {other._chain_id1, other._chain_id2}
        )

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    @staticmethod
    def _residue_is_valid(residue: Residue) -> bool:
        if residue.amino_acid is None:
            return False

        if residue not in residue.chain.pssm:
            _log.debug("%s not in pssm", residue)
            return False

        return True

    def build_graph(self) -> Graph:
        """Builds the residue graph.

        Returns(deeprank graph object): the resulting graph
        """
        # pylint: disable=too-many-locals
        # mccabe: disable=MC0001
        # get residues from the pdb

        interface_pairs = get_residue_contact_pairs(
            self._pdb_path,
            self.model_id,
            self._chain_id1,
            self._chain_id2,
            self._interface_distance_cutoff,
        )
        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        residues_selected = set([])
        for residue1, residue2 in interface_pairs:
            residues_selected.add(residue1)
            residues_selected.add(residue2)
        residues_selected = list(residues_selected)

        # build graph from residues
        contacts = get_residue_contacts(residues_selected)
        graph = self._build_graph_from_contacts(self.get_query_id(), contacts, self._interface_distance_cutoff)
        self._set_graph_targets(graph)

        model = contacts[0].residue1.chain.model
        chain1 = model.get_chain(self._chain_id1)
        chain2 = model.get_chain(self._chain_id2)

        # read the pssm
        if self._pssm_paths is not None:
            for chain in (chain1, chain2):
                pssm_path = self._pssm_paths[chain.id]

                with open(pssm_path, "rt", encoding = "utf-8") as f:
                    chain.pssm = parse_pssm(f, chain)

        # get bsa
        pdb = pdb2sql.interface(self._pdb_path)
        try:
            bsa_calc = BSA.BSA(self._pdb_path, pdb)
            bsa_calc.get_structure()
            bsa_calc.get_contact_residue_sasa(cutoff=self._interface_distance_cutoff)
            bsa_data = bsa_calc.bsa_data
        finally:
            pdb._close()

        # get biopython features
        if self._use_biopython:
            bio_model = BioWrappers.get_bio_model(self._pdb_path)
            residue_depths = BioWrappers.get_depth_contact_res(
                bio_model,
                [
                    (
                        residue.chain.id,
                        residue.number,
                        residue.amino_acid.three_letter_code,
                    )
                    for residue in residues_by_node.values()
                ],
            )
            hse = BioWrappers.get_hse(bio_model)

        # add edge features
        for edge in graph.edges:
            contact = edge.id
            if contact.residue1.chain == contact.residue2.chain:
                edge.features[FEATURENAME_EDGETYPE] = EDGETYPE_INTERNAL
            else:
                edge.features[FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE

            edge.features[FEATURENAME_EDGEDISTANCE] = contact.distance

        # define a numerical value for each chain
        chain_codes = {chain1: 0.0, chain2: 1.0}

        # add node features
        for node in graph.nodes:
            residue = node.id

            bsa_key = (
                residue.chain.id,
                residue.number,
                residue.amino_acid.three_letter_code,
            )
            bio_key = (residue.chain.id, residue.number)

            pssm_row = residue.get_pssm()
            pssm_value = [
                pssm_row.conservations[amino_acid]
                for amino_acid in self.amino_acid_order
            ]

            node.features[FEATURENAME_CHAIN] = chain_codes[residue.chain]
            node.features[FEATURENAME_POSITION] = numpy.mean([atom.position for atom in residue.atoms], axis=0)
            node.features[FEATURENAME_AMINOACID] = residue.amino_acid.onehot
            node.features[FEATURENAME_CHARGE] = residue.amino_acid.charge
            node.features[FEATURENAME_POLARITY] = residue.amino_acid.polarity.onehot
            node.features[FEATURENAME_BURIEDSURFACEAREA] = bsa_data[bsa_key]

            if self._pssm_paths is not None:
                node.features[FEATURENAME_PSSM] = pssm_value
                node.features[FEATURENAME_CONSERVATION] = pssm_row.conservations[residue.amino_acid]
                node.features[FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content

            if self._use_biopython:
                node.features[FEATURENAME_RESIDUEDEPTH] = residue_depths[residue] if residue in residue_depths else 0.0
                node.features[FEATURENAME_HALFSPHEREEXPOSURE] = hse[bio_key] if bio_key in hse else (0.0, 0.0, 0.0)

        return graph

    amino_acid_order = [
        alanine,
        arginine,
        asparagine,
        aspartate,
        cysteine,
        glutamine,
        glutamate,
        glycine,
        histidine,
        isoleucine,
        leucine,
        lysine,
        methionine,
        phenylalanine,
        proline,
        serine,
        threonine,
        tryptophan,
        tyrosine,
        valine,
    ]


class QueryDataset:
    "represents a collection of data queries"

    def __init__(self):
        self._queries = []

    def add(self, query: Query):
        self._queries.append(query)

    @property
    def queries(self) -> List[Query]:
        return self._queries

    def __contains__(self, query: Query) -> bool:
        return query in self._queries

    def __iter__(self) -> Iterator[Query]:
        return iter(self._queries)