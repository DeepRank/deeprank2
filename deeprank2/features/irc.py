import logging
from itertools import combinations_with_replacement as combinations

import pdb2sql

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain.aminoacidlist import amino_acids_by_code
from deeprank2.molstruct.aminoacid import Polarity
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.graph import Graph

_log = logging.getLogger(__name__)
SAFE_MIN_CONTACTS = 5


def _id_from_residue(residue: tuple[str, int, str]) -> str:
    """Create and id from pdb2sql rendered residues that is similar to the id of residue nodes.

    Args:
        residue: Input residue as rendered by pdb2sql: ( str(<chain>), int(<residue_number>), str(<three_letter_code> )
            For example: ('A', 27, 'GLU').

    Returns:
        str: Output id in form of '<chain><residue_number>'. For example: 'A27'.
    """
    return residue[0] + str(residue[1])


class _ContactDensity:
    """Internal class that holds contact density information for a given residue."""

    def __init__(self, residue: tuple[str, int, str], polarity: Polarity):
        self.res = residue
        self.polarity = polarity
        self.id = _id_from_residue(self.res)
        self.densities = {pol: 0 for pol in Polarity}
        self.densities["total"] = 0
        self.connections = {pol: [] for pol in Polarity}
        self.connections["all"] = []


def get_IRCs(pdb_path: str, chains: list[str], cutoff: float = 5.5) -> dict[str, _ContactDensity]:
    """Get all close contact residues from the opposite chain.

    Args:
        pdb_path: Path to pdb file to read molecular information from.
        chains: list (or list-like object) containing strings of the chains to be considered.
        cutoff: Cutoff distance (in Ångström) to be considered a close contact. Defaults to 10.

    Returns:
        Dict[str, _ContactDensity]:
            keys: ids of residues in form returned by id_from_residue.
            items: _ContactDensity objects, containing all contact density information for the residue.
    """
    residue_contacts: dict[str, _ContactDensity] = {}

    sql = pdb2sql.interface(pdb_path)
    pdb2sql_contacts = sql.get_contact_residues(
        cutoff=cutoff,
        chain1=chains[0],
        chain2=chains[1],
        return_contact_pairs=True,
    )

    for chain1_res, chain2_residues in pdb2sql_contacts.items():
        aa1_code = chain1_res[2]
        try:
            aa1 = amino_acids_by_code[aa1_code]
        except IndexError:
            continue  # skip keys that are not an amino acid

        # add chain1_res to residue_contact dict
        contact1_id = _id_from_residue(chain1_res)
        residue_contacts[contact1_id] = _ContactDensity(chain1_res, aa1.polarity)

        for chain2_res in chain2_residues:
            aa2_code = chain2_res[2]
            try:
                aa2 = amino_acids_by_code[aa2_code]
            except IndexError:
                continue  # skip keys that are not an amino acid

            # populate densities and connections for chain1_res
            residue_contacts[contact1_id].densities["total"] += 1
            residue_contacts[contact1_id].densities[aa2.polarity] += 1
            residue_contacts[contact1_id].connections["all"].append(chain2_res)
            residue_contacts[contact1_id].connections[aa2.polarity].append(chain2_res)

            # add chain2_res to residue_contact dict if it doesn't exist yet
            contact2_id = _id_from_residue(chain2_res)
            if contact2_id not in residue_contacts:
                residue_contacts[contact2_id] = _ContactDensity(chain2_res, aa2.polarity)

            # populate densities and connections for chain2_res
            residue_contacts[contact2_id].densities["total"] += 1
            residue_contacts[contact2_id].densities[aa1.polarity] += 1
            residue_contacts[contact2_id].connections["all"].append(chain1_res)
            residue_contacts[contact2_id].connections[aa1.polarity].append(chain1_res)

    return residue_contacts


def add_features(  # noqa: C901, D103
    pdb_path: str,
    graph: Graph,
    single_amino_acid_variant: SingleResidueVariant | None = None,
) -> None:
    if not single_amino_acid_variant:  # VariantQueries do not use this feature
        polarity_pairs = list(combinations(Polarity, 2))
        polarity_pair_string = [f"irc_{x[0].name.lower()}_{x[1].name.lower()}" for x in polarity_pairs]

        total_contacts = 0
        residue_contacts = get_IRCs(pdb_path, graph.get_all_chains())

        for node in graph.nodes:
            if isinstance(node.id, Residue):
                residue = node.id
            elif isinstance(node.id, Atom):
                atom = node.id
                residue = atom.residue
            else:
                msg = f"Unexpected node type: {type(node.id)}"
                raise TypeError(msg)

            contact_id = residue.chain.id + residue.number_string  # reformat id to be in line with residue_contacts keys

            # initialize all IRC features to 0
            for IRC_type in Nfeat.IRC_FEATURES:
                node.features[IRC_type] = 0

            # load correct values to IRC features
            try:
                node.features[Nfeat.IRCTOTAL] = residue_contacts[contact_id].densities["total"]
                for i, pair in enumerate(polarity_pairs):
                    if residue_contacts[contact_id].polarity == pair[0]:
                        node.features[polarity_pair_string[i]] = residue_contacts[contact_id].densities[pair[1]]
                    elif residue_contacts[contact_id].polarity == pair[1]:
                        node.features[polarity_pair_string[i]] = residue_contacts[contact_id].densities[pair[0]]
                total_contacts += 1
            except KeyError:  # node has no contact residues and all counts remain 0
                pass

        if total_contacts < SAFE_MIN_CONTACTS:
            _log.warning(f"Few ({total_contacts}) contacts detected for {pdb_path}.")
