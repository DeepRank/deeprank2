from deeprank_gnn.models.query import Query, QueryType


class SingleResidueVariant(Query):
    def __init__(self, residue, wildtype_amino_acid, variant_amino_acid, target=None):
        Query.__init__(QueryType.SINGLE_RESIDUE_VARIANT, 10.0, residue.model, target)

        self._residue = residue
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

    def __eq__(self, other):
        return type(self) == type(other) and self._residue == other._residue and \
            self._wildtype_amino_acid == other.wildtype_amino_acid and \
            self._variant_amino_acid == other.variant_amino_acid

    def __hash__(self):
        return hash((self._residue, self._wildtype_amino_acid, self._variant_amino_acid))

    def __repr__(self):
        return "{}:{}->{}".format(self._residue, self._wildtype_amino_acid, self._variant_amino_acid)
