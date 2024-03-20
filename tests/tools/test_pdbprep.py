from pathlib import Path

import pytest

from deeprank2.tools.pdbprep.preprocess import preprocess_pdbs


@pytest.fixture(scope="module")
def pdb_file() -> Path:
    return Path("tests/data/pdb/3C8P/3C8P.pdb")
    # with Path("tests/data/pdb/3C8P/3C8P.pdb").open('r') as f:
    #     records =


def test_pdbtools(pdb_file: Path) -> None:
    processed = preprocess_pdbs(pdb_file).splitlines()

    with pdb_file.open("r") as pdb:
        original = pdb.read().splitlines()

    resname_cols = slice(17, 20)
    altloc_cols = slice(16, 17)  # noqa: F841
    coordinate_cols = slice(31, 54)  # noqa: F841

    # check that only atomic records were preserved
    original_openings = [r.split()[0] for r in original]
    processed_openings = [r.split()[0] for r in processed]

    scraped_record_types = ("HEADER", "TITLE", "COMPND", "REMARK")
    kept_record_types = ("ATOM",)

    for record in scraped_record_types:
        assert record in original_openings
        assert record not in processed_openings

    for record in kept_record_types:
        assert record in original_openings
        assert record in processed_openings

    # check that no water remains
    original_resnames = [r[resname_cols] for r in original]
    processed_resnames = [r[resname_cols] for r in processed]
    assert "HOH" in original_resnames
    assert "HOH" not in processed_resnames

    # untested (but confirmed in Jupyter notebook):
    # - select altloc (this file)
    # - residue renumbering (this file)
    # - atom renumbering (file 1ak4)
    # - replace residue names (with dummy names)
    #
    # untested and no good test data:
    # - fix insertion codes
    # - sort
    # - tidy (not sure what it does)
