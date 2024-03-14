from dataclasses import dataclass
from tempfile import TemporaryFile

from openmm import LangevinIntegrator, unit
from openmm import app as mmapp
from pdb2pqr.config import FORCE_FIELDS
from pdb2pqr.main import main_driver as pdb2pqr

_MIN_ATOMS_TO_PROTONATE = 5  # TODO: why do we need this check?
_TEMPERATURE = 310


def add_hydrogens(
    pdb_str: str,
    max_iterations: int = 100,
    constraint_tolerance: float = 1e-05,
    random_seed: int = 917,
) -> str:
    """Add hydrogens to pdb file.

    Args:
        pdb_str: String representation of pdb file, preprocessed using deeprank2.tools.pdbprep.preprocess.
        max_iterations: Maximum number of iterations to perform during energy minimization. Defaults to 100.
        constraint_tolerance: Distance tolerance of LangevinIntegrator within which constraints are maintained, as a
            fraction of the constrained distance. Defaults to  1e-05.
        random_seed: Random seed for LangevinIntegrator.
    """
    with TemporaryFile(mode="w", suffix="pdb") as input_pdb, TemporaryFile(mode="r") as output_pdb:
        input_pdb.write(pdb_str)

        # PARAMETERS
        forcefield_model = "amber14-all.xml"  #'charmm36.xml'
        water_model = "amber14/tip3p.xml"  #'charmm36/tip3p-pme-b.xml'
        platform_properties = {"Threads": str(1)}

        # PREPARES MODEL
        forcefield = mmapp.ForceField(forcefield_model, water_model)
        structure = mmapp.PDBFile(input_pdb)
        protonated_sequence = _detect_protonation_state(pdb_str)

        model = mmapp.Modeller(structure.topology, structure.positions)
        model.addHydrogens(forcefield=forcefield, variants=protonated_sequence)

        structure.positions = model.positions
        structure.topology = model.topology

        system = forcefield.createSystem(structure.topology)

        integrator = LangevinIntegrator(
            temperature=_TEMPERATURE * unit.kelvin,
            frictionCoeff=1.0 / unit.picosecond,
            stepSize=2.0 * unit.femtosecond,
        )

        integrator.setRandomNumberSeed(random_seed)
        integrator.setConstraintTolerance(constraint_tolerance)

        simulation = mmapp.Simulation(
            structure.topology,
            system,
            integrator,
            platformProperties=platform_properties,
        )

        context = simulation.context
        context.setPositions(model.positions)

        state = context.getState(getEnergy=True)
        ini_ene = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)  # noqa:F841 TODO: check if this line is needed
        simulation.minimizeEnergy(maxIterations=max_iterations)
        structure.positions = context.getState(getPositions=True).getPositions()

        # TODO: check whether these lines need to be repeated or whether that's a typo.
        state = context.getState(getEnergy=True)
        simulation.minimizeEnergy(maxIterations=max_iterations)
        structure.positions = context.getState(getPositions=True).getPositions()

        mmapp.PDBFile.writeFile(structure.topology, structure.positions, output_pdb)

        return output_pdb.read()


def _detect_protonation_state(pdb_str: str) -> list[str | None]:
    """Detect protonation states and return them as a sequence of alternative residue names."""
    _calculate_protonation_state(pdb_str)

    pdb_lines = pdb_str.splitlines()
    protonable_residues = ("HIS", "ASP", "GLU", "CYS", "LYS")

    # initialize
    prev_resid = None
    prev_resname = None
    atoms_in_residue = set()
    residues = []

    for i, line in enumerate(pdb_lines):
        if not line.startswith("ATOM"):
            continue

        resid = line[21:26]  # chain ID + res number
        resname = line[17:20]
        atom_name = line[12:16].strip()

        if (resid != prev_resid and len(atoms_in_residue) >= _MIN_ATOMS_TO_PROTONATE) or i == len(pdb_lines):
            if prev_resname in protonable_residues:
                residues.append(_protonation_resname(prev_resname, atoms_in_residue))
            else:
                residues.append(None)
            atoms_in_residue.clear()

        atoms_in_residue.add(atom_name)
        prev_resid = resid
        prev_resname = resname

    return residues


def _protonation_resname(  # noqa:PLR0911
    resname: str,
    atoms_in_residue: list[str],
) -> str:
    """Return alternate residue name based on protonation state."""
    if resname == "HIS":
        if "HD1" in atoms_in_residue and "HE2" in atoms_in_residue:
            return "HIP"
        if "HD1" in atoms_in_residue:
            return "HID"
        if "HE2" in atoms_in_residue:
            return "HIE"
        return "HIN"

    if resname == "ASP" and ("HD2" in atoms_in_residue or "HD1" in atoms_in_residue):
        return "ASN"

    if resname == "GLU" and ("HE2" in atoms_in_residue or "HE1" in atoms_in_residue):
        return "GLH"

    if resname == "LYS" and not all(_a in atoms_in_residue for _a in ("HZ1", "HZ2", "HZ3")):
        return "LYN"

    if resname == "CYS" and "HG" not in atoms_in_residue:
        return "CYX"

    return resname


def _calculate_protonation_state(
    pdb_str: str,
    forcefield: str = "AMBER",
) -> str:
    """Calculate the protonation states using PDB2PQR.

    PDB2PQR can only use files (no strings) as input and output, which is why this function is wrapped inside
    TemporaryFile context managers.
    """
    with TemporaryFile(mode="w", suffix="pdb") as input_pdb, TemporaryFile(mode="r") as output_pdb:
        input_pdb.write(pdb_str)

        input_args = _Pdb2pqrArgs(input_pdb, output_pdb, forcefield)
        pdb2pqr(input_args)

        return output_pdb.read()


@dataclass
class _Pdb2pqrArgs:
    """Input arguments to `main_driver` function of PDB2PQR.

    These are usually given via CLI using argparse. All arguments, including those kept as default need to be given to
    `main_driver` if called from script.
    The argument given to `main_driver` is accessed via dot notation and is iterated over, which is why this is created
    as a dataclass with an iterator.

    Args*:
        input_path: Input file path.
        output_pqr: Output file path.
        ff: Name of the selected forcefield.

        *all other arguments should remain untouched.

    Raises:
        ValueError: if the forcefield is not recognized
    """

    input_path: str
    output_pqr: str
    ff: str = "AMBER"

    # arguments set different from default
    debump: bool = True
    keep_chain: bool = True
    log_level: str = "CRITICAL"

    # arguments kept as default
    ph: float = 7.0
    assign_only: bool = False
    clean: bool = False
    userff: None = None
    ffout: None = None
    usernames: None = None
    ligand: None = None
    neutraln: bool = False
    neutralc: bool = False
    drop_water: bool = False
    pka_method: None = None
    opt: bool = True
    include_header: bool = False
    whitespace: bool = False
    pdb_output: None = None
    apbs_input: None = None

    def __post_init__(self):
        self._index = 0
        if self.ff.lower() not in FORCE_FIELDS:
            msg = f"Forcefield {self.ff} not recognized. Valid options: {FORCE_FIELDS}."
            raise ValueError(msg)
        if self.ff.lower() != "amber":
            msg = f"Forcefield given as {self.ff}. Currently only AMBER forcefield is implemented."
            raise NotImplementedError(msg)

    def __iter__(self):
        return self

    def __next__(self):
        settings = vars(self)
        if self._index < len(settings):
            setting = list(settings)[self._index]
            self._index += 1
            return setting
        raise StopIteration


# Part of this module is modified from https://github.com/DeepRank/pdbprep/blob/main/
# original module names: detect_protonation.py and add_hydrogens.py),
# written by João M.C. Teixeira (https://github.com/joaomcteixeira)
# publishd under the following license:

#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS

#    APPENDIX: How to apply the Apache License to your work.

#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.

#    Copyright [yyyy] [name of copyright owner]

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
