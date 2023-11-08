# `deeprank2` developer documentation

If you're looking for user documentation, go [here](README.md).

## Code editor

We use [Visual Studio Code (VS Code)](https://code.visualstudio.com/) as code editor.
The VS Code settings for this project can be found in [.vscode](.vscode).
The settings will be automatically loaded and applied when you open the project with VS Code.
See [the guide](https://code.visualstudio.com/docs/getstarted/settings) for more info about workspace settings of VS Code.

## Package setup

After having followed the [installation instructions](https://github.com/DeepRank/deeprank2#installation) and installed all the dependencies of the package, the repository can be cloned and its editable version can be installed:

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e .'[test]'
```

## Running the tests

You can check that all components were installed correctly, using pytest.
The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

Run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).

## Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests. In an activated conda environment with the development tools installed, inside the package directory, run:

```bash
coverage run -m pytest
```

This runs tests and stores the result in a `.coverage` file. To see the results on the command line, run:

```bash
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

## Linting

We use [prospector](https://pypi.org/project/prospector/) with pyroma for linting. For running it locally, use `prospector` or `prospector <filepath_or_folderpath>` for specific files/folders.

## Versioning

Bumping the version across all files is done before creating a new package release, running `bump2version [part]` from command line after having installed [bump2version](https://pypi.org/project/bump2version/) on your local environment. Instead of `[part]`, type the part of the version to increase, e.g. minor. The settings in `.bumpversion.cfg` will take care of updating all the files containing version strings.

## Branching workflow

We use a [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)-inspired branching workflow for development. DeepRank2's repository is based on two main branches with infinite lifetime:
- `main` — this branch contains production (stable) code. All development code is merged into `main` in sometime.
- `dev` — this branch contains pre-production code. When the features are finished then they are merged into `dev`.
During the development cycle, three main supporting branches are used:
- Feature branches - Branches that branch off from `dev` and must merge into `dev`: used to develop new features for the upcoming releases. 
- Hotfix branches - Branches that branch off from `main` and must merge into `main` and `dev`: necessary to act immediately upon an undesired status of `main`.
- Release branches - Branches that branch off from `dev` and must merge into `main` and `dev`: support preparation of a new production release. They allow many minor bug to be fixed and preparation of meta-data for a release.

### Development conventions 

- Branching
  - When creating a new branch, please use the following convention: `<issue_number>_<description>_<author_name>`.
  - Always branch from `dev` branch, unless there is the need to fix an undesired status of `main`. See above for more details about the branching workflow adopted. 
- Pull Requests
  - When creating a pull request, please use the following convention: `<type>: <description>`. Example _types_ are `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).

## Making a release 

1. Branch from `dev` and prepare the branch for the release (e.g., removing the unnecessary dev files such as the current one, fix minor bugs if necessary).
2. [Bump the version](https://github.com/DeepRank/deeprank2/blob/dev/README.dev.md#versioning). 
3. Verify that the information in `CITATION.cff` is correct, and that `.zenodo.json` contains equivalent data.
4. Merge the release branch into `main` (and `dev`), and [run the tests](https://github.com/DeepRank/deeprank2/blob/dev/README.dev.md#running-the-tests).
5. Go to https://github.com/DeepRank/deeprank2/releases and draft a new release; create a new tag for the release, generate release notes automatically and adjust them, and finally publish the release as latest. This will trigger [a GitHub action](https://github.com/DeepRank/deeprank2/actions/workflows/release.yml) that will take care of publishing the package on PyPi.   

## UML

Code-base class diagrams updated on 02/11/2023, generated with https://www.gituml.com:
- [Data processing classes and functions](https://gituml-media.s3.ap-southeast-2.amazonaws.com/production_diagram_2026.svg?AWSAccessKeyId=AKIA5BNPSF2PVKDZ4QNO&Signature=1AfRXogJj5JwWqIehv8vzdUC5So%3D&Expires=1698939357)
- [ML pipeline classes and functions](https://gituml-media.s3.ap-southeast-2.amazonaws.com/production_diagram_2025.svg?AWSAccessKeyId=AKIA5BNPSF2PVKDZ4QNO&Signature=eeYNfbm8yYfYfkVgRJyRhv9D1zs%3D&Expires=1698939046](https://plantuml.atug.com/svg/xLhTRziu4dyNiF-3WNlPFLoCsaLVYY90jacp1T9SaMHnZmGZqJ8HYjIIb1FlNl_t6tv9f2JB7vlUmztwnUHmXfmP_ZWSaiePLcg64ELK8uJgxUp3XqeK3IFdvr2S_VZb7qBcQrhyD0F2townVBed-5eImcT_P3Fu8RGXKb714Sg-PRdjxvAGMcB-T8arLaHxJjj6L5n6Mi3G1Ncnn89iQ4v2qJ4MA_ktBbRlK8trsb6j329SI-exh0ZchgIQHnZJ5ZMMPL94AyeqaPwYdbn1Oigx4Yf0Qrjj547wjwBoTZeV_VYJg-PhaZyXgnTqIbHAftnD9DMa0AKbhdgjPcY4UO6y6zfMKTMD9iWOWvGkMhea6CY9rQ7DMOISVO5grGBVrZ083aH6k2yfl25eJPM6-hSi2zFquTzd_QdB-bFvzxEpyvasc9YxwPgz3_FMPvtFpsVhT_LsVYr5KmEVqNoBIbFBF5Ak9IrIb1HaXHkcqGPBDvyhiBsH9AsWWkR9nCBKSRKIScgoWh2dWbZmZC054fo0RhJmFkaPOtmJATlfZ-jrQW3UOChm8oE852L1YVfT6nT5hordZOAPfBo4-TThKOoxerqPmzw7pcafgNkwnJ9KuvNz2UwPScZdFU_GmjLEYKJNTvylqTUBkuVF5pVeqzN5mwztL-ZsuiiL41yUhkvkdUvCbBPmTdOMBGQpO9x8LZbb8otcyq4uUp_B1TSGAjJilCSQutcIBOBDBZwpqa3P4X0n1z_UG958u6vC4EtnIJ8tmGC2qGu6kn2lNW1W7BCRofy2N_wdTe1lhoJ_aVCxKWAMp4uIzWl1Dmnl8RYzUVDcG5aCACi1vQrR17dQIrTVZ88RKF-CPT67DpBBmlIEHACXJ7kovtQuSO6Hy_aNwm4ByiIqB3KqYgOWz14wVriuF4IMJubymIzVXM03gSNn0orF6kZjf1G0ZM23sNWEZXHOP1r6Ffe2NT4Sw_z5dAJcPKE33yNBfDWaPeuVS7doW6_tIkx2px522myX24m4z3W-IcMH9CJl5C9901ymzdHDc5ydU5xVtjuOGqC2eZLaMYOZhrZTfGgXOhVyK74GRQizY3ivOv0dEDc8a41SDHMaBahXibqCJLsOpDN8-Qqr1NrGFQpn-wOc4eZU51YAGPfgoxM4NLZaA34c41EZUbPu7b3C4sJ3aLv36bgk4IVwMSYdCTs2Ba6t2qPBNe5hE_rw8oLhl3N5z1NB-kvm8F7JqXs4S0DfehJdhgpVcqao0YmIo8qmjfhkOqmyC0BaJm2768cF58Vp6gICc7F2L81DIdZEtHyVZRJ8pE28b-ra4flbc5CMWUn6xGwmhNPPfwUD75-3CDW4YaorygbMQsxaYjlew1jo-tRlwVQbfTzj5lv_kHTQkwo30RCmiDMkK_Qz7tIoUPasZqNKnNwMfNV12XXFm3RO5qa2wdfQ1iWDcMxxoRGFZv-XmMnTCJHNkVz3RRWC11jWW2eFW-xt0MkgT1QRSWYc5eYyw1x2RAcb5rBKOCkhOC-gbJdbB2nwBW_WsyUntD-7rNzlFvRBkGoEJoTXi9KpaNHaa82n0T__4mJ_mjm4vhZmI_Haz2Kz78J3KOanHDvYkY6cAy4QWppCOcYcITPaTfLCL3G_EmR8TWriFNjFVclCIi0CTEwKE4AcLJMsyzObUQrn_XfaaLOz3ksTrKdM-c6oP4ohuou3tiVAd2Is715BFTBPupp23JygEJy1beaim6ktWX7WHeM2mYFGlBUtOnSX9fwMK3Qye9BachXhr6T2oxN-Zi5oJEbir99XKkbqEumvpZ7lhoyU_di9fXajiqCwa8sf6uBkGMpBY6qyrPnk3tccAl4f6tmGCmUWdKe6P2RCeDG8R1_Cur4NGVt9chpKGXheEFnYdQ-HehyJf8a_ryQqzYNATeaSCf8NixYSlfwDln_aJU4UiynzCDBRchGr0-rMYvRGgX0e6ynSVUHrRlW6GMfXhC4RKhYB0WUkqMUAz8s12QLM3S_RF8YvxATxhEZwioLadncKDXSCxOi7jmhuOYCnu_u1a9TrqzrX90g9Mjk-oA0tQA4Lp8oCBqlK6dilrf9OVmLRxGkbkmfPDwiL8_vQPE91ClL1zoO9y-9qabtTOCJLx3jbr0nprBL2XHPnguCHAaYEjn5XF7TqJpjXyc4deT0VuJaPPu36fcsolcyhq1HTbLQmW9BR8ZUkxqSbn9KKLMX96lgBgciCHhUNF0uTyLsLewJliOxKcyQkiNGcMLq25RZzKcifvWRDpZyiIBTCGfDzN8exXR2neYL03xNO2ieOTyQBm9FDr8UYqT95w11j9jvZ4vP_MizTUSTz58oHtDz73aDL8ccUgMuic80tnkK4LupFtMm9K4SCWd2iHBT3TdgPb-NU4vf7IyAI3EQ7SR7QJVVHYNb0XIs4zqE85XiA8QN3gzb0km5WomXA9Rj6n3BvqjVZu_KS4CnBigCz5_KMe41gKtZTHxvSL5K3FMnHBGGp6yq9_Wjkwpeh8BM90J_IWcemq6wK0nwlIlS6RHQeZ-RfEgsPqF6BzAWvriKlSrNRZPJXBIoErekx11nSA_oobnEwji3TozWsp1_DaeEefQbVhtj7W2YaSbflaG8nHH2M4c_t2bRvt5oCLpaxXDK-SZdtghtyFvkPsyjbbciAs2iisVPUYxe6jd04i3jXBYJnG6Xu-A87egp_SeSOGTZCXJl4mSoKs1SXC7Kf0kpNrEnj8Rhca1b0z6ttL1EEKUKJeWsHZq8HlslccB7e4D5kejz2TcvrK5xv4_BJWJm4ajGX8qbe6kP3o7978I6tNOA2p7T4H4OMa9VaL9Ha1LJPRvF0M-RRbMUgmphi7bP0ur4TegrcAZC-JX-eninZMZpOpUDHm9dcPygntARlS7qNFPkljUHtSqrlNinvwa05iqCDSSUjfgg0vL_M6hEtUxl2YJ4tLutoAqJ2lWDSeaEDvI7ZHpPYjf_37Jb5xXcSYIvDWjwYC6s-ckDmhlspVXonAPZxo-MdTwChmdzRvRujhDUuNTN7MBXFiU7esOX2TjhgOcwYtoTppZV9p6gA4nOzcEpOVAW9I-daXCR4cxWJY3MzqPDWCysRP9SJKeSYUxAJCKXF2enuTZV_P6QMMxJCLaAOlsi2MHYS7elPutPsJVMl5PizFp_FIwgRYfdFitx8O1x8Z5IrtiAnjdeqmUy_)https://plantuml.atug.com/svg/xLhTRziu4dyNiF-3WNlPFLoCsaLVYY90jacp1T9SaMHnZmGZqJ8HYjIIb1FlNl_t6tv9f2JB7vlUmztwnUHmXfmP_ZWSaiePLcg64ELK8uJgxUp3XqeK3IFdvr2S_VZb7qBcQrhyD0F2townVBed-5eImcT_P3Fu8RGXKb714Sg-PRdjxvAGMcB-T8arLaHxJjj6L5n6Mi3G1Ncnn89iQ4v2qJ4MA_ktBbRlK8trsb6j329SI-exh0ZchgIQHnZJ5ZMMPL94AyeqaPwYdbn1Oigx4Yf0Qrjj547wjwBoTZeV_VYJg-PhaZyXgnTqIbHAftnD9DMa0AKbhdgjPcY4UO6y6zfMKTMD9iWOWvGkMhea6CY9rQ7DMOISVO5grGBVrZ083aH6k2yfl25eJPM6-hSi2zFquTzd_QdB-bFvzxEpyvasc9YxwPgz3_FMPvtFpsVhT_LsVYr5KmEVqNoBIbFBF5Ak9IrIb1HaXHkcqGPBDvyhiBsH9AsWWkR9nCBKSRKIScgoWh2dWbZmZC054fo0RhJmFkaPOtmJATlfZ-jrQW3UOChm8oE852L1YVfT6nT5hordZOAPfBo4-TThKOoxerqPmzw7pcafgNkwnJ9KuvNz2UwPScZdFU_GmjLEYKJNTvylqTUBkuVF5pVeqzN5mwztL-ZsuiiL41yUhkvkdUvCbBPmTdOMBGQpO9x8LZbb8otcyq4uUp_B1TSGAjJilCSQutcIBOBDBZwpqa3P4X0n1z_UG958u6vC4EtnIJ8tmGC2qGu6kn2lNW1W7BCRofy2N_wdTe1lhoJ_aVCxKWAMp4uIzWl1Dmnl8RYzUVDcG5aCACi1vQrR17dQIrTVZ88RKF-CPT67DpBBmlIEHACXJ7kovtQuSO6Hy_aNwm4ByiIqB3KqYgOWz14wVriuF4IMJubymIzVXM03gSNn0orF6kZjf1G0ZM23sNWEZXHOP1r6Ffe2NT4Sw_z5dAJcPKE33yNBfDWaPeuVS7doW6_tIkx2px522myX24m4z3W-IcMH9CJl5C9901ymzdHDc5ydU5xVtjuOGqC2eZLaMYOZhrZTfGgXOhVyK74GRQizY3ivOv0dEDc8a41SDHMaBahXibqCJLsOpDN8-Qqr1NrGFQpn-wOc4eZU51YAGPfgoxM4NLZaA34c41EZUbPu7b3C4sJ3aLv36bgk4IVwMSYdCTs2Ba6t2qPBNe5hE_rw8oLhl3N5z1NB-kvm8F7JqXs4S0DfehJdhgpVcqao0YmIo8qmjfhkOqmyC0BaJm2768cF58Vp6gICc7F2L81DIdZEtHyVZRJ8pE28b-ra4flbc5CMWUn6xGwmhNPPfwUD75-3CDW4YaorygbMQsxaYjlew1jo-tRlwVQbfTzj5lv_kHTQkwo30RCmiDMkK_Qz7tIoUPasZqNKnNwMfNV12XXFm3RO5qa2wdfQ1iWDcMxxoRGFZv-XmMnTCJHNkVz3RRWC11jWW2eFW-xt0MkgT1QRSWYc5eYyw1x2RAcb5rBKOCkhOC-gbJdbB2nwBW_WsyUntD-7rNzlFvRBkGoEJoTXi9KpaNHaa82n0T__4mJ_mjm4vhZmI_Haz2Kz78J3KOanHDvYkY6cAy4QWppCOcYcITPaTfLCL3G_EmR8TWriFNjFVclCIi0CTEwKE4AcLJMsyzObUQrn_XfaaLOz3ksTrKdM-c6oP4ohuou3tiVAd2Is715BFTBPupp23JygEJy1beaim6ktWX7WHeM2mYFGlBUtOnSX9fwMK3Qye9BachXhr6T2oxN-Zi5oJEbir99XKkbqEumvpZ7lhoyU_di9fXajiqCwa8sf6uBkGMpBY6qyrPnk3tccAl4f6tmGCmUWdKe6P2RCeDG8R1_Cur4NGVt9chpKGXheEFnYdQ-HehyJf8a_ryQqzYNATeaSCf8NixYSlfwDln_aJU4UiynzCDBRchGr0-rMYvRGgX0e6ynSVUHrRlW6GMfXhC4RKhYB0WUkqMUAz8s12QLM3S_RF8YvxATxhEZwioLadncKDXSCxOi7jmhuOYCnu_u1a9TrqzrX90g9Mjk-oA0tQA4Lp8oCBqlK6dilrf9OVmLRxGkbkmfPDwiL8_vQPE91ClL1zoO9y-9qabtTOCJLx3jbr0nprBL2XHPnguCHAaYEjn5XF7TqJpjXyc4deT0VuJaPPu36fcsolcyhq1HTbLQmW9BR8ZUkxqSbn9KKLMX96lgBgciCHhUNF0uTyLsLewJliOxKcyQkiNGcMLq25RZzKcifvWRDpZyiIBTCGfDzN8exXR2neYL03xNO2ieOTyQBm9FDr8UYqT95w11j9jvZ4vP_MizTUSTz58oHtDz73aDL8ccUgMuic80tnkK4LupFtMm9K4SCWd2iHBT3TdgPb-NU4vf7IyAI3EQ7SR7QJVVHYNb0XIs4zqE85XiA8QN3gzb0km5WomXA9Rj6n3BvqjVZu_KS4CnBigCz5_KMe41gKtZTHxvSL5K3FMnHBGGp6yq9_Wjkwpeh8BM90J_IWcemq6wK0nwlIlS6RHQeZ-RfEgsPqF6BzAWvriKlSrNRZPJXBIoErekx11nSA_oobnEwji3TozWsp1_DaeEefQbVhtj7W2YaSbflaG8nHH2M4c_t2bRvt5oCLpaxXDK-SZdtghtyFvkPsyjbbciAs2iisVPUYxe6jd04i3jXBYJnG6Xu-A87egp_SeSOGTZCXJl4mSoKs1SXC7Kf0kpNrEnj8Rhca1b0z6ttL1EEKUKJeWsHZq8HlslccB7e4D5kejz2TcvrK5xv4_BJWJm4ajGX8qbe6kP3o7978I6tNOA2p7T4H4OMa9VaL9Ha1LJPRvF0M-RRbMUgmphi7bP0ur4TegrcAZC-JX-eninZMZpOpUDHm9dcPygntARlS7qNFPkljUHtSqrlNinvwa05iqCDSSUjfgg0vL_M6hEtUxl2YJ4tLutoAqJ2lWDSeaEDvI7ZHpPYjf_37Jb5xXcSYIvDWjwYC6s-ckDmhlspVXonAPZxo-MdTwChmdzRvRujhDUuNTN7MBXFiU7esOX2TjhgOcwYtoTppZV9p6gA4nOzcEpOVAW9I-daXCR4cxWJY3MzqPDWCysRP9SJKeSYUxAJCKXF2enuTZV_P6QMMxJCLaAOlsi2MHYS7elPutPsJVMl5PizFp_FIwgRYfdFitx8O1x8Z5IrtiAnjdeqmUy_)
