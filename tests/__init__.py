from pathlib import Path

import pkg_resources as pkg

__all__ = ["PATH_DEEPRANK_CORE", "PATH_TEST"]

# Environment data
PATH_DEEPRANK_CORE = Path(pkg.resource_filename("deeprankcore", ""))
ROOT = PATH_DEEPRANK_CORE.parent

PATH_TEST = ROOT / "tests"
