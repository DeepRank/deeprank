Developer Guideline
===================

Here we provide some details about guideline for internal developers. Most of the
choices are explained in the [guide](https://guide.esciencecenter.nl).

For a quick reference on software development, we refer to [the software
guide checklist](https://guide.esciencecenter.nl/best_practices/checklist.html).

Version control
---------------

We use [git](http://git-scm.com/) and [github](https://github.com/) for developing
[DeepRank](https://github.com/DeepRank/deeprank). And [Github Desktop](https://desktop.github.com/) is recommennded to simplify development workflow.

To get onboard, first ask project manager to get access to the DeepRank repo (short for repository), then clone the repo to your local machine.

**The development workflow is as the following**:
1. Create an issue in [the repo Issues](https://github.com/DeepRank/deeprank/issues) for any idea, bug, feature, improvement, etc.
2. Comment and mention (i.e. @) someone in the issue for discussion if necessary.
3. Assign the issue to yourself or someone after asking
4. Create a new branch from the `development` branch for this issue, and name it in a format of `issue{ID}_*`, e.g. `issue7_format_docstring`.
5. Work in this issue branch to solve the issue. Check this [guide](https://google.github.io/eng-practices/review/developer/) to see how to make good commits.
6. Create a Pull Request to merge this issue branch to the `development` branch when last step is completed.
7. Assign someone but not yourself to review the pull request. For reviewers, check this [guide](https://google.github.io/eng-practices/review/reviewer/) to see how to do a code review.
8. Follow reviewer's comments to fix the code until reviewer approve you to merge. Check this [guide](https://google.github.io/eng-practices/review/developer/handling-comments.html) to see how to handle reviewer comments.
9. Merge the issue branch to `development` branch and delete the issue branch.
10. Close the issue after leavning a comment of the related pull request ID.

Repeat the 1-10 steps for next issue.

Note that try to keep the issue and pull request small for efficient code review. In principle, it should take ≤30 mins to review a pull request.


Package management and dependencies
-----------------------------------

You can use either `pip` or `conda` for
installing dependencies and package management. This repository does not
force you to use one or the other, as project requirements differ. For
advice on what to use, please check [the relevant section of the
guide](https://guide.esciencecenter.nl/best_practices/language_guides/python.html#dependencies-and-package-management).

-   Dependencies should be added to `setup.py` in the
    `install_requires` list.

Packaging/One command install
-----------------------------

You can distribute your code using pipy or conda. Again, the project
template does not enforce the use of either one. [The
guide](https://guide.esciencecenter.nl/best_practices/language_guides/python.html#building-and-packaging-code)
can help you decide which tool to use for packaging.

If you decide to use pypi for distributing you code, you can configure
travis to upload to pypi when you make a release. If you specified your
pypi user name during generation of this package, the `.travis.yml` file
contains a section that looks like:

``` {.sourceCode .yaml}
deploy:
  provider: pypi
  user: no
  password:
    secure: FIXME; see README for more info
 on:
    tags: true
    branch: master
```

Before this actually works, you need to add an encrypted password for
your pypi account. The [travis
documentation](https://docs.travis-ci.com/user/deployment/pypi/)
specifies how to do this.

Testing and code coverage
-------------------------

-   Tests should be put in the `test` folder.
-   The `test` folder contains:
    -   Example tests that you should replace with your own meaningful
        tests (file: `test_learn.py`)
-   The testing framework used is [PyTest](https://pytest.org)
    -   [PyTest
        introduction](http://pythontesting.net/framework/pytest/pytest-introduction/)
-   Tests can be run with `python setup.py test`
    -   This is configured in `setup.py` and `setup.cfg`
-   Use [Travis CI](https://travis-ci.com/) to automatically run tests
    and to test using multiple Python versions
    -   Configuration can be found in `.travis.yml`
    -   [Getting started with Travis
        CI](https://docs.travis-ci.com/user/getting-started/)
-   TODO: add something about code quality/coverage tool?
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/best_practices/language_guides/python.html#testing)

Documentation
-------------

-   Documentation should be put in the `docs` folder. The contents have
    been generated using `sphinx-quickstart` (Sphinx version 1.6.5).
-   We recommend writing the documentation using Restructured Text
    (reST) and Google style docstrings.
    -   [Restructured Text (reST) and Sphinx
        CheatSheet](http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html)
    -   [Google style docstring
        examples](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
-   The documentation is set up with the Read the Docs Sphinx Theme.
    -   Check out the [configuration
        options](https://sphinx-rtd-theme.readthedocs.io/en/latest/).
-   To generate html documentation run `python setup.py build_sphinx`
    -   This is configured in `setup.cfg`
    -   Alternatively, run `make html` in the `docs` folder.
-   The `docs/_templates` directory contains an (empty) `.gitignore`
    file, to be able to add it to the repository. This file can be
    safely removed (or you can just leave it there).
-   To put the documentation on [Read the
    Docs](https://readthedocs.org), log in to your Read the Docs
    account, and import the repository (under 'My Projects').
    -   Include the link to the documentation in this [README](https://guide.esciencecenter.nl/best_practices/language_guides/python.html#writingdocumentation).
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/best_practices/language_guides/python.html#writingdocumentation)

Coding style conventions and code quality
-----------------------------------------

-   Check your code style with `prospector`
-   You may need run `pip install .[dev]` first, to install the required
    dependencies
-   You can use `autopep8` to fix the readability of your code style and
    `isort` to format and group your imports
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/best_practices/language_guides/python.html#coding-style-conventions)

Package version number
----------------------

-   We recommend using [semantic
    versioning](https://guide.esciencecenter.nl/best_practices/releases.html#semantic-versioning).
-   For convenience, the package version is stored in a single place:
    `deeprank/__version__.py`. For updating the
    version number, you only have to change this file.
-   Don't forget to update the version number before [making a
    release](https://guide.esciencecenter.nl/best_practices/releases.html)!

Logging
-------

-   We recommend using the `logging` module for getting
    useful information from your module (instead of using
    `print`).
-   The project is set up with a logging example.
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/best_practices/language_guides/python.html#logging)

CHANGELOG.rst
-------------

-   Document changes to your software package
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/software/releases.html#changelogmd)

CITATION.cff
------------

-   To allow others to cite your software, add a `CITATION.cff` file
-   It only makes sense to do this once there is something to cite
    (e.g., a software release with a DOI).
-   Follow the [making software
    citable](https://guide.esciencecenter.nl/citable_software/making_software_citable.html)
    section in the guide.

CODE_OF_CONDUCT.rst
---------------------

-   Information about how to behave professionally
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/software/documentation.html#code-of-conduct)

CONTRIBUTING.rst
----------------

-   Information about how to contribute to this software package
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/software/documentation.html#contribution-guidelines)

MANIFEST.in
-----------

-   List non-Python files that should be included in a source
    distribution
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/best_practices/language_guides/python.html#building-and-packaging-code)

NOTICE
------

-   List of attributions of this project and Apache-license dependencies
-   [Relevant section in the
    guide](https://guide.esciencecenter.nl/best_practices/licensing.html#notice)
