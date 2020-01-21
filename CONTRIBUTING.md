# Contributing guidelines

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](https://github.com/zurutech/ashpy/blob/master/CONTRIBUTING.md).
- Read [Code of Conduct](https://github.com/zurutech/ashpy/blob/master/CODE_OF_CONDUCT.md).
- Check if my changes are consistent with the [guidelines](https://github.com/zurutech/ashpy/blob/master/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution).
- Changes are consistent with the [Coding Style](https://github.com/zurutech/ashpy/blob/master/CONTRIBUTING.md#python-coding-style).
- Run [Tests](https://github.com/zurutech/ashpy/blob/master/CONTRIBUTING.md#running-tests).

## How to become a contributor and submit your own code

### Contributing code

If you have improvements to AshPy, send us your pull requests! For those
just getting started, please refer to [this](https://github.com/zurutech/ashpy/blob/master/.github/pull_request_template.md) pull request template.

AshPy team members will be assigned to review your pull requests. Once the
pull requests are approved and pass all the tests, your pull request will
be merged into the official codebase.

If you want to contribute, start working through the AshPy codebase,
navigate to the
[Github "issues" tab](https://github.com/zurutech/ashpy/issues) and start
looking through interesting issues.

### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/zurutech/ashpy/pulls),
make sure your changes are consistent with the guidelines and follow the
AshPy coding style.

#### General guidelines and philosophy for contribution

*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
*   When you contribute a new feature to AshPy, the maintenance burden is
    (by default) transferred to the AshPy team. This means that the benefit
    of the contribution must be compared against the cost of maintaining the
    feature.

#### License

We'd love to accept your patches! Before we can take them, you need to understand
that the code you provide will be included with the Apache License.

You can see an example of the licence [here](https://github.com/zurutech/ashpy/blob/master/LICENSE).

Please, put the following lincense header in your files.

```
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

#### Python coding style

- Black is our one and only formatter
- We are more lenient with the length of the line
- Changes to AshPy Python code should conform to
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

To run all the lint against your current codebase:

```bash
tox -e black,pylint,flake8
```

#### Running tests

In order to run the tests and the doctests:

```bash
tox -e testenv
```

#### Documentation

To generate the documentation either look up the Sphinx docs or simply:

```bash
tox -e docs
```

## Attribution
This Contributing file is adapted from the TensorFlow Contributing available at https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md
