---
name: Bug/Performance Issue
about: Use this template for reporting a bug or a performance issue.
title: "[BUG/PERFORMANCE] - Short meaningful description of bug/performance issue"
labels: ''
assignees: ''

---

<em>Please make sure that this is a bug. As per our [GitHub Policy](https://github.com/zurutech/ashpy/blob/master/ISSUES.md), we only address code/doc bugs, performance issues, feature requests and build/installation issues on GitHub. </em>

**System information**
- OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
- AshPy version:
- TensorFlow version (use command below):
- Python version:
- CUDA/cuDNN version:
- GPU model and memory:

You can collect TensorFlow informations from their environment capture [script](https://github.com/tensorflow/tensorflow/tree/master/tools/tf_env_collect.sh). You can also obtain the TensorFlow version with: `python -c "import tensorflow as tf; print(tf.version.GIT_VERSION, tf.version.VERSION)"`

**Describe the bug**
A clear and concise description of what the bug is.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate the problem.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Other info / logs**
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached.
