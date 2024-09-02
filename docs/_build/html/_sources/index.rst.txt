:html_theme.sidebar_secondary.remove: true

.. module:: pyneon

PyNeon documentation
====================

PyNeon is a light-weight library to work with Neon (Pupil Labs) multi-modal
eye-tracking data. It is originally developed by Qian Chu and Jan-Gabriel
Hartel from the Max Planck Institute for Empirical Aesthetics.

Installation
============

To install PyNeon, clone the PyNeon repository from
https://github.com/NCCLabMPI/pyneon and run:

.. code-block:: bash

   pip install .

PyPI and conda releases are planned for the future.

Data format
===========

PyNeon works with the "Timeseries" data format as exported from Pupil Clouds. The data could be from a single recording or from a project with multiple recordings.

.. toctree::
   :maxdepth: 1
   :hidden:

   Tutorials <tutorials/index>
   API reference <reference/index>