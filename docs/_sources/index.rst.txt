:html_theme.sidebar_secondary.remove: true

.. module:: pyneon

Welcome to PyNeon documentation
===============================

PyNeon is a light-weight library to work with Neon (Pupil Labs) multi-modal
eye-tracking data. It is a community-driven effort to provide a versatile set
of tools to work with the rich data (gaze, eye states, IMU, events, etc.)
provided by Neon.

PyNeon is licensed under the MIT License.

Here we provide tutorials and API reference to help you get started with PyNeon.

Installation
============

To install PyNeon, clone the PyNeon repository from
https://github.com/NCCLabMPI/PyNeon and run:

.. code-block:: bash

   pip install .

PyPI and conda releases are planned for the future.

Data format
===========

PyNeon works with the "Timeseries Data" or "Timeseries Data + Scene Video" formats 
as exported from Pupil Clouds. The data could be from a single recording or from a 
project with multiple recordings.

Credits
=======

PyNeon was inspired by https://github.com/pupil-labs/pl-neon-recording/

.. toctree::
   :maxdepth: 1
   :hidden:

   Tutorials <tutorials/index>
   API reference <reference/index>