"""The ``mjolnir`` command bodies — one module per job, composed by ``cli.py``.

``_common`` is what the command modules share; the sub-apps are one module
per group (``serve``, ``bench`` + ``bench_report``, ``model``, ``image``,
``platform``, ``litellm``, ``perses``, ``hw``), the root commands one per
topic (``window``, ``calibrate``, ``results``, ``version``). A command
module imports only ``_common`` from here (``bench`` also ``bench_report``);
only ``cli.py`` imports this package. Nothing is re-exported: importing this
package must stay cheap (no charts/matplotlib on the ``mjolnir version``
path), so the modules are imported directly from their users.
"""
