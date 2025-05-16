**************
GDB NESOASSERT
**************

The default behaviour of ``NESOASSERT`` is to print the user provided error message to stdout then call ``MPI_Abort``.
By default GDB may not consider ``MPI_Abort`` a break point and will simply exit.
Try the following in ``~/.gdbinit`` to make GDB stop on ``MPI_Abort``.
::

  set breakpoint pending on
  b MPI_Abort



