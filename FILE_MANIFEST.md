# File manifest and packaging notes

Online Resource 1 contains editable source files, measurement CSV files, processed XLSX workbooks, a text correctness report, and the Python figure-generation script.

Included source/configuration formats: `.sln`, `.vcxproj`, `.filters`, `.cpp`, `.cu`, `.cuh`, `.h`, `.hpp`, `.ps1`, `.py`, `.md`, `.txt`, `.gitignore`, and `.gitattributes`.

Included data formats: `.csv` and `.xlsx`.

The package intentionally excludes generated or machine-specific content: Visual Studio/CUDA build directories (`bin`, `obj`, and `x64`), executables, object files, program databases, incremental-link artifacts, dependency caches, tracking logs, user settings, and embedded Git metadata. These exclusions do not remove source code or experimental data and reduce the risk of leaking local absolute paths.

The original retained project may contain build outputs, but the submitted archive is source-and-data only. Rebuild instructions are in `code/README.md`.
