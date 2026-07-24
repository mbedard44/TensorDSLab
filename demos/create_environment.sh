#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [environment-name]" >&2
    exit 2
fi

environment_name="${1:-tensor_dslab}"
conda_executable="${CONDA_EXE:-conda}"

if [[ -z "${environment_name}" ]]; then
    echo "environment name must not be empty" >&2
    exit 2
fi

if ! command -v "${conda_executable}" >/dev/null 2>&1; then
    echo "Conda executable not found: ${conda_executable}" >&2
    exit 1
fi

script_directory="$(
    CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd -P
)"
repository_root="$(
    CDPATH= cd -- "${script_directory}/.." >/dev/null 2>&1
    pwd -P
)"

environment_listing="$("${conda_executable}" env list)"
if awk -v requested="${environment_name}" \
    '$1 == requested { found = 1 } END { exit(found ? 0 : 1) }' \
    <<<"${environment_listing}"
then
    echo "Conda environment already exists: ${environment_name}" >&2
    exit 1
fi

"${conda_executable}" create \
    --yes \
    --name "${environment_name}" \
    --no-default-packages \
    --override-channels \
    --channel conda-forge \
    "python=3.14.6" \
    pip

"${conda_executable}" run --name "${environment_name}" \
    python -m pip install \
    --disable-pip-version-check \
    --no-input \
    "${repository_root}[demos]"

"${conda_executable}" run --name "${environment_name}" \
    python -c \
    'import platform; from importlib.metadata import version; import tensor_dslab; from tensor_dslab.readout.profiles import ds20k_veto; assert platform.python_version() == "3.14.6"; assert version("tensor-dslab") == "0.1.0"; assert version("tensor-core") == "0.16.0"; assert type(ds20k_veto()).__name__ == "ReadoutConfig"; print("TensorDSLab", version("tensor-dslab"), "TensorCore", version("tensor-core"), "Python", platform.python_version())'

echo "Environment '${environment_name}' is ready."
echo "Run:"
printf '  conda activate %q\n' "${environment_name}"
printf '  cd %q\n' "${repository_root}"
echo "  python demos/readout.py"
