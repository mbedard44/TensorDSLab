#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [environment-name]" >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    environment_name="tensor_dslab"
else
    environment_name="$1"
fi
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
repository_root="${script_directory}"

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
    "${repository_root}"

smoke_directory="$(mktemp -d "/tmp/tensor-dslab-smoke.XXXXXX")"
cleanup_smoke_directory() {
    rm -rf -- "${smoke_directory}"
}
trap cleanup_smoke_directory EXIT

(
    cd -- "${smoke_directory}"
    "${conda_executable}" run --name "${environment_name}" \
        python -c \
        'import json, platform, site, sys; from importlib.metadata import distribution, version; from pathlib import Path; import torch, tensor_dslab; from tensor_dslab import Charge, ChargeSpec, unit_registry; repository_root = Path(sys.argv[1]).resolve(); module_path = Path(tensor_dslab.__file__).resolve(); site_roots = tuple(Path(path).resolve() for path in site.getsitepackages()); direct_url_text = distribution("tensor-core").read_text("direct_url.json"); assert direct_url_text is not None; direct_url = json.loads(direct_url_text); assert platform.python_version() == "3.14.6"; assert version("tensor-dslab") == "0.2.0"; assert version("tensor-core") == "0.22.0"; assert direct_url["vcs_info"]["commit_id"] == "19bfae35fbc773b55cac7bcd659dda57c4dee6d6"; assert not module_path.is_relative_to(repository_root); assert any(module_path.is_relative_to(root) for root in site_roots); spec = ChargeSpec(axes=(), device=torch.device("cpu"), dtype=torch.float64, unit=unit_registry.Unit("avalanche")); product = Charge(tensor=torch.tensor(1.0, dtype=torch.float64), spec=spec); assert product.spec is spec; assert tuple(tensor_dslab.__all__); print("TensorDSLab", version("tensor-dslab"), "TensorCore", version("tensor-core"), "Python", platform.python_version())' \
        "${repository_root}"
)

cleanup_smoke_directory
trap - EXIT

echo "Environment '${environment_name}' is ready."
echo "Run:"
printf '  conda activate %q\n' "${environment_name}"
printf '  cd %q\n' "${repository_root}"
echo "  python -c 'import tensor_dslab; print(tensor_dslab.__all__)'"
