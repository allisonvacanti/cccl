#!/usr/bin/env bash

set -euo pipefail

readonly script_path=$(cd $(dirname ${0}); pwd -P)
readonly docs_build_path=../build/docs
readonly image_path=${docs_build_path}/img
readonly cub_docs_repo=${docs_build_path}/cubimg

cd ${script_path}

# Install prerequisites
# sudo apt update
sudo apt install -y --no-install-recommends doxygen
pip install sphinx breathe nvidia-sphinx-theme

## Clean image directory, without this any artifacts will prevent fetching
rm -rf "${image_path}"
mkdir -p "${image_path}"

# Pull cub images
if [ ! -d "${cub_docs_repo}" ]; then
    git clone --depth 1 -b gh-pages https://github.com/NVlabs/cub.git "${cub_docs_repo}"
fi

if [ ! -n "$(find ${cub_docs_repo} -name 'example_range.png')" ]; then
    wget -q https://raw.githubusercontent.com/NVIDIA/NVTX/release-v3/docs/images/example_range.png -O "${cub_docs_repo}/example_range.png"
fi

if [ ! -n "$(find ${image_path} -name '*.png')" ]; then
    wget -q https://docs.nvidia.com/cuda/_static/Logo_and_CUDA.png -O "${image_path}/logo.png"

    # Parse files and collects unique names ending with .png
    imgs=( $(grep -R -o -h '[[:alpha:][:digit:]_]*.png' ../cub/cub | uniq) )
    imgs+=( "cub_overview.png" "nested_composition.png" "tile.png" "blocked.png" "striped.png" )

    for img in "${imgs[@]}"
    do
        echo ${img}
        cp "${cub_docs_repo}/${img}" "${image_path}/${img}"
    done
fi


# Generate doxygen xml output for sphinx/breathe:
run_doxygen() {
    local project=$1
    local project_dir="${docs_build_path}/${project}"
    echo "Generating doxygen xml for ${project}"
    mkdir -p "${project_dir}/doxygen/xml"
    doxygen doxyfiles/${project}.conf | tee "${project_dir}/doxygen/log.txt
}

run_doxygen cub
run_doxygen cudax
run_doxygen thrust

sphinx-build -M html . "${docs_build_path}/sphinx" | tee "${docs_build_path}/sphinx/log.txt"
