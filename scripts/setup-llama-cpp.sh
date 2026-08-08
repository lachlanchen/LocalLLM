#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_root/.local/tools/llama.cpp"

# Official ggml-org/llama.cpp release b10327, published 2026-08-08.
llama_cpp_url="https://github.com/ggml-org/llama.cpp.git"
llama_cpp_tag="b10327"
llama_cpp_commit="69bf6437914596fbbc4caf09a7ac16f2acdd1a94"
cuda_architecture="${LOCALLLM_LLAMA_CPP_CUDA_ARCH:-89}"
cuda_arch_slug="${cuda_architecture//;/_}"
build_dir="$project_root/.local/build/llama.cpp-b10327-cuda-sm${cuda_arch_slug}"
install_dir="$project_root/.local/opt/llama.cpp-b10327"

die() {
  echo "setup-llama-cpp: $*" >&2
  exit 1
}

for prerequisite in cmake c++ git install ninja; do
  command -v "$prerequisite" >/dev/null || die "missing prerequisite: $prerequisite"
done

nvcc_bin="${CUDACXX:-}"
if [[ -n "$nvcc_bin" && ! -x "$nvcc_bin" ]]; then
  nvcc_bin="$(command -v "$nvcc_bin" || true)"
fi
if [[ -z "$nvcc_bin" ]]; then
  nvcc_bin="$(command -v nvcc || true)"
fi
if [[ -z "$nvcc_bin" && -x /usr/local/cuda/bin/nvcc ]]; then
  nvcc_bin="/usr/local/cuda/bin/nvcc"
fi
[[ -n "$nvcc_bin" && -x "$nvcc_bin" ]] ||
  die "CUDA nvcc is required; install the CUDA toolkit or set CUDACXX"

if [[ -e "$source_dir" && ! -d "$source_dir/.git" ]]; then
  die "$source_dir exists but is not a Git checkout"
fi

if [[ ! -d "$source_dir/.git" ]]; then
  mkdir -p "$source_dir"
  git -C "$source_dir" init -q
  git -C "$source_dir" remote add origin "$llama_cpp_url"
fi

origin_url="$(git -C "$source_dir" remote get-url origin 2>/dev/null || true)"
[[ "$origin_url" == "$llama_cpp_url" ]] ||
  die "origin mismatch: expected $llama_cpp_url, found ${origin_url:-none}"

if ! git -C "$source_dir" show-ref --verify --quiet "refs/tags/$llama_cpp_tag"; then
  git -C "$source_dir" fetch --depth 1 origin \
    "refs/tags/$llama_cpp_tag:refs/tags/$llama_cpp_tag"
fi

resolved_commit="$(git -C "$source_dir" rev-list -n 1 "refs/tags/$llama_cpp_tag")"
[[ "$resolved_commit" == "$llama_cpp_commit" ]] ||
  die "$llama_cpp_tag did not resolve to the pinned commit"

current_commit="$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || true)"
if [[ "$current_commit" != "$llama_cpp_commit" ]]; then
  if [[ -n "$(git -C "$source_dir" status --porcelain --untracked-files=all)" ]]; then
    die "the llama.cpp checkout has local changes; refusing to replace them"
  fi
  git -C "$source_dir" checkout --detach "$llama_cpp_commit"
fi

if [[ -n "$(git -C "$source_dir" status --porcelain --untracked-files=all)" ]]; then
  die "the pinned llama.cpp checkout has local changes; refusing a non-reproducible build"
fi

jobs="${LOCALLLM_LLAMA_CPP_JOBS:-}"
if [[ -z "$jobs" ]]; then
  cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  if ((cpu_count > 6)); then
    jobs=6
  else
    jobs="$cpu_count"
  fi
fi
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "LOCALLLM_LLAMA_CPP_JOBS must be a positive integer"
[[ "$cuda_architecture" =~ ^[0-9]+(\;[0-9]+)*$ ]] ||
  die "LOCALLLM_LLAMA_CPP_CUDA_ARCH contains an unsupported value"

mkdir -p "$build_dir" "$install_dir"

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$nvcc_bin" \
  -DCMAKE_CUDA_ARCHITECTURES="$cuda_architecture" \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CCACHE=OFF \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_COMMIT="$llama_cpp_commit" \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_NUMBER="${llama_cpp_tag#b}" \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF

cmake --build "$build_dir" --parallel "$jobs" \
  --target llama-server llama-cli llama-bench

mkdir -p "$install_dir/bin"
for binary in llama-server llama-cli llama-bench; do
  install -m 0755 "$build_dir/bin/$binary" "$install_dir/bin/$binary"
done

for binary in llama-server llama-cli llama-bench; do
  [[ -x "$install_dir/bin/$binary" ]] || die "install produced no $binary"
done

installed_version="$($install_dir/bin/llama-server --version 2>&1)"
grep -Fq "$llama_cpp_commit" <<<"$installed_version" ||
  die "installed llama-server does not report pinned commit $llama_cpp_commit"

echo "$installed_version"
echo "llama.cpp source: $source_dir"
echo "llama.cpp binaries: $install_dir/bin"
