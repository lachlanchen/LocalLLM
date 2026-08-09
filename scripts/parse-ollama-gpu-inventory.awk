function field_value(field, value) {
  value = field
  sub(/^[^=]*=/, "", value)
  sub(/^"/, "", value)
  sub(/"$/, "", value)
  return value
}

function is_legacy_accelerator_library(library) {
  library = tolower(library)
  return library == "cuda" || library == "rocm" || library == "metal" || \
    library == "vulkan" || library == "oneapi" || library == "opencl" || \
    library == "cann"
}

/msg="inference compute"/ {
  library = ""
  compute_id = ""
  pci_id = ""
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^library=/) library = field_value($i)
    if ($i ~ /^id=/) compute_id = field_value($i)
    if ($i ~ /^pci_id=/) pci_id = field_value($i)
  }

  normalized_library = tolower(library)
  if (normalized_library == "cpu" || normalized_library ~ /^cpu[-_]/) next

  key = ""
  if (pci_id != "") {
    key = "pci_id=" pci_id
  } else if (compute_id != "" && is_legacy_accelerator_library(normalized_library)) {
    # Older Ollama records may omit pci_id. Only a known accelerator library is
    # safe to count in that format; a bare id also appears on CPU fallback rows.
    key = "library=" normalized_library ",id=" compute_id
  }

  if (key != "" && !seen[key]++) ordered[++count] = key
}

END {
  printf "%d", count + 0
  for (i = 1; i <= count; i++) printf " %s", ordered[i]
}
