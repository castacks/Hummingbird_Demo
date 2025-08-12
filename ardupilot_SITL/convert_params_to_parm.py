with open('/home/tyler/Downloads/original_params.params', 'r') as f:
    lines = f.readlines()

with open('/home/tyler/Downloads/output_params.parm', 'w') as f:
    for line in lines:
        if line.startswith('#'):
            # Skip comment lines
            continue
        parts = line.strip().split('\t')
        # Compose with first and third part, separated by a tab
        if len(parts) == 5:
            f.write(f"{parts[2]}\t{parts[3]}\n")
        else:
            raise ValueError(f"Unexpected line format: {line.strip()}")
