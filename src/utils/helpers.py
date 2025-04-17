def vprint(message, is_verbose):
    if is_verbose:
        print(message)

def get_filename_to_id_mapping(filepath, reverse=False):
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            image_id, filename = line.strip().split()
            if not reverse:
                mapping[filename] = int(image_id)-1
            else:
                mapping[int(image_id)-1] = filename

    return mapping