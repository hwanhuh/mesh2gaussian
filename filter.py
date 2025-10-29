import numpy as np
import struct
from pathlib import Path

def read_3dgs_ply_header(ply_path):
    # Read PLY header and extract properties
    properties = []
    vertex_count = 0
    format_binary = True
    little_endian = True
    
    with open(ply_path, 'rb') as f:
        while True:
            line_bytes = f.readline()
            line = line_bytes.decode('ascii', errors='ignore').strip()
            
            if line == 'end_header':
                break
                
            if line.startswith('format'):
                parts = line.split()
                if len(parts) >= 2 and 'ascii' in parts[1]:
                    format_binary = False
                if len(parts) >= 2 and 'big' in parts[1]:
                    little_endian = False
                    
            elif line.startswith('element vertex'):
                parts = line.split()
                if len(parts) >= 3:
                    vertex_count = int(parts[2])
                
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_type, prop_name))
    
    return properties, vertex_count, format_binary, little_endian

def read_3dgs_ply_binary(ply_path, properties, vertex_count, little_endian=True):
    # Read binary PLY data into numpy arrays
    property_sizes = {
        'float': 4,
        'uchar': 1,
        'int': 4,
        'uint': 4
    }
    
    vertex_size = 0
    for prop_type, prop_name in properties:
        if prop_type in property_sizes:
            vertex_size += property_sizes[prop_type]
    
    with open(ply_path, 'rb') as f:
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line == 'end_header':
                break
        
        data = f.read()
    
    expected_size = vertex_size * vertex_count
    if len(data) < expected_size:
        vertex_count = len(data) // vertex_size
    
    endian_char = '<' if little_endian else '>'
    dtype_list = []
    
    for prop_type, prop_name in properties:
        if prop_type == 'float':
            dtype_list.append((prop_name, f'{endian_char}f4'))
        elif prop_type == 'uchar':
            dtype_list.append((prop_name, f'{endian_char}u1'))
        elif prop_type in ['int', 'uint']:
            dtype_list.append((prop_name, f'{endian_char}i4'))
    
    dtype = np.dtype(dtype_list)
    data = data[:vertex_count * vertex_size]
    array_data = np.frombuffer(data, dtype=dtype, count=vertex_count)
    
    result = {}
    for prop_type, prop_name in properties:
        result[prop_name] = array_data[prop_name].copy()
    
    return result

def filter_3dgs_ply(input_path, output_path, percentile=98.0, min_scale=0.0005, max_scale=10.0):
    # Filter Gaussian splats based on scale criteria
    properties, vertex_count, is_binary, is_little_endian = read_3dgs_ply_header(input_path)
    
    if not is_binary:
        raise ValueError("Only binary PLY format is supported")
    
    data = read_3dgs_ply_binary(input_path, properties, vertex_count, is_little_endian)
    
    scales = np.column_stack([data['scale_0'], data['scale_1'], data['scale_2']])
    exp_scales = np.exp(scales)
    
    max_scales = np.max(exp_scales, axis=1)
    min_scales = np.min(exp_scales, axis=1)
    scale_ratios = max_scales / np.clip(min_scales, 1e-10, None)
    mean_scales = np.mean(exp_scales, axis=1)
    
    ratio_threshold = np.percentile(scale_ratios, percentile)
    
    invalid_mask = (
        (scale_ratios >= ratio_threshold) | 
        (mean_scales <= min_scale) | 
        (mean_scales >= max_scale)
    )
    
    invalid_count = np.sum(invalid_mask)
    
    print(f"Total points: {vertex_count}")
    print(f"Invalid points: {invalid_count}")
    print(f"Invalid ratio: {invalid_count/vertex_count*100:.2f}%")
    print(f"Using {percentile}% percentile threshold: {ratio_threshold:.2f}")
    
    modified_data = data.copy()
    
    # Set opacity to 0 for invalid points (make them transparent)
    if 'opacity' in modified_data:
        modified_data['opacity'][invalid_mask] = -10.0
    
    write_3dgs_ply(output_path, modified_data, properties, is_little_endian)

def write_3dgs_ply(output_path, data, properties, little_endian=True):
    # Write modified data back to PLY file
    vertex_count = len(next(iter(data.values())))
    
    with open(output_path, 'wb') as f:
        f.write(b"ply\n")
        format_str = "binary_little_endian" if little_endian else "binary_big_endian"
        f.write(f"format {format_str} 1.0\n".encode('ascii'))
        f.write(f"element vertex {vertex_count}\n".encode('ascii'))
        
        for prop_type, prop_name in properties:
            f.write(f"property {prop_type} {prop_name}\n".encode('ascii'))
        
        f.write(b"end_header\n")
        
        endian_char = '<' if little_endian else '>'
        dtype_list = []
        
        for prop_type, prop_name in properties:
            if prop_type == 'float':
                dtype_list.append((prop_name, f'{endian_char}f4'))
            elif prop_type == 'uchar':
                dtype_list.append((prop_name, f'{endian_char}u1'))
            elif prop_type in ['int', 'uint']:
                dtype_list.append((prop_name, f'{endian_char}i4'))
        
        structured_array = np.zeros(vertex_count, dtype=dtype_list)
        for prop_type, prop_name in properties:
            structured_array[prop_name] = data[prop_name]
        
        f.write(structured_array.tobytes())

if __name__ == "__main__":
    input_file = "C:/Users/admin/Desktop/parking_lock_on_13-1.ply"
    output_file = "C:/Users/admin/Desktop/parking_lock_filtered.ply"
    
    filter_3dgs_ply(
        input_path=input_file,
        output_path=output_file,
        percentile=98.0,
        min_scale=0.0005,
        max_scale=10.0
    )