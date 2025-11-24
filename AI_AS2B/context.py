def load_context_from_file(filepath):
    """
    Load context data from input.txt file
    """
    context = {
        "Nodes": [],
        "edge": [],
        "cam": [],
        "start": None,
        "end": [],
        "mult": None,
        "path": []
    }
    
    current_section = None
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Detect sections
            if line == '[NODES]':
                current_section = 'nodes'
                continue
            elif line == '[WAYS]':
                current_section = 'ways'
                continue
            elif line == '[CAMERAS]':
                current_section = 'cameras'
                continue
            elif line == '[META]':
                current_section = 'meta'
                continue
            
            # Parse data based on current section
            if current_section == 'nodes':
                parts = line.split(',', 3)
                node_id = int(parts[0])
                lat = float(parts[1])
                lon = float(parts[2])
                name = parts[3]
                context["Nodes"].append([node_id, lat, lon, name])
            
            elif current_section == 'ways':
                parts = line.split(',', 5)
                way_id = int(parts[0])
                from_node = int(parts[1])
                to_node = int(parts[2])
                road_name = parts[3]
                highway_type = parts[4]
                travel_time = int(parts[5])
                context["edge"].append([way_id, from_node, to_node, road_name, highway_type, travel_time])
            
            elif current_section == 'cameras':
                parts = line.split(',', 1)
                way_id = int(parts[0])
                image_path = parts[1]
                context["cam"].append([way_id, image_path])
            
            elif current_section == 'meta':
                parts = line.split(',')
                if parts[0] == 'START':
                    context["start"] = int(parts[1])
                elif parts[0] == 'GOAL':
                    context["end"] = [int(parts[i]) for i in range(1, len(parts))]
                elif parts[0] == 'ACCIDENT_MULTIPLIER':
                    context["mult"] = int(parts[1])
    
    return context

# Load context from file
context = load_context_from_file('input.txt')