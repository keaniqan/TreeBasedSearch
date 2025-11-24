def split_csv_allow_commas(line, min_fields):
    parts = []
    buf = []
    depth = 0
    for ch in line:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            buf.append(ch)
        elif ch == ',':
            if depth == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    if len(parts) < min_fields:
        raise ValueError(f"Line '{line}' parsed into too few fields: {parts}")
    return parts

def parse_config_file(path):
    """Parses the config file to load initial arguments

    Args:
        path (string): Filepath to the initial configuration txt file

    Returns:
        nodes: List of nodes given as a diction of node id to {lat, lon, label}
        ways: List of ways given as a list of {way_id, from, to, road_name, type, time_min}
        cameras: Dictionary of camera id to way id
        meta: Dictionary of meta information including start node, goal nodes, and accident multiplier
    """    
    section = None
    nodes = {}
    ways = []
    cameras = {}
    meta = {"start": None, "goals": [], "accident_multiplier": None}

    def is_header(line):
        return line.startswith("[") and line.endswith("]")

    def ignore(line):
        return (not line.strip()) or line.strip().startswith("#")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if ignore(line):
                continue
            if is_header(line):
                section = line.upper()
                continue

            if section == "[NODES]":
                p = split_csv_allow_commas(line, 4)
                nid, lat, lon, label = p[0], float(p[1]), float(p[2]), p[3]
                nodes[nid] = {"lat": lat, "lon": lon, "label": label}

            elif section == "[WAYS]":
                # way_id,from_node,to_node,road_name,highway_type,travel_time_min
                p = split_csv_allow_commas(line, 6)
                ways.append({
                    "way_id": p[0],
                    "from": p[1],
                    "to": p[2],
                    "road_name": p[3],
                    "type": p[4],
                    "time_min": float(p[5]),
                })

            elif section == "[CAMERAS]":
                p = split_csv_allow_commas(line, 2)
                cameras[p[0]] = p[1]

            elif section == "[META]":
                p = [x.strip() for x in line.split(",")]
                key = p[0].upper()
                if key == "START":
                    meta["start"] = p[1]
                elif key == "GOAL":
                    meta["goals"] = p[1:]
                elif key == "ACCIDENT_MULTIPLIER":
                    meta["accident_multiplier"] = float(p[1])

    return nodes, ways, cameras, meta