import pandas as pd
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
        nodes: Pandas DataFrame of nodes (index: node id, columns: lat, lon, label)
        ways: Pandas DataFrame of ways (columns: id, from, to, name, type, base_time, accident_severity, final_time)
        cameras: Dictionary of camera id to way id
        meta: Dictionary of meta information including start node, goal nodes, and accident multiplier
    """
    section = None
    nodes = {}
    ways = []
    cameras_df = pd.DataFrame(columns=['way_id', 'image_path'])
    start = None
    goals = []
    accident_multiplier = None

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
                nodes[nid] = {"id": nid, "lat": lat, "lon": lon, "label": label}

            elif section == "[WAYS]":
                p = split_csv_allow_commas(line, 6)
                ways.append({
                    "id": int(p[0]),
                    "from": int(p[1])-1,
                    "to": int(p[2])-1,
                    "name": p[3],
                    "type": p[4],
                    "base_time": float(p[5]),
                    "accident_severity": 0,
                    "final_time": float(p[5])
                })

            elif section == "[CAMERAS]":
                p = split_csv_allow_commas(line, 2)
                cameras_df.loc[-1] = {'way_id': int(p[0]), 'image_path': p[1]}
                cameras_df.index = cameras_df.index + 1
                cameras_df = cameras_df.sort_index()

            elif section == "[META]":
                p = [x.strip() for x in line.split(",")]
                key = p[0].upper()
                if key == "START":
                    start = p[1]
                elif key == "GOAL":
                    goals = p[1:]
                elif key == "ACCIDENT_MULTIPLIER":
                    accident_multiplier = float(p[1])

    # Convert nodes and ways to pandas DataFrames
    nodes_df = pd.DataFrame.from_dict(nodes, orient="index")
    nodes_df.index.name = "id"
    ways_df = pd.DataFrame(ways)
    return nodes_df, ways_df, cameras_df, start, goals, accident_multiplier