import re
line_break_regex = re.compile(r'(?<=\n)')

def process_lean_file(file_contents, byte_idx_1, byte_idx_2):
    def get_line(lines, line_number):
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
        else:
            raise IndexError("Line number out of range")

    def convert_pos(lines, byte_idx):
        num_bytes = [len(line.encode('utf-8')) for line in lines]
        n = 0
        for i, num_bytes_in_line in enumerate(num_bytes, start=1):
            n += num_bytes_in_line
            if n > byte_idx:
                line_byte_idx = byte_idx - (n - num_bytes_in_line)
                if line_byte_idx == 0:
                    return i, 1

                line = get_line(lines, i)
                m = 0

                for j, c in enumerate(line, start=1):
                    m += len(c.encode("utf-8"))
                    if m >= line_byte_idx:
                        return i, j + 1
        return len(lines), len(lines[-1])

    def extract_string_between_positions(lines, byte_idx_1, byte_idx_2):
        line_1, column_1 = convert_pos(lines, byte_idx_1)
        line_2, column_2 = convert_pos(lines, byte_idx_2)

        extracted_string = []

        if line_1 == line_2:
            # If both positions are in the same line
            substring = lines[line_1 - 1][column_1 - 1:column_2 - 1]
            extracted_string.append(substring)
        else:
            # If positions span multiple lines
            extracted_string.append(lines[line_1 - 1][column_1 - 1:])
            for line in range(line_1 + 1, line_2):
                extracted_string.append(lines[line - 1])
            extracted_string.append(lines[line_2 - 1][:column_2 - 1])

        return ''.join(extracted_string)

    def convert_line_col_to_char_idx(lines, line, col):
        char_idx = 0
        for i in range(line - 1):
            char_idx += len(lines[i])
        char_idx += col - 1
        return char_idx

    lines = line_break_regex.split(file_contents)

    line_1, column_1 = convert_pos(lines, byte_idx_1)
    line_2, column_2 = convert_pos(lines, byte_idx_2)

    extracted_string = extract_string_between_positions(lines, byte_idx_1, byte_idx_2)

    char_idx_1 = convert_line_col_to_char_idx(lines, line_1, column_1)
    char_idx_2 = convert_line_col_to_char_idx(lines, line_2, column_2)

    return extracted_string, line_1, column_1, line_2, column_2, char_idx_1, char_idx_2

def extract_positions(node):
    positions = []
    if isinstance(node, dict):
        if 'info' in node and 'original' in node['info']:
            info = node['info']['original']
            positions.append((info.get('pos'), info.get('endPos')))
        for key, value in node.items():
            positions.extend(extract_positions(value))
    elif isinstance(node, list):
        for item in node:
            positions.extend(extract_positions(item))
    return positions

def extract_vals(data):
    vals = []
    if isinstance(data, dict):
        if "val" in data:
            vals.append(data["val"].strip())
        for value in data.values():
            vals.extend(extract_vals(value))
    elif isinstance(data, list):
        for item in data:
            vals.extend(extract_vals(item))
    return vals

def find_doccomment_vals(data):
    vals = []
    positions = []

    if isinstance(data, dict):
        if data.get("kind") == "Lean.Parser.Command.docComment":
            args = data.get("args", [])
            for arg in args:
                if "atom" in arg:
                    vals.append(arg["atom"]["val"])
                    positions.append((arg["atom"]["info"]["original"]["pos"], arg["atom"]["info"]["original"]["endPos"]))
        for value in data.values():
            v, p = find_doccomment_vals(value)
            vals.extend(v)
            positions.extend(p)
    elif isinstance(data, list):
        for item in data:
            v, p = find_doccomment_vals(item)
            vals.extend(v)
            positions.extend(p)

    return vals, positions

def find_attributes_vals(data):
    vals = []
    positions = []

    if isinstance(data, dict):
        if data.get("kind") in ["Lean.Parser.Term.attributes"]:
            args = data.get("args", [])
            for arg in args:
                vals.extend(extract_vals(arg))
                positions.extend(extract_positions(arg))
        for value in data.values():
            v, p = find_attributes_vals(value)
            vals.extend(v)
            positions.extend(p)
    elif isinstance(data, list):
        for item in data:
            v, p = find_attributes_vals(item)
            vals.extend(v)
            positions.extend(p)

    return vals, positions

def find_pripro_vals(data):
    vals = []
    positions = []

    if isinstance(data, dict):
        if data.get("kind") in ["Lean.Parser.Command.private", "Lean.Parser.Command.protected"]:
            args = data.get("args", [])
            for arg in args:
                vals.extend(extract_vals(arg))
                positions.extend(extract_positions(arg))
        for value in data.values():
            v, p = find_pripro_vals(value)
            vals.extend(v)
            positions.extend(p)
    elif isinstance(data, list):
        for item in data:
            v, p = find_pripro_vals(item)
            vals.extend(v)
            positions.extend(p)

    return vals, positions

def extract_other_vals(data):
    vals = []
    if isinstance(data, dict):
        if "val" in data:
            vals.append(data["val"])
        for value in data.values():
            vals.extend(extract_other_vals(value))
    elif isinstance(data, list):
        for item in data:
            vals.extend(extract_other_vals(item))
    return vals

def find_kind_name_theorem_lemma_abbrev_def_instance_inductive(file_content, data):
    kind = None
    name = None
    kind_pos = None
    kind_end = None
    name_pos = None
    name_end = None

    if isinstance(data, dict):
        if len(data.get("node", {}).get("args", [])) > 1:
            second_node=data["node"]["args"][1]["node"]
            if second_node["kind"]== "Lean.Parser.Command.instance":
                atom = second_node.get("args", [])[1].get("atom", {})
                kind = atom.get("val")
                kind_pos = atom.get("info", {}).get("original", {}).get("pos")
                kind_end = atom.get("info", {}).get("original", {}).get("endPos")
                for arg in second_node.get("args", []):
                    if arg.get("node"):
                        if arg.get("node")["args"]:
                            if arg.get("node")["args"][0].get("node",{}).get("kind") == "Lean.Parser.Command.declId":
                                ident = arg.get("node")["args"][0]["node"]["args"][0]["ident"]
                                name = ident.get("val")
                                # print(name)
                                name_pos = ident.get("info", {}).get("original", {}).get("pos")
                                # print(name_pos)
                                name_end = ident.get("info", {}).get("original", {}).get("endPos")
                                break
            else:
                atom = second_node.get("args", [])[0].get("atom", {})
                kind = atom.get("val")
                kind_pos = atom.get("info", {}).get("original", {}).get("pos")
                kind_end = atom.get("info", {}).get("original", {}).get("endPos")
                for arg in second_node.get("args", []):
                    if arg.get("node", {}).get("kind") == "Lean.Parser.Command.declId":
                        ident = arg.get("node", {}).get("args", [])[0].get("ident", {})
                        name = ident.get("val")
                        name_pos = ident.get("info", {}).get("original", {}).get("pos")
                        name_end = ident.get("info", {}).get("original", {}).get("endPos")
                        break

    if kind_pos and kind_end:
        kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2 = process_lean_file(file_content, kind_pos, kind_end)
    else:
        kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2 = None,None,None, None,None,None,None

    if name_pos and name_end:
        name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2 = process_lean_file(file_content, name_pos, name_end)
    else:
        name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2 = None,None,None, None,None,None,None
    return kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2, name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2

def find_statement_theorem_lemma_abbrev(file_content,data):
    explicitBinder_list = []
    type_list=[]
    second_node=data["node"]["args"][1]["node"]

    for arg in second_node.get("args", []):
        if arg.get("node", {}).get("kind") in ["Lean.Parser.Command.declSig", "Lean.Parser.Command.optDeclSig"]:
            node_declSig = arg.get("node")
    statement_positions= extract_positions(node_declSig)
    node_1=node_declSig["args"][0]["node"]
    node_2=node_declSig["args"][1]["node"] if len(node_declSig["args"]) > 1 else None
    for arg in node_1.get("args", []):
        if arg.get("node", {}).get("kind") in[ "Lean.Parser.Term.explicitBinder","Lean.Parser.Term.implicitBinder","Lean.Parser.Term.instBinder"]:
            node=arg.get("node", {})
            val=extract_vals(node)
            val=' '.join(val)
            positions=extract_positions(node)
            start_pos = min(pos[0] for pos in positions)
            end_pos = max(pos[1] for pos in positions)
            explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
            explicitBinder_list.append({
                "explicitBinder":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })
    if node_2 is not None and second_node["kind"] in ["Lean.Parser.Command.theorem", "group", "Lean.Parser.Command.abbrev"]:
        if node_2.get("kind") in ["Lean.Parser.Term.typeSpec"]:
            node=node_2
            val=extract_vals(node)
            val=' '.join(val)
            positions=extract_positions(node)
            start_pos = min(pos[0] for pos in positions)
            end_pos = max(pos[1] for pos in positions)
            explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
            type_list.append({
                "typeSpec":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })

    # print(explicitBinder_list,statement_positions)
    return explicitBinder_list, type_list, statement_positions

def find_proof(file_content, data):
    vals = []
    positions = []
    if isinstance(data, dict):
        second_node = data["node"]["args"][1]["node"]
        for arg in second_node.get("args", []):
            if arg.get("node", {}).get("kind") in ["Lean.Parser.Command.declValSimple", "Lean.Parser.Command.declValEqns"]:
                vals.extend(extract_vals(arg))
                positions.extend(extract_positions(arg))
                break  # 找到后即可退出
    if positions:
        proof_start_pos = min(pos[0] for pos in positions)
        proof_end_pos = max(pos[1] for pos in positions)
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = process_lean_file(file_content, proof_start_pos, proof_end_pos)
    else:
        proof_start_pos = None
        proof_end_pos = None
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = None,None,None, None,None,None, None


    return proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2

def process_modifier(file_content, declaration, tactics):
    doccomment_vals, doccomment_positions = find_doccomment_vals(declaration)
    if doccomment_vals:
        comment_start_pos = min(pos[0] for pos in doccomment_positions)
        comment_end_pos = max(pos[1] for pos in doccomment_positions)
        comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2 = process_lean_file(file_content,comment_start_pos,comment_end_pos)
    else:
        comment = None
        comment_start_pos = None
        comment_end_pos = None
        comment= None
        comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2 = None, None, None, None, None, None

    attributes_vals, attributes_positions = find_attributes_vals(declaration)
    if attributes_vals:
        attributes_start_pos = min(pos[0] for pos in attributes_positions)
        attributes_end_pos = max(pos[1] for pos in attributes_positions)
        attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2 = process_lean_file(file_content,attributes_start_pos,attributes_end_pos)

    else:
        attributes = None
        attributes_start_pos = None
        attributes_end_pos = None
        attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2 = None, None, None, None, None, None

    pripro_vals, pripro_positions = find_pripro_vals(declaration)
    if pripro_vals:
        pripro_start_pos = min(pos[0] for pos in pripro_positions)
        pripro_end_pos = max(pos[1] for pos in pripro_positions)
        pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2 = process_lean_file(file_content,pripro_start_pos,pripro_end_pos)
    else:
        pripro = None
        pripro_start_pos = None
        pripro_end_pos = None
        pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2 = None, None, None, None, None, None

    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]

    start_pos = min(pos[0] for pos in positions)
    end_pos = max(pos[1] for pos in positions)
    whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2 = process_lean_file(file_content, start_pos, end_pos)

    tactics_list = []
    for tactic in tactics:
        if start_pos <= tactic["pos"] <= end_pos:
            _, A,B,c,d, tactic["pos"], tactic["endPos"] =  process_lean_file(file_content, tactic["pos"], tactic["endPos"])
            tactics_list.append(tactic)

    return tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2

def theorem_lemma_abbrev(file_content, declaration,tactics):
    tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2= process_modifier(file_content, declaration,tactics)

    kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2,name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2  = find_kind_name_theorem_lemma_abbrev_def_instance_inductive(file_content, declaration)

    explicitBinder_list, type_list, statement_positions = find_statement_theorem_lemma_abbrev(file_content, declaration)
    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]
    if not positions:
        return {"kind" : kind}

    whole_start_pos = min(pos[0] for pos in positions)
    whole_end_pos = max(pos[1] for pos in positions)
    if statement_positions:
        statement_start_pos = whole_start_pos
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)
    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = find_proof(file_content, declaration)
    declaration_info = {
        "kind" : kind,
        "whole_start_pos" : {
            "position" : whole_char_idx_1,
            "line" : whole_line_1,
            "column" : whole_column_1
        },

        "whole_end_pos" :{
            "position" : whole_char_idx_2,
            "line" : whole_line_2,
            "column" : whole_column_2
        },

        "attributes": attributes,
        "comment": {
            "content": comment,
            "start_pos": {
                "position" : comment_char_idx_1,
                "line" : comment_line_1,
                "column" : comment_column_1
            },
            "end_pos": {
                "position" : comment_char_idx_2,
                "line" : comment_line_2,
                "column" : comment_column_2
            }
        },
        "private_protected": pripro,
        "name": {
            "content": name,
            "start_pos": {
                "position" : name_char_idx_1,
                "line" : name_line_1,
                "column" : name_column_1
            },
            "end_pos": {
                "position" : name_char_idx_2,
                "line" : name_line_2,
                "column" : name_column_2
            }
        },
        "parameters": explicitBinder_list,
        "Type":type_list,
        "statement":{
            "content": statement_val_str,
            "start_pos": {
                "position" :  statement_char_idx_1,
                "line" :  statement_line_1,
                "column" :  statement_column_1
            },
            "end_pos": {
                "position" :  statement_char_idx_2,
                "line" :  statement_line_2,
                "column" :  statement_column_2
            }
        },
        "proof" :{
            "content": proof,
            "start_pos": {
                "position" :  proof_char_idx_1,
                "line" :  proof_line_1,
                "column" :  proof_column_1
            },
            "end_pos": {
                "position" :  proof_char_idx_2,
                "line" :  proof_line_2,
                "column" :  proof_column_2
            }
        }
    }
    return declaration_info

def find_statement_def(file_content,data):
    explicitBinder_list = []
    type_list=[]
    second_node=data["node"]["args"][1]["node"]

    for arg in second_node.get("args", []):
        if arg.get("node", {}).get("kind") in ["Lean.Parser.Command.declSig", "Lean.Parser.Command.optDeclSig"]:
            node_declSig = arg.get("node")
    statement_positions= extract_positions(node_declSig)
    node_1=node_declSig["args"][0]["node"]
    node_2=node_declSig["args"][1]["node"]
    for arg in node_1.get("args", []):
        if arg.get("node", {}).get("kind") in[ "Lean.Parser.Term.explicitBinder","Lean.Parser.Term.implicitBinder","Lean.Parser.Term.instBinder"]:
            node=arg.get("node", {})
            val=extract_vals(node)
            val=' '.join(val)
            positions=extract_positions(node)
            start_pos = min(pos[0] for pos in positions)
            end_pos = max(pos[1] for pos in positions)
            explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
            # explicitBinder_list.append({"explicitBinder": [explicit, explicit_char_idx_1,  explicit_line_1,  explicit_column_1, explicit_char_idx_2, explicit_line_2,  explicit_column_2]})
            explicitBinder_list.append({
                "explicitBinder":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })

    if second_node["kind"] in[ "Lean.Parser.Command.definition"] :
        for arg in node_2.get("args", []):
            if arg.get("node", {}).get("kind") in ["Lean.Parser.Term.typeSpec"]:
                node=arg.get("node", {})
                val=extract_vals(node)
                val=' '.join(val)
                positions=extract_positions(node)
                start_pos = min(pos[0] for pos in positions)
                end_pos = max(pos[1] for pos in positions)
                explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
                type_list.append({
                "typeSpec":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })

    elif second_node["kind"] in[ "Lean.Parser.Command.instance"]:
        val=extract_vals(node_2)
        val=' '.join(val)
        positions=extract_positions(node_2)
        start_pos = min(pos[0] for pos in positions)
        end_pos = max(pos[1] for pos in positions)
        explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
        type_list.append({
                "typeSpec":{
                    "content":explicit,
                    "start_pos": {
                        "position" :explicit_char_idx_1,
                        "line" : explicit_line_1,
                        "column" : explicit_column_1
                    },
                    "end_pos": {
                        "position" : explicit_char_idx_2,
                        "line" : explicit_line_2,
                        "column" : explicit_column_2
                    }
                }
            })
    return explicitBinder_list,type_list, statement_positions

def definition_instance(file_content, declaration,tactics):
    tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2= process_modifier(file_content, declaration,tactics)

    kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2,name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2  = find_kind_name_theorem_lemma_abbrev_def_instance_inductive(file_content, declaration)

    explicitBinder_list,type_list, statement_positions = find_statement_def(file_content,declaration)
    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]
    if not positions:
        return {"kind" : kind}

    whole_start_pos = min(pos[0] for pos in positions)
    whole_end_pos = max(pos[1] for pos in positions)
    if statement_positions:
        statement_start_pos = whole_start_pos
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)

    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = find_proof(file_content, declaration)
    declaration_info = {
        "kind" : kind,
        "whole_start_pos" : {
            "position" : whole_char_idx_1,
            "line" : whole_line_1,
            "column" : whole_column_1
        },

        "whole_end_pos" :{
            "position" : whole_char_idx_2,
            "line" : whole_line_2,
            "column" : whole_column_2
        },

        "attributes": attributes,
        "comment": {
            "content": comment,
            "start_pos": {
                "position" : comment_char_idx_1,
                "line" : comment_line_1,
                "column" : comment_column_1
            },
            "end_pos": {
                "position" : comment_char_idx_2,
                "line" : comment_line_2,
                "column" : comment_column_2
            }
        },
        "private_protected": pripro,
        "name": {
            "content": name,
            "start_pos": {
                "position" : name_char_idx_1,
                "line" : name_line_1,
                "column" : name_column_1
            },
            "end_pos": {
                "position" : name_char_idx_2,
                "line" : name_line_2,
                "column" : name_column_2
            }
        },
        "parameters": explicitBinder_list,
        "Type":type_list,
        "statement":{
            "content": statement_val_str,
            "start_pos": {
                "position" :  statement_char_idx_1,
                "line" :  statement_line_1,
                "column" :  statement_column_1
            },
            "end_pos": {
                "position" :  statement_char_idx_2,
                "line" :  statement_line_2,
                "column" :  statement_column_2
            }
        },
        "proof" :{
            "content": proof,
            "start_pos": {
                "position" :  proof_char_idx_1,
                "line" :  proof_line_1,
                "column" :  proof_column_1
            },
            "end_pos": {
                "position" :  proof_char_idx_2,
                "line" :  proof_line_2,
                "column" :  proof_column_2
            }
        }
    }
    return declaration_info

def find_kind_name_structure(file_content, data):
    kind = None
    name = None
    kind_pos = None
    kind_end = None
    name_pos = None
    name_end = None

    if isinstance(data, dict):
        if len(data.get("node", {}).get("args", [])) > 1:
            second_node=data["node"]["args"][1]["node"]
            if second_node["kind"]==  "Lean.Parser.Command.structure":
                node_structureTk=second_node["args"][0]["node"]
                atom = node_structureTk["args"][0]["atom"]
                kind = atom.get("val")
                kind_pos = atom.get("info", {}).get("original", {}).get("pos")
                kind_end = atom.get("info", {}).get("original", {}).get("endPos")

                for arg in second_node.get("args", []):
                    if arg.get("node"):
                        if arg.get("node").get("kind") == "Lean.Parser.Command.declId":
                            node_declId = second_node["args"][1]["node"]
                            val=extract_vals(node_declId)
                            val=' '.join(val)
                            positions=extract_positions(node_declId)
                            name_pos = min(pos[0] for pos in positions)
                            name_end = max(pos[1] for pos in positions)

    if kind_pos and kind_end:
        kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2 = process_lean_file(file_content, kind_pos, kind_end)
    else:
        kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2 = None,None,None, None,None,None,None

    if name_pos and name_end:
        name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2 = process_lean_file(file_content, name_pos, name_end)
    else:
        name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2 = None,None,None, None,None,None,None
    return kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2, name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2

def find_statement_structure(file_content,data):
    explicitBinder_list = []
    second_node=data["node"]["args"][1]["node"]
    statement_positions=None
    for arg in second_node.get("args", []):
        if arg.get("node"):
            if arg.get("node")["args"]:
                if arg.get("node")["args"][0].get("node"):
                    if arg.get("node")["args"][0].get("node")["kind"] in [ "Lean.Parser.Term.explicitBinder","Lean.Parser.Term.implicitBinder","Lean.Parser.Term.instBinder"]:
                        node_all= arg.get("node")
                        statement_positions= extract_positions(node_all)
                        # print(statement_positions)
                        for arg in node_all.get("args", []):
                            if arg.get("node", {}).get("kind") in[ "Lean.Parser.Term.explicitBinder","Lean.Parser.Term.implicitBinder","Lean.Parser.Term.instBinder"]:
                                node=arg.get("node", {})
                                val=extract_vals(node)
                                val=' '.join(val)
                                positions=extract_positions(node)
                                start_pos = min(pos[0] for pos in positions)
                                end_pos = max(pos[1] for pos in positions)
                                explicit, explicit_line_1,  explicit_column_1,  explicit_line_2,  explicit_column_2,  explicit_char_idx_1,  explicit_char_idx_2 =process_lean_file(file_content,start_pos, end_pos)
                                # explicitBinder_list.append({"explicitBinder": [explicit, explicit_char_idx_1,  explicit_line_1,  explicit_column_1, explicit_char_idx_2, explicit_line_2,  explicit_column_2]})
                                explicitBinder_list.append({
                                    "explicitBinder":{
                                        "content":explicit,
                                        "start_pos": {
                                            "position" :explicit_char_idx_1,
                                            "line" : explicit_line_1,
                                            "column" : explicit_column_1
                                        },
                                        "end_pos": {
                                            "position" : explicit_char_idx_2,
                                            "line" : explicit_line_2,
                                            "column" : explicit_column_2
                                        }
                                    }
                                })

    return explicitBinder_list, statement_positions

def find_proof_structure(file_content, declaration):
    vals = []
    positions = []
    second_node=declaration["node"]["args"][1]["node"]
    for arg in second_node.get("args", []):
        if arg.get("node"):
            if arg.get("node")["args"]:
                if arg.get("node")["args"][0].get("atom"):
                    if arg.get("node")["args"][0].get("atom").get("val")== "where":
                        vals.extend(extract_vals(arg))
                        positions.extend(extract_positions(arg))
                        break
    if positions:
        proof_start_pos = min(pos[0] for pos in positions)
        proof_end_pos = max(pos[1] for pos in positions)
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = process_lean_file(file_content, proof_start_pos, proof_end_pos)
    else:
        proof_start_pos = None
        proof_end_pos = None
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = None,None,None, None,None,None, None


    return proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2

def structure(file_content, declaration,tactics):
    tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2= process_modifier(file_content, declaration,tactics)

    kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2,name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2  = find_kind_name_structure(file_content, declaration)

    explicitBinder_list, statement_positions = find_statement_structure(file_content, declaration)
    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]
    if not positions:
        return {"kind" : kind}

    whole_start_pos = min(pos[0] for pos in positions)
    whole_end_pos = max(pos[1] for pos in positions)
    if statement_positions:
        statement_start_pos = whole_start_pos
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)

    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    if statement_positions:
        statement_start_pos = min(pos[0] for pos in statement_positions)
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)
    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    parts = [attributes, pripro, kind, name, statement_val_str]
    non_none_parts = [part for part in parts if part is not None]
    statement = ' '.join(non_none_parts).strip()

    proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = find_proof_structure(file_content, declaration)
    declaration_info = {
        "kind" : kind,
        "whole_start_pos" : {
            "position" : whole_char_idx_1,
            "line" : whole_line_1,
            "column" : whole_column_1
        },

        "whole_end_pos" :{
            "position" : whole_char_idx_2,
            "line" : whole_line_2,
            "column" : whole_column_2
        },

        "attributes": attributes,
        "comment": {
            "content": comment,
            "start_pos": {
                "position" : comment_char_idx_1,
                "line" : comment_line_1,
                "column" : comment_column_1
            },
            "end_pos": {
                "position" : comment_char_idx_2,
                "line" : comment_line_2,
                "column" : comment_column_2
            }
        },
        "private_protected": pripro,
        "name": {
            "content": name,
            "start_pos": {
                "position" : name_char_idx_1,
                "line" : name_line_1,
                "column" : name_column_1
            },
            "end_pos": {
                "position" : name_char_idx_2,
                "line" : name_line_2,
                "column" : name_column_2
            }
        },
        "parameters": explicitBinder_list,
        "statement":{
            "content": statement_val_str,
            "start_pos": {
                "position" :  statement_char_idx_1,
                "line" :  statement_line_1,
                "column" :  statement_column_1
            },
            "end_pos": {
                "position" :  statement_char_idx_2,
                "line" :  statement_line_2,
                "column" :  statement_column_2
            }
        },
        "proof" :{
            "content": proof,
            "start_pos": {
                "position" :  proof_char_idx_1,
                "line" :  proof_line_1,
                "column" :  proof_column_1
            },
            "end_pos": {
                "position" :  proof_char_idx_2,
                "line" :  proof_line_2,
                "column" :  proof_column_2
            }
        }

    }
    return declaration_info

def find_proof_inductive(file_content, declaration):
    vals = []
    positions = []
    second_node=declaration["node"]["args"][1]["node"]
    for arg in second_node.get("args", []):
        if arg.get("node"):
            if arg.get("node")["args"]:
                if arg.get("node")["args"][0].get("node",{}).get("kind",{}) == "Lean.Parser.Command.ctor":
                    vals.extend(extract_vals(arg))
                    positions.extend(extract_positions(arg))
                    break
    if positions:
        proof_start_pos = min(pos[0] for pos in positions)
        proof_end_pos = max(pos[1] for pos in positions)
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = process_lean_file(file_content, proof_start_pos, proof_end_pos)
    else:
        proof_start_pos = None
        proof_end_pos = None
        proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = None,None,None, None,None,None, None
    return proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2

def inductive(file_content, declaration,tactics):
    tactics_list, comment, comment_line_1, comment_column_1, comment_line_2, comment_column_2, comment_char_idx_1, comment_char_idx_2, attributes, attributes_line_1, attributes_column_1, attributes_line_2, attributes_column_2, attributes_char_idx_1, attributes_char_idx_2, pripro, pripro_line_1, pripro_column_1, pripro_line_2, pripro_column_2, pripro_char_idx_1, pripro_char_idx_2, whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2= process_modifier(file_content, declaration,tactics)

    kind, kind_line_1, kind_column_1, kind_line_2, kind_column_2, kind_char_idx_1, kind_char_idx_2,name, name_line_1, name_column_1, name_line_2, name_column_2, name_char_idx_1, name_char_idx_2  = find_kind_name_theorem_lemma_abbrev_def_instance_inductive(file_content, declaration)

    explicitBinder_list,type_list, statement_positions = find_statement_theorem_lemma_abbrev(file_content, declaration)
    positions = extract_positions(declaration)
    positions = [(start, end) for start, end in positions if start is not None and end is not None]
    if not positions:
        return {"kind" : kind}

    whole_start_pos = min(pos[0] for pos in positions)
    whole_end_pos = max(pos[1] for pos in positions)
    # statement_val_str = ' '.join(statement_vals)
    if statement_positions:
        statement_start_pos = whole_start_pos
        statement_end_pos = max(pos[1] for pos in statement_positions)
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 = process_lean_file(file_content, statement_start_pos, statement_end_pos)
        # print(statement_val_str)
    else:
        statement_start_pos = None
        statement_end_pos = None
        statement_val_str, statement_line_1, statement_column_1, statement_line_2, statement_column_2, statement_char_idx_1, statement_char_idx_2 =  None, None, None, None, None, None,None

    parts = [attributes, pripro, kind, name, statement_val_str]
    non_none_parts = [part for part in parts if part is not None]
    statement = ' '.join(non_none_parts).strip()

    proof, proof_line_1, proof_column_1, proof_line_2, proof_column_2, proof_char_idx_1, proof_char_idx_2 = find_proof_inductive(file_content, declaration)
    declaration_info = {
        "kind" : kind,
        "whole_start_pos" : {
            "position" : whole_char_idx_1,
            "line" : whole_line_1,
            "column" : whole_column_1
        },

        "whole_end_pos" :{
            "position" : whole_char_idx_2,
            "line" : whole_line_2,
            "column" : whole_column_2
        },

        "attributes": attributes,
        "comment": {
            "content": comment,
            "start_pos": {
                "position" : comment_char_idx_1,
                "line" : comment_line_1,
                "column" : comment_column_1
            },
            "end_pos": {
                "position" : comment_char_idx_2,
                "line" : comment_line_2,
                "column" : comment_column_2
            }
        },
        "private_protected": pripro,
        "name": {
            "content": name,
            "start_pos": {
                "position" : name_char_idx_1,
                "line" : name_line_1,
                "column" : name_column_1
            },
            "end_pos": {
                "position" : name_char_idx_2,
                "line" : name_line_2,
                "column" : name_column_2
            }
        },
        "parameters": explicitBinder_list,
        "Type":type_list,
        "statement":{
            "content": statement_val_str,
            "start_pos": {
                "position" :  statement_char_idx_1,
                "line" :  statement_line_1,
                "column" :  statement_column_1
            },
            "end_pos": {
                "position" :  statement_char_idx_2,
                "line" :  statement_line_2,
                "column" :  statement_column_2
            }
        },
        "proof" :{
            "content": proof,
            "start_pos": {
                "position" :  proof_char_idx_1,
                "line" :  proof_line_1,
                "column" :  proof_column_1
            },
            "end_pos": {
                "position" :  proof_char_idx_2,
                "line" :  proof_line_2,
                "column" :  proof_column_2
            }
        }
    }
    return declaration_info

def lean4_parser(file_content, data):
    tactics = data.get("tactics")
    premises = data.get("premises")
    command_asts = data.get("commandASTs")

    declarations_info = []
    for i, declaration in enumerate(command_asts):
        kind = declaration.get("node", {}).get("kind")
        if kind in ["Lean.Parser.Command.declaration", "lemma"]:
            if len(declaration.get("node", {}).get("args", [])) > 1 and declaration["node"]["args"][1]["node"]["kind"] in ["Lean.Parser.Command.theorem","Lean.Parser.Command.example", "group", "Lean.Parser.Command.abbrev"]:
                declaration_info = theorem_lemma_abbrev(file_content, declaration,tactics)
                declarations_info.append(declaration_info)
            elif len(declaration.get("node", {}).get("args", [])) > 1 and declaration["node"]["args"][1]["node"]["kind"] in [ "Lean.Parser.Command.definition","Lean.Parser.Command.instance"]:
                declaration_info = definition_instance(file_content, declaration,tactics)
                declarations_info.append(declaration_info)

            elif len(declaration.get("node", {}).get("args", [])) > 1 and declaration["node"]["args"][1]["node"]["kind"] in [ "Lean.Parser.Command.structure"]:
                declaration_info = structure(file_content, declaration,tactics)
                declarations_info.append(declaration_info)

            elif len(declaration.get("node", {}).get("args", [])) > 1 and declaration["node"]["args"][1]["node"]["kind"] in [ "Lean.Parser.Command.inductive"]:
                declaration_info = inductive(file_content, declaration,tactics)
                declarations_info.append(declaration_info)

        else:
            positions = extract_positions(declaration)
            positions = [(start, end) for start, end in positions if start is not None and end is not None]
            if positions:
                start_pos = min(pos[0] for pos in positions)
                end_pos = max(pos[1] for pos in positions)
                whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2 = process_lean_file(file_content, start_pos, end_pos)
            else:
                start_pos=None
                end_pos =None
                whole, whole_line_1, whole_column_1, whole_line_2, whole_column_2, whole_char_idx_1, whole_char_idx_2=None,None,None,None,None,None,None
            declaration_info={
                "kind": kind,
                "content":whole,
                "whole_start_pos" : {
                    "position" : whole_char_idx_1,
                    "line" : whole_line_1,
                    "column" : whole_column_1
                },
                "whole_end_pos" :{
                    "position" : whole_char_idx_2,
                    "line" : whole_line_2,
                    "column" : whole_column_2
                }
            }
            declarations_info.append(declaration_info)

    return dict(
        tactics=tactics,
        premises=premises,
        declarations=declarations_info,
    )