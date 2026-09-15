import xml.etree.ElementTree as ET
import sys
import re
import glob
import os
import warnings

MAP_TOC_LEVEL_TO_HEADING_SYMBOLS = {0: "=", 1: "-", 2: "^", 3: '"'}


if __name__ == "__main__":

    assert len(sys.argv) > 1, "No XML directory passed."
    assert len(sys.argv) > 2, "No output directory passed."
    xml_directory = sys.argv[1]
    output_directory = sys.argv[2]

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    assert os.path.exists(output_directory), "Output directory does not exist and could not be created."

    index_xml_filename = os.path.join(xml_directory, "index.xml")
    assert os.path.exists(
        index_xml_filename
    ), "index.xml not found in directory {}".format(xml_directory)

    index_tree = ET.parse(index_xml_filename)
    index_root = index_tree.getroot()

    group_filenames = []
    map_group_ids_to_group_names = {}

    for node in index_root.iter():
        if node_kind := node.attrib.get("kind", "") == "group":
            refid = node.attrib.get("refid", "")
            group_filename = os.path.join(xml_directory, refid + ".xml")
            if os.path.exists(group_filename):
                group_filenames.append(group_filename)
            else:
                warnings.warn(
                    "Expected the file: {} to exist but it does not.".format(
                        group_filename
                    )
                )
            for child in node:
                if child.tag == "name":
                    map_group_ids_to_group_names[refid] = child.text

    index_root = None
    index_tree = None

    map_groups_to_inner_groups = {}
    map_id_to_roots = {}
    groups_with_parents = set()
    map_group_ids_to_titles = {}

    for group_filename in group_filenames:
        print("Found:", group_filename)
        group_tree = ET.parse(group_filename)
        group_root = group_tree.getroot()

        group_node = None
        group_id = None
        for gx in group_root.iter():
            if gx.attrib.get("kind", "") == "group":
                group_node = gx
                group_id = gx.attrib.get("id")
                if group_id in map_groups_to_inner_groups.keys():
                    warnings.warn(
                        "Group with id {} already seen.".format(group_id)
                    )
                else:
                    map_groups_to_inner_groups[group_id] = []
                    map_id_to_roots[group_id] = group_node
                break

        if group_node is None:
            warnings.warn("Could not find group inside group xml file.")

        for child in group_node:
            if child.tag == "innergroup":
                child_id = child.attrib["refid"]
                map_groups_to_inner_groups[group_id].append(child_id)
                groups_with_parents.add(child_id)
            if child.tag == "title":
                map_group_ids_to_titles[group_id] = child.text

    def recurse_groups(group_id, level, rst_source):

        group_root = map_id_to_roots[group_id]
        assert group_root.attrib["id"] == group_id, "group_root != group_id"

        for child in group_root.iter():
            if child.attrib.get("kind", "") == "group":

                child_group_id = child.attrib["id"]
                title = map_group_ids_to_titles[child_group_id]
                symbol = MAP_TOC_LEVEL_TO_HEADING_SYMBOLS.get(level, '"')
                underline = len(title) * symbol

                rst_source += """
{}
{}
.. doxygengroup:: {}
  :content-only:
""".format(
                    title,
                    underline,
                    map_group_ids_to_group_names[child_group_id],
                )

        for child_id in map_groups_to_inner_groups[group_id]:
            rst_source = recurse_groups(child_id, level + 1, rst_source)
        return rst_source

    for group_id in map_groups_to_inner_groups.keys():
        if not group_id in groups_with_parents:
            rst_source = ""
            print("Generating:", group_id)
            rst_source = recurse_groups(group_id, 0, rst_source)
            print("--------------")
            print(rst_source)
            print("~~~~~~~~~~~~~~")
        
            output_file = os.path.join(output_directory, group_id + ".rst")
            with open(output_file, "w") as fh:
                fh.write(rst_source)




