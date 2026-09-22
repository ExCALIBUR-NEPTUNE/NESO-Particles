"""
This script should be called with three arguments. The first is the directory
containing the xml files produced by doxygen (i.e. ./build/doxygen/xml). The
second is the sphinx source directory in which the generated RST should be
written (i.e. ./sphinx/source/guide-user).
The third argument is the directory containing the Doxyfile. RST source files
for groups etc should be placed in the Doxyfile directory.

This script parses the XML produced by Doxygen to find "groups" (see Doxygen
documentation) and then generates RST source for each group. For each group a
RST section will be created with TOC tree structure given by the group
structure.

Below we create a group neso_particles_core. With a description that doxygen
will use provided in @details. The description provided in @np_rst_block will
be directly inserted into the generated RST instead of the @details block.
/**
 * @defgroup neso_particles_core Core Types and Functions
 * @details Here we describe the core types and functions.
 * @np_rst_block{
 * Here we describe the core types and functions of ``NESO::Particles``.
 * }
 */

We can also point to a source file for the RST as follows. This is much more robust than placing RST source in the Doxygen comments.
/**
 * @defgroup particle_pair_loop Particle Pair Loop
 * @details This section contains documentation for particle pair looping.
 * Particle pair looping is a looping type similar to particle loop except that
 * the kernel operates on two particles.
 * @np_rst_source_file{particle_pair_loop_group_description.rst}
 */
"""

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
    assert len(sys.argv) > 3, "No Doxyfile source directory passed."
    xml_directory = sys.argv[1]
    output_directory = sys.argv[2]
    doxygen_directory = sys.argv[3]

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    assert os.path.exists(
        output_directory
    ), "Output directory does not exist and could not be created."

    index_xml_filename = os.path.join(xml_directory, "index.xml")
    assert os.path.exists(
        index_xml_filename
    ), "index.xml not found in directory {}".format(xml_directory)

    index_tree = ET.parse(index_xml_filename)
    index_root = index_tree.getroot()

    group_filenames = []
    map_group_ids_to_group_names = {}

    # The index.xml defines which groups exist and what the unique ID is for
    # each group. This unique ID is the filename that contains the info for the
    # group.
    #
    # Unfortunately the sub-group (innergroup in Doxygen terminology) is not
    # given in index.xml but in the actual xml file for each group.
    #
    # Here we visit the nodes in the index.xml to find possible groups.
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

    # If a group has a parent node then the group is not a top level group and
    # hence we should not start recursion from that node to generate an RST
    # source.
    groups_with_parents = set()
    map_group_ids_to_titles = {}
    map_group_ids_to_description = {}

    # For each group found in index.xml read the corresponding XML tree and
    # discover group members and inner groups.
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

        map_group_ids_to_description[group_id] = ""
        found_rst_source = False

        for child in group_node:
            if child.tag == "innergroup":
                child_id = child.attrib["refid"]
                map_groups_to_inner_groups[group_id].append(child_id)
                groups_with_parents.add(child_id)
            if child.tag == "title":
                map_group_ids_to_titles[group_id] = child.text
            if child.tag == "detaileddescription":
                for para in child:
                    if not found_rst_source:
                        map_group_ids_to_description[group_id] += (
                            para.text.strip() + "\n"
                        )

                    for candidate_rst_source in para:
                        if (
                            candidate_rst_source.tag
                            == "NESO_PARTICLES_RST_SOURCE"
                        ):
                            found_rst_source = True
                            map_group_ids_to_description[group_id] = (
                                candidate_rst_source.text
                            )
                        elif (
                            candidate_rst_source.tag
                            == "NESO_PARTICLES_RST_SOURCE_FILE"
                        ):
                            filename = candidate_rst_source.text.strip()
                            filename = os.path.join(doxygen_directory, filename)
                            file_exists = os.path.exists(filename)
                            if not file_exists:
                                warning.warn(
                                    "Source Doxygen lists a file containing rst source ({}) but no file was found.".format(
                                        filename
                                    )
                                )
                            else:
                                print("Found:", filename)
                                t = None
                                with open(filename) as fh:
                                    t = fh.read()
                                found_rst_source = True
                                map_group_ids_to_description[group_id] = t

    # For each group generate the directive for the group node then visit all
    # the children (recursively) and generate the directives for the children.
    def recurse_groups(group_id, level, rst_source):

        group_root = map_id_to_roots[group_id]
        assert group_root.attrib["id"] == group_id, "group_root != group_id"

        for child in group_root.iter():
            if child.attrib.get("kind", "") == "group":

                child_group_id = child.attrib["id"]
                title = map_group_ids_to_titles[child_group_id]
                symbol = MAP_TOC_LEVEL_TO_HEADING_SYMBOLS.get(level, '"')
                underline = len(title) * symbol

                description = map_group_ids_to_description[child_group_id]

                rst_source += """
{}
{}

{}

.. doxygengroup:: {}
  :content-only:
  :members:
""".format(
                    title,
                    underline,
                    description,
                    map_group_ids_to_group_names[child_group_id],
                )

        for child_id in map_groups_to_inner_groups[group_id]:
            rst_source = recurse_groups(child_id, level + 1, rst_source)
        return rst_source

    # For each node that has no parent group generate an RST source file that
    # contains the directives for that node and all the child nodes.
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
