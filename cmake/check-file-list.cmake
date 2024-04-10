# Function to compare files found using a recursive glob for an extension in a
# directory against a given list. The ignore files parameter indicates files in
# the directory which are not expected to be found by the glob.
function(CHECK_ADDED_FILE_LIST SRC_DIR EXT LISTED_FILES IGNORE_FILES)

  file(GLOB_RECURSE FOUND_FILES CONFIGURE_DEPENDS ${SRC_DIR}/*.${EXT})
  list(REMOVE_ITEM FOUND_FILES ${LISTED_FILES})
  list(REMOVE_ITEM FOUND_FILES ${IGNORE_FILES})

  list(LENGTH FOUND_FILES FOUND_FILES_LENGTH)
  if (${FOUND_FILES_LENGTH})
    message(WARNING "The following files are not included in a cmake list:")
    foreach(FILEX ${FOUND_FILES})
      message(${FILEX})
    endforeach()
    message(WARNING "End of list.")
  endif()

endfunction()
