if (FILE_TO_REMOVE)
    if (EXISTS ${FILE_TO_REMOVE})
        message(STATUS "File to remove exists, attempting to remove: "
            ${FILE_TO_REMOVE})
        file(REMOVE ${FILE_TO_REMOVE})
    endif()
endif()

