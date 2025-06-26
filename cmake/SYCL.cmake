
# This is a fix for an issue where the AdaptiveCpp nvc++ backend fails to link
# the libNESO-Particles.so library with an error of "/usr/bin/ld: input file
# 'libNESO-Particles.so' is the same as output file" when libNESO-Particles.so
# already exists in the build directory. The issue is possible due to bad
# arguments or bad argument order when nvc++ is called via acpp/cmake.
#
# What we do is detect if the linker is GNU ld which we know also accepts
# -soname= and replace the -soname <foo> with -soname=<foo> as the linker
# argument that sets the soname.
macro(fix_linker_for_nvcxx)
    # Test if the linker is GNU, or assume it is if we are pre cmake 3.29
    set(NESO_PARTICLES_LINKER_IS_GNU TRUE)
    if(DEFINED CMAKE_CXX_COMPILER_LINKER_ID)
      if(NOT CMAKE_CXX_COMPILER_LINKER_ID STREQUAL "GNU")
        set(NESO_PARTICLES_LINKER_IS_GNU FALSE)
      endif()
    endif()
    # Replace the linker args if the linker is GNU and the soname args are
    # -Wl,-soname
    if(NESO_PARTICLES_LINKER_IS_GNU AND (CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG
                                         STREQUAL "-Wl,-soname,"))
      message(STATUS "Setting CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG to -Wl,-soname=")
      set(CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Wl,-soname=")
    else()
      message(STATUS "Keeping existing CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG")
    endif()
endmacro()

#Macro to find a sycl implimentation
#Called without arguments will try to find a compiler appropriate sycl
#With one argument finds that specific implimentation
#With 2 arguments is as above but second argument is treated like version
#Not happy with this yet - 
# - should be able to accept lists of implementaions and versions
# - and target architectures CPU,GPU etc
macro(find_sycl)
    set(SYCL_FOUND FALSE )
    if (${ARGC} EQUAL 0)
        #If intel compiler try to find the intel 
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            #TODO IntelDPCPP is deprecated remove
            list(APPEND candidates "IntelSYCL" "IntelDPCPP")
            foreach(candidate IN LISTS candidates)
                find_package(${candidate} QUIET)
                if(${candidate}_FOUND)
                    set(SYCL_FOUND TRUE )
                    set(SYCL_IMPLEMENTATION ${candidate} )
                    break()
                endif()
            endforeach()
        else()
            #not intel try one of these
            list(APPEND candidates "AdaptiveCPP" "hipSYCL")
            list(APPEND versions   ""            "0.9.4")
            foreach(candidate version IN ZIP_LISTS candidates versions)
                find_package(${candidate} ${version} QUIET)
                if(${candidate}_FOUND)
                    set(SYCL_FOUND TRUE )
                    set(SYCL_IMPLEMENTATION ${candidate} )
                    fix_linker_for_nvcxx()
                    break()
                endif()
            endforeach()
        endif()
    elseif (${ARGC} EQUAL 1)
        find_package(${ARGV0} REQUIRED)
        if(${ARGV0}_FOUND)
            set(SYCL_FOUND TRUE )
            set(SYCL_IMPLEMENTATION ${ARGV0} )
        endif()
    elseif (${ARGC} EQUAL 2)
        find_package(${ARGV0} ${ARGV1} REQUIRED)
        if(${ARGV0}_FOUND)
            set(SYCL_FOUND TRUE )
            set(SYCL_IMPLEMENTATION ${ARGV0} )
        endif()
    endif()
    if (NOT COMMAND add_sycl_to_target) 
        function(add_sycl_to_target)
        endfunction()
    endif()
    if (SYCL_FOUND) 
        message(STATUS "Found SYCL implimentation ${SYCL_IMPLEMENTATION}")
    elseif()
        message(WARNING "No SYCL implimentation found - "
            "Proceeding on the assumption that the CXX compiler is a SYCL2020 compiler")
    endif()
endmacro()

macro(find_sycl_if_required)
    if (COMMAND add_sycl_to_target) 
        #Not perfect method but if in a submodule dont want to use a different
        #sycl
        message(STATUS "add_sycl_to_target macro exists")
    else()
        message(STATUS "add_sycl_to_target macro NOT FOUND looking for SYCL implementation...")
        find_sycl()
    endif()
endmacro()

