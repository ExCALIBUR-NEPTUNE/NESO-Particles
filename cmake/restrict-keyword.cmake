# try to find an acceptable CXX restrict keyword if not user provided
INCLUDE(CheckCXXSourceCompiles)

if(NOT DEFINED RESTRICT)
    message(STATUS "Detecting restrict keyword")
    check_cxx_source_compiles(
        "
        int f(void *restrict x);
        int main(void) {return 0;}
        "
        HAVERESTRICT
    )
    check_cxx_source_compiles(
        "
        int f(void * __restrict x);
        int main(void) {return 0;}
        "
        HAVE__RESTRICT
    )
    check_cxx_source_compiles(
        "
        int f(void * __restrict__ x);
        int main(void) {return 0;}
        "
        HAVE__RESTRICT__
    )

    if (HAVERESTRICT)
        set(RESTRICT "restrict")
    elseif(HAVE__RESTRICT)
        set(RESTRICT "__restrict")
    elseif(HAVE__RESTRICT__)
        set(RESTRICT "__restrict__")
    else()
        set(RESTRICT "")
    endif()
endif()

message(STATUS "Using restrict keyword: " ${RESTRICT})
