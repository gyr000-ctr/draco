
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was draco-config.cmake.template                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include("${CMAKE_CURRENT_LIST_DIR}/draco-targets.cmake")

if(OFF)
  if(NOT TARGET draco::pmc)
    add_library(draco::pmc STATIC IMPORTED)
    set_target_properties(draco::pmc PROPERTIES
      IMPORTED_LOCATION "${PACKAGE_PREFIX_DIR}/lib/"
    )
  endif()
  if(TARGET draco::draco_static)
    target_link_libraries(draco::draco_static INTERFACE draco::pmc)
  elseif(NOT  AND TARGET draco::draco)
    target_link_libraries(draco::draco INTERFACE draco::pmc)
  endif()
endif()
