# Install script for directory: /Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/build/solver_code/lib/qdldl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/build/solver_code/lib/amd/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/build/out/libqocostatic.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libqocostatic.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libqocostatic.a")
    execute_process(COMMAND "/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libqocostatic.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/build/out/libqoco.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libqoco.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libqoco.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/build/solver_code/lib/qdldl/out"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libqoco.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libqoco.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/qoco" TYPE FILE FILES
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/qoco.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/qoco_api.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/input_validation.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/qoco_linalg.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/kkt.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/cone.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/qoco_status.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/equilibration.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/enums.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/definitions.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/structs.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/timer.h"
    "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/solver_code/include/qoco_utils.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/su781/Desktop/gitclonescp/TrajOpt_CT-STL/di_until_aug_solver/c/build/solver_code/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
