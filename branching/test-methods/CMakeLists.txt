cmake_minimum_required(VERSION 3.3)

project(sparse)

set(SCIP_DIR /Users/selinbayramoglu/opt/anaconda3/envs/branching-env/lib/cmake/scip)
#set(DYLD_LIBRARY_PATH /Users/selinbayramoglu/opt/anaconda3/envs/branching-env/lib)

if(TARGET SCIP::SCIP)
  # find package by SCIP PATH
  find_package(SCIP CONFIG PATHS ${SCIP_BINARY_DIR} REQUIRED)
else()
  find_package(SCIP REQUIRED)
endif()

include_directories(${SCIP_INCLUDE_DIRS})
message( ${SCIP_INCLUDE_DIRS} )

add_executable(sparse
   src/branch_sparse.c
   src/cmain.c)

# link to math library if it is available
find_library(LIBM m)
if(NOT LIBM)
  set(LIBM "")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -Wextra -Wshadow -O2")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -Wshadow -O3")
set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -Wshadow -O0 -g")
set(CMAKE_VERBOSE_MAKEFILE)

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
message("CMAKE_C_FLAGS_DEBUG is ${CMAKE_C_FLAGS_DEBUG}")
message("CMAKE_C_FLAGS_RELEASE is ${CMAKE_C_FLAGS_RELEASE}")

target_link_libraries(sparse ${SCIP_LIBRARIES} ${LIBM})
