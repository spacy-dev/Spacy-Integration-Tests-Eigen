cmake_minimum_required(VERSION 3.8)

project(Spacy-Integration-Tests-Eigen VERSION 1.0.0 )

# Set path to cmake scripts
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake
    )

find_package(Spacy CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

aux_source_directory(Mock MOCK_SRC_LIST)
add_library(mocks ${MOCK_SRC_LIST})
target_compile_features(mocks PUBLIC cxx_std_14)

include_directories(${EIGEN3_INCLUDE_DIR})

aux_source_directory(Eigen SRC_LIST)
include(CTest)
enable_testing()
foreach(TEST ${SRC_LIST})
  get_filename_component(TEST_NAME ${TEST} NAME_WE)
  get_filename_component(TEST_DIR ${TEST} DIRECTORY)
  string(REGEX REPLACE "/" "_" TEST_DIR ${TEST_DIR})
  set(TEST_UNIQUE_NAME "${TEST_DIR}_${TEST_NAME}")
  add_executable(${TEST_UNIQUE_NAME} ${TEST})
  target_include_directories(${TEST_UNIQUE_NAME} PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
  target_link_libraries(${TEST_UNIQUE_NAME} PRIVATE mocks Spacy::Spacy GTest::GTest GTest::Main Threads::Threads)
add_test(${TEST_UNIQUE_NAME} ${PROJECT_BINARY_DIR}/${TEST_UNIQUE_NAME})
endforeach()
