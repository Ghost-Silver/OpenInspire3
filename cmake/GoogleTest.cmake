# Wrapper around CMake's built-in GoogleTest module.
# When TEST_ENABLED is OFF (as in OpenInspire3), prevent submodules from
# registering gtest targets with CTest via gtest_discover_tests / gtest_add_tests.
if(TEST_ENABLED)
    include("${CMAKE_ROOT}/Modules/GoogleTest.cmake")
else()
    macro(gtest_add_tests)
    endmacro()
    macro(gtest_discover_tests)
    endmacro()
endif()
