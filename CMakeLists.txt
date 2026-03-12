cmake_minimum_required(VERSION 3.19)
project(DIP_1_2 LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

qt_standard_project_setup()

qt_add_executable(DIP_1_2
    WIN32 MACOSX_BUNDLE
    main.cpp
    mainwindow.cpp
    mainwindow.h
    ImgPcAlg.h
    ImgPcAlg.cpp
    ImgPcAlg_2.cpp
    task.h task.cpp
    roi.h roi.cpp
)

target_link_libraries(DIP_1_2
    PRIVATE
        Qt::Core
        Qt::Widgets
        ${OpenCV_LIBS}
)
target_link_libraries(DIP_1_2 PRIVATE Qt6::Widgets)
target_link_libraries(DIP_1_2 PRIVATE Qt6::Core)

target_include_directories(DIP_1_2
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

include(GNUInstallDirs)

install(TARGETS DIP_1_2
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET DIP_1_2
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
