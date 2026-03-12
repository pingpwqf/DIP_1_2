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

if(WIN32)
    # 1. 精准拷贝第三方 DLL (如 OpenCV)
    install(FILES $<TARGET_RUNTIME_DLLS:DIP_1_2>
            DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # 2. 依然保留 Qt 的部署脚本，处理复杂的插件依赖
    qt_generate_deploy_app_script(
        TARGET DIP_1_2
        OUTPUT_SCRIPT deploy_script
        NO_UNSUPPORTED_PLATFORM_ERROR
    )
    install(SCRIPT ${deploy_script})
endif()

if(WIN32)
    # 1. 定义我们要找的“特殊名单”
    set(MANIFEST_MISSING_DLLS
        "avif.dll"
        "dav1d.dll"
    )

    # 2. 在环境变量 PATH 中搜索这些 DLL
    foreach(dll ${MANIFEST_MISSING_DLLS})
        find_file(FOUND_PATH_${dll} NAMES ${dll} PATHS $ENV{PATH})

        if(FOUND_PATH_${dll})
            message(STATUS "找到关键依赖: ${FOUND_PATH_${dll}}")
            # 强制将其安装到输出目录的 bin 文件夹
            install(FILES ${FOUND_PATH_${dll}} DESTINATION ${CMAKE_INSTALL_BINDIR})
        else()
            message(WARNING "警告：仍未找到 ${dll}，请确认其是否在环境变量 PATH 中！")
        endif()
    endforeach()
endif()
