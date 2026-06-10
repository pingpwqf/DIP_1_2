### 本项目是一个专门针对光纤散斑图传感器（Fiber Specklegram Sensor, FSS）设计的跨平台图像批处理与特征分析系统。

---

## 核心功能特性

1. **统一的算法执行接口 (`AlgInterface`)**
   - 采用工厂/注册机设计模式（`AlgRegistry`），支持算法的动态注册与获取，具有极高的扩展性。
   - 所有匹配与特征提取算法统一继承自 `AlgInterface::process` 接口。

2. **丰富的图像分析算法**
   - **NIPC (归一化内积系数)**：提取并计算图像间的归一化内积系数（基于 L2 范数拉伸归一化）。
   - **ZNCC (零均值归一化互相关)**：使用 OpenCV GPU 加速（`cv::matchTemplate` 与 `cv::UMat`）实现高速、高精度的相关系数计算。
   - **MSV (平均绝对差)**：计算参考图与输入图之间的 L1 范数平均绝对误差。
   - **GLCM (灰度共生矩阵)**：基于相位谱图像（Phase Spectrum）提取纹理特征，支持多线程缓存复用计算 **GLCMcorr (相关性)** 和 **GLCMhomo (同质性)**。

3. **智能的预处理与 ROI 划取**
   - **经典预处理策略**：集成了基于 GPU 的双方向卷积边缘增强（Roberts算子变形）与最大值自适应阈值过滤。
   - **交互式 ROI (感兴趣区域) 裁剪**：支持在主界面弹窗中左键拖动自由划取 ROI，具备边界溢出限制，并支持逆向划动自动归一化处理。

4. **高效的多线程批处理架构 (`TaskManager`)**
   - 自动获取系统理想线程数，并自动限制最大线程池容量为**理想核心数的一半（最小为2）**，主动预留 CPU 资源以彻底防止 UI 界面产生卡顿。
   - 采用生产者-消费者异步架构：`ProcessingSession` 分发任务，`ProcessingTask` 在后台执行图像加载与计算。
   - **GLCM 计算优化**：在同一个图像任务内部，自动缓存相位谱和 GLCM 矩阵，避免重复计算。
   - **安全的文件刷盘机制**：`ResultCollector` 集中接收并分类保存结果，采用多线程互斥锁（`QMutex`）保护，输出为标准的 `.csv` 表格文件，且支持随时中断取消（`cancel`）并安全释放文件句柄。

5. **Windows 平台中文路径适配**
   - 针对 OpenCV 默认不支持 Windows 下含有中文或特殊字符路径的问题，项目内置了 `imread_safe` 函数，底层通过 Qt 的 `QFile` 读取二进制再由 `cv::imdecode` 进行内存解码，彻底杜绝了中文路径引起的读取失败。

---

## 开发与构建环境要求

- **C++ 标准**：C++17 及以上
- **CMake**：最低要求 VERSION 3.19
- **Qt 框架**：Qt 6.5+ (REQUIRED Components: `Core`, `Widgets`)
- **OpenCV 框架**：OpenCV 4.12 (REQUIRED Components: `core`, `imgproc`, `imgcodecs`)
- **编译器**：MSVC (Windows) / GCC (Linux) / Clang (macOS)

---

## 编译与打包布署

项目支持通过 `CMake` 自动调用 Qt 官方的部署脚本。在 Windows 平台上还会自动扫描环境变量，确保拷贝关键的第三方 DLL（如 `avif.dll`, `dav1d.dll`）。

## 1. 编译项目
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
### 2. 独立打包部署
```bash
cmake --install . --prefix "D:/DIP_Distribute"
```
