#include "ImgPcAlg.h"

QString MSVNAME = "MSV",
    NIPCNAME = "NIPC",
    ZNCCNAME = "ZNCC";

const int factor = 1;

PreTreatClass<PreTreatMethod::Classic> globalScheme;
const double threshold = 0.25;

void applyThreshold(cv::UMat& m, double ratio)
{
    double maxVal;
    cv::minMaxLoc(m, nullptr, &maxVal);
    cv::threshold(m, m, maxVal * ratio, 0, cv::THRESH_TOZERO);
}

cv::UMat preTreat(const cv::UMat& src, PreTreatClass<PreTreatMethod::Classic> Scheme)
{
    cv::UMat grad1, grad2, totalGrad;

    // 定义卷积核 (留在 CPU 定义，但 filter2D 会将其上传并应用于 GPU 上的 src)
    cv::Mat k1 = (cv::Mat_<float>(2,2) << 1, 0, 0, -1);
    cv::Mat k2 = (cv::Mat_<float>(2,2) << 0, 1, -1, 0);

    // 利用 GPU 进行卷积运算
    // ddepth 使用 CV_32F 以处理减法产生的负值
    cv::filter2D(src, grad1, CV_32F, k1, cv::Point(0,0));
    cv::filter2D(src, grad2, CV_32F, k2, cv::Point(0,0));

    // 计算绝对值之和
    cv::absdiff(grad1, cv::Scalar::all(0), grad1);
    cv::absdiff(grad2, cv::Scalar::all(0), grad2);
    cv::add(grad1, grad2, totalGrad);

    applyThreshold(totalGrad, threshold);

    return totalGrad;
}

BaseAlg::BaseAlg(cv::InputArray img, int f) : m_factor(std::max(1, f))
{
    if (img.empty()) throw std::invalid_argument("Reference image is empty.");
    img.getUMat().convertTo(m_refImg, CV_32F);          // 转换图像数据类型为浮点数
    cv::normalize(m_refImg, m_refImg, 0, 255, cv::NORM_MINMAX, CV_32F);     // 将图像拉伸至[0, 255]
    cv::UMat tmpImg = preTreat(m_refImg, globalScheme);           // 图像预处理
    downsample(tmpImg, m_downRef);          // 获得下采样数据
}

void BaseAlg::downsample(const cv::UMat& src, cv::UMat& dst) const
{
    if (m_factor > 1) {
        cv::resize(src, dst, cv::Size(src.cols / m_factor, src.rows / m_factor), 0, 0, cv::INTER_AREA);
    }
    else {
        src.copyTo(dst);
    }
}

cv::UMat BaseAlg::prepareInput(cv::InputArray input) const
{
    cv::UMat in = input.getUMat();
    if (in.size() != m_refImg.size()) throw std::invalid_argument("Input size mismatch.");
    cv::normalize(in, in, 0, 255, cv::NORM_MINMAX, CV_32F);
    return in;
}

// NIPC: 归一化图像相位相关
NIPCAlg::NIPCAlg(cv::InputArray img, int f)
    : BaseAlg(img, f)
{
    m_refNorm = cv::norm(m_downRef, cv::NORM_L2);
    if (m_refNorm < 1e-9) throw std::runtime_error("Reference image is invalid (too dark).");
}

double NIPCAlg::process(cv::InputArray input) const
{
    ensureInputNotEmpty(input);         // 确保process接受的输入非空
    cv::UMat downInput;
    cv::UMat img = prepareInput(input); // 检查输入的图像尺寸是否与参考图像一致，返回拉伸至[0, 255]的input
    downsample(preTreat(img, globalScheme), downInput);     // 预处理和下采样
    double inNorm = cv::norm(downInput, cv::NORM_L2);       // 计算下采样后的欧几里得范数
    if (inNorm < 1e-9) return 0.0;
    return m_downRef.dot(downInput) / (m_refNorm * inNorm); // 计算NIPC
}

// ZNCC: 零均值归一化互相关 (优化 GPU 提取分数)
double ZNCCAlg::process(cv::InputArray input) const
{
    ensureInputNotEmpty(input);
    cv::UMat downInput;
    cv::UMat img(prepareInput(input));
    downsample(preTreat(img, globalScheme), downInput);

    cv::UMat result;
    cv::matchTemplate(downInput, m_downRef, result, cv::TM_CCOEFF_NORMED);

    // 关键优化：直接使用 minMaxLoc 拿结果，避免显存到内存的碎片拷贝
    double maxVal;
    cv::minMaxLoc(result, nullptr, &maxVal);
    return std::isnan(maxVal) ? 0.0 : maxVal;
}

// MSV: 平均绝对差
double MSVAlg::process(cv::InputArray input) const
{
    ensureInputNotEmpty(input);         // 确保process接受的输入非空
    cv::UMat in = prepareInput(input);  // 检查尺寸并拉伸
    return cv::norm(m_refImg, in, cv::NORM_L1) / static_cast<double>(m_refImg.total());
}
