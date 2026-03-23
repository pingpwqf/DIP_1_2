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
    const int W = src.cols - 1,
        H = src.rows - 1;

    // f(x+1, y+1) 和 f(x,y)
    cv::Rect r1(1, 1, W, H);
    cv::Rect r_tl(0, 0, W, H);

    // f(x+1, y) 和 f(x, y+1)
    cv::Rect r_x1(1, 0, W, H);
    cv::Rect r_y1(0, 1, W, H);

    cv::UMat diff1, diff2, grad;

    // 计算 |f(x,y) - f(x+1,y+1)|
    cv::absdiff(src(r_tl), src(r1), diff1);

    // 计算 |f(x+1,y) - f(x,y+1)|
    cv::absdiff(src(r_x1), src(r_y1), diff2);

    // 求和
    cv::add(diff1, diff2, grad);
    applyThreshold(grad, threshold);

    return grad;
}

BaseAlg::BaseAlg(cv::InputArray img, int f) : m_factor(std::max(1, f)) {
    if (img.empty()) throw std::invalid_argument("Reference image is empty.");
    img.getUMat().convertTo(m_refImg, CV_32F);          //转换图像数据类型为浮点数
    cv::normalize(m_refImg, m_refImg, 0, 255, cv::NORM_MINMAX, CV_32F);     //将图像拉伸至[0, 255]
    cv::UMat tmpImg = preTreat(m_refImg, globalScheme);           //图像预处理
    downsample(tmpImg, m_downRef);          //获得下采样数据
}

void BaseAlg::downsample(const cv::UMat& src, cv::UMat& dst) const {
    if (m_factor > 1) {
        cv::resize(src, dst, cv::Size(src.cols / m_factor, src.rows / m_factor), 0, 0, cv::INTER_AREA);
    }
    else {
        src.copyTo(dst);
    }
}

cv::UMat BaseAlg::prepareInput(cv::InputArray input) const {
    cv::UMat in = input.getUMat();
    if (in.size() != m_refImg.size()) throw std::invalid_argument("Input size mismatch.");
    cv::normalize(in, in, 0, 255, cv::NORM_MINMAX, CV_32F);
    return in;
}

// NIPC: 归一化图像相位相关
NIPCAlg::NIPCAlg(cv::InputArray img, int f) : BaseAlg(img, f) {
    m_refNorm = cv::norm(m_downRef, cv::NORM_L2);
    if (m_refNorm < 1e-9) throw std::runtime_error("Reference image is invalid (too dark).");
}

double NIPCAlg::process(cv::InputArray input) const {
    ensureInputNotEmpty(input);
    cv::UMat downInput;
    cv::UMat img = prepareInput(input);
    downsample(preTreat(img, globalScheme), downInput);
    double inNorm = cv::norm(downInput, cv::NORM_L2);
    if (inNorm < 1e-9) return 0.0;
    return m_downRef.dot(downInput) / (m_refNorm * inNorm);
}

// ZNCC: 零均值归一化互相关 (优化 GPU 提取分数)
double ZNCCAlg::process(cv::InputArray input) const {
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
double MSVAlg::process(cv::InputArray input) const {
    ensureInputNotEmpty(input);
    cv::UMat in = prepareInput(input);
    return cv::norm(m_refImg, in, cv::NORM_L1) / static_cast<double>(m_refImg.total());
}
