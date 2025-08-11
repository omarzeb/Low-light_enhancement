#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

static inline double calcMeanLuminanceBGR(const cv::Mat& imgBGR) {
    CV_Assert(imgBGR.type() == CV_8UC3);
    cv::Scalar bgrMean = cv::mean(imgBGR);
    // Normalize 0..1 and convert BGR -> luminance
    double B = bgrMean[0] / 255.0;
    double G = bgrMean[1] / 255.0;
    double R = bgrMean[2] / 255.0;
    return 0.299 * R + 0.587 * G + 0.114 * B;
}

static inline double calcAlpha(double p0, double mean) {
    const double p1 = -0.018;
    return p0 + (p1 * mean);
}

// Percentile for CV_8U channel via histogram (fast, no sorting)
static inline double percentileU8(const cv::Mat& u8, double percentile) {
    CV_Assert(u8.type() == CV_8UC1);
    percentile = std::min(std::max(percentile, 0.0), 100.0);
    int hist[256] = {0};

    if (u8.isContinuous()) {
        const uchar* p = u8.ptr<uchar>(0);
        size_t n = static_cast<size_t>(u8.total());
        for (size_t i = 0; i < n; ++i) hist[p[i]]++;
    } else {
        for (int r = 0; r < u8.rows; ++r) {
            const uchar* p = u8.ptr<uchar>(r);
            for (int c = 0; c < u8.cols; ++c) hist[p[c]]++;
        }
    }

    const size_t n = static_cast<size_t>(u8.total());
    const size_t rank = static_cast<size_t>(std::round((percentile / 100.0) * (n - 1)));
    size_t cum = 0;
    for (int v = 0; v < 256; ++v) {
        cum += static_cast<size_t>(hist[v]);
        if (cum > rank) return static_cast<double>(v);
    }
    return 255.0;
}

// Percentile for float channel using nth_element (O(n))
static inline float percentileFloat(const cv::Mat& f32, double percentile) {
    CV_Assert(f32.type() == CV_32F);
    percentile = std::min(std::max(percentile, 0.0), 100.0);
    const int total = f32.rows * f32.cols;
    std::vector<float> vals;
    vals.reserve(total);

    if (f32.isContinuous()) {
        const float* p = f32.ptr<float>(0);
        vals.insert(vals.end(), p, p + total);
    } else {
        for (int r = 0; r < f32.rows; ++r) {
            const float* p = f32.ptr<float>(r);
            vals.insert(vals.end(), p, p + f32.cols);
        }
    }

    int idx = static_cast<int>(std::round((percentile / 100.0) * (vals.size() - 1)));
    std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
    return vals[idx];
}

// Enhancement using robust percentiles and temporally smoothed per-channel gains
static inline void enhancement(
    const cv::Mat& imgBGR, double alpha,
    cv::Mat& outBGR,
    cv::Vec3d& prevScales, bool hasPrevScales,
    double gainSmooth = 0.85, double percentile = 99.5
) {
    CV_Assert(imgBGR.type() == CV_8UC3);
    std::vector<cv::Mat> ch(3);
    cv::split(imgBGR, ch);

    double vb = percentileU8(ch[0], percentile);
    double vg = percentileU8(ch[1], percentile);
    double vr = percentileU8(ch[2], percentile);

    const double eps = 1e-6;
    const double EbCurr = alpha * (160.4 / (vb + 15.81 + eps));
    const double EgCurr = alpha * (179.3 / (vg + 15.42 + eps));
    const double ErCurr = alpha * (170.7 / (vr + 15.49 + eps));

    cv::Vec3d scales;
    if (!hasPrevScales) {
        scales = cv::Vec3d(EbCurr, EgCurr, ErCurr);
    } else {
        const double s = std::min(std::max(gainSmooth, 0.0), 0.999);
        scales[0] = s * prevScales[0] + (1.0 - s) * EbCurr;
        scales[1] = s * prevScales[1] + (1.0 - s) * EgCurr;
        scales[2] = s * prevScales[2] + (1.0 - s) * ErCurr;
    }
    prevScales = scales;

    cv::Mat chOut[3];
    cv::convertScaleAbs(ch[0], chOut[0], scales[0], 0);
    cv::convertScaleAbs(ch[1], chOut[1], scales[1], 0);
    cv::convertScaleAbs(ch[2], chOut[2], scales[2], 0);
    cv::merge(chOut, 3, outBGR);
}

// Simple color balance on float channel -> uint8 (clip percentiles and rescale to [0,255])
static inline void simpleColorBalancePerChannel(
    const cv::Mat& channelF32, float lowClip, float highClip, cv::Mat& outU8
) {
    CV_Assert(channelF32.type() == CV_32F);
    if (lowClip <= 0.0f && highClip <= 0.0f) {
        double minv, maxv;
        cv::minMaxLoc(channelF32, &minv, &maxv);
        if (maxv <= minv) {
            outU8 = cv::Mat::zeros(channelF32.size(), CV_8U);
            return;
        }
        cv::Mat scaled;
        channelF32.convertTo(scaled, CV_32F, 255.0 / (maxv - minv), -static_cast<float>(minv) * 255.0f / static_cast<float>(maxv - minv));
        scaled.convertTo(outU8, CV_8U, 1.0, 0.0);
        return;
    }

    double lowP = 100.0 * std::min(std::max(static_cast<double>(lowClip), 0.0), 0.49);
    double highP = 100.0 * (1.0 - std::min(std::max(static_cast<double>(highClip), 0.0), 0.49));

    float lo = percentileFloat(channelF32, lowP);
    float hi = percentileFloat(channelF32, highP);
    if (hi <= lo) {
        cv::Mat clipped;
        cv::min(channelF32, 255.0f, clipped);
        cv::max(clipped, 0.0f, clipped);
        clipped.convertTo(outU8, CV_8U, 1.0, 0.0);
        return;
    }

    cv::Mat clipped = channelF32.clone();
    cv::threshold(clipped, clipped, hi, hi, cv::THRESH_TRUNC);
    cv::threshold(clipped, clipped, lo, lo, cv::THRESH_TOZERO);
    cv::Mat scaled;
    clipped.convertTo(scaled, CV_32F, 255.0 / (hi - lo), -lo * (255.0f / (hi - lo)));
    scaled.convertTo(outU8, CV_8U, 1.0, 0.0);
}

// MSRCR implementation for BGR uint8
static inline cv::Mat msrcr(
    const cv::Mat& imgBGR,
    const std::vector<double>& sigmas = {15.0, 80.0, 250.0},
    const std::vector<double>& weightsIn = {},
    double alpha = 125.0,
    double beta = 46.0,
    double gain = 1.0,
    double offset = 0.0,
    float lowClip = 0.01f,
    float highClip = 0.01f
) {
    CV_Assert(imgBGR.type() == CV_8UC3);
    cv::Mat imgF;
    imgBGR.convertTo(imgF, CV_32F, 1.0, 0.0);

    std::vector<cv::Mat> ch(3);
    cv::split(imgF, ch);

    std::vector<double> weights;
    if (weightsIn.empty()) {
        weights.assign(sigmas.size(), 1.0 / std::max<size_t>(1, sigmas.size()));
    } else {
        double sumw = std::accumulate(weightsIn.begin(), weightsIn.end(), 0.0);
        if (sumw <= 0.0) weights.assign(sigmas.size(), 1.0 / std::max<size_t>(1, sigmas.size()));
        else {
            weights.resize(weightsIn.size());
            for (size_t i = 0; i < weightsIn.size(); ++i) weights[i] = weightsIn[i] / sumw;
        }
    }

    const float eps = 1e-6f;
    cv::Mat Bp = ch[0] + 1.0f, Gp = ch[1] + 1.0f, Rp = ch[2] + 1.0f;

    auto retinexMulti = [&](const cv::Mat& channelPlus) -> cv::Mat {
        cv::Mat logChannel;
        cv::log(channelPlus, logChannel);
        cv::Mat ret = cv::Mat::zeros(channelPlus.size(), CV_32F);
        cv::Mat blurred, logBlur, tmp;
        for (size_t i = 0; i < sigmas.size(); ++i) {
            cv::GaussianBlur(channelPlus, blurred, cv::Size(0, 0), sigmas[i]);
            blurred += eps;
            cv::log(blurred, logBlur);
            cv::subtract(logChannel, logBlur, tmp);
            ret += static_cast<float>(weights[i]) * tmp;
        }
        return ret;
    };

    cv::Mat Rr = retinexMulti(Rp);
    cv::Mat Gr = retinexMulti(Gp);
    cv::Mat Br = retinexMulti(Bp);

    cv::Mat sumRGB = Rp + Gp + Bp;
    cv::Mat logSum, logRp, logGp, logBp;
    cv::log(sumRGB, logSum);

    cv::log(Rp * static_cast<float>(alpha), logRp);
    cv::log(Gp * static_cast<float>(alpha), logGp);
    cv::log(Bp * static_cast<float>(alpha), logBp);

    cv::Mat Rc = static_cast<float>(beta) * (logRp - logSum);
    cv::Mat Gc = static_cast<float>(beta) * (logGp - logSum);
    cv::Mat Bc = static_cast<float>(beta) * (logBp - logSum);

    cv::Mat Rmsrcr = static_cast<float>(gain) * (Rc.mul(Rr)) + static_cast<float>(offset);
    cv::Mat Gmsrcr = static_cast<float>(gain) * (Gc.mul(Gr)) + static_cast<float>(offset);
    cv::Mat Bmsrcr = static_cast<float>(gain) * (Bc.mul(Br)) + static_cast<float>(offset);

    cv::Mat outB, outG, outR;
    simpleColorBalancePerChannel(Bmsrcr, lowClip, highClip, outB);
    simpleColorBalancePerChannel(Gmsrcr, lowClip, highClip, outG);
    simpleColorBalancePerChannel(Rmsrcr, lowClip, highClip, outR);

    cv::Mat out;
    std::vector<cv::Mat> mergeCh = {outB, outG, outR};
    cv::merge(mergeCh, out);
    return out;
}

int main(int argc, char** argv) {
    // Fast paths
    cv::setUseOptimized(true);
    cv::ocl::setUseOpenCL(true);
    try {
        int nthreads = std::max(1, cv::getNumberOfCPUs());
        cv::setNumThreads(nthreads);
    } catch (...) {}

    // Parameters (mirroring Python defaults)
    int cameraIndex = 0;
    double p0 = 1.6;
    int width = 640;
    int height = 480;
    int targetFps = 30;
    bool fastMean = true;
    int meanDownsample = 8;
    int alphaUpdateInterval = 3;
    double alphaSmooth = 0.9;
    double alphaMin = 0.8;
    double alphaMax = 2.5;
    double gainSmooth = 0.85;
    double gainPercentile = 99.5;
    bool statsBlur = true;
    double statsBlurSigma = 1.0;
    bool useMSRCR = true; // true: MSRCR, false: percentile method

    std::vector<double> retinexSigmas = {15.0, 80.0, 250.0};
    std::vector<double> retinexWeights; // equal by default
    double retinexAlpha = 125.0, retinexBeta = 46.0, retinexGain = 1.0, retinexOffset = 0.0;
    float retinexLowClip = 0.01f, retinexHighClip = 0.01f;

    // Optional simple CLI to switch algorithm: pass "perc" to use percentile method
    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "perc" || mode == "PERC" || mode == "percentile") useMSRCR = false;
        if (mode == "msrcr" || mode == "MSRCR") useMSRCR = true;
    }

    cv::VideoCapture cap;
#ifdef _WIN32
    cap.open(cameraIndex, cv::CAP_DSHOW);
    if (!cap.isOpened()) cap.open(cameraIndex, cv::CAP_MSMF);
#else
    cap.open(cameraIndex, cv::CAP_V4L2);
    if (!cap.isOpened()) cap.open(cameraIndex);
#endif
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera." << std::endl;
        return 1;
    }

#ifdef _WIN32
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
#else
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
#endif

    if (width > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    if (height > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    if (targetFps > 0) cap.set(cv::CAP_PROP_FPS, targetFps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    int actualW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actualH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actualFPS = cap.get(cv::CAP_PROP_FPS);

    std::string windowName = useMSRCR ? "Enhanced (MSRCR) - q to quit" : "Enhanced (Percentile) - q to quit";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, std::min(actualW, 1280), std::min(actualH, 720));

    auto prevTime = std::chrono::high_resolution_clock::now();
    cv::Mat outBuffer;
    cv::Vec3d prevScales(1.0, 1.0, 1.0);
    bool hasPrevScales = false;
    double prevAlpha = -1.0;
    double alpha = 1.0;
    int frameCount = 0;

    for (;;) {
        cv::Mat frameBGR;
        if (!cap.read(frameBGR) || frameBGR.empty()) {
            cv::Mat placeholder((height > 0 ? height : 480),
                                (width > 0 ? width : 640),
                                CV_8UC3, cv::Scalar(0,0,0));
            cv::putText(placeholder, "No frame from camera", {20,40}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,255}, 2);
            cv::imshow(windowName, placeholder);
            if (cv::waitKey(1) == 'q') break;
            continue;
        }

        if (!useMSRCR) {
            if (frameCount % alphaUpdateInterval == 0) {
                cv::Mat small = frameBGR;
                if (fastMean && meanDownsample > 1) {
                    cv::resize(
                        frameBGR, small,
                        cv::Size(std::max(1, frameBGR.cols / meanDownsample),
                                 std::max(1, frameBGR.rows / meanDownsample)),
                        0, 0, cv::INTER_AREA
                    );
                }
                if (statsBlur && statsBlurSigma > 0.0) {
                    cv::GaussianBlur(small, small, cv::Size(0,0), statsBlurSigma);
                }
                double meanValue = calcMeanLuminanceBGR(small);
                double alphaRaw = calcAlpha(p0, meanValue);
                alphaRaw = std::max(alphaMin, std::min(alphaRaw, alphaMax));
                if (prevAlpha < 0.0) alpha = alphaRaw;
                else {
                    const double s = std::min(std::max(alphaSmooth, 0.0), 0.999);
                    alpha = s * prevAlpha + (1.0 - s) * alphaRaw;
                }
                prevAlpha = alpha;
            }
        }

        if (useMSRCR) {
            outBuffer = msrcr(frameBGR, retinexSigmas, retinexWeights,
                              retinexAlpha, retinexBeta, retinexGain, retinexOffset,
                              retinexLowClip, retinexHighClip);
        } else {
            enhancement(frameBGR, alpha, outBuffer, prevScales, hasPrevScales, gainSmooth, gainPercentile);
            hasPrevScales = true;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double fps = 1.0 / std::max(1e-6, std::chrono::duration<double>(now - prevTime).count());
        prevTime = now;

        char info[256];
        std::snprintf(info, sizeof(info), "%dx%d @%.0f alg=%s fps=%.1f",
                      actualW, actualH, actualFPS, (useMSRCR ? "MSRCR" : "PERC"), fps);
        cv::putText(outBuffer, info, {10,30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,0}, 2);

        cv::imshow(windowName, outBuffer);
        frameCount++;
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}