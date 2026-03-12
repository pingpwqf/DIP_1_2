#include "task.h"
#include <QThreadPool>
#include <QFileInfo>
#include <QDebug>
#include <QCoreApplication>
#include <QMessageBox>

#include "ImgPcAlg.h"

// 辅助函数：处理 OpenCV 在 Windows 下的中文路径读取问题
cv::Mat imread_safe(const QString& path)
{
    // 1. 使用 QFile 读取二进制数据，Qt 会自动处理各种平台的路径编码（包括 Windows 的 UTF-16）
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) { return cv::Mat(); }

    QByteArray data = file.readAll();
    file.close();

    // 2. 将 QByteArray 转换为 std::vector<uchar>
    std::vector<uchar> buffer(data.begin(), data.end());

    // 3. 使用 imdecode 从内存中解码图像
    // 这种方式完全避开了 OpenCV 对文件路径字符串的平台差异处理
    return cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
}

void ProcessingTask::run()
{
    if (m_pCancelled && m_pCancelled->load()) {
        emit resultsSkipped(m_algNames.size());
        emit finished();
        return;
    }

    try {
        cv::Mat fullImg = imread_safe(m_path);
        if (fullImg.empty()) {
            emit resultsSkipped(m_algNames.size());
            emit finished();
            return;
        }

        QString fileName = QFileInfo(m_path).fileName();

        // --- 核心改动：应用 ROI ---
        cv::Mat img;
        if (m_roi.width > 0 && m_roi.height > 0) {
            img = fullImg(m_roi).clone();            // 只处理 ROI 区域
            emit imageReady(img, fileName, true);
        } else {
            img = fullImg;
            emit imageReady(img, fileName, false);
        }
        if (img.empty()) {
            emit resultsSkipped(m_algNames.size());
            emit finished();
            return;
        }

        // GLCM 缓存逻辑
        std::shared_ptr<GLCM::GLCmat> sharedGlcm = nullptr;
        bool needsGlcm = m_algNames.contains(CORRNAME) || m_algNames.contains(HOMONAME);

        if (needsGlcm) {
            sharedGlcm = GLCM::getPSGLCM(img, 32, 0, 5);
        }

        for (const QString& algName : m_algNames) {
            if (m_pCancelled && m_pCancelled->load()) break;

            std::unique_ptr<AlgInterface> alg;
            // 如果是 GLCM 类算法且有缓存
            if ((algName == CORRNAME || algName == HOMONAME) && sharedGlcm) {
                if (algName == CORRNAME) alg = std::make_unique<GLCM::GLCMcorrAlg>(sharedGlcm);
                else alg = std::make_unique<GLCM::GLCMhomoAlg>(sharedGlcm);
            } else {
                // 普通算法通过注册机获取
                alg = AlgRegistry<QString>::instance().get(algName, m_refImg);
            }

            if (alg) {
                double val = alg->process(img); // 统一调用！
                emit resultReady(algName, fileName, val);
            }
        }
    }
    catch (const std::exception& e) {
        qDebug() << "TaskError:" << e.what();
        emit resultsSkipped(m_algNames.size());
    }
    catch (...) {
        emit resultsSkipped(m_algNames.size());
    }

    emit finished();
}

/**********TaskManager************/
TaskManager::TaskManager(QString outputPath)
    : m_collector(new ResultCollector(this))
    // m_session(createSession())
{
    m_collector->setOutputDir(outputPath);
    m_collector->prepare();
}

ProcessingSession* TaskManager::execute(const BatchConfig& config)
{
    ProcessingSession* session = new ProcessingSession(m_collector);
    session->setROI(config.roi);

    connect(session, &ProcessingSession::sessionFinished,
            m_collector, &ResultCollector::closeAll);

    session->start(config.refImg, config.files, config.dir, config.algorithms);

    return session;
}


/********ProcessingSession********/
void ProcessingSession::start(const cv::Mat& refImg, const QStringList& files, const QDir& dir, const QVector<QString>& algs)
{
    m_totalTasks = files.size();
    m_activeTasks = m_totalTasks;

    if (m_totalTasks == 0) {
        emit sessionFinished();
        return;
    }

    m_collector->resetExpectedCount(m_totalTasks * algs.size());

    for (const QString& fileName : files) {
        if(m_pCancelled->load()) {
            // 补偿未提交任务的计数，确保 activeTasks 最终能归零
            m_activeTasks--;
            m_collector->decrementExpectedCount(algs.size());
            continue;
        }

        ProcessingTask* task = new ProcessingTask(dir.absoluteFilePath(fileName), algs, refImg);
        task->setPCancelled(m_pCancelled);
        task->setROI(roi4Task);
        connect(task, &ProcessingTask::imageReady, m_collector, &ResultCollector::saveImage);
        connect(task, &ProcessingTask::resultReady, m_collector, &ResultCollector::handleResult);
        // 如果任务内部失败，也要同步计数
        connect(task, &ProcessingTask::resultsSkipped, m_collector, &ResultCollector::decrementExpectedCount);
        connect(task, &ProcessingTask::finished, this, &ProcessingSession::onTaskFinished);
        QThreadPool::globalInstance()->start(task);
    }
    if (m_activeTasks <= 0) { emit sessionFinished(); }
}

void ProcessingSession::onTaskFinished()
{
    m_activeTasks--;
    emit progressUpdated(m_totalTasks - m_activeTasks, m_totalTasks);

    if (m_activeTasks <= 0) emit sessionFinished();
}

void ProcessingSession::cancel()
{
    if(m_pCancelled) m_pCancelled->store(true);

    if (m_collector) {
        m_collector->abort(); // 立即强行释放文件句柄
    }
}

/********ResultCollector********/
void ResultCollector::setOutputDir(QString path)
{
    QMutexLocker locker(&m_mutex);
    m_outputDir = path;
}

void ResultCollector::prepare()
{
    m_isAborted = false; // <--- 关键：重置幽灵状态
    m_expectedResults = 0; // 确保计数器也是干净的

    closeAll();
    if (!m_outputDir.isEmpty()) {
        QDir dir;
        dir.mkpath(m_outputDir+"/RawImages");
        dir.mkpath(m_outputDir+"/NormalizeImages");
        if (!dir.exists(m_outputDir)) {
            if (dir.mkpath(m_outputDir)) {
                qDebug() << "Created output directory:" << m_outputDir;
            } else {
                qDebug() << "Critical: Could not create output directory!";
            }
        }
    }
}

void ResultCollector::closeAll()
{
    QMutexLocker locker(&m_mutex);

    m_streams.clear();

    for (auto f : m_files) {
        if (f && f->isOpen()) {
            f->flush(); // 显式刷盘
            f->close();
        }
    }
    m_files.clear();
    // QCoreApplication::processEvents();
}

void ResultCollector::abort()
{
    QMutexLocker locker(&m_mutex);
    m_isAborted = true;
    // m_expectedResults = 0; // 清空预期，防止后续 handleResult 继续工作
    // closeAll();            // 立即关闭所有文件句柄
}

void ResultCollector::resetExpectedCount(int count)
{
    QMutexLocker locker(&m_mutex);
    m_expectedResults = count;
}

void ResultCollector::decrementExpectedCount(int count)
{
    QMutexLocker locker(&m_mutex);
    m_expectedResults -= count;
    if (m_expectedResults <= 0) {
        // 虽然在子线程，但因为是 DirectConnection 或简单计数，我们最好通过信号去通知
        QMetaObject::invokeMethod(this, "allResultsSaved", Qt::QueuedConnection);
    }
}

void ResultCollector::handleResult(QString algName, QString fileName, double value)
{
    QMutexLocker locker(&m_mutex);

    if (m_isAborted) {
        m_expectedResults--;
        if (m_expectedResults <= 0) emit allResultsSaved();
        return;
    }

    if (!m_streams.contains(algName)) {
        QString fullPath = m_outputDir + "/" + algName + ".csv"; // 建议用 csv 方便表格打开
        auto file = QSharedPointer<QFile>::create(fullPath);

        // 使用 Append 模式，并在文件开头写入表头
        bool isNew = !file->exists();
        if (file->open(QIODevice::Append | QIODevice::Text)) {
            m_files[algName] = file;
            auto stream = QSharedPointer<QTextStream>::create(file.data());
            if (isNew) *stream << "FileName,Value\n";
            m_streams[algName] = stream;
        } else {
            m_expectedResults--;
            qDebug() << "Failed to open output file:" << fullPath;
            // QMessageBox::warning(nullptr, "fail",
            //                      tr(info.toLatin1()));
            return;
        }
    }

    if (m_streams.contains(algName)) {
        *m_streams[algName] << fileName << "," << QString::number(value, 'f', 6) << "\n";
        // m_streams[algName]->flush(); // 强制刷盘，防止崩溃丢失数据
    }

    m_expectedResults--;
    if (m_expectedResults <= 0) {
        emit allResultsSaved();
    }
}

void ResultCollector::saveImage(cv::InputArray image, QString fileName, bool ifROI)
{
    cv::Mat newImage(image.getMat().clone()), normalized;
    QString outputDir;
    cv::normalize(newImage, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    {
        QMutexLocker locker(&m_mutex);
        outputDir = m_outputDir;
    }

    if(ifROI){
        QString rawPath = QString("%1/RawImages/%2").arg(outputDir, fileName);

        if(!cv::imwrite(rawPath.toStdString(), newImage)) {
            qDebug() << "Failed to write raw image:" << rawPath;
        }
    }


    QString normPath = QString("%1/NormalizeImages/%2").arg(outputDir, fileName);
    cv::imwrite(normPath.toStdString(), normalized);
}
