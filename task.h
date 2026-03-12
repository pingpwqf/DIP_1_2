#ifndef TASK_H
#define TASK_H

#include <QRunnable>
#include <QObject>
#include <QFile>
#include <QTextStream>
#include <QMap>
#include <QSharedPointer>
#include <QDir>
#include <QMutex>
#include <opencv2/opencv.hpp>

cv::Mat imread_safe(const QString& path);

// 结果收集器：负责将不同线程产生的数据分类写入文件
class ResultCollector : public QObject {
    Q_OBJECT
public:
    explicit ResultCollector(QObject* parent = nullptr) : QObject(parent) {}
    ~ResultCollector() { closeAll(); }

    void setOutputDir(QString path);
    void prepare(); // 准备工作：检查并创建目录
    void abort();

    void resetExpectedCount(int count);
    void decrementExpectedCount(int count);

public slots:
    // 增加 fileName 参数，让结果知道对应哪张图
    void handleResult(QString algName, QString fileName, double value);
    void saveImage(cv::InputArray);
    void closeAll();

private:
    QMutex m_mutex;
    std::atomic<int> m_expectedResults{0};
    std::atomic<bool> m_isAborted{false};

    QString m_outputDir;
    QMap<QString, QSharedPointer<QFile>> m_files;
    QMap<QString, QSharedPointer<QTextStream>> m_streams;

signals:
    void allResultsSaved();
};

/*************************************************/
// 具体的处理任务
class ProcessingTask : public QObject, public QRunnable {
    Q_OBJECT
public:
    // 传递算法名称和参考图，而不是直接传递算法实例，以保证线程安全
    ProcessingTask(QString imgPath, QVector<QString> algNames, cv::Mat refImg)
        : m_path(imgPath), m_algNames(algNames), m_refImg(refImg) {
        setAutoDelete(true);
    }

    void run() override;
    void setPCancelled(std::shared_ptr<std::atomic<bool>> pFlag) {m_pCancelled = pFlag;}
    void setROI(cv::Rect roi) {m_roi = roi;}

private:
    std::shared_ptr<std::atomic<bool>> m_pCancelled = nullptr;
    cv::Rect m_roi;

    QString m_path;
    QVector<QString> m_algNames;
    cv::Mat m_refImg;

signals:
    void resultReady(QString algName, QString fileName, double value);
    void imageReady(cv::InputArray img);
    void finished();
    void errorOccurred(QString msg);
    void resultsSkipped(unsigned size);
};

/****************************************************/
class ProcessingSession : public QObject {
    Q_OBJECT
public:
    explicit ProcessingSession(ResultCollector* rc, QObject* parent = nullptr)
        : QObject(parent), m_collector(rc), m_activeTasks(0), m_totalTasks(0)
    {
        m_pCancelled = std::make_shared<std::atomic<bool>>(false);
    }

    void start(const cv::Mat& refImg, const QStringList& files, const QDir& dir, const QVector<QString>& algs);
    void cancel();

    void setROI(cv::Rect roi) { roi4Task = roi; }
    std::shared_ptr<std::atomic<bool>> getPCancelled() const {return m_pCancelled;}
signals:
    void sessionFinished(); // 整个批处理完成
    void progressUpdated(int current, int total); // 可选：进度条支持

private slots:
    void onTaskFinished();

private:
    std::shared_ptr<std::atomic<bool>> m_pCancelled;

    ResultCollector* m_collector;
    cv::Rect roi4Task;
    int m_activeTasks;
    int m_totalTasks;
};

struct BatchConfig {
    cv::Mat refImg;
    QStringList files;
    QDir dir;
    QVector<QString> algorithms;
    cv::Rect roi;
};

// 任务管理器
class TaskManager : public QObject
{
    Q_OBJECT
public:
    TaskManager(QString outputPath);
    ProcessingSession* execute(const BatchConfig& config);


private:
    ResultCollector* m_collector;
};

#endif // TASK_H
