#include "mainwindow.h"
#include "ImgPcAlg.h"
#include "roi.h"

#include <QAction>
#include <QFileDialog>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>
#include <QThreadPool>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("DIP");
    CreateAction();
    CreateMenu();
    CreateLabel();
    CreateButton();
    CreateLineEdit();
    CreateGraphicsView();
    CreateLayout();
    resize(500, 300);

    int idealThreadCount = QThread::idealThreadCount();
    int maxThreads = qMax(2, idealThreadCount / 2);
    QThreadPool::globalInstance()->setMaxThreadCount(maxThreads);

    RegisterAlgs();
}

MainWindow::~MainWindow() {}

void MainWindow::CreateAction()
{
    actionROI = new QAction("ROI", this);
    actionNIPC = new QAction("NIPC", this);
    actionZNCC = new QAction("ZNCC", this);
    actionMSV = new QAction("MSC", this);
    actionGLCMcorr = new QAction("correlation", this);
    actionGLCMhomo = new QAction("homogeneity", this);

    actionNIPC->setCheckable(true);
    actionZNCC->setCheckable(true);
    actionMSV->setCheckable(true);
    actionGLCMcorr->setCheckable(true);
    actionGLCMhomo->setCheckable(true);

    connect(actionROI, &QAction::triggered, this, &MainWindow::selectROI);
}

void MainWindow::CreateMenu()
{
    preProcessMenu = new QMenu(tr("预处理"), this);
    algSelectMenu = new QMenu(tr("算法选择"), this);
    GLCMmenu = new QMenu("GLCM", this);

    preProcessMenu->addAction(actionROI);
    algSelectMenu->addAction(actionNIPC);
    algSelectMenu->addAction(actionZNCC);
    algSelectMenu->addMenu(GLCMmenu);
    algSelectMenu->addAction(actionMSV);
    GLCMmenu->addAction(actionGLCMcorr);
    GLCMmenu->addAction(actionGLCMhomo);

    menuBar()->addMenu(preProcessMenu);
    menuBar()->addMenu(algSelectMenu);
}

void MainWindow::CreateButton()
{
    QFont dotFont("MicroSoft YaHei UI", 14);
    refBtn = new QPushButton("...", this);
    sourceBtn = new QPushButton("...", this);
    outputBtn = new QPushButton("...", this);
    okBtn = new QPushButton(tr("确定"), this);
    cancelBtn = new QPushButton(tr("取消"), this);

    refBtn->setFixedSize(QSize(40, 25));
    refBtn->setFont(dotFont);
    sourceBtn->setFixedSize(QSize(40, 25));
    sourceBtn->setFont(dotFont);
    outputBtn->setFixedSize(QSize(40, 25));
    outputBtn->setFont(dotFont);

    connect(refBtn, &QPushButton::clicked, this, &MainWindow::showFile);
    connect(sourceBtn, &QPushButton::clicked, this, &MainWindow::showDir);
    connect(outputBtn, &QPushButton::clicked, this, &MainWindow::showOutDir);
    connect(okBtn, &QPushButton::clicked,
            this, &MainWindow::MainExecute);
    connect(cancelBtn, &QPushButton::clicked, this, [this](){
        okBtn->setEnabled(true);
        if (taskEngine) {
            statusBar()->showMessage(tr("正在取消"), 1000);
        } else {
            // 若未运行，则执行重置逻辑：清空路径
            refEdit->clear();
            sourceEdit->clear();
        }
    });
}

void MainWindow::CreateLabel()
{
    roiLabel = new QLabel(tr("ROI范围显示："), this);
    refLabel = new QLabel(tr("选择参考图像："), this);
    sourceLabel = new QLabel(tr("选择源图像文件夹："), this);
    outputLabel = new QLabel(tr("选择输出文件夹："), this);
}

void MainWindow::CreateLineEdit()
{
    refEdit = new QLineEdit(this);
    sourceEdit = new QLineEdit(this);
    outputEdit = new QLineEdit(this);
}

void MainWindow::CreateGraphicsView()
{
    graphicsView = new QGraphicsView(this);
    graphicsView->setFrameStyle(QFrame::StyledPanel|QFrame::Sunken);
    myScene = new QGraphicsScene(this);
    graphicsView->setScene(myScene);
}

void MainWindow::CreateLayout()
{
    QGridLayout *refLayout, *sourceLayout, *outputLayout;
    QHBoxLayout  *yesOrNoLayout;
    QVBoxLayout *roiLayout, *fileSelectLayout;
    QSpacerItem *horizontalSpacer;
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    mainLayout = new QGridLayout(centralWidget);

    refLayout = new QGridLayout;
    sourceLayout = new QGridLayout;
    outputLayout = new QGridLayout;
    yesOrNoLayout = new QHBoxLayout;
    roiLayout = new QVBoxLayout;
    fileSelectLayout = new QVBoxLayout;

    refLayout->addWidget(refLabel, 0, 0, Qt::AlignLeft);
    refLayout->addWidget(refEdit, 1, 0);
    refLayout->addWidget(refBtn, 1, 1);
    refLayout->setSpacing(5);
    sourceLayout->addWidget(sourceLabel, 0, 0, Qt::AlignLeft);
    sourceLayout->addWidget(sourceEdit, 1, 0);
    sourceLayout->addWidget(sourceBtn, 1, 1);
    outputLayout->addWidget(outputLabel, 0, 0, Qt::AlignLeft);
    outputLayout->addWidget(outputEdit, 1, 0);
    outputLayout->addWidget(outputBtn, 1, 1);
    yesOrNoLayout->addStretch();
    yesOrNoLayout->addWidget(okBtn);
    yesOrNoLayout->addWidget(cancelBtn);
    yesOrNoLayout->setSpacing(20);
    roiLayout->addWidget(roiLabel);
    roiLayout->addWidget(graphicsView);
    fileSelectLayout->addLayout(refLayout);
    fileSelectLayout->addLayout(sourceLayout);
    fileSelectLayout->addLayout(outputLayout);
    fileSelectLayout->setSpacing(10);
    mainLayout->addLayout(roiLayout, 0, 0);
    mainLayout->addLayout(fileSelectLayout, 0, 1);
    mainLayout->addLayout(yesOrNoLayout, 1, 0, 1, 2);
    mainLayout->setColumnStretch(0, 2);
    mainLayout->setColumnStretch(1, 3);
}

void MainWindow::RegisterAlgs()
{
    AlgRegistry<QString>::instance().Register(MSVNAME, [](cv::InputArray img){
        return std::make_unique<MSVAlg>(img);
    });
    AlgRegistry<QString>::instance().Register(NIPCNAME, [](cv::InputArray img){
        return std::make_unique<NIPCAlg>(img);
    });
    AlgRegistry<QString>::instance().Register(ZNCCNAME, [](cv::InputArray img){
        return std::make_unique<ZNCCAlg>(img);
    });
}

void MainWindow::showFile()
{
    QString newPath = QFileDialog::getOpenFileName(this, tr("Open Image"),"/",
                                                   tr("Image Files (*.bmp *.png)"));
    if (!newPath.isEmpty()) {
        filePath = newPath;
        refEdit->setText(filePath);

        // 清空预览区域
        if (myScene) {
            myScene->clear();
        }

        // 重置保存的 ROI 变量，防止用户用旧 ROI 算新图
        this->currentROI = cv::Rect();
    }
}

void MainWindow::showDir()
{
    dirPath = QFileDialog::getExistingDirectory(this, tr("Open Directory"),filePath+"/..",
                                                QFileDialog::ShowDirsOnly|QFileDialog::DontResolveSymlinks);
    sourceEdit->setText(dirPath);
}

void MainWindow::showOutDir()
{
    dirOutPath = QFileDialog::getExistingDirectory(this, tr("Open Directory"),"/",
                                                   QFileDialog::ShowDirsOnly|QFileDialog::DontResolveSymlinks);
    outputEdit->setText(dirOutPath);
}

void MainWindow::selectROI()
{

    if (filePath.isEmpty()) {
        QMessageBox::warning(this, "NoRef", tr("haven't select a reference image!"));
        return;
    }

    // 1. 加载参考图并转换为 QImage
    cv::Mat ref = imread_safe(filePath);
    QImage qimg = QImage(ref.data, ref.cols, ref.rows, ref.step, QImage::Format_Grayscale8).copy();

    // 2. 弹出 ROI 窗口
    ROI roiDlg(qimg, this);
    roiDlg.show();
    if (roiDlg.exec() == QDialog::Accepted) {
        QRect r = roiDlg.getSelectedRect();

        // 3. 将 QRect 转换为 cv::Rect
        cv::Rect cvROI(r.x(), r.y(), r.width(), r.height());

        // 4. 在主界面预览 ROI 区域
        cv::Mat croppedRef = ref(cvROI).clone();
        QImage qPreview(croppedRef.data,croppedRef.cols,croppedRef.rows,
                        croppedRef.step,QImage::Format_Grayscale8); // 转换为QImage

        myScene->clear();
        myScene->addPixmap(QPixmap::fromImage(qPreview));
        graphicsView->fitInView(myScene->itemsBoundingRect(), Qt::KeepAspectRatio);

        // 5. 保存这个 cvROI，后续传入 TaskManager
        this->currentROI = cvROI;
    }

}

void MainWindow::MainExecute()
{
    if(filePath.isEmpty()) QMessageBox::warning(this, "noRef",
                             tr("haven't select reference image!"));
    else if(dirPath.isEmpty()) QMessageBox::warning(this, "noSource",
                             tr("haven't select Source images!"));
    else if(dirOutPath.isEmpty()) QMessageBox::warning(this, "noOutPath",
                             tr("haven't select output path!"));
    else{
        okBtn->setEnabled(false); // 冻结按钮

        taskEngine = std::make_unique<TaskManager>(dirOutPath);
        ProcessingSession* session = taskEngine->createSession();
        taskEngine->setROI(currentROI);
        connect(cancelBtn, &QPushButton::clicked, session, &ProcessingSession::cancel);

        connect(session, &ProcessingSession::sessionFinished, this, [this, session](){
            okBtn->setEnabled(true); // 解冻
            // collector.closeAll();               // 关闭文件
            statusBar()->showMessage(tr("批处理完成！"), 3000);

            session->deleteLater(); // 销毁 Session 对象
        });

        QDir dir(dirPath);
        QStringList files = dir.entryList({"*.bmp", "*.png", "*.jpg"}, QDir::Files);
        cv::Mat refImg = imread_safe(filePath);
        refImg = currentROI.width > 0 && currentROI.height > 0 ?
                     refImg(currentROI).clone() : refImg;

        selectedChoices.clear();
        if(actionMSV->isChecked())selectedChoices.emplaceBack(MSVNAME);
        if(actionNIPC->isChecked())selectedChoices.emplaceBack(NIPCNAME);
        if(actionZNCC->isChecked())selectedChoices.emplaceBack(ZNCCNAME);
        if(actionGLCMcorr->isChecked())selectedChoices.emplaceBack(CORRNAME);
        if(actionGLCMhomo->isChecked())selectedChoices.emplaceBack(HOMONAME);
        // taskEngine->ExecuteSelected(filePath, dirPath, selectedChoices);
        if(selectedChoices.isEmpty()) {
            QMessageBox::warning(this, "noChoice",
                                 tr("haven't choose any processing method!"));
            okBtn->setEnabled(true);
        }else session->start(refImg, files, dir, selectedChoices);
    }
}
