#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "task.h"

class QPushButton;
class QLabel;
class QLineEdit;
class QMenu;
class QAction;
class QGraphicsView;
class QGraphicsScene;
class QGridLayout;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void CreateAction();
    void CreateMenu();
    void CreateButton();
    void CreateLabel();
    void CreateLineEdit();
    void CreateGraphicsView();
    void CreateLayout();

    void RegisterAlgs();

private:
    QString filePath, dirPath, dirOutPath;
    QStringList inFileList, outFileList;
    cv::Rect currentROI;
    QGraphicsScene* myScene;

    QVector<QString> selectedChoices;
    // ResultCollector collector;
    std::unique_ptr<TaskManager> taskEngine;

    //buttons
    QPushButton *refBtn, *sourceBtn, *outputBtn;
    QPushButton *okBtn, *cancelBtn;
    //labels
    QLabel *roiLabel, *refLabel, *sourceLabel, *outputLabel;
    //lineEdits
    QLineEdit *refEdit, *sourceEdit, *outputEdit;
    //menus
    QMenu *preProcessMenu, *algSelectMenu, *GLCMmenu;
    //actions
    QAction *actionROI, *actionNIPC, *actionZNCC, *actionMSV;
    QAction *actionGLCMcorr, *actionGLCMhomo;
    //graphicsView
    QGraphicsView *graphicsView;
    //layouts
    QGridLayout *mainLayout;

private slots:
    void showFile();
    void showDir();
    void showOutDir();

    void selectROI();
    void MainExecute();
};
#endif // MAINWINDOW_H
