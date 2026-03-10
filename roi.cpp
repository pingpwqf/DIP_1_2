#include "roi.h"

#include <QEvent>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGraphicsRectItem>

ROI::ROI(const QImage& image, QWidget* parent)
    :QDialog(parent), m_roiRectItem(nullptr), m_isDrawing(false)
{
    setWindowTitle(tr("请划取 ROI 区域 (左键拖动)"));
    setMinimumSize(400, 300);

    // 1. UI 布局
    QVBoxLayout* layout = new QVBoxLayout(this);
    QHBoxLayout* btnLayout = new QHBoxLayout(this);
    m_view = new QGraphicsView(this);
    m_scene = new QGraphicsScene(this);
    m_view->setScene(m_scene);

    // 加载图片
    m_pixmapItem = m_scene->addPixmap(QPixmap::fromImage(image));

    layout->addWidget(m_view);

    QPushButton* okBtn = new QPushButton(tr("确定"), this);
    QPushButton* cancelBtn = new QPushButton(tr("取消"), this);
    connect(okBtn, &QPushButton::clicked, this, &QDialog::accept);
    connect(cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
    btnLayout->addWidget(okBtn);
    btnLayout->addWidget(cancelBtn);
    layout->addLayout(btnLayout);

    // 2. 安装事件过滤器，捕获视图上的鼠标动作
    m_view->viewport()->installEventFilter(this);
}

bool ROI::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == m_view->viewport()) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        switch (event->type()) {
        case QEvent::MouseButtonPress:
            mousePressEvent(mouseEvent);
            return true;
        case QEvent::MouseMove:
            mouseMoveEvent(mouseEvent);
            return true;
        case QEvent::MouseButtonRelease:
            mouseReleaseEvent(mouseEvent);
            return true;
        default:
            break;
        }
    }
    return QDialog::eventFilter(obj, event);
}

void ROI::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        m_isDrawing = true;
        // 转换点击位置到场景坐标（即图片像素坐标）
        m_startPoint = m_view->mapToScene(event->pos());

        // 清理旧矩形
        if (m_roiRectItem) {
            m_scene->removeItem(m_roiRectItem);
            delete m_roiRectItem;
            m_roiRectItem = nullptr;
        }

        // 创建新矩形，初始大小为 0
        m_roiRectItem = m_scene->addRect(QRectF(m_startPoint, QSizeF(0, 0)),
                                         QPen(Qt::red, 2, Qt::SolidLine));
    }
}

void ROI::mouseMoveEvent(QMouseEvent* event)
{
    if (m_isDrawing && m_roiRectItem) {
        QPointF currentPoint = m_view->mapToScene(event->pos());

        // --- 边界限制逻辑 ---
        // 确保矩形不会划出图片的边界，防止 OpenCV 处理时崩溃
        QRectF imgBounds = m_pixmapItem->boundingRect();
        qreal boundedX = qBound(imgBounds.left(), currentPoint.x(), imgBounds.right());
        qreal boundedY = qBound(imgBounds.top(), currentPoint.y(), imgBounds.bottom());
        QPointF finalPoint(boundedX, boundedY);

        // --- 矩形生成逻辑 ---
        // 使用 normalized()。这很重要！
        // 它能处理“从右往左”或“从下往上”的逆向划动，自动计算出正确的左上角坐标和宽高
        QRectF newRect = QRectF(m_startPoint, finalPoint).normalized();
        m_roiRectItem->setRect(newRect);
    }
}

void ROI::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && m_isDrawing) {
        m_isDrawing = false;
        if (m_roiRectItem) {
            // 保存最终的矩形区域（转换为整数像素坐标）
            m_finalRect = m_roiRectItem->rect().toRect().intersected(m_pixmapItem->pixmap().rect());

            // 可以在释放时把虚线变实线，表示选定
            m_roiRectItem->setPen(QPen(Qt::red, 2, Qt::SolidLine));
            m_roiRectItem->setBrush(QColor(255, 0, 0, 40)); // 半透明填充
        }
    }
}
