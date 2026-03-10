#ifndef ROI_H
#define ROI_H

#include <QDialog>
#include <QGraphicsView>
#include <QMouseEvent>

class ROI : public QDialog
{
    Q_OBJECT
public:
    explicit ROI(const QImage& img, QWidget* parent = nullptr);
    QRect getSelectedRect() const { return m_finalRect; }

private:
    QGraphicsView* m_view;
    QGraphicsScene* m_scene;
    QGraphicsPixmapItem* m_pixmapItem;
    QGraphicsRectItem* m_roiRectItem;

    QPointF m_startPoint;
    QRect m_finalRect;
    bool m_isDrawing = false;

protected:
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    bool eventFilter(QObject *obj, QEvent *event) override;

};

#endif // ROI_H
