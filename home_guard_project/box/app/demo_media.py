"""Generated camera placeholders. No external photos or devices."""
from PySide6.QtCore import Qt,QRectF
from PySide6.QtGui import QPixmap,QPainter,QColor,QLinearGradient,QPen,QFont

def picture(index=0,aspect="16:9"):
    pix=QPixmap(1280,960 if aspect=="4:3" else 720);p=QPainter(pix);p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if aspect=="4:3": p.scale(1,4/3)
    sky=QLinearGradient(0,0,0,720);sky.setColorAt(0,QColor('#182d3c'));sky.setColorAt(1,QColor('#53656a'));p.fillRect(QRectF(0,0,1280,720),sky)
    p.setPen(Qt.PenStyle.NoPen)
    for x,y,r in ((110,62,1),(920,100,2),(1040,44,1),(730,75,1),(290,112,2)):
        p.setBrush(QColor('#b1c2ca'));p.drawEllipse(x,y,r*2,r*2)
    p.setBrush(QColor('#283d39'));p.drawRect(0,320,1280,400)
    p.setBrush(QColor('#73817c'));p.drawPolygon(__import__('PySide6.QtGui',fromlist=['QPolygonF']).QPolygonF([__import__('PySide6.QtCore',fromlist=['QPointF']).QPointF(x,y) for x,y in ((500,450),(720,450),(1200,720),(100,720))]))
    p.setBrush(QColor('#8a9696'));p.drawRoundedRect(300,210,670,270,8,8)
    p.setBrush(QColor('#293940'));p.drawPolygon(__import__('PySide6.QtGui',fromlist=['QPolygonF']).QPolygonF([__import__('PySide6.QtCore',fromlist=['QPointF']).QPointF(x,y) for x,y in ((250,215),(620,110),(1030,215))]))
    for x in (355,780):
        p.setBrush(QColor('#f1d29c'));p.drawRect(x,260,120,120);p.setPen(QPen(QColor('#5c686c'),7));p.drawLine(x+60,260,x+60,380);p.drawLine(x,320,x+120,320);p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor('#273c42'));p.drawRect(585,278,100,200)
    p.setBrush(QColor('#d9c5a0'));p.drawEllipse(666,378,5,5)
    for x in (70,1110):
        p.setBrush(QColor('#152d29'));p.drawRoundedRect(x,190,100,330,48,48)
    p.setBrush(QColor('#152129'));p.drawRoundedRect(530,340,240,80,22,22);p.drawRoundedRect(570,304,155,70,28,28)
    p.setBrush(QColor('#637d88'));p.drawRoundedRect(587,314,119,37,12,12)
    p.setBrush(QColor('#09151c'));p.drawEllipse(552,393,48,48);p.drawEllipse(706,393,48,48)
    x=160+index*65;p.setBrush(QColor('#c4a795'));p.drawEllipse(x-15,238,30,34)
    p.setBrush(QColor('#2c5464'));p.drawRoundedRect(x-25,271,52,125,20,20)
    p.setPen(QPen(QColor('#1b2a32'),16));p.drawLine(x-10,390,x-20,518);p.drawLine(x+10,390,x+28,518)
    p.setPen(QPen(QColor('#2c5464'),12));p.drawLine(x-20,285,x-35,357);p.drawLine(x+20,285,x+45,350)
    p.setPen(QColor('#bdc9cf'));p.setFont(QFont('Segoe UI',14));p.drawText(32,42,'HOME GUARD / DEMONSTRATION')
    p.end()
    return pix
