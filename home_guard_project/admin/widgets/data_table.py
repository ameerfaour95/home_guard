from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableView, QHeaderView, QAbstractItemView
from ..theme import PALETTES


class RowsModel(QAbstractTableModel):
    def __init__(self, columns):
        super().__init__()
        self.columns, self.items = columns, []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section][0]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.items[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return row
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return str(self.columns[index.column()][1](row))

    def replace(self, items, append=False):
        if append and items:
            start = len(self.items)
            self.beginInsertRows(QModelIndex(), start, start+len(items)-1); self.items.extend(items); self.endInsertRows()
        elif not append:
            self.beginResetModel(); self.items = list(items); self.endResetModel()


def data_table(model, stretch=0):
    table = QTableView(); table.setModel(model)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setShowGrid(False); table.verticalHeader().hide(); table.verticalHeader().setDefaultSectionSize(64)
    table.setWordWrap(False)
    table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    table.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    return table
