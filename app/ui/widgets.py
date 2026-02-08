from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog

class FilePicker(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)

        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse)

        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_btn)
        self.setLayout(layout)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if path:
            self.path_edit.setText(path)

    def path(self) -> str:
        return self.path_edit.text().strip()
