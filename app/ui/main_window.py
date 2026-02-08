from __future__ import annotations
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QListWidget, QListWidgetItem, QSplitter, QFileDialog
)
from PySide6.QtCore import Qt

from app.ui.widgets import FilePicker
from app.ui.mpl_canvas import MplCanvas
from app.core.loader import load_csv, validate_schema
from app.core.drift_engine import compute_drift
from app.core.report import export_report_json, export_report_md

import pandas as pd
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Drift Desktop Dashboard (Offline)")
        self.resize(1200, 720)

        self.baseline_picker = FilePicker("Baseline CSV:")
        self.current_picker = FilePicker("Current CSV:")

        self.load_btn = QPushButton("Load & Compute Drift")
        self.load_btn.clicked.connect(self.on_load)

        self.export_btn = QPushButton("Export Report")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.on_export)

        self.status = QLabel("Select baseline + current CSV, then click Load.")
        self.status.setWordWrap(True)

        top = QVBoxLayout()
        top.addWidget(QLabel("<h2>Data Drift Desktop Dashboard</h2>"))
        top.addWidget(QLabel("Offline desktop app to compare baseline vs current datasets and visualize drift."))
        top.addWidget(self.baseline_picker)
        top.addWidget(self.current_picker)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch(1)
        top.addLayout(btn_row)
        top.addWidget(self.status)

        self.feature_search = QLineEdit()
        self.feature_search.setPlaceholderText("Search features…")
        self.feature_search.textChanged.connect(self.on_search)

        self.feature_list = QListWidget()
        self.feature_list.currentItemChanged.connect(self.on_feature_selected)

        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Features</b>"))
        left.addWidget(self.feature_search)
        left.addWidget(self.feature_list, 1)
        left_widget = QWidget()
        left_widget.setLayout(left)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Feature", "Kind", "Score", "Missing Δ", "Key metric"])
        self.table.setSortingEnabled(True)

        self.canvas = MplCanvas()

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Top Drift Summary</b>"))
        right.addWidget(self.table, 2)
        right.addWidget(QLabel("<b>Feature View</b>"))
        right.addWidget(self.canvas, 3)
        right_widget = QWidget()
        right_widget.setLayout(right)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        root = QWidget()
        root_layout = QVBoxLayout()
        root_layout.addLayout(top)
        root_layout.addWidget(splitter, 1)
        root.setLayout(root_layout)
        self.setCentralWidget(root)

        # state
        self.baseline_path = ""
        self.current_path = ""
        self.baseline_df: pd.DataFrame | None = None
        self.current_df: pd.DataFrame | None = None
        self.schema = []
        self.drifts = []
        self._filtered_names: list[str] = []

    def on_load(self):
        self.baseline_path = self.baseline_picker.path()
        self.current_path = self.current_picker.path()
        try:
            b = load_csv(self.baseline_path).df
            c = load_csv(self.current_path).df
            ok, msg = validate_schema(b, c)
            if not ok:
                raise ValueError(msg)

            self.schema, self.drifts = compute_drift(b, c)
            self.baseline_df, self.current_df = b, c

            self.status.setText(
                f"Loaded OK.\nBaseline: {b.shape[0]} rows × {b.shape[1]} cols | "
                f"Current: {c.shape[0]} rows × {c.shape[1]} cols\n"
                f"Top drifted: {self.drifts[0].name if self.drifts else 'N/A'}"
            )
            self.export_btn.setEnabled(True)

            self.populate_lists()
            self.populate_table(top_n=30)

            # auto-select first feature
            if self.feature_list.count() > 0:
                self.feature_list.setCurrentRow(0)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.status.setText("Error loading files. See message.")
            self.export_btn.setEnabled(False)

    def populate_lists(self):
        self.feature_list.clear()
        self._filtered_names = [d.name for d in self.drifts]
        for name in self._filtered_names:
            self.feature_list.addItem(QListWidgetItem(name))

    def populate_table(self, top_n: int = 30):
        self.table.setRowCount(0)
        for d in self.drifts[:top_n]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(d.name))
            self.table.setItem(row, 1, QTableWidgetItem(d.kind))
            self.table.setItem(row, 2, QTableWidgetItem(f"{d.score:.4f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{d.missing_delta:.4f}"))

            key = ""
            if d.kind == "numeric":
                key = f"PSI={d.details.get('psi', 0):.4f}, KS p={d.details.get('ks_pvalue', 1):.3g}"
            elif d.kind == "categorical":
                key = f"JSD={d.details.get('js_divergence', 0):.4f}, χ² p={d.details.get('chi2_pvalue', 1):.3g}"
            self.table.setItem(row, 4, QTableWidgetItem(key))

        self.table.resizeColumnsToContents()

    def on_search(self, text: str):
        text = (text or "").strip().lower()
        self.feature_list.clear()
        if not text:
            names = [d.name for d in self.drifts]
        else:
            names = [d.name for d in self.drifts if text in d.name.lower()]
        self._filtered_names = names
        for name in names:
            self.feature_list.addItem(QListWidgetItem(name))

    def on_feature_selected(self, current: QListWidgetItem, _previous: QListWidgetItem):
        if current is None or self.baseline_df is None or self.current_df is None:
            return
        feature = current.text()
        d = next((x for x in self.drifts if x.name == feature), None)
        if d is None:
            return
        self.plot_feature(d)

    def plot_feature(self, d):
        b = self.baseline_df[d.name]
        c = self.current_df[d.name]
        self.canvas.ax.clear()

        if d.kind == "numeric":
            bnum = pd.to_numeric(b, errors="coerce").dropna()
            cnum = pd.to_numeric(c, errors="coerce").dropna()
            if bnum.empty or cnum.empty:
                self.canvas.ax.text(0.5, 0.5, "No numeric data to plot.", ha="center", va="center")
            else:
                self.canvas.ax.hist(bnum.values, bins=30, alpha=0.5, label="baseline")
                self.canvas.ax.hist(cnum.values, bins=30, alpha=0.5, label="current")
                self.canvas.ax.set_title(f"{d.name} (numeric) — PSI={d.details.get('psi', 0):.4f}")
                self.canvas.ax.legend()
        elif d.kind == "categorical":
            bc = b.astype("object").fillna("∅NA").value_counts().head(12)
            cc = c.astype("object").fillna("∅NA").value_counts().reindex(bc.index, fill_value=0)

            x = np.arange(len(bc.index))
            self.canvas.ax.bar(x - 0.2, bc.values, width=0.4, label="baseline")
            self.canvas.ax.bar(x + 0.2, cc.values, width=0.4, label="current")
            self.canvas.ax.set_xticks(x)
            self.canvas.ax.set_xticklabels([str(v)[:18] for v in bc.index], rotation=25, ha="right")
            self.canvas.ax.set_title(f"{d.name} (categorical) — JSD={d.details.get('js_divergence', 0):.4f}")
            self.canvas.ax.legend()
        else:
            self.canvas.ax.text(0.5, 0.5, f"{d.name} ({d.kind})\nPlot not implemented in v0.", ha="center", va="center")

        self.canvas.draw_idle()

    def on_export(self):
        if not self.drifts:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not out_dir:
            return
        try:
            json_path = f"{out_dir}/drift_report.json"
            md_path = f"{out_dir}/drift_report.md"
            export_report_json(json_path, self.baseline_path, self.current_path, self.schema, self.drifts, top_n=25)
            export_report_md(md_path, self.baseline_path, self.current_path, self.drifts, top_n=25)
            QMessageBox.information(self, "Export complete", f"Saved:\n- {json_path}\n- {md_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

def run_app():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
