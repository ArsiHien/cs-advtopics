from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QLineEdit, QPushButton, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect


class GazeCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_point = None
        self.trail_points = []
        self.max_trail_points = 64
        self.displayed_text = "LOOK HERE"
        self.text_color = QColor(0, 0, 0)
        self.text_font = QFont("Arial", 24, QFont.Bold)
        self.screen_width = parent.screen_width
        self.screen_height = parent.screen_height
        self.setStyleSheet("background-color: transparent;")

    def set_gaze_point(self, point):
        self.current_point = point
        self.trail_points.append(point)
        if len(self.trail_points) > self.max_trail_points:
            self.trail_points.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw crosshair grid lines
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawLine(self.width() // 2, 0,
                         self.width() // 2, self.height())
        painter.drawLine(0, self.height() // 2,
                         self.width(), self.height() // 2)

        # Draw trail
        if len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                opacity = i / len(self.trail_points)
                thickness = round((i / len(self.trail_points)) * 8) + 2
                trail_pen = QPen(QColor(0, 0, 255, int(255 * opacity)))
                trail_pen.setWidth(thickness)
                painter.setPen(trail_pen)
                painter.drawLine(self.trail_points[i-1], self.trail_points[i])

        # Draw gaze point
        if self.current_point:
            painter.setPen(QPen(QColor(0, 0, 0), 5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self.current_point, 60, 60)
            painter.setPen(QPen(QColor(0, 150, 0), 4))
            painter.setBrush(QBrush(QColor(0, 150, 0, 100)))
            painter.drawEllipse(self.current_point, 40, 40)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 0)))
            painter.drawEllipse(self.current_point, 30, 30)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 0, 0)))
            painter.drawEllipse(self.current_point, 15, 15)
            painter.setFont(self.text_font)
            painter.setPen(QPen(self.text_color, 2))
            text_rect = painter.fontMetrics().boundingRect(self.displayed_text)
            text_rect.moveCenter(self.current_point + QPoint(0, 70))
            painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            backdrop_rect = QRect(text_rect)
            backdrop_rect.adjust(-15, -10, 15, 10)
            painter.drawRoundedRect(backdrop_rect, 10, 10)
            painter.setPen(QPen(self.text_color, 2))
            painter.drawText(text_rect, Qt.AlignCenter, self.displayed_text)


class GazeVisualizerGUI(QMainWindow):
    def __init__(self, screen_resolution=None):
        super().__init__()
        self.setWindowTitle("Gaze Tracking Visualization")

        if screen_resolution is None:
            screen = QApplication.primaryScreen().size()
            self.screen_width, self.screen_height = screen.width(), screen.height()
        else:
            self.screen_width, self.screen_height = screen_resolution

        self.resize(self.screen_width, self.screen_height)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #EEEEEE;")

        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: transparent;")
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()

        self.gaze_canvas = GazeCanvas(self)
        layout.addWidget(self.gaze_canvas, stretch=1)

        # Create metrics panel with grid layout for better organization
        metrics_panel = QWidget()
        metrics_layout = QGridLayout()
        metrics_panel.setLayout(metrics_layout)
        metrics_panel.setStyleSheet(
            "background-color: rgba(40, 40, 40, 200); border-radius: 10px; margin: 10px;")

        # Labels for metrics
        self.info_label = QLabel("GAZE TRACKING ACTIVE", self)
        self.info_label.setStyleSheet(
            "color: white; padding: 10px; font-weight: bold;")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.setFont(QFont("Arial", 16))

        self.coord_label = QLabel("X: 0, Y: 0", self)
        self.coord_label.setStyleSheet(
            "color: white; padding: 8px; font-weight: bold;")
        self.coord_label.setFont(QFont("Arial", 14, QFont.Bold))

        # Add eye metrics labels
        self.distance_label = QLabel("Distance: -- cm", self)
        self.distance_label.setStyleSheet(
            "color: white; padding: 8px; font-weight: bold;")
        self.distance_label.setFont(QFont("Arial", 14, QFont.Bold))

        self.blink_label = QLabel("Blinks: 0", self)
        self.blink_label.setStyleSheet(
            "color: white; padding: 8px; font-weight: bold;")
        self.blink_label.setFont(QFont("Arial", 14, QFont.Bold))

        self.eye_status_label = QLabel("Eyes: Open", self)
        self.eye_status_label.setStyleSheet(
            "color: white; padding: 8px; font-weight: bold;")
        self.eye_status_label.setFont(QFont("Arial", 14, QFont.Bold))

        # Add all labels to the metrics grid
        metrics_layout.addWidget(self.info_label, 0, 0, 1, 2)
        metrics_layout.addWidget(self.coord_label, 1, 0)
        metrics_layout.addWidget(self.distance_label, 1, 1)
        metrics_layout.addWidget(self.blink_label, 2, 0)
        metrics_layout.addWidget(self.eye_status_label, 2, 1)

        # Add metrics panel to main layout
        layout.addWidget(metrics_panel)

        # Text input and button
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText(
            "Enter text to display at gaze point")
        self.text_input.setText("LOOK HERE")
        self.text_input.setStyleSheet(
            "color: black; background-color: white; padding: 15px; border: 3px solid black; font-size: 16px;")
        self.text_input.setFixedWidth(400)
        self.text_input.returnPressed.connect(self.update_displayed_text)

        self.update_button = QPushButton("Update Text", self)
        self.update_button.setStyleSheet(
            "color: white; background-color: #0066cc; padding: 15px; font-weight: bold; font-size: 16px;")
        self.update_button.setFixedWidth(150)
        self.update_button.clicked.connect(self.update_displayed_text)

        # Add text input layout
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.text_input)
        input_layout.addWidget(self.update_button)
        input_layout.addStretch()
        layout.addLayout(input_layout)

        self.central_widget.setLayout(layout)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.gaze_canvas.update)
        self.update_timer.start(16)
        print("Timer started")

        self.showMaximized()

    def update_displayed_text(self):
        self.gaze_canvas.displayed_text = self.text_input.text()
        self.gaze_canvas.update()

    def update_gaze_point(self, x, y, fps=0):
        window_width = self.gaze_canvas.width()
        window_height = self.gaze_canvas.height()
        x = (x / self.screen_width) * window_width
        y = (y / self.screen_height) * window_height

        self.gaze_canvas.set_gaze_point(QPoint(int(x), int(y)))
        self.coord_label.setText(
            f"GAZE POINT: X={int(x)}, Y={int(y)} | FPS: {fps:.1f}")

    def update_eye_metrics(self, distance=None, blink_count=0, is_blinking=False):
        """Update the eye metrics display"""
        if distance:
            self.distance_label.setText(f"Distance: {distance:.2f} cm")
        else:
            self.distance_label.setText("Distance: -- cm")

        self.blink_label.setText(f"Blinks: {blink_count}")

        if is_blinking:
            self.eye_status_label.setText("Eyes: Blinking")
            self.eye_status_label.setStyleSheet(
                "color: yellow; padding: 8px; font-weight: bold;")
        else:
            self.eye_status_label.setText("Eyes: Open")
            self.eye_status_label.setStyleSheet(
                "color: white; padding: 8px; font-weight: bold;")

    def closeEvent(self, event):
        """Handle the window close event - terminate application when window is closed"""
        print("Window closed, terminating application...")
        # Accept the close event
        event.accept()
        # Terminate the application
        QApplication.instance().quit()
