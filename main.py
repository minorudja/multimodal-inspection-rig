import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from gui import VisionDashboard # Custom UI class from gui.py

if __name__ == "__main__":
    # Standard entry point for the application
    app = QApplication(sys.argv)

    # Setting 'Fusion' because it's the most consistent style across Windows, Mac, and Linux
    app.setStyle("Fusion")

    # --- Theme Configuration (Dark Mode) ---
    # Manually defining a dark theme palette since PyQt doesn't have a 'dark_mode=True' toggle.
    palette = QPalette()
    
    # Main window background and text colors
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    
    # Text input fields, lists, and tables background
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    
    # Tooltip styling
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    
    # General text and button colors
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    
    # Highlight colors (for selected text or links)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    # Apply the custom palette to the whole application
    app.setPalette(palette)

    # Initialize the main dashboard window
    window = VisionDashboard()
    window.show()

    # app.exec() starts the event loop; sys.exit ensures the process closes cleanly
    sys.exit(app.exec())