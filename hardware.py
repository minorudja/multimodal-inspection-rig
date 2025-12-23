import serial
import time
from typing import Dict

# =========================================================
# ACTUATOR CONTROLLER
# =========================================================

# Calibration constants: adjust these if the hardware 
# components (motors/lead screws) change.
STEPS_PER_DEGREE = 4.4444  
STEPS_PER_MM = 200         

class ActuatorController:
    """ 
    Manages serial communication for motor control. 
    Protocol format: "ID,Direction,Steps,Speed\n"
    """
    def __init__(self, port: str, baud_rate: int = 9600):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        
        # We only track positions for ID 2 and 3 (the linear slides).
        # ID 1 (Turntable) isn't tracked here as it's typically used 
        # for relative rotations rather than absolute positioning.
        self._linear_positions: Dict[int, int] = {2: 0, 3: 0}
        
        self._connect()
    
    def _connect(self):
        """ Opens the serial port and handles the hardware reset delay. """
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            # Arduinos usually reboot when the serial port opens. 
            # 2 seconds gives the bootloader time to finish.
            time.sleep(2)  
            print(f"ACTUATOR: Connected on {self.port}")
        except serial.SerialException:
            self.ser = None
    
    def close(self):
        """ Cleanly close the serial connection. """
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _send_command(self, actuator_id: int, steps: int, speed_factor: float):
        """ 
        The primary communication bridge. 
        Converts movement requests into the CSV-style string the firmware expects.
        """
        if not self.ser or not self.ser.is_open:
            return
        
        # Firmware expects 1 for 'forward/up' and 0 for 'backward/down'
        direction = 1 if steps >= 0 else 0
        abs_steps = abs(steps)
        
        # Clamp speed between 1% and 100% to avoid firmware math errors
        constrained_speed = max(0.01, min(1.0, speed_factor))
        
        # Build the command string. Example: "2,1,400,0.50\n"
        command = f"{actuator_id},{direction},{int(abs_steps)},{constrained_speed:.2f}\n"
        
        try:
            self.ser.write(command.encode())
        except Exception:
            # If the write fails (e.g. cable unplugged), we fail silently 
            # to prevent the whole control loop from crashing.
            pass

    def rotate_turntable(self, angle: float, speed: float = 1.0):
        """ 
        Converts degrees to steps and commands the turntable (ID 1).
        Positive angle = Clockwise, Negative = Counter-Clockwise.
        """
        steps = int(round(angle * STEPS_PER_DEGREE))
        self._send_command(1, steps, speed)

    def move_linear_actuator(self, actuator_id: int, distance: float, speed: float = 1.0):
        """ 
        Moves Linear Actuator (ID 2 or 3) and updates internal software tracking.
        Distance is in millimeters.
        """
        if actuator_id not in [2, 3]: 
            return
        
        steps = int(round(distance * STEPS_PER_MM))
        self._send_command(actuator_id, steps, speed)
        
        # Update the software-side odometer so we know where the actuator is.
        # Note: This assumes the motor actually finishes the move successfully.
        self._linear_positions[actuator_id] += steps

    def set_zero_position(self, actuator_id: int):
        """ 
        Manually overrides the current software position to 0. 
        Useful after performing a physical homing routine.
        """
        if actuator_id in [2, 3]:
            self._linear_positions[actuator_id] = 0

    def get_current_position(self, actuator_id: int) -> float:
        """ 
        Converts tracked steps back into millimeters for the UI/Logic.
        """
        if actuator_id not in [2, 3]: 
            return 0.0
        return self._linear_positions[actuator_id] / STEPS_PER_MM

    def stop_actuator(self, actuator_id: int):
        """ Immediately halts a specific motor by sending a 0-step command. """
        if actuator_id not in [1, 2, 3]:
            return
        self._send_command(actuator_id, steps=0, speed_factor=1.0)

    def emergency_stop(self):
        """ 
        Sends a broadcast stop to ID 99. 
        The firmware should be configured to kill all power to all motors upon 
        receiving this ID.
        """
        self._send_command(actuator_id=99, steps=0, speed_factor=1.0)