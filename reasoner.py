import json
import ollama
from PyQt6.QtCore import QThread, pyqtSignal

class LLMWorker(QThread):
    finished_signal = pyqtSignal(dict) 

    def __init__(self):
        super().__init__()
        # No complex init needed for pure ollama library
        self.current_data = None

    def set_data(self, data_dict):
        """Receive the defect metrics dictionary"""
        self.current_data = data_dict

    def run(self):
        if not self.current_data:
            return

        # 1. Define the System Prompt
        system_prompt = """
        You are an expert Industrial Inspector for End-of-Life Components. Your job is to analyze defects and output a JSON response.
        
        RULES FOR ANALYSIS:
        1. **Root Cause Hypothesis**:
           - Distinguish between 'Manufacturing Defect' (process failure, machine error) and 'Handling Defect' (impact, human error).
           - Logic: High surface roughness often implies impact/tear (Handling). High volume/depth implies structural failure (Manufacturing). Coil overlaps are usually winding tension issues (Manufacturing).
        
        2. **Sensor Escalation**:
           - Recommend the top 1 or 2 sensors for clearer inspection based on physics.
           - Logic: 
             - Deep/Internal defects -> Ultrasonic Testing (UT) or Industrial CT Scan (for fracture).
             - Surface texture/roughness -> 3D Laser Profilometer or White Light Interferometry (for fracture with high roughness).
             - Holes/Geometry -> Optical Coordinate Measuring Machine (CMM) or Bore Gauge (for hole).
             - Conductive material cracks -> Eddy Current (for coil overlap).

        3. **Output Format**:
           - You must return ONLY valid JSON. Do not include markdown formatting like ```json ... ```.
           - Structure: 
             {
               "root_cause": {"hypothesis": "string", "category": "Manufacturing OR Handling", "confidence_score": float (0.0-1.0)},
               "sensor_escalation": [{"sensor_name": "string", "reason": "string"}]
             }
        """

        try:
            # 2. Convert Python Dictionary to JSON String
            user_input_json = json.dumps(self.current_data)

            # 3. Call the Ollama Model
            response = ollama.chat(model='llama3.2:3b', messages=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': f"Analyze this defect data: {user_input_json}",
                },
            ],
            options={
                'temperature': 0,
                'seed': 42
            })

            # 4. Extract and Parse Response
            response_content = response['message']['content']
            
            # Clean up if model adds markdown backticks by mistake
            if response_content.startswith("```"):
                response_content = response_content.strip("`").replace("json\n", "")

            response_dict = json.loads(response_content)
            
            # Emit the dictionary directly
            self.finished_signal.emit(response_dict)

        except Exception as e:
            # Fallback error dict
            self.finished_signal.emit({
                "error": str(e),
                "root_cause": {"hypothesis": "Analysis Failed", "category": "Error", "confidence_score": 0.0},
                "sensor_escalation": []
            })