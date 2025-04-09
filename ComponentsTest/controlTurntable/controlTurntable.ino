/* Pin Definitions */ 
// Button Pin
#define BUTTON_PIN 2

// Stepper Motor Pins
#define TT_DIR_PIN 3
#define TT_STEP_PIN 4

// Status LED Pin
#define STATUS_LED_PIN 13 

// Function Definitions
void rotateTurntable(int angle, float speed);

void setup() {
    Serial.begin(9600);

    // Stepper Motor Pins
    pinMode(TT_DIR_PIN, OUTPUT); 
    pinMode(TT_STEP_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
  
    // Button Pin
    pinMode(BUTTON_PIN, INPUT);
  
}

void loop() {
    // turn status LED on
    digitalWrite(STATUS_LED_PIN,  HIGH);
   
    if (digitalRead(BUTTON_PIN) == HIGH){
        rotateTurntable(10,0.1);
    }
    delay(2000); // delay for button debouncing
}

void rotateTurntable(int angle, float speed){
    // Specify direction of rotation
    if (angle >= 0){
        digitalWrite(TT_DIR_PIN,HIGH);
    } else{
        digitalWrite(TT_DIR_PIN,LOW);
    }

    // Angle Check (Angle below 180 degrees)
    if (abs(angle) > 180){
        angle = 180;
    }

    // Speed Check (Must be between 0.01 and 1)
    if (speed < 0.01){
        speed = 0.01;
    } else if (speed > 1){
        speed = 1;
    }

    // Need to change once stepper motor characteristics known or after calibration
    int steps = abs(angle)*(1/0.225);
    float usDelay = (1/speed) * 70;

    for (int i = 0; i < steps; i++){ 
        digitalWrite(TT_STEP_PIN, HIGH); 
        delayMicroseconds(usDelay); 
    
        digitalWrite(TT_STEP_PIN, LOW); 
        delayMicroseconds(usDelay);
    }
}