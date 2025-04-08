/* Pin Definitions */ 
// Stepper Motor Pins
#define DIR_PIN 2
#define STEP_PIN 3

// Button Pin
#define BUTTONPIN 4

// Status LED Pin
#define PIN_STATUS_LED 13 

// Function Definitions
void rotateTurntable(int angle, int speed);

void setup() {
    Serial.begin(9600);

    // Stepper Motor Pins
    pinMode(DIR_PIN, OUTPUT); 
    pinMode(STEP_PIN, OUTPUT); 
  
    // Button Pin
    pinMode(BUTTONPIN, INPUT);
  
}

void loop() {
    // turn status LED on
    digitalWrite(PIN_STATUS_LED,  HIGH);
   
    if (digitalRead(BUTTONPIN) == HIGH){
        rotateTurntable();    // start photoshooting
    }
    delay(2000); // delay for button debouncing
}

void rotateTurntable(int angle, float speed){
    // Specify direction of rotation
    if (angle >= 0){
        digitalWrite(DIR_PIN,HIGH);
    } else{
        digitalWrite(DIR_PIN,LOW);
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
        digitalWrite(STEP_PIN, HIGH); 
        delayMicroseconds(usDelay); 
    
        digitalWrite(STEP_PIN, LOW); 
        delayMicroseconds(usDelay);
    }
}