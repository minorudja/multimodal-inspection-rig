#include <AccelStepper.h>

// --- Pin Definitions ---
#define TT_DIR_PIN 5
#define TT_STEP_PIN 2
#define LA1_DIR_PIN 6
#define LA1_STEP_PIN 3
#define LA2_DIR_PIN 7
#define LA2_STEP_PIN 4
#define ENA_PIN 8

// --- Configuration ---
const float MAX_SPEED = 1000.0;
const float ACCELERATION = 500.0;
const unsigned long DISABLE_TIMEOUT = 5000; // 5000ms = 5 seconds

// --- Globals ---
unsigned long lastMoveTime = 0; // Tracks the last time a motor moved
bool driversEnabled = true;

AccelStepper stepperTT(AccelStepper::DRIVER, TT_STEP_PIN, TT_DIR_PIN);
AccelStepper stepperLA1(AccelStepper::DRIVER, LA1_STEP_PIN, LA1_DIR_PIN);
AccelStepper stepperLA2(AccelStepper::DRIVER, LA2_STEP_PIN, LA2_DIR_PIN);

void setup() {
    Serial.begin(9600);

    // Setup Enable Pin
    pinMode(ENA_PIN, OUTPUT);
    digitalWrite(ENA_PIN, LOW); // Start Enabled (LOW = ON)

    // Configure Motors
    stepperTT.setMaxSpeed(MAX_SPEED);
    stepperTT.setAcceleration(ACCELERATION);
    
    stepperLA1.setMaxSpeed(MAX_SPEED);
    stepperLA1.setAcceleration(ACCELERATION);

    stepperLA2.setMaxSpeed(MAX_SPEED);
    stepperLA2.setAcceleration(ACCELERATION);

    lastMoveTime = millis(); // Initialize timer
}

void loop() {
    // Check Serial
    if (Serial.available()) {
        handleSerialInput();
    }

    // Check if motors are actually running
    if (stepperTT.isRunning() || stepperLA1.isRunning() || stepperLA2.isRunning()) {
        lastMoveTime = millis(); // Reset the timer constantly while moving
    }

    // Smart Enable/Disable Logic
    if (millis() - lastMoveTime > DISABLE_TIMEOUT) {
        // If exceeded the timeout, DISABLE the drivers.
        if (driversEnabled) {
            digitalWrite(ENA_PIN, HIGH); // HIGH = Disable
            driversEnabled = false;
        }
    } else {
        // If within the active window, ensure drivers are ENABLED
        if (!driversEnabled) {
            digitalWrite(ENA_PIN, LOW); // LOW = Enable
            driversEnabled = true;
            delay(5); // Tiny delay to wake up drivers
        }
    }

    // Run Motors
    stepperTT.run();
    stepperLA1.run();
    stepperLA2.run();
}

void handleSerialInput() {
    String data = Serial.readStringUntil('\n');
    data.trim();
    if (data.length() == 0) return;

    // Force Enable immediately upon receiving a command
    if (!driversEnabled) {
        digitalWrite(ENA_PIN, LOW);
        driversEnabled = true;
        delay(5); 
    }
    
    lastMoveTime = millis(); // Reset timer

    int firstComma  = data.indexOf(',');
    int secondComma = data.indexOf(',', firstComma + 1);
    int thirdComma  = data.indexOf(',', secondComma + 1);

    if (firstComma > 0 && secondComma > 0 && thirdComma > 0) {
        int actuatorNo = data.substring(0, firstComma).toInt();
        int dir = data.substring(firstComma + 1, secondComma).toInt();
        int steps = data.substring(secondComma + 1, thirdComma).toInt();
        float speedFactor = data.substring(thirdComma + 1).toFloat();

        speedFactor = constrain(speedFactor, 0.01, 1.0);
        long targetSteps = (dir == 1) ? steps : -steps;

        switch (actuatorNo) {
            case 1: stepperTT.move(targetSteps); stepperTT.setMaxSpeed(MAX_SPEED * speedFactor); break;
            case 2: stepperLA1.move(targetSteps); stepperLA1.setMaxSpeed(MAX_SPEED * speedFactor); break;
            case 3: stepperLA2.move(targetSteps); stepperLA2.setMaxSpeed(MAX_SPEED * speedFactor); break;
            case 99:
                // Emergency Stop Condition
                stepperTT.moveTo(stepperTT.currentPosition());
                stepperLA1.moveTo(stepperLA1.currentPosition());
                stepperLA2.moveTo(stepperLA2.currentPosition());
                break;
        }
    }
}