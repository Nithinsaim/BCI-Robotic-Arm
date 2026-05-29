/*
 * arduino_control.ino
 * BCI Robotic Arm — Arduino Uno Servo Controller
 * Receives: LEFT | RIGHT | REST via Serial from MATLAB
 * Controls: 4x SG90 Servo Motors on PWM pins D3, D5, D6, D9
 */
#include <Servo.h>

Servo shoulder, elbow, wrist, gripper;

void setup() {
  Serial.begin(9600);
  shoulder.attach(3);
  elbow.attach(5);
  wrist.attach(6);
  gripper.attach(9);
  resetArm();
}

void resetArm() {
  shoulder.write(90); elbow.write(90);
  wrist.write(90);    gripper.write(90);
}

void moveLeft() {
  shoulder.write(45); elbow.write(90);
  wrist.write(90);    gripper.write(45);
  delay(500);         resetArm();
}

void moveRight() {
  shoulder.write(135); elbow.write(90);
  wrist.write(90);     gripper.write(135);
  delay(500);          resetArm();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if      (cmd == "LEFT")  moveLeft();
    else if (cmd == "RIGHT") moveRight();
    else if (cmd == "REST")  resetArm();
  }
}
