#include "driver/i2s.h"
#include <ESP32Servo.h>
#include <TFT_eSPI.h>
#include <math.h>

// --- PIN DEFINITIONS ---
#define I2S_SD  32
#define I2S_WS  15
#define I2S_SCK 14
#define SERVO_PIN 13

// --- AUDIO CONFIG ---
#define SAMPLE_RATE 16000
#define NUM_SAMPLES 512
int32_t samples[NUM_SAMPLES * 2];
float leftChan[NUM_SAMPLES];
float rightChan[NUM_SAMPLES];

// --- OBJECTS ---
TFT_eSPI tft = TFT_eSPI();
Servo panServo;

// --- STATE VARIABLES ---
#define ROBOT_BLUE TFT_CYAN
#define BG_COLOR TFT_BLACK
int eyeY = 30;
int leftEyeBaseX = 38;
int rightEyeBaseX = 103;
int eyeOffset = 0;

float currentAngle = 0;

void setup() {
  Serial.begin(115200);

  // 1. Initialize TFT
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(BG_COLOR);

  // 2. Initialize Servo
  panServo.attach(SERVO_PIN);
  panServo.write(90); // Center

  // 3. Initialize I2S Microphones
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .dma_buf_count = 4,
    .dma_buf_len = NUM_SAMPLES,
    .use_apll = false
  };
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);

  Serial.println("Robot Face Active!");
}

void loop() {
  size_t bytesRead;
  i2s_read(I2S_NUM_0, samples, sizeof(samples), &bytesRead, portMAX_DELAY);

  // --- 1. SIGNAL PROCESSING ---
  for (int i = 0; i < NUM_SAMPLES; i++) {
    leftChan[i] = (float)(samples[i * 2] >> 14);
    rightChan[i] = (float)(samples[i * 2 + 1] >> 14);
  }

  // --- 2. DIRECTION CALCULATION (CROSS-CORRELATION) ---
  float maxCorr = -1e10;
  int bestShift = 0;
  int maxSearch = 12; // Adjusted for 8cm distance

  for (int shift = -maxSearch; shift <= maxSearch; shift++) {
    float corr = 0;
    for (int i = maxSearch; i < NUM_SAMPLES - maxSearch; i++) {
      corr += leftChan[i] * rightChan[i + shift];
    }
    if (corr > maxCorr) {
      maxCorr = corr;
      bestShift = shift;
    }
  }

  // Convert to degrees
  float timeDelay = (float)bestShift / SAMPLE_RATE;
  float sinTheta = (timeDelay * 343.0) / 0.08; // 8cm dist
  if (sinTheta > 1.0) sinTheta = 1.0;
  if (sinTheta < -1.0) sinTheta = -1.0;
  float angleDeg = asin(sinTheta) * 180.0 / PI;

  // --- 3. ACTUATION (SERVO + EYES) ---
  // Only move if the sound is strong enough (maxCorr threshold)
  if (maxCorr > 500000) {
    // Smooth angle update
    currentAngle = (currentAngle * 0.7) + (angleDeg * 0.3);

    // Servo: 0 is Left, 90 Center, 180 Right
    panServo.write(90 + (int)currentAngle);

    // Eyes: set offset based on angle
    if (currentAngle < -15) eyeOffset = -16;      // Look Left
    else if (currentAngle > 15) eyeOffset = 16;   // Look Right
    else eyeOffset = 0;                           // Look Center

    drawFace();
  }
}

void drawFace() {
  // Simple optimized redraw: only clear eyes area to prevent flickering
  tft.fillRect(leftEyeBaseX - 20, eyeY, 130, 60, BG_COLOR);

  int leftX = leftEyeBaseX + eyeOffset;
  int rightX = rightEyeBaseX + eyeOffset;

  // Draw Neutral Eyes
  tft.fillRoundRect(leftX, eyeY, 26, 50, 12, ROBOT_BLUE);
  tft.fillRoundRect(rightX, eyeY, 26, 50, 12, ROBOT_BLUE);
}
