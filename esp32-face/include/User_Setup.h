// TFT_eSPI display configuration for PlatformIO
// Copy the settings from your Arduino IDE TFT_eSPI/User_Setup.h here.
// That file lives at: <Arduino libraries>/TFT_eSPI/User_Setup.h

// ---- DRIVER: uncomment the one matching your display ----
//#define ILI9341_DRIVER
//#define ST7789_DRIVER
//#define ST7735_DRIVER
//#define ILI9163_DRIVER

// ---- SPI PINS: set to your actual wiring ----
#define TFT_MISO 19
#define TFT_MOSI 23
#define TFT_SCLK 18
#define TFT_CS   5
#define TFT_DC   2
#define TFT_RST  -1   // or a real GPIO if you have a reset pin

// ---- OPTIONAL FONTS ----
#define LOAD_GLCD
#define LOAD_FONT2

// ---- SPI SPEED ----
#define SPI_FREQUENCY  40000000
