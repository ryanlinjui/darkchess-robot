#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WiFiMulti.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>

#ifndef SSID_ID
    #define SSID_ID "ssid" // set your ssid
#endif
#ifndef PASSWORD
    #define PASSWORD "password" // set your password
#endif
#ifndef ARM_SYSTEM_RESET_URL
    #define ARM_SYSTEM_RESET_URL "http://<your-system-ip>:8080/reset" // set your arm system ip
#endif
#ifndef RECEIVER_URL
    #define RECEIVER_URL "http://<your-receiver-ip>/receiver" // set your receiver ip
#endif

#define LED_BLUE 13
#define LED_RED 15
#define LED_GREEN 12
#define BUTTON 16

ESP8266WiFiMulti WiFiMulti;

void LED_set_color(String color)
{
    if(color == "r")
    {
        digitalWrite(LED_RED, HIGH);
        digitalWrite(LED_BLUE, LOW);
        digitalWrite(LED_GREEN, LOW);
    }
    else if(color == "g")
    {
        digitalWrite(LED_RED, LOW);
        digitalWrite(LED_GREEN, HIGH);
        digitalWrite(LED_BLUE, LOW);
    }
    else if(color == "b")
    {
        digitalWrite(LED_RED, LOW);
        digitalWrite(LED_GREEN, LOW);
        digitalWrite(LED_BLUE, HIGH);
    }
    else if(color == "w")
    {
        digitalWrite(LED_RED, HIGH);
        digitalWrite(LED_BLUE, HIGH);
        digitalWrite(LED_GREEN, HIGH);
    }
    else if(color == "off")
    {
        digitalWrite(LED_RED, LOW);
        digitalWrite(LED_BLUE, LOW);
        digitalWrite(LED_GREEN, LOW);
    }
}

void setup() 
{
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_BLUE, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(BUTTON, INPUT);
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    WiFiMulti.addAP(SSID_ID, PASSWORD);
    analogWrite(BUTTON, 0);
}

void loop()
{
    if(WiFiMulti.run() != WL_CONNECTED) // WiFi not connected, try to reconnect, LED red
    {   
        LED_set_color("r");
        delay(500);
    }   
    else if(digitalRead(BUTTON) == HIGH) // Button pressed
    { 
        int num = 0;
        String url = RECEIVER_URL;

        while(digitalRead(BUTTON) == HIGH) // Button pressed for 3 seconds, LED blue
        {
            num += 1;
            analogWrite(BUTTON, 0);
            delay(50);
            if(num > 60)
            {
                url = ARM_SYSTEM_RESET_URL;
                LED_set_color("b");
                delay(2500);
                break;
            }
        }
        
        // Button pressed for less than 3 seconds, LED green
        WiFiClient client;
        HTTPClient http;
        if(http.begin(client, url))
        {
            LED_set_color("g");
            delay(1000);
            int httpCode = http.GET();
            if (httpCode > 0) // HTTP request success, LED blink green
            {
                if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_MOVED_PERMANENTLY)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        LED_set_color("g");
                        delay(500);
                        LED_set_color("off");
                        delay(500);
                    }
                }
            }
            else // HTTP request failed, LED red
            {
                LED_set_color("r");
                delay(2500);
            }

            http.end();
            delay(500);
            analogWrite(BUTTON, 0);
        }
    }
    else // WiFi connected, LED white
    {
        LED_set_color("w");
    }
}