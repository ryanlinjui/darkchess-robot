#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>
#include <ESP8266WiFiMulti.h>
#include <ESP8266mDNS.h>
#include <ESP8266HTTPClient.h>
#include <Arduino.h>

#ifndef SSID_ID
    #define SSID_ID "ssid" // set your ssid
#endif
#ifndef PASSWORD
    #define PASSWORD "password" // set your password
#endif
#ifndef ARM_SYSTEM_URL
    #define ARM_SYSTEM_URL "http://<your-system-ip>:8080/arm" // set your arm system ip
#endif

#define LED_BLUE 13
#define LED_RED 15
#define LED_GREEN 12

#define API_ENDPOINT "/receiver"

ESP8266WebServer server(80);
ESP8266WiFiMulti WiFiMulti;

const int led = 13;

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

void handleNotFound()
{
    return;
}

void showIP()
{
    Serial.print("Local IP: ");
    Serial.println(WiFi.localIP());
}

void setup() 
{
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_BLUE, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    
    Serial.begin(115200);
    delay(3000);
    WiFi.mode(WIFI_STA);
    WiFi.begin(SSID_ID, PASSWORD);
    LED_set_color("r");
    while (WiFi.status() != WL_CONNECTED) 
    {
        Serial.println("Connecting to WiFi...");
        Serial.println(SSID_ID);
        Serial.println(PASSWORD);
        delay(2000);
    }
    LED_set_color("w");    
    Serial.println("WiFi connected");
    showIP();

    server.on(API_ENDPOINT,[]()
    {
        LED_set_color("g");
        server.send(200, "text/plain", "received");
        WiFiClient client;
        HTTPClient http;
        if (http.begin(client, ARM_SYSTEM_URL))
        {
            http.setTimeout(60000);
            int httpCode = http.GET();
            if(httpCode > 0)
            {
                if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_MOVED_PERMANENTLY)
                {
                    String payload = http.getString();
                    Serial.print(payload);
                }
            } 
            http.end();
            
            int count = 0;
            while(!(Serial.available()))
            {
                count++;
                if(count > 25)
                {
                    break;
                }
                delay(200);
            }
            LED_set_color("w");
        }
    });
    server.onNotFound(handleNotFound);
    server.begin();
}

void loop()
{
    if (Serial.available() > 0)
    {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "ip")
        {
            Serial.print("Local IP: ");
            Serial.println(WiFi.localIP());
        }
    }
    if(WiFiMulti.run() != WL_CONNECTED)
    {
        LED_set_color("r");
        delay(500);
    }
    server.handleClient();
    MDNS.update();
    delay(500);
}