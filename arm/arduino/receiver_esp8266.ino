#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>
#include <ESP8266WiFiMulti.h>
#include <ESP8266mDNS.h>
#include <ESP8266HTTPClient.h>
#include <Arduino.h>

#ifndef STASSID
    #define STASSID "ssid"      // set your ssid
    #define STAPSK  "password"  // set your password
#endif

#define LED_BLUE 13
#define LED_RED 15
#define LED_GREEN 12

#define DARKCHESS_ROBOT_URL "http://0.0.0.0:8080/arm"
#define ENDPOINT "/receiver"

const char *ssid = STASSID;
const char *password = STAPSK;

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

void setup(void) 
{
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_BLUE, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    
    Serial.begin(115200);
    delay(3000);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    LED_set_color("r");
    while (WiFi.status() != WL_CONNECTED) 
    {
        delay(500);
    }
    LED_set_color("w");
    
    Serial.println(WiFi.localIP());
    
    server.on(ENDPOINT,[]() 
    {
        LED_set_color("g");
        WiFiClient client;
        HTTPClient http;
        if (http.begin(client, DARKCHESS_ROBOT_URL))
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
            
            while(!(Serial.available()))
            {
                delay(200);
            }
            server.send(200, "text/plain", "done");
            LED_set_color("w");
        }
    });
    server.onNotFound(handleNotFound);
    server.begin();
}

void loop(void) 
{
    if(WiFiMulti.run() != WL_CONNECTED)
    {
        LED_set_color("r");
        delay(500);
    }   
    server.handleClient();
    MDNS.update();
}
