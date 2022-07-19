#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WiFiMulti.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>

#define LED_BLUE 13
#define LED_RED 15
#define LED_GREEN 12
#define BUTTON 16

ESP8266WiFiMulti WiFiMulti;

void LED_set_color(String color)
{
  if(color == "r")
  {
    digitalWrite(LED_RED,HIGH);
    digitalWrite(LED_BLUE,LOW);
    digitalWrite(LED_GREEN,LOW);
  }
  else if(color == "g")
  {
    digitalWrite(LED_RED,LOW);
    digitalWrite(LED_GREEN,HIGH);
    digitalWrite(LED_BLUE,LOW);
  }
  else if(color == "b")
  {
    digitalWrite(LED_RED,LOW);
    digitalWrite(LED_GREEN,LOW);
    digitalWrite(LED_BLUE,HIGH);
  }
  else if(color == "w")
  {
    digitalWrite(LED_RED,HIGH);
    digitalWrite(LED_BLUE,HIGH);
    digitalWrite(LED_GREEN,HIGH);
  }
  else if(color == "off")
  {
    digitalWrite(LED_RED,LOW);
    digitalWrite(LED_BLUE,LOW);
    digitalWrite(LED_GREEN,LOW);
  }
}

void setup() 
{
  pinMode(LED_BUILTIN,OUTPUT);
  pinMode(LED_RED,OUTPUT);
  pinMode(LED_BLUE,OUTPUT);
  pinMode(LED_GREEN,OUTPUT);
  pinMode(BUTTON,INPUT);
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFiMulti.addAP("ssid", "password");
  analogWrite(BUTTON,0);
}

void loop()
{
  if(WiFiMulti.run() != WL_CONNECTED)
  {
    LED_set_color("r");
    delay(500);
  }   
  else if(digitalRead(BUTTON)==HIGH)
  { 
    int num = 0;
    String cc = "g";
    String url = "http://172.20.10.5/control";

    while(digitalRead(BUTTON)==HIGH)
    {
      num += 1;
      analogWrite(BUTTON,0);
      delay(50);
      if(num>60)
      {
        cc = "b";
        url = "http://172.20.10.4:814/reset_all";
        LED_set_color(cc);
        delay(2500);
        break;
      }
    }

    WiFiClient client;
    HTTPClient http;

    if(http.begin(client,url))
    {
      int httpCode = http.GET();
      if (httpCode > 0) 
      {
        if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_MOVED_PERMANENTLY)
        {
          LED_set_color(cc);
        }
      } 
      http.end();
      delay(500);
      analogWrite(BUTTON,0);
    }
  }
  else
  {
    LED_set_color("w");
  }
}
