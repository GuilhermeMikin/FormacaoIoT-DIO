int buz = 10;
int led = 11;
int motor = 12;
int sensor = 0;

int temp;

void setup()
{
  pinMode(buz, OUTPUT);
  pinMode(led, OUTPUT);
  pinMode(motor, OUTPUT);
  pinMode(sensor, INPUT);
  
  Serial.begin(9600);
  
  digitalWrite(led, LOW);
  digitalWrite(buz, LOW);
  digitalWrite(motor, LOW);
}

void loop()
{
  temp = analogRead(sensor);
  Serial.print("Valor temp: ");
  Serial.println(temp);
  
  if (temp >= 30 and temp < 50)
  {
    digitalWrite(motor, HIGH);
    digitalWrite(led, LOW);
    digitalWrite(buz, LOW);
  }
  if (temp >= 50)
  {
    digitalWrite(led, HIGH);
    digitalWrite(buz, HIGH);
  }
  if (temp <30)
  {
    digitalWrite(led, LOW);
    digitalWrite(buz, LOW);
    digitalWrite(motor, LOW);
  }
  delay(1000);
}