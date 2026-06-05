#define PI 3.141592

 -- 모터 핀 & 엔코더 핀 --
const byte PWMPin_r = 9, DirPin1_r = 10, DirPin2_r = 11;
const byte PWMPin_l = 6, DirPin1_l = 7, DirPin2_l = 8;

const byte L_Encoder_CHA = 3;
const byte L_Encoder_CHB = 5;
const byte R_Encoder_CHA = 2;
const byte R_Encoder_CHB = 4;

 -- 바퀴 & 엔코더 파라미터 (이미 주어진 값) --
const float wheel_R = 0.034;  바퀴 반지름 (m)
const float PPR = 11.0;
const float REDUCTION = 46.0;
const float ENCODER_CH = 2.0;
const float COUNTS_PER_REV = PPR  ENCODER_CH  REDUCTION;  1012

 -- 속도 제어 파라미터 (실험적으로 변경) --
const int MAX_PWM   = 250;   최대 PWM
const int MIN_PWM   = 50;    최소 PWM (마찰 극복)
const int PIVOT_PWM = 160;   제자리 회전 PWM
const float Kp_pos = 0.4;  잔여 카운트 → PWM 변수 (감속 프로파일)
const float Kb = 1.5;      좌우 균형 보정 변수

volatile long EncoderCount_L = 0;
volatile long EncoderCount_R = 0;


 -- 엔코더 인터럽트  2채널 엔코더의 CHA 핀에서 인터럽트 발생 → 방향 판별 후 카운트 증가감소 --
void EI_L_Encoder()
{
    bool chA = digitalRead(L_Encoder_CHA);
    bool chB = digitalRead(L_Encoder_CHB);
    if (chA == HIGH)
        EncoderCount_L += (chB == LOW)  1  -1;
    else
        EncoderCount_L += (chB == LOW)  -1  1;
}

void EI_R_Encoder()
{
    bool chA = digitalRead(R_Encoder_CHA);
    bool chB = digitalRead(R_Encoder_CHB);
    if (chA == HIGH)
        EncoderCount_R += (chB == LOW)  1  -1;
    else
        EncoderCount_R += (chB == LOW)  -1  1;
}


 -- 모터 제어 함수  PWM과 방향을 설정하여 모터를 구동 --
void setMotor(byte pwmPin, byte dir1, byte dir2, int pwm, bool forward)
{
    analogWrite(pwmPin, pwm);
    digitalWrite(dir1, forward);
    digitalWrite(dir2, !forward);
}

 -- 능동 브레이크 양쪽 핀 HIGH → 역기전력으로 즉시 제동 --
void brakeMotors()
{
    analogWrite(PWMPin_r, 0);
    digitalWrite(DirPin1_r, HIGH);
    digitalWrite(DirPin2_r, HIGH);
    analogWrite(PWMPin_l, 0);
    digitalWrite(DirPin1_l, HIGH);
    digitalWrite(DirPin2_l, HIGH);
}

void stopMotors()
{
    analogWrite(PWMPin_r, 0);
    digitalWrite(DirPin1_r, LOW);
    digitalWrite(DirPin2_r, LOW);
    analogWrite(PWMPin_l, 0);
    digitalWrite(DirPin1_l, LOW);
    digitalWrite(DirPin2_l, LOW);
}

 -- 제자리 피벗 회전 --
 dir  0 우회전 (PWMPin_r 전진, PWMPin_l 후진)
 dir  0 좌회전 (PWMPin_r 후진, PWMPin_l 전진)
 ※ driveContinuous와 동일한 핀 스왑 기준 유지
void pivotTurn(float dir)
{
    bool r_fwd = (dir  0);
    setMotor(PWMPin_r, DirPin1_r, DirPin2_r, PIVOT_PWM,  r_fwd);
    setMotor(PWMPin_l, DirPin1_l, DirPin2_l, PIVOT_PWM, !r_fwd);
}


 -- Setup   초기설정 --
void setup()
{
    Serial.begin(9600);   아두이노 통신
    Serial1.begin(460800);  라즈베리파이 통신

    pinMode(PWMPin_r, OUTPUT);
    pinMode(DirPin1_r, OUTPUT);
    pinMode(DirPin2_r, OUTPUT);
    pinMode(PWMPin_l, OUTPUT);
    pinMode(DirPin1_l, OUTPUT);
    pinMode(DirPin2_l, OUTPUT);

    pinMode(L_Encoder_CHA, INPUT_PULLUP);
    pinMode(L_Encoder_CHB, INPUT_PULLUP);
    pinMode(R_Encoder_CHA, INPUT_PULLUP);
    pinMode(R_Encoder_CHB, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(L_Encoder_CHA), EI_L_Encoder, CHANGE);
    attachInterrupt(digitalPinToInterrupt(R_Encoder_CHA), EI_R_Encoder, CHANGE);

    stopMotors();
}


 -- 연속 주행 제어 (장애물 회피용) --
 steer   -1.0(좌회전) ~ +1.0(우회전)
 speed_f 0.0 ~ 1.0 (정규화 속도)
 forward true=전진, false=후진
 pwm_L = MAX_PWM  speed_f  (1 + steer)
 pwm_R = MAX_PWM  speed_f  (1 - steer)
void driveContinuous(float steer, float speed_f, bool forward)
{
    int base  = (int)(MAX_PWM  speed_f);
    int pwm_L = constrain((int)(base  (1.0f + steer)), MIN_PWM, MAX_PWM);
    int pwm_R = constrain((int)(base  (1.0f - steer)), MIN_PWM, MAX_PWM);
    setMotor(PWMPin_r, DirPin1_r, DirPin2_r, pwm_L, forward);
    setMotor(PWMPin_l, DirPin1_l, DirPin2_l, pwm_R, forward);
}

 -- 통신 타임아웃 (명령이 끊기면 자동 정지) --
const unsigned long TIMEOUT_MS = 2000; 
unsigned long lastCmdMs = 0; 
String rxBuf = ;

 -- Loop   반복실행 (장애물 회피 명령 수신) --
 수신 프로토콜 (Serial1, 460800 baud)
   F steer speedn  → 전진  steer(-1.00~+1.00)  speed(0.00~1.00)
   T dirn          → 제자리 피벗  dir(+1.00=우  -1.00=좌)
   B speedn        → 후진
void loop()
{
     타임아웃 마지막 명령 이후 TIMEOUT_MS ms 초과 시 자동 정지
    if (millis() - lastCmdMs  TIMEOUT_MS)
        stopMotors();

    while (Serial1.available())
    {
        char c = (char)Serial1.read();
        if (c == 'n')
        {
            rxBuf.trim();
            if (rxBuf.length()  0)
            {
                lastCmdMs = millis();
                char cmd = rxBuf.charAt(0);

                if (cmd == 'F' && rxBuf.length()  2)
                {
                     F steer speed
                    String args  = rxBuf.substring(2);
                    int    sp    = args.indexOf(' ');
                    float  steer = (sp = 0)  args.substring(0, sp).toFloat()  0.0f;
                    float  speed = (sp = 0)  args.substring(sp + 1).toFloat()  0.0f;
                    steer = constrain(steer, -1.0f, 1.0f);
                    speed = constrain(speed,  0.0f, 1.0f);
                    driveContinuous(steer, speed, true);
                    
                }
                else if (cmd == 'T' && rxBuf.length()  2)
                {
                     T dir  (+1.00=우회전, -1.00=좌회전)
                    float dir = (rxBuf.substring(2).toFloat() = 0)  1.0f  -1.0f;
                    pivotTurn(dir);
                }
                else if (cmd == 'B' && rxBuf.length()  2)
                {
                     B speed
                    float speed = constrain(rxBuf.substring(2).toFloat(), 0.0f, 1.0f);
                    driveContinuous(0.0f, speed, false);
                }
                else if (cmd == 'S') { brakeMotors(); }
                else
                {
                    Serial.print([ERR] 알 수 없는 명령 ); Serial.println(rxBuf);
                    rxBuf = ;  쓰레기 누적 방어
                }
            }
            rxBuf = ;
        }
        else
        {
            rxBuf += c;
             if (rxBuf.length()  32) rxBuf = ;   쓰레기 누적 방어

        }
    }
}