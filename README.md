# AIoT Darkchess Robot

## Introduce
The concept of Artificial Intelligence of Things (AIoT) is gaining traction in various aspects of our life. In research, we develop an AIoT darkchess Robot. This platform will let developers to utilize cloud API through HTTP protocols, allowing for the creation of diverse darkchess programs and robots. The foundational setup involves a robot equipped with cloud-connected IPCAM and robotic arms. The system retrieves darkchess board images from IPCAM and employs Convolutional Neural Network (CNN) for board recognition. AI-generated move are given to robotic system, and the robotic arm executes actions like lipping, moving, and capturing darkchess. This design not only facilitates building a comprehensive chess database but also enhances AI's chess skills through Deep Reinforcement Learning. With integrated robotic arm gameplay and CNN board recognition, the AIoT architecture aims to make darkchess robots a common presence in various environments.

![darkchess-robot](https://github.com/ryanlinjui/darkchess-robot/blob/main/assets/images/darkchess-robot.png?raw=true)

## Features

### Brain
- Min-Max and Alpha-Beta winrate up to 50%
- Attempt using AlphaZero construct Deep Reinforce Learning system

### Eye
- Convolutional Neural Network recognition up to 98.9 % accuracy
- Smartphone camera as its eye

### Arm

- Construct by 3D drawing & printing and EDA printed circuit board
- XYZrobot Smart Servo [A1-16](https://www.pololu.com/product/3400) as its joint

### AIoT
- API endpoints connecting the brain, eye cloud systems
- With the power of AIoT, developing your own robotic system becomes more convenient


## Reference
- [研究報告書](https://www.mxeduc.org.tw/scienceaward/history/projectDoc/19th/doc/SA19-120_final.pdf)

- [影片介紹](https://www.youtube.com/watch?v=iaBYF3ZuBAg)

- [暗棋機器人-發明專利](https://twpat3.tipo.gov.tw/twpatc/twpatkm?.7bf093500010100000001032000100000005^0000000000000109F104113)
- [機器手臂夾爪-新型專利](https://twpat3.tipo.gov.tw/twpatc/twpatkm?.90230130100000000000^0500000010000234000000000110000599F41b6)

## Awards
- 2020第7屆高雄KIDE國際發明暨設計展
    - 金牌獎（國際 Top 50 / 408）

- [第19屆旺宏科學獎](https://www.mxeduc.org.tw/scienceaward/old.htm)
    - 優等 (全國 Top 20 / 661)

- [第12屆i-ONE國研盃儀器科技創新獎](https://i-one.org.tw/Home/ListContents/107?ATimes=12)
    - 二獎 (全國 2 / 200 up)

- [中華民國第60屆中小學科學展覽會](https://twsf.ntsec.gov.tw/activity/race-1/60/pdf/NPHSF2020-052310.pdf?746)
    - 工程學科（一）第二名  (全國 2 / 151)

- [臺北市第53屆中小學科學展覽會](https://sites.google.com/csjh.tp.edu.tw/science/高級中等學校組/工程學科一?authuser=0#h.6xilplkz0fpy)
    - 工程學科（一）特優

- 臺北市109年度中等學校學生科學研究獎助計畫
    - 決審（未去現場）

- 全國高級中等學校電機與電子群109年專題及創意製作競賽(複賽) 
    - 專題組 佳作

- 全國高級中等學校小論文寫作比賽
    - 1091015梯次 優等 機械手臂在暗棋棋盤上之應用
    - 1090325梯次 甲等 運用類神經網路進行暗棋棋局分類辨識之研究
    - 1081031梯次 甲等 暗棋機器人

- 臺北市立內湖高工專題比賽
    - 校內 特優
    - 資訊科 特優