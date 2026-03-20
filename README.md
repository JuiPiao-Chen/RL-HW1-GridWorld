# HW1 - Reinforcement Learning Grid World

Flask 網頁應用程式，實作強化學習中的 **策略評估 (Policy Evaluation)** 與 **價值迭代 (Value Iteration)**。

## Demo

> **Live Demo:** [https://chen-huan-rl-hw1.onrender.com](https://chen-huan-rl-hw1.onrender.com)

## Features

### HW1-1: Grid Map Development
- 使用者輸入 n (5~9)，產生 n x n 網格地圖
- 點擊設定起點 (綠色)、終點 (紅色)、障礙物 (灰色，最多 n-2 個)
- 再次點擊已標記格子可取消設定

### HW1-2: Random Policy + Policy Evaluation
- 為每個格子隨機指定行動方向 (↑↓←→)
- 使用 Policy Evaluation 演算法計算每個狀態的價值 V(s)
- 參數：Step Reward = -1, Discount Factor γ = 0.9

### HW1-3 (Bonus): Value Iteration → Optimal Policy
- 執行 Value Iteration 演算法求最佳策略
- 顯示每個格子的最佳行動方向與 V(s)
- 以綠色高亮顯示從起點到終點的最佳路徑

## Tech Stack

- **Backend:** Python / Flask
- **Frontend:** HTML / CSS / JavaScript
- **Algorithm:** Policy Evaluation, Value Iteration (MDP)

## Quick Start

```bash
pip install flask gunicorn
cd HW_1
python app.py
```

Open http://localhost:5000 in browser.

## Project Structure

```
HW_1/
├── app.py                 # Flask backend + RL algorithms
├── templates/
│   └── index.html         # Frontend UI
├── requirements.txt       # Python dependencies
├── a.txt                  # Assignment description
├── HW1-1.png              # Reference screenshot
├── HW1-2.png              # Reference screenshot
└── HW1-3.png              # Reference screenshot
```

## Author

CHEN JUI-PIAO
