# trading_robots_project

A reinforcement learning-based trading bot designed to make buy/sell decisions based on technical indicators and historical price data. This project supports both training and interaction with a simulated trading environment.

---

## 📁 Project Structure

```
.
├── drl_agent.py               # Deep RL agent (model & training logic)
├── interact.py                # Interact with the simulated trading website
├── Stock_API.py               # Fetch historical stock data and technical indicators
├── StockTradingEnv.py         # Custom trading environment compatible with Gym-style agents
├── training.py                # Main training script
├── RL_Training_Analysis.ipynb # Jupyter notebook for analyzing results
├── requirement.txt            # Dependencies
├── README.md                  # Project description (this file)
```

---

## 🚀 Usage

### 1. Install dependencies

```bash
pip install -r requirement.txt
```

### 2. Start training the agent

```bash
python training.py
```

The training process will generate `.npy` files for logs and `.pth` files for saved models. (These can be organized later into dedicated folders.)

### 3. Interact with the model in a simulated trading environment

After training completes and a model is saved, run the following to begin trading interaction:

```bash
python interact.py
```

---

## 👤 Author

Developed by [JIA-DE JIANG](https://github.com/p76121699)

---

## 📄 License

This project is provided for academic and educational use only. You may extend or modify it freely.
"# trading_robots_project" 
