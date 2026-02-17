import streamlit as st
import numpy as np

# -----------------------------
# Q-Learning Agent
# -----------------------------
class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((2, 2, 2))  # (holding_state, gain_state, action)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        return np.argmax(self.q_table[state[0], state[1]])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] += self.alpha * (
            reward + self.gamma * best_next -
            self.q_table[state[0], state[1], action]
        )


# -----------------------------
# Investment Environment
# -----------------------------
class InvestmentEnv:
    def __init__(self, initial_investment, stcg_rate, ltcg_rate,
                 transaction_cost_rate, monthly_drift):

        self.initial_investment = initial_investment
        self.stcg_rate = stcg_rate
        self.ltcg_rate = ltcg_rate
        self.transaction_cost_rate = transaction_cost_rate
        self.monthly_drift = monthly_drift

        self.reset()

    def reset(self):
        self.price = 100
        self.holding_period = 0
        self.last_rebalance_price = self.price
        self.portfolio_value = self.initial_investment
        return self.get_state()

    def get_state(self):
        holding_state = 0 if self.holding_period < 12 else 1
        gain = self.price - self.last_rebalance_price
        gain_state = 0 if gain <= 0 else 1
        return (holding_state, gain_state)

    def step(self, action):
        self.price += np.random.normal(self.monthly_drift, 2)
        self.holding_period += 1

        reward = 0

        if action == 1:  # Rebalance
            gain = self.price - self.last_rebalance_price

            if gain > 0:
                if self.holding_period < 12:
                    tax = self.stcg_rate * gain
                else:
                    tax = self.ltcg_rate * gain
            else:
                tax = 0

            transaction_cost = self.transaction_cost_rate * abs(gain)

            reward = gain - tax - transaction_cost

            self.last_rebalance_price = self.price
            self.holding_period = 0

        else:
            reward = -0.05  # waiting penalty

        return self.get_state(), reward, False


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 RL-Based Smart Investment Rebalancing Simulator")

st.sidebar.header("Investor Parameters")

initial_investment = st.sidebar.number_input("Initial Investment", value=10000.0)
stcg_rate = st.sidebar.number_input("Short-Term Tax Rate", value=0.15)
ltcg_rate = st.sidebar.number_input("Long-Term Tax Rate", value=0.10)
transaction_cost_rate = st.sidebar.number_input("Transaction Cost Rate", value=0.01)
monthly_drift = st.sidebar.number_input("Expected Monthly Return", value=1.0)
investment_horizon = st.sidebar.slider("Investment Horizon (Months)", 12, 120, 60)

episodes = st.sidebar.slider("Training Episodes", 1000, 10000, 3000)

if st.button("Run Simulation"):

    env = InvestmentEnv(initial_investment,
                        stcg_rate,
                        ltcg_rate,
                        transaction_cost_rate,
                        monthly_drift)

    agent = QLearningAgent()

    # -------- Training --------
    for episode in range(episodes):
        state = env.reset()

        for step in range(investment_horizon):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

        agent.epsilon *= 0.995
        agent.alpha *= 0.995

    # -------- Testing --------
    state = env.reset()
    total_reward = 0
    rebalance_count = 0

    for step in range(investment_horizon):
        action = np.argmax(agent.q_table[state[0], state[1]])

        if action == 1:
            rebalance_count += 1

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    # -------- Results --------
    st.subheader("📊 Simulation Results")
    st.write("Total Post-Tax Reward:", round(total_reward, 2))
    st.write("Number of Rebalances:", rebalance_count)

    if total_reward > 0:
        st.success("Strategy Generated Profit ✅")
    else:
        st.error("Strategy Resulted in Loss ❌")
