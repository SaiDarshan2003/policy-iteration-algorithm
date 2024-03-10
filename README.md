# POLICY ITERATION ALGORITHM

## AIM:
To find an optimal policy for a given MDP and evaluate its performance. It uses the Policy Iteration algorithm to iteratively improve the policy until convergence, resulting in an optimal policy and its associated state-value function.

## PROBLEM STATEMENT:
The problem being addressed in this code is a specific MDP called "Slippery Walk Five." The MDP is defined by its transition probabilities, rewards, and terminal states. The objective is to find a policy that maximizes the expected cumulative reward when navigating from the initial state to a goal state in this MDP.

## POLICY ITERATION ALGORITHM:

**Initialization**:
- Initialize a policy arbitrarily.
- Initialize a value function arbitrarily.

**Policy Evaluation (Prediction):**

- Given the current policy, calculate the state-value function (V) that estimates the expected cumulative rewards starting from each state.
- Iterate until V converges:
  - For each state, update V based on the expected rewards and transitions following the current policy.

**Policy Improvement (Control):**
- Given the current state-value function (V), update the policy to be greedy with respect to V.
- For each state, choose actions that maximize the expected cumulative reward.

**Iteration:**
- Repeat steps 2 and 3 until the policy no longer changes (convergence).

## POLICY IMPROVEMENT FUNCTION:
```python3
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
    new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))} [s]         
    return new_pi
```

## POLICY ITERATION FUNCTION:
```python3
def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi=lambda s: {s:a for s, a in enumerate(random_actions)} [s]
    while True:
      old_pi={s:pi(s) for s in range(len(P))}
      V=policy_evaluation(pi,P,gamma,theta)
      pi=policy_improvement(V,P,gamma)
      if old_pi=={s:pi(s) for s in range(len(P))}:
        break
    return V, pi
```

## OUTPUT:

![image](https://github.com/SaiDarshan2003/policy-iteration-algorithm/assets/94692595/e886312d-b34d-49d2-82cd-df58be061e0b)


![image](https://github.com/SaiDarshan2003/policy-iteration-algorithm/assets/94692595/84487d56-0268-4c9f-bd54-a7554f377dcb)


![image](https://github.com/SaiDarshan2003/policy-iteration-algorithm/assets/94692595/aa0172fe-80ae-4a0f-9610-4e0a7ee3a93a)



## RESULT:
Thus, policy iteration algorithm is used to iteratively improve the policy until convergence, resulting in an optimal policy and its associated state-value function.
