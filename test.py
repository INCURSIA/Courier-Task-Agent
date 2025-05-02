from environment import movement, actions
import numpy as np

def test_agent(Q, start_state=None, max_steps=100):
    if start_state is None:
        man_row, man_col = (3,0)
        package_status = 0
        check_point_status = 0
    else:
        man_row, man_col, package_status, check_point_status = start_state

    state = (man_row, man_col, package_status, check_point_status)
    steps = 0
    total_reward = 0
    path = []  

    while steps < max_steps:
        action = np.argmax(Q[state[0], state[1], state[2], state[3], :])
        action_name = actions[action]

        print(f"Step {steps}: At {state} → Action: {action_name}")
        next_state, reward, done = movement(state, action)
        total_reward += reward
        state = next_state
        steps += 1
        path.append(state)

        if done:
            print(f"\nAgent successfully completed the mission in {steps} steps!")
            print(f"Total reward: {total_reward}")
            print(f"Path taken: {path}")
            break
    else:
        print("\nAgent failed to complete the mission within step limit.")
        print(f"Final state: {state} | Total reward: {total_reward}")
        print(f"Path taken: {path}")
