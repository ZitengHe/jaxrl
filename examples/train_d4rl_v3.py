import d4rl
env_name = "HalfCheetah-v3"
try:
    score = d4rl.get_normalized_score(env_name, 1000)
    print(f"Normalized Score: {score}")
except Exception as e:
    print(f"Error: {e}")