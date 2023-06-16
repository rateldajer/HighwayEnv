import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env


TRAIN = False #True

if __name__ == '__main__':
    exp_str = "intersection_dqn_w256_w256_b15k_s40k"
    # Create the environment
    # env = gym.make("highway-fast-v0", render_mode="rgb_array")
    # env = gym.make("roundabout-v0", render_mode="rgb_array")
    env = gym.make("intersection-v0", render_mode="rgb_array")
    obs, info = env.reset()

    # Create the model
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                # policy_kwargs=dict(net_arch=[64, 64]),
                learning_rate=5e-4,
                buffer_size=15000, #15000
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log=exp_str+"/")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(4e4)) #2e4
        model.save(exp_str+"/model")
        del model

    # Run the trained model and record video
    model = DQN.load(exp_str+"/model", env=env)
    env = RecordVideo(env, video_folder=exp_str+"/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(200): #why 10? what if I did more?
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
