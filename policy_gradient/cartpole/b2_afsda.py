from gym.wrappers import RecordVideo
import gym

env = gym.make("AlienDeterministic-v4", render_mode="human")
env = preprocess_env(env)  # method with some other wrappers
env = RecordVideo(env, 'video', episode_trigger=lambda x: x == 2)
env.start_video_recorder()

for episode in range(4):
        state = env.reset()[0]
        done = False
        while not done:
            action = self.select_smart_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            env.render()

    env.close()























