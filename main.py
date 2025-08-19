# main.py
import pygame
from environment import PongEnv
from agent import DQNAgent
import random
import pygame
import time
import os
"""
TODO: 
- Give reward for hitting the ball
- Remove opponents y position from state and vice versa
- Fix epsilon decay 
- Refactor for steps
"""

def train(modelPathPlayer=None, modelPathOpponent=None, startFrom=0, humanMode=False):
    env = PongEnv()
    playerAgent = DQNAgent(stateDim=5, actionDim=3)  # State: ball.x, ball.y, ball.speed[0](x), ball.speed[1](y), player.y Actions: 0=up, 1=down, 2=stay
    #opponentAgent = DQNAgent(stateDim=5, actionDim=3)  # For right paddle 
    clock = pygame.time.Clock()
    episodes = 10000
    batchSize = 64

    if modelPathOpponent and modelPathPlayer:
        try:
            playerAgent.load(modelPathPlayer)
            #opponentAgent.load(modelPathOpponent)
            if startFrom > 10:
                playerAgent.epsilon = 0.0
                #opponentAgent.epsilon = 0.0
            print(f"Models loaded from {modelPathPlayer} and {modelPathOpponent}")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            env.close()
            return
    else:
        print("No models provided, starting training from scratch.")

    print("Pure self play as same agent controls both paddles.")

    for episode in range(startFrom, episodes):
        env.reset()
        playerState = env.getState(perspective='player')
        opponentState = env.getState(perspective='opponent')
        playerTotalReward = 0
        opponentTotalReward = 0

        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        humanMode = not humanMode

            # Player action
            if humanMode:
                keys = pygame.key.get_pressed()
                playerAction = 2  # Stay
                if keys[pygame.K_UP]:
                    playerAction = 0  # Up
                if keys[pygame.K_DOWN]:
                    playerAction = 1  # Down
            else:
                playerAction = playerAgent.act(playerState)

            # Opponent action (self-play or random in human mode)
            opponentAction = playerAgent.act(opponentState)  #opentAgent.act(opponentState)

            playerNextState, opponentNextState, playerReward, opponentReward, done = env.step(playerAction, opponentAction)
            playerAgent.replayBuffer.push(playerState, playerAction, playerReward, playerNextState, done)
            playerAgent.replayBuffer.push(opponentState, opponentAction, opponentReward, opponentNextState, done) # oppenentAgent.replayBufffer
            playerState = playerNextState
            opponentState = opponentNextState
            playerTotalReward += playerReward
            opponentTotalReward += opponentReward
            playerAgent.train(batchSize)
            #print("Player trained")
            playerAgent.train(batchSize) # opponentAgent.train(batchSize)
            #print("Opponent trained")
            # Render and control frame rate
            env.render(episode=episode + 1)
            clock.tick(60)

            if done:
                #playerAgent.updateTargetModel()
                #opponentAgent.updateTargetModel()
                print(f"Episode {episode + 1}, Player Reward: {playerTotalReward}, Opponent Reward: {opponentTotalReward}, "
                      f"Player Epsilon: {playerAgent.epsilon:.2f}, Opponent Epsilon: {playerAgent.epsilon:.2f}") # opponentAgent.epsilon:.2f}")
                break

        # Save model periodically
        if episode % 10 == 0:
            playerAgent.save(f"models/v3/pong_model_{episode}_v3.pth")
            #opponentAgent.save(f"models/v2/pong_opponent_model_{episode}_v2.pth")
            # remove previous model 
            #if episode > 10: 
            #    os.remove(f"models/v3/pong_model_{episode - 5}_v3.pth")

def test(modelPathPlayer="pong_model_100.pth", modelPathOpponent="pong_opponent_model_100.pth", humanMode=False):
    env = PongEnv()
    playerAgent = DQNAgent(stateDim=5, actionDim=3)  # Actions: 0=up, 1=down, 2=stay
    opponentAgent = DQNAgent(stateDim=5, actionDim=3)  # For right paddle 

    try:
        playerAgent.load(modelPathPlayer)
        opponentAgent.load(modelPathOpponent)
        print(f"Models loaded from {modelPathPlayer} and {modelPathOpponent}")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        env.close()
        return

    # Disable exploration (set epsilon to 0 for deterministic actions)
    playerAgent.epsilon = 0.0
    opponentAgent.epsilon = 0.0

    clock = pygame.time.Clock()
    episodes = 10  # Number of test episodes
    font = pygame.font.Font(None, 74)
    playerScore, oppScore = 0, 0

    for episode in range(episodes):
        env.reset()  # Reset entire environment (ball, paddles, scores)
        playerState = env.getState(perspective='player')
        opponentState = env.getState(perspective='opponent')
        playerTotalReward = 0
        opponentTotalReward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        humanMode = not humanMode

            # Player action
            if humanMode:
                keys = pygame.key.get_pressed()
                playerAction = 2  # Stay
                if keys[pygame.K_UP]:
                    playerAction = 0  # Up
                if keys[pygame.K_DOWN]:
                    playerAction = 1  # Down
            else:
                playerAction = playerAgent.act(playerState)

            # Opponent action
            opponentAction = opponentAgent.act(opponentState)

            # Apply actions
            if playerAction == 0:
                env.movePaddle(env.player, up=True)
            elif playerAction == 1:
                env.movePaddle(env.player, up=False)
            if opponentAction == 0:
                env.movePaddle(env.opponent, up=True)
            elif opponentAction == 1:
                env.movePaddle(env.opponent, up=False)

            # Update game state
            prevPlayerScore = env.playerScore
            prevOpponentScore = env.opponentScore
            env.moveBall()
            player_reward = 1 if env.playerScore > prevPlayerScore else -1 if env.opponentScore > prevOpponentScore else 0
            opponent_reward = -player_reward
            player_next_state = env.getState(perspective='player')
            opponent_next_state = env.getState(perspective='opponent')
            done = env.playerScore >= 10 or env.opponentScore >= 10

            playerState = player_next_state
            opponentState = opponent_next_state
            playerTotalReward += player_reward
            opponentTotalReward += opponent_reward

            # Render and control frame rate
            env.render(episode)
            clock.tick(60)

            if done:
                print(f"Test Episode {episode + 1}, Player Score: {env.playerScore}, Opponent Score: {env.opponentScore}")
                playerScore += 1 if env.playerScore > env.opponentScore else 0
                oppScore += 1 if env.opponentScore > env.playerScore else 0
                print(f"Score: {playerScore} - {oppScore}")
                break



        # Display game over screen and wait for user input
        env.render(episode=episode + 1)  # Render final state
        gameOverText = font.render("Game Over! Press SPACE to continue", False, env.WHITE)
        env.screen.blit(gameOverText, (env.WIDTH // 2 - gameOverText.get_width() // 2, env.HEIGHT // 2))
        pygame.display.flip()

        # Wait for SPACE key or QUIT event
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    if event.key == pygame.K_h:
                        human_mode = not human_mode

        # Clear event queue to prevent residual inputs
        pygame.event.clear()


if __name__ == "__main__":
    #train()
    #train(modelPathPlayer="./models/v2/pong_model_600_v2.pth", modelPathOpponent="./models/v2/pong_opponent_model_600_v2.pth", startFrom=601)
    print("Testing 1000 episodes V2 model vs 30 episodes V3 model")
    test(modelPathPlayer="./models/v2/pong_model_1000_v2.pth", modelPathOpponent="./models/v3/pong_model_90_v3.pth", humanMode=False)