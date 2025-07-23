# environment.py
import pygame
import random
import numpy as np

class PongEnv:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.PADDLEWIDTH, self.PADDLEHEIGHT = 15, 90
        self.BALLSIZE = 15
        self.PADDLESPEED = 7
        self.BALLSPEED = 7
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Pong")

        self.reset()

        self.font = pygame.font.Font(None, 74)

    def reset(self):
        """Reset the environment: ball, paddles, and scores."""
        # Reset paddles to center
        self.player = pygame.Rect(50, self.HEIGHT // 2 - self.PADDLEHEIGHT // 2, self.PADDLEWIDTH, self.PADDLEHEIGHT)
        self.opponent = pygame.Rect(self.WIDTH - 50 - self.PADDLEWIDTH, self.HEIGHT // 2 - self.PADDLEHEIGHT // 2, self.PADDLEWIDTH, self.PADDLEHEIGHT)
        # Reset ball
        self.ball = pygame.Rect(self.WIDTH // 2 - self.BALLSIZE // 2, self.HEIGHT // 2 - self.BALLSIZE // 2, self.BALLSIZE, self.BALLSIZE)
        
        self.playerScore = 0
        self.opponentScore = 0
        
        #self.ballSpeed = [self.BALLSPEED * random.choice((1, -1)), self.BALLSPEED * random.choice((1, -1))]
        self.resetBall()
        # Reset scores
        

    def movePaddle(self, paddle, up=True):
        """Move the paddle up or down within screen bounds."""
        if up and paddle.top > 0:
            paddle.y -= self.PADDLESPEED
        if not up and paddle.bottom < self.HEIGHT:
            paddle.y += self.PADDLESPEED

    def moveBall(self):
        """Update ball position and handle collisions."""
        self.ball.x += self.ballSpeed[0]
        self.ball.y += self.ballSpeed[1]

        playerHit = False
        opponentHit = False
        # Bounce off top/bottom walls
        if self.ball.top <= 0 or self.ball.bottom >= self.HEIGHT:
            self.ballSpeed[1] = -self.ballSpeed[1]

        # Paddle collisions
        if self.ball.colliderect(self.player) and self.ballSpeed[0] < 0:
            self.ballSpeed[0] = -self.ballSpeed[0]
            playerHit = True
            #print('Player hit the ball!')
        if self.ball.colliderect(self.opponent) and self.ballSpeed[0] > 0:
            self.ballSpeed[0] = -self.ballSpeed[0]
            opponentHit = True
            #print('Opponent hit the ball!')

        # Scoring and reset
        if self.ball.left <= 0:
            self.opponentScore += 1
            #self.resetPaddles()
            self.resetBall()
            
        if self.ball.right >= self.WIDTH:
            self.playerScore += 1
            #self.resetPaddles()
            self.resetBall()
        
        return playerHit, opponentHit

    def step(self, player_action, opponent_action):
        """Apply actions, update state, and return next states, rewards, and done."""
        # Apply actions
        if player_action == 0:
            self.movePaddle(self.player, up=True)
        elif player_action == 1:
            self.movePaddle(self.player, up=False)
        if opponent_action == 0:
            self.movePaddle(self.opponent, up=True)
        elif opponent_action == 1:
            self.movePaddle(self.opponent, up=False)

        # Update game state
        prevPlayerScore = self.playerScore
        prevOpponentScore = self.opponentScore
        playerHit, opponentHit = self.moveBall()

        # Calculate rewards
        playerReward = 1 if self.playerScore > prevPlayerScore else -1 if self.opponentScore > prevOpponentScore else 0
        #if playerHit:
        #    playerReward += 0.1
        opponentReward = -1 if self.playerScore > prevPlayerScore else 1 if self.opponentScore > prevOpponentScore else 0
        #if opponentHit:
        #    opponentReward += 0.1

        # Get next states
        playerNextState = self.getState(perspective='player')
        opponentNextSTate = self.getState(perspective='opponent')

        # Check if episode is done
        done = self.playerScore >= 10 or self.opponentScore >= 10

        return playerNextState, opponentNextSTate, playerReward, opponentReward, done

    def resetPaddles(self):
        """Reset paddles to their initial positions."""
        self.player.y = self.HEIGHT // 2 - self.PADDLEHEIGHT // 2
        self.opponent.y = self.HEIGHT // 2 - self.PADDLEHEIGHT // 2

    def resetBall(self):
        """Reset ball to center with random direction."""
        self.resetPaddles()
        self.ball.center = (self.WIDTH // 2, self.HEIGHT // 2)
        if self.playerScore == 0 and self.opponentScore == 0:
            # Horizontal trajectory at game start
            #print("Start ball with horizontal trajectory")
            self.ballSpeed = [self.BALLSPEED * random.choice((1, -1)), 0]
        else:
            self.ballSpeed = [self.BALLSPEED * random.choice((1, -1)), self.BALLSPEED * random.choice((1, -1))]

    def getState(self, perspective='player'):
        """Return normalized game state for the agent."""
        if perspective == 'player':
            return np.array([
                self.ball.x / self.WIDTH,
                self.ball.y / self.HEIGHT,
                self.ballSpeed[0] / self.BALLSPEED,
                self.ballSpeed[1] / self.BALLSPEED,
                self.player.y / self.HEIGHT,
                #self.opponent.y / self.HEIGHT
            ])
        else:  # Opponent perspective (right paddle)
            return np.array([
                (self.WIDTH - self.ball.x) / self.WIDTH,  # Mirrored ball x
                self.ball.y / self.HEIGHT,                # Ball y
                -self.ballSpeed[0] / self.BALLSPEED,    # Mirrored ball velocity x
                self.ballSpeed[1] / self.BALLSPEED,     # Ball velocity y
                self.opponent.y / self.HEIGHT,            # Opponent paddle y
                #self.player.y / self.HEIGHT               # Player paddle y
            ])


    def render(self, episode):
        """Draw the game elements on the screen."""
        self.screen.fill(self.BLACK)
        pygame.draw.rect(self.screen, self.WHITE, self.player)
        pygame.draw.rect(self.screen, self.WHITE, self.opponent)
        pygame.draw.ellipse(self.screen, self.WHITE, self.ball)
        pygame.draw.aaline(self.screen, self.WHITE, (self.WIDTH // 2, 0), (self.WIDTH // 2, self.HEIGHT))
        playerText = self.font.render(str(self.playerScore), False, self.WHITE)
        opponentText = self.font.render(str(self.opponentScore), False, self.WHITE)
        episodeText = self.font.render(f"Episode: {episode}", True, self.RED)
        self.screen.blit(playerText, (self.WIDTH // 4, 20))
        self.screen.blit(episodeText, (300, 550))
        self.screen.blit(opponentText, (3 * self.WIDTH // 4, 20))
        pygame.display.flip()

    def close(self):
        """Clean up and close the game."""
        pygame.quit()