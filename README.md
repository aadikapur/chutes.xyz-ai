## AI for chutes.xyz
AI for chutes.xyz using a deep q-learning model.</br></br>
Trains on AWS SageMaker, convertModel converts keras model to tf_js model and the static file on the model is uploaded to and read by the server for chutes.xyz.

### Updates
ai-v-ai-version-1
<ol>
  <li>Created an AI vs Random model with small rewards for capturing bases and a big reward for winning the game. Epsilon decayed every game.</li>
  <li>Made AI vs AI because AI won a lot of games against random but it only seemed to put parachutes down</li>
  <li>Changed epsilon to increase if an agent lost over the past 50 games so that it can try new tactics against the successful bot</li>
  <li>Epsilon starts at 1, increases if AI loses and decreases if AI wins but never goes over 1</li>
  <li>Changed small rewards to only be given when the opponent has less bases after the turn to increase aggression</li>
  <li>Changed model structure from 34(number of squares on the board + 1 for bias)->100->100(hidden layers)->546(number of outputs) to 34->9(number of pieces)->204(max number of possible moves for a piece [tank move])->546 so that the model decides what piece to move and then figures out how to move it.</li>
</ol>
ai-v-ai-version-2
<ol>
  <li>Built the game environment to only include tanks at start and changed the legal moves to only moving tanks so that the agents can learn how to move tanks first. Then I add everything but parachutes or the ability to put parachutes down with other moves allowed. Only base-related intermediate rewards are available and the reward to win is not.</li>
</ol>
