## AI for chutes.xyz
AI for chutes.xyz using a deep q-learning model.</br></br>
Trains on AWS SageMaker, convertModel converts keras model to tf_js model and the static directory is uploaded to and read by the server for chutes.xyz.

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
Conclusion: The AI knows how to put down parachutes around bases but it doesn't bomb, it doesn't upgrade parachutes until all squares are taken and it doesn't move units once it upgrades to them.</br></br>
ai-v-ai-version-2</br>
<ol>
  <li>First the agents are trained only with base-related intermediate rewards in a randomly generated map with only tanks. These agents are saved and loaded again and again into a map with only soldiers and spies, then a map with everything but parachutes, and finally the original game map. There are no rewards for winning until the original game environment.</li>
</ol>
ai-v-ai-version-3</br>
<ol>
  <li>New agents were trained with only intermediate rewards: lowering opponent's bases, winning bases, moving complex pieces and a ratio of moves predicted that were legal moves to those that were not. The last reward though I thought would encourage the model to fit around legality but instead the AI trained to just play the same move over and over again to maximize the ratio for this single reward and did not consider the other rewards.
  </li>
</ol>
