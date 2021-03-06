## AI for chutes.xyz
AI for chutes.xyz using a deep q-learning model.</br></br>
Trains on AWS SageMaker, convertModel converts keras model to tf_js model and the static directory is uploaded to and read by the server for chutes.xyz.

### Updates
version 1
<ol>
  <li>Created an AI vs Random model with small rewards for capturing bases and a big reward for winning the game. Epsilon decayed every game.</li>
  <li>Made AI vs AI because AI won a lot of games against random but it only seemed to put parachutes down</li>
  <li>Changed epsilon to increase if an agent lost over the past 50 games so that it can try new tactics against the successful bot</li>
  <li>Epsilon starts at 1, increases if AI loses and decreases if AI wins but never goes over 1</li>
  <li>Changed small rewards to only be given when the opponent has less bases after the turn to increase aggression</li>
  <li>Changed model structure from 34(number of squares on the board + 1 for bias)->100->100(hidden layers)->546(number of outputs) to 34->9(number of pieces)->204(max number of possible moves for a piece [tank move])->546 so that the model decides what piece to move and then figures out how to move it.</li>
</ol>
Conclusion: The AI knows how to put down parachutes around bases but it doesn't bomb, it doesn't upgrade parachutes until all squares are taken and it doesn't move units once it upgrades to them.</br></br>
version 2</br>
<ol>
  <li>First the agents are trained only with base-related intermediate rewards in a randomly generated map with only tanks. These agents are saved and loaded again and again into a map with only soldiers and spies, then a map with everything but parachutes, and finally the original game map. There are no rewards for winning until the original game environment.</li>
  <li>If they do not train a lot, they play badly because they do not move a lot. If they train a lot, the old values get overwritten. AI still does not prefer to move often. So it might be impossible to ever lower the complexity around moving pieces and make the training natural.</li>
  <li>Added a reward with the ratio of moves predicted that were legal moves to those that were not (though I thought this would encourage the model to fit around legality, instead the AI trained to just play the same move over and over again to maximize the ratio for this single reward and did not consider the other rewards). So it seems that encouraging the agent to learn to play moves legally and precisely is difficult. So this is the end of this version.</li>
</ol>
Conclusion: This version still didn't encourage the AI to move more. I will need to artificially reward the AI to move pieces to balance out the negative rewards associated with them in the beginning due to bad moves. Both environment changes and rewards based on accuracy were not able to make the AI's move-piece moves any better.</br></br>
version 3</br>
<ol>
  <li>New agents were trained with only intermediate rewards: lowering opponent's bases, winning bases and moving complex pieces (at 0.3 for tanks because they have around triple the number of possible moves compared to soldiers and spies). The latter was done by imposing a tax for standing advanced pieces.
  </li>
  <li>Bombing is now done algorithmically before the moves from the neural network are fed. If a bombing opportunity is available, a bomb is set on that square.</li>
  <li>After the new agents are trained, they are trained with both p1 and p2 adopting p2's model but p1 playing without any random exploratory moves while p2 starts at epsilon=1. Through this, I aim to fix p2's strategy's weaknesses as it plays against itself as p1.</li>
</ol>
Conclusion: This version is playable. Soldiers are often used improperly and spies are never used but parachutes and tanks are played well.
