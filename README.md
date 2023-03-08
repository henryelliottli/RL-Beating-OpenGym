# RL-Beating-OpenGym

How to run:
Ensure all packages have been installed.


To train an image based model and output the metrics run `python main.py <Game Name>`  
  ex: `python main.py Breakout`  
  
Then once the model has been fully trained and saved run   
  `python test.py <Game Name>`  
in order to render the images and watch the model play the game.   
  
Lastly, given the trained model run TSNE by following the example in `example_tsne.ipynb` in the tsne folder   

## TSNE Example

![My Image](tsne/breakout-tsne.png)
- TSNE clustering of RAM based model’s best action(labels) Q value pairs in the game

## Examples

# Mario N64
![My Image](examples/mario-run.gif)

# Boxing Atari
![My Image](examples/boxing-game.gif)
