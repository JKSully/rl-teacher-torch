# Developing notes (active tally):

* `CatTensors` transform to take in observation, action into what I need.
* Transform is my current path [Example](https://pytorch.org/rl/stable/tutorials/pendulum.html#pendulum-tuto) has been really helpful
* `torchrl.record.VideoRecorder`


# Thoughts:
What if I create a logger that inputs into `VideoRecorder` that uploads to the database for comparison collections? Then I need to somehow check if there are comparible sequences for use in network.