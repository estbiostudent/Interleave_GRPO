# Interleave_GRPO
The interleave project uses the task of interleaving multiple texts, of increasing length, one word at a time, to measure a model's ability to track and maintain latent variables and states.

## Overview

A model's capacity to handle complex tasks is dependent on its ability to create and maintain ideas that are not explicitly present in the input. It must maintain these ideas despite having nothing like explicit memory within which to store them. These ideas can take many forms, including, for example, intermediate numbers in a series of computations, a name or particular attribute associated with a name, or where the model is in the sequence of steps of a recipe or in the middle of solving a complex math problem. 

We can think of them as any particular thought that we might hold in our mind while thinking through a problem. “Daisy is here. She is wearing a red dress” gives us the latent variable that Daisy has the property of wearing a red dress, despite this relationship never having been directly stated. 400 x 37 will include the idea of 400 x 30, and since we know that 4 x 3 is 12, and there are three 0s involved, we can know that the answer will be greater than, and in some way dependent on, the number 12,000, without having had to explicitly write down any of these facts.    

Because these values are not explicitly written down anywhere, we call them latent, and because they change based on the situation, both in kind and value, it is convenient to call them variables. Measuring a model’s ability to maintain and manipulate these latent variables is challenging, as by definition they are latent. It’s also difficult to tease apart measuring the ability of a model to maintain latent variables from a more general notion of complexity, since it is often the case that solving “complex” problems also requires maintaining fidelity of latent variables. The interleave project measures this ability more directly. Models are asked to interleave multiple texts, alternating between them one word at a time, over longer and longer contexts.



## Hypotheses

Hypothesis 1: The ability to interleave the execution and output of two, or more, different tasks is a measurement of a model’s ability to maintain latent states.

Hypothesis 2: An architecture has a limit of both depth (length of text, steps in a calculation, etc) and breadth (number of simultaneous worlds/processes) that it can track.

Hypothesis 2a: This ability can be trained.

Hypothesis 2b: The architectures limit can be mapped

Hypothesis 2b1: The length of texts (or sequence of calculations) that the model can recite successfully for a given training regimen is described by the logistic function

Hypothesis 2b2: the rate of transition of success for a given depth is an indicator of approaching the boundary: shallower curves indicate training is still possible, steeper curves indicate approach of architectural capacity.

Hypothesis 3: Training on interleaving can lead to formation of independent thinking structures.

Hypothesis 4: The wording of the prompts during training affects the nature of the structure: “processes” vs “pointers” vs “worlds”




