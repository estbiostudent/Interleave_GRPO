# Interleave_GRPO
The interleave project measures how well LLMs track and maintain latent variables. Models interleave n texts one word at a time, cycling between them, over increasing lengths. This is a task where the surface operation is trivial copying, so failure isolates state maintenance. Frontier models handle two streams without difficulty but begin breaking down at three. Smaller models are trained on the task with GRPO, using performance dynamics across successive rounds to probe architectural limits.

## Overview

A model's capacity to handle complex tasks is dependent on its ability to create and maintain ideas that are not explicitly present in the input. In the AI field, these non-explicit ideas are called latent states or variables: latent because they are never externalized explicitly and states or variables because they have specific values that apply to the unique situation at hand. The model must maintain these latent states and variables despite having nothing like explicit memory within which to store them. This is similar to any particular thought or connection that we might hold in our mind while thinking through a problem. 

These latent states and variables can take many forms, including, for example, intermediate numbers in a series of computations, a name or a particular attribute associated with a name, or where the model is in the sequence of steps of a recipe or in the middle of solving a complex math problem. “Daisy is here. She is wearing a red dress” gives us the latent variable that Daisy has the property of wearing a red dress, despite the explicit relationship never having been directly stated. 400 x 37 will include the idea of 400 x 30, and since we know that 4 x 3 is 12, and there are three 0s involved, we can know that the answer will be greater than, and in some way dependent on, the number 12,000, without having had to explicitly write down any of these facts.

Measuring a model’s ability to maintain and manipulate these latent variables is challenging, as by definition they are latent, not stored anywhere explicitly. Additionally, most tasks that challenge a model’s abilities around latent states, do so in the context of additional complexity that is independent of the proper interaction with and maintenance of the latent states and variables. 

The interleave project measures a model’s ability to maintain and interact with latent states and variables directly. In its simplest form, models are asked to interleave multiple texts, alternating between them one word at a time, over longer and longer contexts. The model’s attempt at recreating the interleaved version of the two texts is compared to a programmatically constructed ground truth, and a score is determined by calculating how close the two versions are, using a modified Needleman-Wunsch, the algorithm that used to measure the similarity of two different strands of DNA.

Preliminary data indicate that current frontier models successfully interleave two source texts beyond 10,000 total words with minimal or no errors. With three texts, degradation appears at roughly two orders of magnitude fewer words. Small models trained on the task show a transition from near-perfect to complete failure as length increases, with successful interleaving extending well past the lengths used in training.




## Hypotheses

Hypothesis 1: The ability to interleave the execution and output of two, or more, different tasks is a measurement of a model’s ability to maintain latent states.

Hypothesis 2: An architecture has a limit of both depth (length of text, steps in a calculation, etc) and breadth (number of simultaneous worlds/processes) that it can track.

Hypothesis 2a: This ability can be trained.

Hypothesis 2b: The architectures limit can be mapped

Hypothesis 2b1: The length of texts (or sequence of calculations) that the model can recite successfully for a given training regimen is described by the logistic function

Hypothesis 2b2: the rate of transition of success for a given depth is an indicator of approaching the boundary: shallower curves indicate training is still possible, steeper curves indicate approach of architectural capacity.

Hypothesis 3: Training on interleaving can lead to formation of independent thinking structures.

Hypothesis 4: The wording of the prompts during training affects the nature of the structure: “processes” vs “pointers” vs “worlds”




